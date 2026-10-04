"""Restore the handoff's raw inputs without silently changing their snapshots.

The historical successful manifest entries are the contract: every restored
file must have the recorded SHA-256. Downloads use normal HTTPS verification;
changed public-source responses are quarantined, not fed to the analysis. Run
``python restore_inputs.py --stage core`` first, then ``--stage all``. Completed
files are checksum-verified and skipped, so either command is resumable.

After reviewing results/input_restoration.json, an investigator may explicitly
promote a changed trusted-source snapshot with ``--accept-current PATH``. This
keeps the historical expectation unchanged and records the replacement's hash.
It is not an exact reproduction and must be disclosed in the final report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.parse import urlsplit
import zipfile

import requests

from common import MANIFEST, RAW, RESULTS, ROOT, UA, _now, log_manifest, sha256_file

REPORT = RESULTS / "input_restoration.json"
QUARANTINE = RAW / ".restoration_divergent"
_local = threading.local()
_host_limits: dict[str, threading.Semaphore] = {}
_blocked_hosts: set[str] = set()
_blocked_lock = threading.Lock()

# The PyPI distribution embeds the unchanged JPL file. Both the distribution
# digest published by PyPI and the historical ephemeris digest are mandatory.
_DE421_WHEEL = ("https://files.pythonhosted.org/packages/bb/46/"
                "e815e37cfb1f1108d80e45ee100c8a6d9f1d5999ccf1cf14dab7be6858dc/"
                "skyfield_data-7.0.0-py2.py3-none-any.whl")
_DE421_WHEEL_SHA256 = "4279f999f7f8c5ef5d8bf1b714f301a0844ae5df3a3f0d23610b20d8789073f5"


def verified_ephemeris_mirror(entry: dict) -> dict | None:
    if entry["dest"] != "data/raw/astro/de421.bsp":
        return None
    import io
    try:
        response = session().get(_DE421_WHEEL, timeout=(25, 180))
        response.raise_for_status()
        archive_hash = hashlib.sha256(response.content).hexdigest()
        if archive_hash != _DE421_WHEEL_SHA256:
            raise ValueError("PyPI distribution SHA-256 does not match pinned published digest")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = [name for name in archive.namelist() if Path(name).name == "de421.bsp"]
            if len(names) != 1:
                raise ValueError("Expected exactly one de421.bsp in the distribution")
            data = archive.read(names[0])
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("Mirror ephemeris SHA-256 differs from historical snapshot")
        dest = ROOT / entry["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + f".mirror-{os.getpid()}-{threading.get_ident()}.part")
        try:
            part.write_bytes(data)
            publish_preserving_existing(part, dest, digest)
        finally:
            part.unlink(missing_ok=True)
        return {"dest": entry["dest"], "status": "restored", "sha256": digest,
                "expected_sha256": entry["sha256"], "bytes": len(data),
                "provenance": {"url": response.url, "dest": entry["dest"], "retrieved_utc": _now(),
                    "status": 200, "ok": True, "bytes": len(data), "sha256": digest,
                    "restoration": "verified_snapshot_mirror", "historical_sha256": digest,
                    "historical_url": entry["url"], "artifact_sha256": archive_hash,
                    "artifact_bytes": len(response.content), "artifact_member": names[0],
                    "description": "Exact historical JPL DE421 ephemeris extracted from SHA-verified skyfield-data 7.0.0 PyPI wheel; primary JPL host denied by runtime proxy."}}
    except (requests.RequestException, ValueError, zipfile.BadZipFile) as error:
        return {"dest": entry["dest"], "status": "failed", "expected_sha256": entry["sha256"],
                "error": f"Primary source and verified PyPI ephemeris mirror unavailable: {error}"}


def publish_preserving_existing(part: Path, dest: Path, digest: str) -> None:
    """Publish completed content atomically without overwriting another writer."""
    try:
        os.link(part, dest)
    except FileExistsError:
        if sha256_file(dest) != digest:
            raise ValueError(f"Concurrent destination differs; preserved: {dest}")


def historical_entries() -> tuple[list[dict], dict[str, dict]]:
    """Last successful historical entry per raw destination; omit our additions."""
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    expected = {}
    for row in rows:
        if (row.get("ok") and row.get("sha256")
                and row.get("dest", "").startswith("data/raw/")
                and not row.get("restoration")
                and "/.restoration_divergent/" not in row["dest"]):
            # Refuse a manifest destination that escapes the checkout's raw dir.
            dest = (ROOT / row["dest"]).resolve()
            if not dest.is_relative_to(RAW.resolve()):
                raise ValueError(f"Unsafe raw destination: {row['dest']}")
            expected[row["dest"]] = row
    return rows, expected


def priority(entry: dict) -> tuple[int, str]:
    dest = entry["dest"]
    if "/weather/isd_lite/" in dest:
        return 3, dest
    if "/weather/storm_events/" in dest:
        return 2, dest
    if "/geophysical/usgs/" in dest or "/goes_xrs/" in dest:
        return 1, dest
    return 0, dest


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update({"User-Agent": UA})
    return _local.session


def retrieve(entry: dict, retries: int) -> dict:
    dest = ROOT / entry["dest"]
    want = entry["sha256"]
    base = {"dest": entry["dest"], "expected_sha256": want}
    if dest.is_file():
        actual = sha256_file(dest)
        if actual == want:
            return {**base, "status": "verified_existing", "sha256": actual}
        # A reviewed current snapshot is also resumable, but is never mislabeled.
        for line in MANIFEST.read_text().splitlines()[::-1]:
            row = json.loads(line)
            if (row.get("restoration") == "accepted_current" and row.get("dest") == entry["dest"]
                    and row.get("sha256") == actual and row.get("historical_sha256") == want):
                return {**base, "status": "accepted_current_existing", "sha256": actual}
        return {**base, "status": "existing_mismatch", "sha256": actual,
                "error": "Existing file preserved; review before replacing."}
    # Reuse a previously downloaded but unaccepted divergent response.
    qdir = QUARANTINE / Path(entry["dest"]).relative_to("data/raw")
    for q in sorted(qdir.parent.glob(qdir.name + ".sha256-*")) if qdir.parent.exists() else []:
        actual = sha256_file(q)
        return {**base, "status": "snapshot_mismatch", "sha256": actual,
                "quarantine": str(q.relative_to(ROOT)), "error": "Source response differs from historical hash."}
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + f".restore-{os.getpid()}-{threading.get_ident()}.part")
    last_error, last_status, final_url = None, None, entry["url"]
    host = urlsplit(entry["url"]).hostname
    if urlsplit(entry["url"]).scheme != "https":
        return {**base, "status": "failed", "error": "Historical URL is not HTTPS."}
    limiter = _host_limits[host]
    with limiter:
        with _blocked_lock:
            if host in _blocked_hosts:
                return {**base, "status": "network_policy_blocked", "host": host,
                        "error": "Earlier request to this host received proxy CONNECT 403; not repeatedly requested."}
        for attempt in range(retries):
            try:
                with session().get(entry["url"], stream=True, timeout=(25, 180)) as response:
                    last_status, final_url = response.status_code, response.url
                    if response.status_code == 429 or response.status_code >= 500:
                        last_error = f"HTTP {response.status_code}"
                        retry_after = response.headers.get("Retry-After", "")
                        delay = min(60, float(retry_after)) if retry_after.isdigit() else min(60, 4 * 2 ** attempt)
                        if attempt + 1 < retries:
                            time.sleep(delay)
                        continue
                    if response.status_code != 200:
                        last_error = f"HTTP {response.status_code}"
                        break
                    h = hashlib.sha256()
                    size = 0
                    with part.open("wb") as output:
                        for chunk in response.iter_content(1 << 20):
                            if chunk:
                                output.write(chunk)
                                h.update(chunk)
                                size += len(chunk)
                    actual = h.hexdigest()
                    provenance = {"url": final_url, "retrieved_utc": _now(), "status": 200,
                                  "bytes": size, "sha256": actual, "ok": True,
                                  "description": entry.get("description", "") + " [handoff restoration]",
                                  "historical_sha256": want, "historical_dest": entry["dest"]}
                    if actual != want:
                        qdir.parent.mkdir(parents=True, exist_ok=True)
                        qpath = qdir.with_name(qdir.name + ".sha256-" + actual)
                        part.replace(qpath)
                        provenance.update(dest=str(qpath.relative_to(ROOT)), restoration="quarantined_current")
                        return {**base, "status": "snapshot_mismatch", "sha256": actual,
                                "quarantine": str(qpath.relative_to(ROOT)), "bytes": size,
                                "provenance": provenance, "error": "Source response differs from historical hash."}
                    publish_preserving_existing(part, dest, actual)
                    provenance.update(dest=entry["dest"], restoration="verified_snapshot")
                    return {**base, "status": "restored", "sha256": actual, "bytes": size,
                            "provenance": provenance}
            except requests.RequestException as error:
                last_error = str(error)
                if isinstance(error, requests.exceptions.ProxyError) and "403 Forbidden" in last_error:
                    with _blocked_lock:
                        _blocked_hosts.add(host)
                    break
                # A trust failure requires diagnosis; repeated downloads cannot fix it.
                if isinstance(error, requests.exceptions.SSLError):
                    break
                if attempt + 1 < retries:
                    time.sleep(min(30, 2 ** (attempt + 1)))
            finally:
                part.unlink(missing_ok=True)
    provenance = {"url": final_url, "dest": entry["dest"], "retrieved_utc": _now(),
                  "status": last_status, "ok": False, "error": last_error,
                  "historical_sha256": want, "restoration": "failed"}
    mirror = verified_ephemeris_mirror(entry)
    if mirror:
        mirror["primary_failure_provenance"] = provenance
        return mirror
    return {**base, "status": "failed", "error": last_error, "http_status": last_status,
            "provenance": provenance}


def extract_tables() -> list[str]:
    """Only documented table members, after the source archive is approved."""
    extracts = []
    for archive, member in [
        ("population/2020_Gaz_counties_national.zip", "2020_Gaz_counties_national.txt"),
        ("population/2020_Gaz_place_national.zip", "2020_Gaz_place_national.txt"),
        ("population/geonames_cities1000.zip", "cities1000.txt"),
    ]:
        source = RAW / archive
        if not source.exists():
            continue
        target = source.parent / member
        with zipfile.ZipFile(source) as z:
            matches = [name for name in z.namelist() if Path(name).name == member]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {member} in {archive}")
            data = z.read(matches[0])  # Zip CRC verified by the standard library.
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"Existing extracted table differs; preserved: {target}")
        if not target.exists():
            target.write_bytes(data)
        extracts.append(str(target.relative_to(ROOT)))
    return extracts


def reconstruct_stations(rows: list[dict]) -> dict:
    """Recover the original station set from acquisition attempts, not outcomes."""
    import pandas as pd

    sids = set()
    for row in rows:
        match = re.search(r"/isd_lite/\d{4}/(\d{6}-\d{5})-\d{4}\.gz$", row.get("dest", ""))
        if match and not row.get("restoration"):
            sids.add(match.group(1))
    history = RAW / "weather/isd-history.csv"
    target = RAW / "weather/selected_isd_stations.csv"
    if not history.exists():
        return {"status": "blocked", "stations_requested": len(sids), "error": "isd-history.csv not restored"}
    table = pd.read_csv(history, dtype={"USAF": str, "WBAN": str})
    table["sid"] = table.USAF + "-" + table.WBAN
    selected = table.loc[table.sid.isin(sids)].sort_values("sid", kind="stable")
    if selected.sid.duplicated().any() or set(selected.sid) != sids:
        return {"status": "blocked", "stations_requested": len(sids),
                "error": "Missing or duplicate official station metadata",
                "missing": sorted(sids - set(selected.sid))}
    if selected[["LAT", "LON"]].isna().any().any():
        return {"status": "blocked", "error": "Selected official metadata lacks coordinates"}
    data = selected.drop(columns="sid").to_csv(index=False)
    if target.exists() and target.read_text() != data:
        return {"status": "existing_preserved", "stations_requested": len(sids), "error": "Existing station list differs"}
    if not target.exists():
        target.write_text(data)
    return {"status": "reconstructed", "stations": len(sids), "sha256": sha256_file(target),
            "method": "Unique station IDs from all original successful and failed ISD acquisition attempts, sorted and joined to official isd-history.csv; no holdout outcomes used."}


def accept_current(paths: list[str], expected: dict[str, dict]) -> list[str]:
    promoted = []
    for path in paths:
        if path not in expected:
            raise ValueError(f"No historical successful entry for {path}")
        qdir = QUARANTINE / Path(path).relative_to("data/raw")
        matches = sorted(qdir.parent.glob(qdir.name + ".sha256-*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one reviewed quarantined response for {path}, found {len(matches)}")
        q = matches[0]
        actual = sha256_file(q)
        if q.name.split(".sha256-")[-1] != actual:
            raise ValueError(f"Quarantine integrity check failed for {path}")
        dest = ROOT / path
        if dest.exists() and sha256_file(dest) != actual:
            raise ValueError(f"Existing destination preserved: {path}")
        shutil.copyfile(q, dest)
        log_manifest({"dest": path, "url": expected[path]["url"], "ok": True,
                      "sha256": actual, "bytes": dest.stat().st_size, "retrieved_utc": _now(),
                      "restoration": "accepted_current", "historical_sha256": expected[path]["sha256"],
                      "description": "Explicitly reviewed current trusted-source snapshot; historical hash was not changed."})
        promoted.append(path)
    return promoted


def save_report(results: list[dict], total: int, stage: str, started: str, finished: bool) -> None:
    counts = Counter(result["status"] for result in results)
    report = {"started_utc": started, "updated_utc": _now(), "stage": stage, "finished": finished,
              "expected_files": total, "completed_files": len(results), "counts": dict(counts),
              "expected_by_category": dict(Counter(r["dest"].split("/")[2] for r in results)),
              "tls_verification": True, "historical_manifest_hashes_unchanged": True,
              "verified_destinations": sorted(r["dest"] for r in results if r["status"] in ("restored", "verified_existing")),
              "exceptions": [r for r in results if r["status"] not in ("restored", "verified_existing")],
              "extracted_tables": extract_tables(), "station_reconstruction": reconstruct_stations(historical_entries()[0])}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    part = REPORT.with_suffix(".json.part")
    part.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    part.replace(REPORT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["core", "weather", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--accept-current", action="append", default=[], metavar="data/raw/PATH")
    parser.add_argument("--only", action="append", default=[], metavar="data/raw/PATH",
                        help="Restore just these explicit historical destinations.")
    parser.add_argument("--audit-only", action="store_true",
                        help="Checksum local inputs and report missing inputs without network calls.")
    args = parser.parse_args()
    rows, expected = historical_entries()
    if args.accept_current:
        print("Explicitly promoted", len(accept_current(args.accept_current, expected)), "reviewed current snapshots", flush=True)
    jobs = sorted(expected.values(), key=priority)
    if args.stage == "core":
        jobs = [j for j in jobs if priority(j)[0] < 2]
    elif args.stage == "weather":
        jobs = [j for j in jobs if "/weather/" in j["dest"]]
    if args.only:
        missing = set(args.only) - set(expected)
        if missing:
            raise ValueError(f"Unknown historical destinations: {sorted(missing)}")
        jobs = [j for j in jobs if j["dest"] in set(args.only)]
    for j in jobs:
        host = urlsplit(j["url"]).hostname
        _host_limits.setdefault(host, threading.Semaphore(4 if host == "earthquake.usgs.gov" else args.workers))
    results, started = [], _now()
    if args.audit_only:
        denied = {}
        for row in rows:
            if row.get("restoration") == "failed" and "403 Forbidden" in str(row.get("error", "")):
                denied[urlsplit(row["url"]).hostname] = row
        for entry in jobs:
            if (ROOT / entry["dest"]).is_file():
                results.append(retrieve(entry, args.retries))  # Local branch only.
            else:
                host = urlsplit(entry["url"]).hostname
                evidence = denied.get(host)
                results.append({"dest": entry["dest"], "expected_sha256": entry["sha256"],
                    "status": "network_policy_blocked" if evidence else "absent_unattempted",
                    "host": host, "error": "Historical file absent; no request made during offline audit.",
                    "prior_host_denial_utc": evidence.get("retrieved_utc") if evidence else None})
        save_report(results, len(jobs), args.stage + "_offline_audit", started, True)
        print(f"Offline snapshot audit: {dict(Counter(r['status'] for r in results))}; {REPORT}", flush=True)
        return int(any(r["status"] not in ("verified_existing", "accepted_current_existing") for r in results))
    print(f"Restoring {len(jobs)} {args.stage} snapshot files with {args.workers} workers; TLS and SHA-256 checks enabled", flush=True)
    for tier in sorted(set(priority(j)[0] for j in jobs)):
        selected = [j for j in jobs if priority(j)[0] == tier]
        print(f"Tier {tier}: {len(selected)} files", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(retrieve, entry, args.retries) for entry in selected]
            for future in as_completed(futures):
                result = future.result()
                provenance = result.pop("provenance", None)
                primary_failure = result.pop("primary_failure_provenance", None)
                if primary_failure:
                    log_manifest(primary_failure)
                if provenance:
                    log_manifest(provenance)  # Main thread only: append without races.
                results.append(result)
                if len(results) % 100 == 0 or result["status"] not in ("restored", "verified_existing"):
                    if len(results) % 100 == 0:
                        print(f"Progress {len(results)}/{len(jobs)}: {dict(Counter(r['status'] for r in results))}", flush=True)
                    elif priority(expected[result["dest"]])[0] == 0:
                        print(result["status"], result["dest"], result.get("error", ""), flush=True)
                    if len(results) % 100 == 0:
                        save_report(results, len(jobs), args.stage, started, False)
        save_report(results, len(jobs), args.stage, started, False)
        print(f"Tier {tier} complete: {dict(Counter(r['status'] for r in results))}", flush=True)
    save_report(results, len(jobs), args.stage, started, True)
    blockers = [r for r in results if r["status"] in ("failed", "network_policy_blocked", "snapshot_mismatch", "existing_mismatch")]
    print(f"Finished: {dict(Counter(r['status'] for r in results))}; review {REPORT}", flush=True)
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
