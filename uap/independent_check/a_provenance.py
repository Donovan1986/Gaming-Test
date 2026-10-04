"""Independent check A: environment + provenance.

A1: python / package versions vs requirements.txt
A2: recompute sha256 of every manifest entry with ok=true whose dest exists
A3: sanity-parse key raw files (Kp, OMNI2, GCAT launch.tsv, CNEOS json, ISS TLE)

Writes: independent_check/out/a_env.json, a_hash_check.csv, a_parse_checks.json
Reads only raw files / manifest; writes only under independent_check/.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def a1_env():
    req = {}
    for line in open(ROOT / "requirements.txt"):
        line = line.strip()
        if not line or line.startswith("#"):
            if "Python" in line:
                req["__python__"] = line.split("Python")[-1].strip()
            continue
        name, ver = line.split("==")
        req[name] = ver
    rows = []
    for name, ver in req.items():
        if name == "__python__":
            inst = platform.python_version()
        else:
            try:
                inst = metadata.version(name)
            except metadata.PackageNotFoundError:
                inst = None
        rows.append(dict(package=name, required=ver, installed=inst, match=(inst == ver)))
    extra = {}
    for n in ("tzdata", "pytz", "jplephem", "python-dateutil", "scipy"):
        try:
            extra[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            extra[n] = None
    res = dict(python=sys.version, executable=sys.executable, packages=rows, other=extra)
    json.dump(res, open(OUT / "a_env.json", "w"), indent=1)
    return res


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def a2_hashes():
    import csv
    rows = []
    for line in open(ROOT / "data" / "manifest.jsonl"):
        e = json.loads(line)
        if not e.get("ok"):
            continue
        p = ROOT / e["dest"]
        if not p.exists():
            rows.append(dict(dest=e["dest"], status="missing", expected=e.get("sha256"), got="", bytes_manifest=e.get("bytes"), bytes_disk=""))
            continue
        got = sha256(p)
        rows.append(dict(dest=e["dest"], status="match" if got == e.get("sha256") else "mismatch",
                         expected=e.get("sha256"), got=got, bytes_manifest=e.get("bytes"), bytes_disk=p.stat().st_size))
    with open(OUT / "a_hash_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    return dict(counts=dict(c), n=len(rows),
                mismatches=[r for r in rows if r["status"] == "mismatch"][:50],
                missing=[r["dest"] for r in rows if r["status"] == "missing"][:50])


def a3_parse():
    import numpy as np
    import pandas as pd
    out = {}
    raw = ROOT / "data" / "raw"
    # GFZ Kp
    n, bad, years, last = 0, 0, set(), None
    for line in open(raw / "spaceweather" / "Kp_ap_Ap_SN_F107_since_1932.txt"):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 28:
            bad += 1
            continue
        n += 1
        years.add(int(f[0]))
        kps = [float(x) for x in f[7:15]]
        if any((k > 9.0) or (k < 0 and k != -1) for k in kps):
            bad += 1
        last = f[:3]
    out["gfz_kp"] = dict(rows=n, bad_rows=bad, first_year=min(years), last_year=max(years), last_date=last)
    # OMNI2
    o = pd.read_csv(raw / "spaceweather" / "omni2_all_years.dat", sep=r"\s+", header=None, usecols=[0, 1, 2, 40])
    out["omni2"] = dict(rows=len(o), ncols_check=int(pd.read_csv(raw / "spaceweather" / "omni2_all_years.dat", sep=r"\s+", header=None, nrows=3).shape[1]),
                        year_min=int(o[0].min()), year_max=int(o[0].max()),
                        hour_range=[int(o[2].min()), int(o[2].max())], doy_range=[int(o[1].min()), int(o[1].max())],
                        dst_valid_frac=float((o[40] < 99999).mean()),
                        dst_valid_range=[float(o.loc[o[40] < 99999, 40].min()), float(o.loc[o[40] < 99999, 40].max())])
    # GCAT launch.tsv
    L = pd.read_csv(raw / "space" / "gcat_launch.tsv", sep="\t", dtype=str)
    L.columns = [c.lstrip("#").strip() for c in L.columns]
    out["gcat_launch"] = dict(rows=len(L), columns=list(L.columns)[:20],
                              first_rows_launch_date=L["Launch_Date"].head(3).tolist())
    # CNEOS
    d = json.load(open(raw / "space" / "cneos_fireballs.json"))
    df = pd.DataFrame(d["data"], columns=d["fields"])
    t = pd.to_datetime(df["date"], utc=True)
    out["cneos"] = dict(count_field=d.get("count"), rows=len(df), fields=d["fields"], first=str(t.min()), last=str(t.max()),
                        lat_missing=int(df["lat"].isna().sum()))
    # ISS TLE
    from sgp4.api import Satrec
    l = [x.rstrip("\n") for x in open(raw / "space" / "iss_25544_tle_mcdowell.txt")]
    l1 = [x for x in l if x.startswith("1 ")]
    l2 = [x for x in l if x.startswith("2 ")]
    eps, nbad, ncheck_bad = [], 0, 0

    def checksum_ok(s):
        s = s[:69]
        if len(s) < 69 or not s[68].isdigit():
            return None
        c = sum(int(ch) if ch.isdigit() else (1 if ch == "-" else 0) for ch in s[:68]) % 10
        return c == int(s[68])
    i = 0
    lines = [x for x in l if x.startswith(("1 ", "2 "))]
    pairs = 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            pairs += 1
            ok = checksum_ok(lines[i]), checksum_ok(lines[i + 1])
            if False in ok:
                ncheck_bad += 1
            try:
                s = Satrec.twoline2rv(lines[i][:69].ljust(69), lines[i + 1][:69].ljust(69))
                eps.append(s.jdsatepoch + s.jdsatepochF)
                if s.satnum != 25544:
                    nbad += 1
            except Exception:
                nbad += 1
            i += 2
        else:
            i += 1
    eps = np.sort(np.array(eps))
    gaps = np.diff(eps)
    ep_dt = pd.to_datetime((eps - 2440587.5) * 86400, unit="s", utc=True)
    out["iss_tle"] = dict(line1=len(l1), line2=len(l2), pairs=pairs, parse_fail_or_wrong_satnum=nbad,
                          checksum_fail_pairs=ncheck_bad, first_epoch=str(ep_dt.min()), last_epoch=str(ep_dt.max()),
                          max_gap_days=float(gaps.max()), n_gaps_gt_5d=int((gaps > 5).sum()),
                          gaps_gt_5d=[(str(ep_dt[k]), float(gaps[k])) for k in np.where(gaps > 5)[0][:20]])
    json.dump(out, open(OUT / "a_parse_checks.json", "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("env", "all"):
        r = a1_env()
        print(json.dumps(r, indent=1))
    if which in ("parse", "all"):
        print(json.dumps(a3_parse(), indent=1, default=str))
    if which in ("hash", "all"):
        r = a2_hashes()
        json.dump(r, open(OUT / "a_hash_summary.json", "w"), indent=1)
        print(json.dumps(r, indent=1))
