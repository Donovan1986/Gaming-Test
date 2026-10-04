"""Resume the registered investigation; unavailable inputs block research (exit 2)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
PYTHON = "/workspace/.venvs/gaming-test/bin/python"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    part.replace(path)


def preflight(root, manifest_prefix=None):
    """Verify actual files; a stale restoration report never supplies readiness."""
    manifest = root / "uap/data/manifest.jsonl"
    whole = manifest.read_bytes()
    contract = whole if manifest_prefix is None else whole[:manifest_prefix]
    if manifest_prefix is not None and len(contract) != manifest_prefix:
        raise ValueError("Initial manifest prefix was truncated")
    expected, accepted = {}, {}
    for line in contract.decode().splitlines():
        row = json.loads(line)
        name = row.get("dest", "")
        if not (row.get("ok") and row.get("sha256") and name.startswith("data/raw/")):
            continue
        path = (root / "uap" / name).resolve()
        if not path.is_relative_to((root / "uap/data/raw").resolve()):
            raise ValueError(f"Unsafe manifest destination: {name}")
        if not row.get("restoration") and "/.restoration_divergent/" not in name:
            expected[name] = row
        elif row.get("restoration") == "accepted_current":
            accepted[name] = row
    if not expected:
        raise ValueError("No successful historical raw inputs in the manifest")
    missing, mismatched, verified, replacements = [], [], {}, []
    for name, row in sorted(expected.items()):
        path = root / "uap" / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = digest(path)
        approval = accepted.get(name, {})
        if actual != row["sha256"]:
            if approval.get("sha256") != actual or approval.get("historical_sha256") != row["sha256"]:
                mismatched.append(name)
                continue
            replacements.append(name)
        verified[name] = actual
    report_path = root / "uap/results/input_restoration.json"
    restoration = json.loads(report_path.read_text()) if report_path.exists() else {}
    return {"ready": not missing and not mismatched, "expected_files": len(expected),
            "verified_files": len(verified), "missing_files": len(missing), "mismatched_files": len(mismatched),
            "missing_sample": missing[:15], "mismatch_sample": mismatched[:15],
            "missing_hosts": sorted({urlsplit(expected[n]["url"]).hostname for n in missing}),
            "accepted_current_snapshots": replacements, "data_fingerprint": fingerprint(verified),
            "manifest_sha256": hashlib.sha256(whole).hexdigest(), "manifest_prefix_bytes": len(contract),
            "manifest_prefix_sha256": hashlib.sha256(contract).hexdigest(),
            "restoration_report_counts": restoration.get("counts", {}),
            "restoration_report_updated_utc": restoration.get("updated_utc")}


def supplementary_manifest(root, readiness):
    """Appended measurements are audited separately from the frozen input contract."""
    whole = (root / "uap/data/manifest.jsonl").read_bytes()
    size = readiness["manifest_prefix_bytes"]
    if hashlib.sha256(whole[:size]).hexdigest() != readiness["manifest_prefix_sha256"]:
        raise ValueError("Initial manifest prefix changed; refusing resume")
    addition = whole[size:]
    rows = [json.loads(line) for line in addition.decode().splitlines() if line.strip()]
    measurements = {}
    for row in rows:
        name = row.get("dest", "")
        if row.get("ok") and row.get("sha256") and name.startswith("data/raw/"):
            path = (root / "uap" / name).resolve()
            if not path.is_relative_to((root / "uap/data/raw").resolve()):
                raise ValueError(f"Unsafe supplementary destination: {name}")
            if not path.is_file() or digest(path) != row["sha256"]:
                raise ValueError(f"Supplementary raw input missing or changed: {name}")
            measurements[name] = row["sha256"]
    return {"manifest_sha256": hashlib.sha256(whole).hexdigest(), "appended_sha256": hashlib.sha256(addition).hexdigest(),
            "appended_entries": rows, "verified_measurements": measurements}


def versions(root, python):
    code = "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version,'pins':{" + ",".join(
        f"{name!r}:m.version({name!r})" for name in requirement_pins(root)) + "}}))"
    result = subprocess.run([python, "-c", code], text=True, capture_output=True, check=True)
    installed = json.loads(result.stdout)
    if not installed["python"].startswith("3.11.15 ") or installed["pins"] != requirement_pins(root):
        raise ValueError("Runtime differs from Python 3.11.15 and exact requirements pins")
    return installed


def requirement_pins(root):
    return dict(line.strip().split("==", 1) for line in (root / "uap/requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#"))


def plan(args, identity):
    stages = []
    def stage(name, script, outputs, code=None):
        command = [args.python, "-c", code] if code else [args.python, script + ".py"]
        stages.append({"name": name, "command": command, "outputs": outputs.split()})
    stage("prepare", "restore_inputs", "data/raw/weather/selected_isd_stations.csv",
          "import restore_inputs as r, re, pandas as pd; r.extract_tables(); "
          "rows=r.historical_entries()[0]; s=r.reconstruct_stations(rows); "
          "want={m.group(1) for x in rows for m in [re.search(r'/isd_lite/\\d{4}/(\\d{6}-\\d{5})-\\d{4}\\.gz$', x.get('dest',''))] if m and not x.get('restoration')}; "
          "have=pd.read_csv(r.RAW/'weather/selected_isd_stations.csv',dtype=str); have=set(have.USAF+'-'+have.WBAN); "
          "assert s['status']=='reconstructed' or (s['status']=='existing_preserved' and have==want), (s, len(have), len(want))")
    stage("reports", "build_reports", "data/processed/reports.parquet results/geocoding_decisions.csv results/excluded_records.csv")
    stage("events", "build_events", "data/processed/events.parquet results/splits_manifest.json results/quality_score_formula.md")
    stage("features", "analysis_data", "data/processed/points_core.parquet",
          f"import analysis_data as a; original=a.compute_features; "
          f"a.compute_features=lambda pts: original(pts,chunk={args.feature_chunk}); a.main()")
    stage("weather", "analysis_data", "data/processed/points.parquet", "import analysis_data as a; a.add_weather()")
    stage("positive_controls", "positive_controls", "results/positive_controls.csv results/placebo_calibration.csv results/placebo_calibration_summary.csv results/detection_power.csv")
    stage("discovery", "discovery_tests", "results/discovery_results.csv")
    stage("ml_discovery", "ml_discovery", "results/ml_performance.csv results/ml_feature_importance.csv results/ml_shap_interactions.csv results/ml_interaction_tests.csv results/ml_spline_doseresponse.csv")
    stage("sequences", "sequences", "results/seq_superposed_epoch_summary.csv results/seq_quake_pre_post_asymmetry.csv results/seq_ordered_pairs.csv")
    stage("spatial_discovery", "spatial_analysis", "results/spatial_autocorrelation.csv results/spatial_scan_clusters.csv results/spatial_knox.csv")
    metadata = {"runner_identity": identity, "master_seed": 20261004,
                "validation_plan": {"exposure_shift_diagnostics_requested": args.permutations,
                                    "temporal_unique_shift_limit": 30, "prospective_prediction": "requires_separate_qualification"}}
    freeze = ("import select_candidates as s,hashlib,json; from common import RESULTS; "
              f"metadata=json.loads({json.dumps(metadata)!r}); "
              "metadata['splits_manifest_sha256']=hashlib.sha256((RESULTS/'splits_manifest.json').read_bytes()).hexdigest(); "
              "import build_frozen as b; b.main(metadata); import validate; validate.load_frozen()")
    stage("freeze", "select_candidates", "results/candidate_selection_audit.csv results/frozen_hypotheses.json results/frozen_hypotheses.sha256", freeze)
    stage("validate", "validate", "results/validation_results.csv results/validation_sensitivity.csv results/validation_permutation.csv",
          f"import validate; validate.main(n_perm={args.permutations})")
    stage("starlink_integrity", "positive_controls", "results/positive_controls_starlink_integrity.csv",
          "import positive_controls as p; p.post_freeze_starlink_integrity()")
    for name, outputs in [
        ("solar_cycle_systems", "results/solar_cycle_across_systems.csv"),
        ("convergence", "results/convergence_events.csv results/convergence_score_distribution.csv"),
        ("natural_experiments", "results/natural_experiments.csv"),
        ("media_waves", "results/media_event_study.csv results/media_attention_elasticity.csv results/exposure_annual_rates.csv"),
        ("make_data_sources", "report/DATA_SOURCES.md")]:
        stage(name, name, outputs)
    # Frozen adjudication hypotheses (declared in the freeze; run after validation)
    stage("adjudicate", "adjudicate", "results/adjudication_urban.csv results/adjudication_same_mechanism.csv")
    stage("spatial_holdout", "spatial_analysis", "results/spatial_scan_clusters_holdout.csv results/spatial_knox_holdout.csv",
          "import spatial_analysis as s; s.main_holdout()")
    stage("ne_replication", "natural_experiments", "results/natural_experiments_replication.csv",
          "import natural_experiments as n; n.main(splits=('validation','holdout_random'), tag='replication')")
    return stages


def artifacts(root, names):
    return {name: digest(root / "uap" / name) for name in names}


def check_freeze(root, identity):
    base = root / "uap/results"
    payload = (base / "frozen_hypotheses.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != (base / "frozen_hypotheses.sha256").read_text().strip():
        raise ValueError("Frozen hypothesis checksum changed")
    spec = json.loads(payload)
    if spec.get("runner_identity") != identity:
        raise ValueError("Frozen hypotheses belong to a different code/data/runtime design")
    if spec.get("splits_manifest_sha256") != digest(base / "splits_manifest.json"):
        raise ValueError("Split manifest changed after freezing")


def invoke(command, root, log):
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="4", MPLBACKEND="Agg",
               PYTHONUNBUFFERED="1", XDG_CACHE_HOME="/workspace/.cache", MPLCONFIGDIR="/workspace/.cache/matplotlib",
               NUMBA_CACHE_DIR="/workspace/.cache/numba")
    with log.open("a") as stream:
        return subprocess.run(command, cwd=root / "uap/src", env=env, stdout=stream, stderr=subprocess.STDOUT).returncode


def reset_caches(root, journal, record):
    """Archive generated intermediates once, before rebuilding with corrected code."""
    if record.get("cache_reset", {}).get("completed"):
        return
    if "cache_reset" not in record:
        record["cache_reset"] = {"completed": False, "files": [
            {"source": str(p.relative_to(root)), "sha256": digest(p),
             "archive": f"uap/logs/{record['run_id']}-cache-{p.name}.backup.log"}
            for p in sorted((root / "uap/data/interim").glob("*.parquet"))]}
        save(journal, record)
    for item in record["cache_reset"]["files"]:
        source, archive = root / item["source"], root / item["archive"]
        if source.exists():
            if digest(source) != item["sha256"]:
                raise ValueError(f"Generated cache changed during archival: {source}")
            if not archive.exists():
                shutil.copyfile(source, archive)
        if not archive.exists() or digest(archive) != item["sha256"]:
            raise ValueError(f"Cache archive missing or changed: {archive}")
        source.unlink(missing_ok=True)
    record["cache_reset"]["completed"] = True
    save(journal, record)


def run(args, root=REPO):
    status_path = root / "uap/results/investigation_status.json"
    prior_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    journal_ref = {"journal": prior_status["journal"]} if prior_status.get("journal") else {}
    resume_record, prefix = None, None
    if args.resume and journal_ref:
        journal = root / journal_ref["journal"]
        if not journal.resolve().is_relative_to((root / "uap/logs").resolve()):
            raise ValueError("Unsafe resume journal path")
        resume_record = json.loads(journal.read_text())
        resume_record["supplementary_manifest"] = supplementary_manifest(root, resume_record["readiness"])
        prefix = resume_record["readiness"]["manifest_prefix_bytes"]
    if args.restore:
        restore_log = root / "uap/logs/runner-restoration.log"
        restore_log.parent.mkdir(parents=True, exist_ok=True)
        rc = invoke([args.python, "restore_inputs.py", "--stage", "all"], root, restore_log)
        print(f"Restoration exit {rc}; checking actual inputs", flush=True)
    readiness = preflight(root, prefix)
    if not readiness["ready"]:
        status = {"state": "BLOCKED_INPUTS", "updated_utc": now(), "readiness": readiness,
                  "exit_code": 2, "research_executed": False, "scientific_conclusion": "NOT_EVALUATED", **journal_ref}
        save(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
        return 2
    runtime = versions(root, args.python)
    sources = {str(p.relative_to(root)): digest(p) for p in sorted((root / "uap/src").glob("*.py"))}
    sources["uap/requirements.txt"] = digest(root / "uap/requirements.txt")
    identity = {"code": fingerprint(sources), "data": readiness["data_fingerprint"], "runtime": fingerprint(runtime),
                "feature_chunk": args.feature_chunk, "permutations": args.permutations}
    if args.preflight_only:
        save(status_path, {"state": "INPUTS_VERIFIED", "updated_utc": now(), "readiness": readiness,
                           "identity": identity, "exit_code": 0, "research_executed": False, **journal_ref})
        print("Inputs and pinned runtime verified; research not run", flush=True)
        return 0
    stages = plan(args, identity)
    if resume_record:
        record = resume_record
        if record["identity"] != identity:
            raise ValueError("Resume code/data/runtime/options fingerprint differs; preserved previous outputs")
    else:
        if (root / "uap/results/frozen_hypotheses.json").exists():
            raise ValueError("Existing freeze without matching resume journal; cannot reopen discovery")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
        journal = root / f"uap/logs/investigation-{run_id}.json.log"
        record = {"run_id": run_id, "identity": identity, "readiness": readiness, "runtime": runtime,
                  "code_files": sources, "started_utc": now(), "stages": {}}
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        record["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    reset_caches(root, journal, record)
    save(journal, record)
    frozen = False
    for stage in stages:
        name = stage["name"]
        previous = record["stages"].get(name, {})
        if previous.get("exit_code") == 0:
            if previous.get("command") != stage["command"] or previous.get("outputs") != artifacts(root, stage["outputs"]):
                raise ValueError(f"Completed stage {name} output/command changed; refusing stale resume")
            if name == "freeze":
                check_freeze(root, identity)
                frozen = True
            print(f"Verified completed stage: {name}", flush=True)
            continue
        if frozen:
            check_freeze(root, identity)
        log = root / f"uap/logs/{record['run_id']}-{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        for output in stage["outputs"] + ["results/hypothesis_registry.csv"]:
            path = root / "uap" / output
            backup = log.with_name(f"{record['run_id']}-{name}-{Path(output).name}.backup.log")
            if path.exists() and not backup.exists():
                shutil.copyfile(path, backup)
        started = time.monotonic()
        item = {"command": stage["command"], "started_utc": now(), "log": str(log.relative_to(root)), "exit_code": None}
        record["stages"][name] = item
        save(journal, record)
        save(status_path, {"state": "RUNNING", "updated_utc": now(), "stage": name,
                           "journal": str(journal.relative_to(root)), "identity": identity})
        print(f"Running {name}; log {log}", flush=True)
        rc = invoke(stage["command"], root, log)
        item.update(exit_code=rc, duration_seconds=round(time.monotonic() - started, 3), finished_utc=now())
        try:
            record["supplementary_manifest"] = supplementary_manifest(root, record["readiness"])
            item["manifest_sha256"] = record["supplementary_manifest"]["manifest_sha256"]
            item["supplementary_sha256"] = record["supplementary_manifest"]["appended_sha256"]
        except (OSError, ValueError) as error:
            rc = item["exit_code"] = 1
            item["error"] = str(error)
        if rc == 0:
            try:
                item["outputs"] = artifacts(root, stage["outputs"])
                if name == "freeze":
                    check_freeze(root, identity)
                    frozen = True
            except (OSError, ValueError) as error:
                rc = item["exit_code"] = 1
                item["error"] = str(error)
        save(journal, record)
        if rc:
            save(status_path, {"state": "FAILED_STAGE", "stage": name, "updated_utc": now(), "exit_code": rc,
                               "journal": str(journal.relative_to(root)), "identity": identity})
            return 1
    save(status_path, {"state": "COMPUTATION_COMPLETE_REVIEW_REQUIRED", "updated_utc": now(), "exit_code": 0,
                       "journal": str(journal.relative_to(root)), "identity": identity,
                       "completed_stages": list(record["stages"]), "prospective_prediction": "NOT_QUALIFIED_BY_RUNNER"})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--restore", action="store_true", help="Attempt manifest restoration before preflight")
    parser.add_argument("--resume", action="store_true", help="Reuse only matching journaled stages and output hashes")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--feature-chunk", type=int, default=50_000)
    parser.add_argument("--permutations", type=int, default=30, help="Exposure-shift diagnostics; temporal shifts cap at 30 unique offsets")
    args = parser.parse_args()
    if args.feature_chunk <= 0 or args.permutations <= 0:
        parser.error("Chunk size and diagnostic count must be positive")
    try:
        return run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
