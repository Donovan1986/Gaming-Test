"""Shared paths, provenance logging and constants for the UAP investigation.

Every network retrieval goes through `fetch()`, which appends a JSON line to
data/manifest.jsonl recording URL, retrieval timestamp (UTC), HTTP status,
byte size and SHA-256 of the saved file. Nothing is downloaded silently.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
LOGS = ROOT / "logs"
MANIFEST = ROOT / "data" / "manifest.jsonl"

for _p in (RAW, INTERIM, PROCESSED, RESULTS, REPORT, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# Global seed. Every stochastic step derives its generator from this value plus
# a fixed, documented offset so that results are reproducible.
MASTER_SEED = 20261004

UA = "uap-empirical-investigation/1.0 (research; contact via repository)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_manifest(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def fetch(url: str, dest: Path, desc: str = "", params: dict | None = None,
          overwrite: bool = False, timeout: int = 180, retries: int = 4,
          sleep_between: float = 0.0) -> Path | None:
    """Download `url` to `dest`, logging provenance. Returns path or None."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": UA}, stream=True)
            status = r.status_code
            if status == 429 or status >= 500:
                last_err = f"HTTP {status}"
                time.sleep(2 ** (attempt + 1) * 2)
                continue
            if status != 200:
                log_manifest({"url": r.url, "dest": str(dest.relative_to(ROOT)),
                              "status": status, "retrieved_utc": _now(),
                              "description": desc, "ok": False})
                return None
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            log_manifest({"url": r.url, "dest": str(dest.relative_to(ROOT)),
                          "status": status, "retrieved_utc": _now(),
                          "bytes": dest.stat().st_size,
                          "sha256": sha256_file(dest), "description": desc,
                          "ok": True})
            if sleep_between:
                time.sleep(sleep_between)
            return dest
        except requests.RequestException as e:  # network error: back off
            last_err = repr(e)
            time.sleep(2 ** (attempt + 1))
    log_manifest({"url": url, "dest": str(dest.relative_to(ROOT)),
                  "status": None, "retrieved_utc": _now(), "description": desc,
                  "ok": False, "error": last_err})
    return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (numpy arrays or scalars)."""
    import numpy as np
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
