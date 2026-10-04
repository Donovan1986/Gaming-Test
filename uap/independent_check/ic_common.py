"""Shared helpers for the independent check.

HOLDOUT GUARD: every read of events/points goes through these functions, which
apply pyarrow row filters so only SPLIT == "discovery" & SOURCE == "NUFORC"
events (and points whose EVENT_ID is in that set) ever reach memory as data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
SEED = 777123  # independent-check seed (deliberately different from pipeline MASTER_SEED)

_DISC_FILTER = [("SPLIT", "=", "discovery"), ("SOURCE", "=", "NUFORC")]


def disc_events(columns) -> pd.DataFrame:
    cols = list(dict.fromkeys(["EVENT_ID", "SPLIT", "SOURCE"] + list(columns)))
    t = pq.read_table(PROC / "events.parquet", columns=cols, filters=_DISC_FILTER)
    df = t.to_pandas()
    assert (df.SPLIT == "discovery").all() and (df.SOURCE == "NUFORC").all()
    return df


def disc_ids() -> np.ndarray:
    return disc_events([]).EVENT_ID.to_numpy()


def disc_points(columns, strategies=None, ids=None) -> pd.DataFrame:
    """Points rows whose EVENT_ID is in the discovery NUFORC set (optionally a subset `ids`)."""
    if ids is None:
        ids = disc_ids()
    ids = pa.array(list(map(str, ids)), type=pa.large_string())
    cols = list(dict.fromkeys(["EVENT_ID", "strategy", "is_case", "SPLIT", "SOURCE"] + list(columns)))
    filt = pc.field("EVENT_ID").isin(ids)
    if strategies is not None:
        filt = filt & pc.field("strategy").isin(pa.array(list(strategies), type=pa.large_string()))
    ds = pq.ParquetDataset(PROC / "points.parquet", filters=filt)
    df = ds.read(columns=cols).to_pandas()
    assert (df.SPLIT == "discovery").all() and (df.SOURCE == "NUFORC").all()
    return df.drop(columns=["SPLIT", "SOURCE"])


def to_sec(ts) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).as_unit("ns")
    out = idx.asi8.astype("float64") / 1e9
    out[idx.isna()] = np.nan
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
