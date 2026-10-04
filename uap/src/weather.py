"""Surface weather at points from NOAA ISD-Lite (nearest selected station
within MAX_KM) and convective activity from NOAA Storm Events.

Sky cover code (ISD-Lite): 0 = SKC/CLR ... 8 = OVC, 9/10 = obscured/partial
obscuration; -9999 missing. We map 9/10 -> 8 (obscured sky is not clear sky).
If no station within MAX_KM, or the station has no observation within +-90
min, weather fields are NaN with status DATA_UNAVAILABLE (not "clear").
"""
from __future__ import annotations

import gzip
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from common import RAW, INTERIM

W = RAW / "weather"
MAX_KM = 50.0
WEATHER_COLUMNS = (
    "wx_temp_c", "wx_dewpt_spread_c", "wx_slp_hpa", "wx_wind_ms", "wx_sky_oktas",
    "wx_precip_1h_mm", "wx_dslp_3h", "wx_dslp_24h", "wx_dtemp_24h", "wx_dsky_6h",
    "wx_sky_mean_prior6h", "wx_precip_prior24h_mm", "wx_inversion_proxy",
)


@lru_cache(maxsize=1)
def stations():
    s = pd.read_csv(W / "selected_isd_stations.csv", dtype={"USAF": str, "WBAN": str})
    s["sid"] = s.USAF + "-" + s.WBAN
    return s


def load_station(sid: str, years=range(1995, 2024)) -> pd.DataFrame:
    parts = []
    for y in years:
        f = W / "isd_lite" / str(y) / f"{sid}-{y}.gz"
        if not f.exists():
            continue
        try:
            a = np.loadtxt(gzip.open(f), dtype=float)
        except Exception:
            continue
        if a.ndim == 1:
            a = a[None, :]
        parts.append(a)
    if not parts:
        return pd.DataFrame()
    a = np.vstack(parts)
    df = pd.DataFrame(a, columns=["y", "m", "d", "h", "t", "td", "slp", "wd", "ws", "sky", "p1", "p6"])
    df = df.replace(-9999.0, np.nan)
    df["time"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d, hour=df.h), utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    df["t"] /= 10; df["td"] /= 10; df["slp"] /= 10; df["ws"] /= 10; df["p1"] /= 10; df["p6"] /= 10
    # Only observed obscured-sky codes map to overcast; missing stays missing.
    df["sky"] = df["sky"].replace({9.0: 8.0, 10.0: 8.0})
    df.loc[~df["sky"].between(0, 8), "sky"] = np.nan
    full = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC"))
    out = pd.DataFrame(index=full.index)
    out["wx_temp_c"] = full.t
    out["wx_dewpt_spread_c"] = full.t - full.td
    out["wx_slp_hpa"] = full.slp
    out["wx_wind_ms"] = full.ws
    out["wx_sky_oktas"] = full.sky
    out["wx_precip_1h_mm"] = full.p1
    out["wx_dslp_3h"] = full.slp - full.slp.shift(3)
    out["wx_dslp_24h"] = full.slp - full.slp.shift(24)
    out["wx_dtemp_24h"] = full.t - full.t.shift(24)
    out["wx_dsky_6h"] = full.sky - full.sky.shift(6)
    out["wx_sky_mean_prior6h"] = full.sky.rolling(6, min_periods=3).mean()
    out["wx_precip_prior24h_mm"] = full.p1.rolling(24, min_periods=6).sum()
    # Clear-sky/light-wind component; downstream wx_inversion_night adds
    # the Sun-altitude requirement after weather and astronomy are joined.
    out["wx_inversion_proxy"] = ((full.sky <= 2) & (full.ws <= 2.0)).astype(float)
    out.loc[full.sky.isna() | full.ws.isna(), "wx_inversion_proxy"] = np.nan
    # Keep source measurement gaps. Match actual observation timestamps in
    # weather_features rather than impute every variable on the hourly grid.
    return out.loc[df.index]


def assign_station(lat, lon):
    s = stations()
    tree = BallTree(np.radians(s[["LAT", "LON"]].to_numpy()), metric="haversine")
    d, i = tree.query(np.radians(np.c_[lat, lon]), k=1)
    d = d[:, 0] * 6371.0088
    sid = s.sid.to_numpy()[i[:, 0]]
    return np.where(d <= MAX_KM, sid, None), d


def weather_features(pts_t, lat, lon) -> pd.DataFrame:
    """pts_t: float seconds UTC. Returns per-point weather columns (NaN if unavailable)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    tt = np.asarray(pts_t, float)
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(tt)
    sid = np.full(len(lat), None, dtype=object)
    dist = np.full(len(lat), np.nan)
    if ok.any():
        s_, d_ = assign_station(lat[ok], lon[ok])
        sid[ok] = s_; dist[ok] = d_
    res = {column: np.full(len(lat), np.nan) for column in WEATHER_COLUMNS}
    observation_age = np.full(len(lat), np.nan)
    times = pd.to_datetime(tt, utc=True, unit="s")
    for s in pd.unique(sid[pd.notna(sid)]):
        m = np.where(sid == s)[0]
        df = load_station(s)
        if df.empty:
            continue
        indexer = df.index.get_indexer(times[m], method="nearest", tolerance=pd.Timedelta("90min"))
        usable = indexer >= 0
        if not usable.any():
            continue
        target = m[usable]
        sub = df.iloc[indexer[usable]]
        for c in df.columns:
            res.setdefault(c, np.full(len(lat), np.nan))[target] = sub[c].to_numpy()
        observation_age[target] = (times[target] - sub.index).total_seconds() / 60
    out = pd.DataFrame(res, index=range(len(lat)))
    out["wx_station"] = sid
    out["wx_station_km"] = dist
    # Negative age identifies observations after the event. Prospective rules
    # must require nonnegative age; retrospective nearest-weather matches may
    # use either side of the documented 90-minute tolerance.
    out["wx_observation_age_min"] = observation_age
    return out


@lru_cache(maxsize=1)
def storm_events() -> pd.DataFrame:
    p = INTERIM / "storm_events.parquet"
    if p.exists():
        return pd.read_parquet(p)
    parts = []
    tz_off = {"EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
              "AKST": -9, "AKDT": -8, "HST": -10, "AST": -4, "SST": -11, "GST": 10}
    for f in sorted((W / "storm_events").glob("details_*.csv.gz")):
        d = pd.read_csv(f, usecols=["BEGIN_DATE_TIME", "CZ_TIMEZONE", "EVENT_TYPE", "BEGIN_LAT", "BEGIN_LON"],
                        low_memory=False)
        d = d[d.EVENT_TYPE.isin(["Lightning", "Thunderstorm Wind", "Hail", "Tornado", "Funnel Cloud",
                                 "Marine Thunderstorm Wind", "Marine Hail", "Waterspout"])]
        d = d.dropna(subset=["BEGIN_LAT", "BEGIN_LON"])
        tz = d.CZ_TIMEZONE.str.upper().str.extract(r"([A-Z]+)")[0].str.replace(r"-?\d+", "", regex=True)
        off = tz.map(tz_off)
        off = off.fillna(pd.to_numeric(d.CZ_TIMEZONE.str.extract(r"(-?\d+)")[0], errors="coerce"))
        lt = pd.to_datetime(d.BEGIN_DATE_TIME, format="%d-%b-%y %H:%M:%S", errors="coerce")
        d["time"] = (lt - pd.to_timedelta(off, unit="h")).dt.tz_localize("UTC")
        d = d.rename(columns={"BEGIN_LAT": "lat", "BEGIN_LON": "lon", "EVENT_TYPE": "type"})
        parts.append(d[["time", "lat", "lon", "type"]])
    out = pd.concat(parts, ignore_index=True).dropna(subset=["time"])
    out.to_parquet(p, index=False)
    return out
