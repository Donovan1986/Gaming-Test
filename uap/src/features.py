"""Temporal + spatio-temporal covariates for arbitrary (time, lat, lon) points.

The same functions are applied to UAP events and to their matched controls so
that cases and controls are measured identically (no differential
measurement). Every window is defined relative to the point time t:
  "pre" windows  = [t - W, t]   (could precede the observation)
  "post" windows = [t, t + W]   (follow the observation; used for sequence tests)
Missing covariate data stays NaN (DATA_UNAVAILABLE); counts are only zero
when the catalog covers that period and region (TRUE ZERO).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

import covariates as C

R_EARTH = 6371.0088


def to_sec(ts) -> np.ndarray:
    """UTC timestamps -> float seconds since epoch (unit-safe for pandas 3)."""
    idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).as_unit("ns")
    out = idx.asi8.astype("float64") / 1e9
    out[idx.isna()] = np.nan
    return out


@njit(cache=True)
def _window_counts(pt_t, pt_lat, pt_lon, cat_t, cat_lat, cat_lon, cat_v, w_lo, w_hi, radii, need_loc):
    n, nw, nr = pt_t.shape[0], w_lo.shape[0], radii.shape[0]
    cnt = np.zeros((n, nw, nr), dtype=np.int32)
    vmax = np.full((n, nw, nr), np.nan)
    tmin = w_lo.min()
    tmax = w_hi.max()
    rad = np.pi / 180.0
    for i in range(n):
        t = pt_t[i]
        if np.isnan(t):
            for a in range(nw):
                for b in range(nr):
                    cnt[i, a, b] = -1
            continue
        j0 = np.searchsorted(cat_t, t + tmin)
        j1 = np.searchsorted(cat_t, t + tmax, side="right")
        la1 = pt_lat[i] * rad
        lo1 = pt_lon[i] * rad
        for j in range(j0, j1):
            dt = cat_t[j] - t
            if need_loc:
                la2 = cat_lat[j] * rad
                lo2 = cat_lon[j] * rad
                h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
                d = 2 * R_EARTH * np.arcsin(np.sqrt(min(1.0, h)))
            else:
                d = 0.0
            for a in range(nw):
                if dt >= w_lo[a] and dt <= w_hi[a]:
                    for b in range(nr):
                        if d <= radii[b]:
                            cnt[i, a, b] += 1
                            v = cat_v[j]
                            if np.isnan(vmax[i, a, b]) or v > vmax[i, a, b]:
                                vmax[i, a, b] = v
    return cnt, vmax


def window_counts(pts_t, pts_lat, pts_lon, cat: pd.DataFrame, windows: dict, radii: list, value_col=None,
                  prefix="", need_loc=True, coverage=None):
    """Return DataFrame of counts (and max value) for each window x radius.
    coverage: (start_ts, end_ts) of the catalog; points outside -> NaN (DATA_UNAVAILABLE)."""
    cat = cat.dropna(subset=["time"]).sort_values("time")
    ct = to_sec(cat["time"])
    cla = cat["lat"].to_numpy(float) if need_loc else np.zeros(len(cat))
    clo = cat["lon"].to_numpy(float) if need_loc else np.zeros(len(cat))
    cv = cat[value_col].to_numpy(float) if value_col else np.ones(len(cat))
    names = list(windows)
    wlo = np.array([windows[k][0] for k in names], float)
    whi = np.array([windows[k][1] for k in names], float)
    rr = np.array(radii, float) if need_loc else np.array([0.0])
    cnt, vmax = _window_counts(np.asarray(pts_t, float), np.asarray(pts_lat, float), np.asarray(pts_lon, float),
                               ct, cla, clo, cv, wlo, whi, rr, need_loc)
    out = {}
    for a, w in enumerate(names):
        for b, r in enumerate(rr):
            key = f"{prefix}{w}" + (f"_{int(r)}km" if need_loc else "")
            c = cnt[:, a, b].astype(float)
            c[c < 0] = np.nan
            out[f"n_{key}"] = c
            if value_col:
                out[f"max_{key}"] = vmax[:, a, b]
    df = pd.DataFrame(out)
    if coverage is not None:
        lo, hi = to_sec([coverage[0]])[0], to_sec([coverage[1]])[0]
        bad = (np.asarray(pts_t) < lo + max(0, -wlo.min())) | (np.asarray(pts_t) > hi - max(0, whi.max()))
        df.loc[bad, :] = np.nan
    return df


H = 3600.0
D = 86400.0


# --------------------------------------------------------------------------- space weather
def space_weather_features(pts_t) -> pd.DataFrame:
    t = pd.to_datetime(pd.Series(pts_t) * 1e9, utc=True, unit="ns") if not isinstance(pts_t, pd.Series) else pts_t
    tsec = np.asarray(pts_t, float)
    kp = C.kp3h()
    ks = to_sec(kp.time)
    kv = kp.kp.to_numpy(float)
    i = np.searchsorted(ks, tsec, side="right") - 1
    i = np.clip(i, 0, len(ks) - 1)
    out = {"kp": np.where(np.isfinite(tsec), kv[i], np.nan)}
    # rolling maxima over prior windows using cumulative arrays
    s = pd.Series(kv, index=pd.to_datetime(ks * 1e9, utc=True, unit="ns"))
    for hrs in (24, 72):
        r = s.rolling(f"{hrs}h").max().to_numpy()
        out[f"kp_max_prior{hrs}h"] = np.where(np.isfinite(tsec), r[i], np.nan)
    out["kp_change_prior24h"] = out["kp"] - np.where(i >= 8, kv[np.clip(i - 8, 0, None)], np.nan)
    om = C.omni_hourly()
    os_ = to_sec(om.time)
    j = np.clip(np.searchsorted(os_, tsec, side="right") - 1, 0, len(os_) - 1)
    ok = np.isfinite(tsec) & (np.abs(os_[j] - tsec) <= 2 * H)
    oi = om.set_index(pd.to_datetime(os_ * 1e9, utc=True, unit="ns"))
    for col in ("dst", "ae", "bz_gsm", "v", "np", "pdyn", "pflux10", "efield"):
        out[col] = np.where(ok, om[col].to_numpy(float)[j], np.nan)
    roll = {"dst_min_prior24h": ("dst", "24h", "min"), "ae_max_prior6h": ("ae", "6h", "max"),
            "bz_min_prior3h": ("bz_gsm", "3h", "min"), "v_max_prior24h": ("v", "24h", "max"),
            "pflux10_max_prior24h": ("pflux10", "24h", "max"), "dst_min_prior72h": ("dst", "72h", "min")}
    for name, (col, w, fn) in roll.items():
        r = getattr(oi[col].rolling(w, min_periods=1), fn)().to_numpy()
        out[name] = np.where(ok, r[j], np.nan)
    out["dst_change_prior6h"] = out["dst"] - np.where(ok & (j >= 6), om["dst"].to_numpy(float)[np.clip(j - 6, 0, None)], np.nan)
    sd = C.sw_daily()
    ds = to_sec(pd.to_datetime(sd.date).dt.tz_localize("UTC"))
    k = np.clip(np.searchsorted(ds, tsec, side="right") - 1, 0, len(ds) - 1)
    for col in ("F107", "SN", "Ap"):
        out[col] = np.where(np.isfinite(tsec), sd[col].to_numpy(float)[k], np.nan)
    fl = C.flares()
    fl["v"] = fl.cls.map({"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}) + fl.mag / 10
    fl["lat"] = 0.0
    fl["lon"] = 0.0
    mx = fl[fl.cls.isin(["M", "X"])]
    w = window_counts(tsec, np.zeros(len(tsec)), np.zeros(len(tsec)), mx,
                      {"flareMX_prior24h": (-D, 0), "flareMX_prior72h": (-3 * D, 0)}, [0], value_col="v",
                      need_loc=False, coverage=(fl.time.min(), fl.time.max()))
    df = pd.DataFrame(out)
    return pd.concat([df, w], axis=1)


# --------------------------------------------------------------------------- astronomy & calendar
def astro_calendar_features(pts_t, lat, lon, tz_names=None) -> pd.DataFrame:
    ts = pd.to_datetime(np.asarray(pts_t, float) * 1e9, utc=True, unit="ns")
    ok = np.isfinite(np.asarray(pts_t, float)) & np.isfinite(lat) & np.isfinite(lon)
    ok &= (ts > pd.Timestamp("1900-01-02", tz="UTC")) & (ts < pd.Timestamp("2053-01-01", tz="UTC"))  # DE421 range
    out = pd.DataFrame(index=range(len(ts)))
    if ok.any():
        a = C.astro_features(ts[ok], np.asarray(lat)[ok], np.asarray(lon)[ok])
        for c in a.columns:
            out.loc[ok, c] = a[c].to_numpy()
        s = C.shower_activity(ts[ok])
        for c in s.columns:
            out.loc[ok, c] = s[c].to_numpy()
    # Local solar time (no tz database needed; used for matching/time-of-day)
    lst = (ts.hour + ts.minute / 60.0 + np.asarray(lon) / 15.0) % 24
    out["local_solar_hour"] = np.where(ok, lst, np.nan)
    if tz_names is not None:
        loc_hour, dow, mmdd = [], [], []
        for t, z in zip(ts, tz_names):
            if pd.isna(t) or not isinstance(z, str):
                loc_hour.append(np.nan); dow.append(np.nan); mmdd.append(None)
                continue
            lt = t.tz_convert(z)
            loc_hour.append(lt.hour + lt.minute / 60); dow.append(lt.dayofweek); mmdd.append(lt.strftime("%m-%d"))
        out["local_hour"] = loc_hour
        out["dow"] = dow
        out["mmdd"] = mmdd
        for k, name in C.HOLIDAYS_FIXED.items():
            out[f"hol_{name}"] = (out["mmdd"] == k).astype(float)
        out.loc[out["mmdd"].isna(), [f"hol_{n}" for n in C.HOLIDAYS_FIXED.values()]] = np.nan
    out["year"] = ts.year
    out["month"] = ts.month
    od = pd.to_datetime(C.OUTBURSTS and [o[0] for o in C.OUTBURSTS])
    ob = np.zeros(len(ts))
    for d in od:
        ob = np.maximum(ob, (np.abs((ts.tz_convert(None) - d).total_seconds()) <= 1.5 * D).astype(float))
    out["meteor_outburst_pm1d"] = np.where(ok, ob, np.nan)
    return out


# --------------------------------------------------------------------------- catalogs
QUAKE_WINDOWS = {"pre7d": (-7 * D, 0), "pre1d": (-D, 0), "pre3h": (-3 * H, 0), "post1d": (0, D), "post7d": (0, 7 * D)}
QUAKE_RADII = [25, 50, 100, 250, 500]


def quake_features(pts_t, lat, lon) -> pd.DataFrame:
    q = C.quakes().rename(columns={"latitude": "lat", "longitude": "lon"})
    a = window_counts(pts_t, lat, lon, q, QUAKE_WINDOWS, QUAKE_RADII, value_col="mag", prefix="eq_",
                      coverage=(pd.Timestamp("1973-01-01", tz="UTC"), q.time.max()))
    q4 = q[q.mag >= 4.0]
    b = window_counts(pts_t, lat, lon, q4, QUAKE_WINDOWS, [100, 250, 500], prefix="eq4_",
                      coverage=(pd.Timestamp("1973-01-01", tz="UTC"), q.time.max()))
    return pd.concat([a, b], axis=1)


def fireball_features(pts_t, lat, lon) -> pd.DataFrame:
    f = C.fireballs().dropna(subset=["lat"])
    return window_counts(pts_t, lat, lon, f, {"pm30m": (-1800, 1800), "pm3h": (-3 * H, 3 * H), "pm1d": (-D, D)},
                         [250, 500, 1000], value_col="energy", prefix="fb_",
                         coverage=(pd.Timestamp("1988-04-01", tz="UTC"), f.time.max()))


def launch_features(pts_t, lat, lon) -> pd.DataFrame:
    L = C.launches()
    L = L[L.time_has_clock & L.lat.notna()]
    win = {"lpost3h": (-3 * H, 0.5 * H), "lpost1h": (-1 * H, 0.25 * H), "lpm24h": (-D, D)}
    # window semantics: catalog time minus point time; a launch 0-3 h BEFORE the sighting -> dt in [-3h, +0.5h]
    a = window_counts(pts_t, lat, lon, L, win, [500, 1000, 1500, 2500], prefix="launch_",
                      coverage=(pd.Timestamp("1957-01-01", tz="UTC"), L.time.max()))
    orb = L[L.orbital]
    b = window_counts(pts_t, lat, lon, orb, {"lpost3h": (-3 * H, 0.5 * H)}, [1000, 2500], prefix="launchorb_")
    sub = L[~L.orbital]
    c = window_counts(pts_t, lat, lon, sub, {"lpost3h": (-3 * H, 0.5 * H)}, [1000, 2500], prefix="launchsub_")
    out = pd.concat([a, b, c], axis=1)
    # Starlink: days since the most recent Starlink launch (global)
    st = to_sec(L[L.is_starlink].time)
    st.sort()
    tt = np.asarray(pts_t, float)
    k = np.searchsorted(st, tt, side="right") - 1
    ds = np.where(k >= 0, (tt - st[np.clip(k, 0, None)]) / D, np.nan)
    ds[tt < st[0]] = np.nan  # pre-Starlink era: not applicable
    out["days_since_starlink_launch"] = ds
    for w in (7, 14, 30):
        lo = np.searchsorted(st, tt - w * D)
        out[f"n_starlink_launches_prior{w}d"] = np.where(np.isfinite(tt), k + 1 - lo, np.nan)
    out["starlink_era"] = (tt >= st[0]).astype(float)
    return out


def storm_event_features(pts_t, lat, lon, storms: pd.DataFrame) -> pd.DataFrame:
    return window_counts(pts_t, lat, lon, storms, {"pm1h": (-H, H), "pm3h": (-3 * H, 3 * H), "pm12h": (-12 * H, 12 * H)},
                         [25, 50, 100], prefix="storm_",
                         coverage=(pd.Timestamp("1996-01-01", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")))


def media_features(pts_t) -> pd.DataFrame:
    tt = np.asarray(pts_t, float)
    me = np.sort(to_sec([m[0] for m in C.MEDIA_EVENTS]))
    k = np.searchsorted(me, tt, side="right") - 1
    out = {"days_since_media_event": np.where(k >= 0, (tt - me[np.clip(k, 0, None)]) / D, np.nan)}
    out["media_event_within30d"] = (out["days_since_media_event"] <= 30).astype(float)
    out["media_event_within7d"] = (out["days_since_media_event"] <= 7).astype(float)
    w = C.wiki_pageviews()
    w = w[w.article == "Unidentified_flying_object"].sort_values("date")
    ws = to_sec(w.date.dt.tz_localize("UTC"))
    j = np.searchsorted(ws, tt, side="right") - 1
    ok = (j >= 0) & (tt <= ws[-1] + D)
    out["wiki_ufo_views"] = np.where(ok, w.views.to_numpy(float)[np.clip(j, 0, None)], np.nan)
    return pd.DataFrame(out)
