"""Build case/control point sets and compute identical covariates for each.

Control strategies (every case gets all of them; analyses report each):
  CS1  time-stratified case-crossover: same location, same LOCAL clock time,
       same weekday, all other such days within the same calendar month
       (Janes et al. 2005 referent design; avoids overlap bias).
  CS2  same location, same local clock time, +-364 and +-728 days
       (same weekday, near-identical season; different year).
  CS4  spatial controls: same UTC instant, K=4 populated places drawn from the
       same country, same population-size decile (+-1), >= 200 km away and
       (US) same Census region; place sampled proportional to population.
Cases with no UTC time are excluded here (date-only analyses are separate).

Output: data/processed/points.parquet (one row per case or control point)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features as F
import spatial
import weather
import iss
from common import PROCESSED, INTERIM, MASTER_SEED, haversine_km
import geo

CENSUS_REGION = {  # state -> Census region
    **{s: "NE" for s in "CT ME MA NH RI VT NJ NY PA".split()},
    **{s: "MW" for s in "IL IN MI OH WI IA KS MN MO NE ND SD".split()},
    **{s: "S" for s in "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX".split()},
    **{s: "W" for s in "AZ CO ID MT NV NM UT WY AK CA HI OR WA".split()},
}


def load_cases() -> pd.DataFrame:
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[ev.utc_ts.notna() & ev.LATITUDE.notna() & ev.TIMEZONE.notna()].copy()
    ev["t"] = F.to_sec(ev.utc_ts)
    return ev.reset_index(drop=True)


def _localize_referent(naive: pd.Timestamp, tz: str):
    """Explicit DST policy for a referent wall-clock time:
    nonexistent (spring-forward gap) -> referent DROPPED (no equivalent clock time);
    ambiguous (fall-back repeated hour) -> the first (daylight-time) instance."""
    try:
        return naive.tz_localize(tz, ambiguous=True, nonexistent="NaT").tz_convert("UTC")
    except Exception:
        return pd.NaT


def cs1_referents(ev: pd.DataFrame) -> pd.DataFrame:
    """Same local wall-clock time and weekday, other weeks of the same calendar month.
    Calendar offsets are applied to the NAIVE local time (fixes DST drift; audit item 1)."""
    rows = []
    for e in ev.itertuples():
        naive = e.utc_ts.tz_convert(e.TIMEZONE).tz_localize(None)
        for k in (-4, -3, -2, -1, 1, 2, 3, 4):
            c = naive + pd.Timedelta(days=7 * k)
            if c.month != naive.month:
                continue
            cu = _localize_referent(c, e.TIMEZONE)
            if pd.isna(cu):
                continue
            rows.append((e.EVENT_ID, "CS1", k, cu, e.LATITUDE, e.LONGITUDE))
    return pd.DataFrame(rows, columns=["EVENT_ID", "strategy", "offset", "utc_ts", "lat", "lon"])


def cs2_referents(ev: pd.DataFrame) -> pd.DataFrame:
    """Same local wall-clock time, +-364 / +-728 calendar days (same weekday)."""
    rows = []
    for e in ev.itertuples():
        naive = e.utc_ts.tz_convert(e.TIMEZONE).tz_localize(None)
        for k in (-728, -364, 364, 728):
            cu = _localize_referent(naive + pd.Timedelta(days=k), e.TIMEZONE)
            if pd.isna(cu):
                continue
            rows.append((e.EVENT_ID, "CS2", k, cu, e.LATITUDE, e.LONGITUDE))
    return pd.DataFrame(rows, columns=["EVENT_ID", "strategy", "offset", "utc_ts", "lat", "lon"])


def check_referents(ev: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Verify every referent has the case's local wall-clock time and weekday."""
    tz = ev.set_index("EVENT_ID").TIMEZONE
    case_t = ev.set_index("EVENT_ID").utc_ts
    bad_clock = bad_dow = 0
    for r in ref.itertuples():
        z = tz[r.EVENT_ID]
        a = case_t[r.EVENT_ID].tz_convert(z)
        b = r.utc_ts.tz_convert(z)
        bad_clock += (a.hour, a.minute) != (b.hour, b.minute)
        bad_dow += a.dayofweek != b.dayofweek
    return dict(n=len(ref), clock_mismatch=int(bad_clock), weekday_mismatch=int(bad_dow))


def cs4_spatial(ev: pd.DataFrame, k=4) -> pd.DataFrame:
    rng = np.random.default_rng(MASTER_SEED + 4)
    G = geo.geonames()
    G = G[G.population > 0].copy()
    cc_map = {"USA": "US", "Canada": "CA", "France": "FR", "United Kingdom": "GB"}
    rows = []
    # population of the case location: nearest geonames place within 10 km
    from sklearn.neighbors import BallTree
    for cc, sub in ev.groupby(ev.COUNTRY.replace(cc_map).fillna("")):
        Gc = G[G.country == cc]
        if len(Gc) < 50:
            continue
        tree = BallTree(np.radians(Gc[["lat", "lon"]].to_numpy()), metric="haversine")
        d, i = tree.query(np.radians(sub[["LATITUDE", "LONGITUDE"]].to_numpy()), k=1)
        case_pop = Gc.population.to_numpy()[i[:, 0]]
        dec_edges = np.quantile(np.log10(Gc.population.to_numpy()), np.linspace(0, 1, 11))
        gdec = np.clip(np.searchsorted(dec_edges, np.log10(Gc.population.to_numpy()), side="right") - 1, 0, 9)
        cdec = np.clip(np.searchsorted(dec_edges, np.log10(case_pop), side="right") - 1, 0, 9)
        reg = Gc.admin1.map(CENSUS_REGION).fillna("X").to_numpy() if cc == "US" else np.array(["X"] * len(Gc))
        st = sub.STATE.fillna("").to_numpy()
        creg = np.array([CENSUS_REGION.get(s, "X") for s in st]) if cc == "US" else np.array(["X"] * len(sub))
        glat, glon, gpop = Gc.lat.to_numpy(), Gc.lon.to_numpy(), Gc.population.to_numpy(float)
        for j, e in enumerate(sub.itertuples()):
            m = (np.abs(gdec - cdec[j]) <= 1) & (reg == creg[j])
            cand = np.where(m)[0]
            if len(cand) < k:
                continue
            dd = haversine_km(e.LATITUDE, e.LONGITUDE, glat[cand], glon[cand])
            cand = cand[dd >= 200]
            if len(cand) < k:
                continue
            p = gpop[cand] / gpop[cand].sum()
            pick = rng.choice(cand, size=k, replace=False, p=p)
            for q, g in enumerate(pick):
                rows.append((e.EVENT_ID, "CS4", q, e.utc_ts, glat[g], glon[g]))
    return pd.DataFrame(rows, columns=["EVENT_ID", "strategy", "offset", "utc_ts", "lat", "lon"])


def compute_features(pts: pd.DataFrame, chunk=150_000) -> pd.DataFrame:
    """Chunked to bound memory (session cgroup limit ~14 GB)."""
    outs = []
    for i in range(0, len(pts), chunk):
        print(f"  chunk {i // chunk + 1}/{(len(pts) - 1) // chunk + 1}", flush=True)
        o = _compute_features(pts.iloc[i:i + chunk])
        for c in o.columns:
            if o[c].dtype == "float64":
                o[c] = o[c].astype("float32")
        outs.append(o)
    return pd.concat(outs)


def _compute_features(pts: pd.DataFrame) -> pd.DataFrame:
    t = F.to_sec(pts.utc_ts)
    la, lo = pts.lat.to_numpy(float), pts.lon.to_numpy(float)
    tz = pts.TIMEZONE.to_numpy()
    parts = [F.space_weather_features(t)]
    parts.append(F.astro_calendar_features(t, la, lo, tz))
    parts += [F.quake_features(t, la, lo), F.fireball_features(t, la, lo), F.launch_features(t, la, lo),
              F.storm_event_features(t, la, lo, weather.storm_events()), F.media_features(t)]
    parts.append(pd.DataFrame({"iss_visible_win": iss.iss_visible_window(t, la, lo, np.full(len(t), 10.0))}))
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    out.index = pts.index
    return out


def main():
    ev = load_cases()
    print("cases with UTC+coords:", len(ev), ev.SOURCE.value_counts().to_dict(), flush=True)
    case_pts = pd.DataFrame({"EVENT_ID": ev.EVENT_ID, "strategy": "CASE", "offset": 0, "utc_ts": ev.utc_ts,
                             "lat": ev.LATITUDE, "lon": ev.LONGITUDE})
    print("CS1", flush=True); c1 = cs1_referents(ev)
    print("CS2", flush=True); c2 = cs2_referents(ev)
    print("CS4", flush=True); c4 = cs4_spatial(ev)
    pts = pd.concat([case_pts, c1, c2, c4], ignore_index=True)
    meta = ev.set_index("EVENT_ID")[["SOURCE", "SPLIT", "TIMEZONE", "YEAR", "HIGH_QUALITY", "HQ_UNEXPLAINED",
                                     "UNEXPLAINED", "MULTI_SENSOR", "MULTI_SENSOR_RADAR", "EVENT_QUALITY",
                                     "INFORMATION_QUALITY", "P_EXPLAINED", "TIME_UNCERTAINTY_MIN", "COUNTRY",
                                     "GEO_REGION", "GEIPAN_CLASS", "REPORT_COUNT", "SHAPE"]]
    pts = pts.join(meta, on="EVENT_ID")
    pts["is_case"] = (pts.strategy == "CASE").astype(int)
    pts["YEAR_PT"] = pts.utc_ts.dt.year
    print("points:", len(pts), pts.strategy.value_counts().to_dict(), flush=True)
    feats = compute_features(pts)
    print("  spatial", flush=True)
    sp = spatial.spatial_features(pts.lat.to_numpy(), pts.lon.to_numpy(), pts.YEAR_PT.to_numpy())
    sp.index = pts.index
    out = pd.concat([pts, feats, sp], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.to_parquet(PROCESSED / "points_core.parquet", index=False)
    print("saved", out.shape)


def add_weather():
    pts = pd.read_parquet(PROCESSED / "points_core.parquet")
    t = F.to_sec(pts.utc_ts)
    w = weather.weather_features(t, pts.lat.to_numpy(float), pts.lon.to_numpy(float))
    w.index = pts.index
    out = pd.concat([pts, w], axis=1)
    out.to_parquet(PROCESSED / "points.parquet", index=False)
    print("saved with weather", out.shape, "weather coverage", w.wx_sky_oktas.notna().mean())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "weather":
        add_weather()
    else:
        main()
