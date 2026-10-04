"""Multi-dataset convergence events, ranked separately from population trends.

For every event, independent indicators (each from a different data stream):
  I1  multiple independent witnesses / reports (REPORT_COUNT >= 2 from >= 2
      separated locations, or INDEPENDENT_SOURCE_COUNT >= 2)
  I2  radar involvement reported (structured > text)
  I3  photographic / video evidence reported
  I4  trained observer (pilot, ATC, astronomer, police, military)
  I5  no conventional match: P_EXPLAINED < 0.1 AND no launch/fireball/ISS/
      Starlink/Venus context flag
  I6  precise time (<= 15 min) and location (<= 10 km)
  I7  ground magnetometer anomaly: max |dH/dt| at the nearest USGS observatory
      within +-30 min exceeds the same window on all 14 surrounding days (rank test) (queried only for the top candidates; NaN = DATA_UNAVAILABLE)
  I8  unusual local atmosphere: |24h pressure change| >= 8 hPa or radiative
      inversion proxy at night (from ISD; NaN if no station)
Score = number of indicators present. Ranked list -> results/convergence_events.csv
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from common import PROCESSED, RESULTS, RAW, fetch, haversine_km

USGS_OBS = {  # USGS geomagnetic observatories (lat, lon) — public metadata
    "BOU": (40.137, -105.237), "FRD": (38.205, -77.373), "FRN": (37.091, -119.719), "TUC": (32.174, -110.734),
    "BSL": (30.350, -89.640), "NEW": (48.265, -117.123), "SIT": (57.058, -135.327), "CMO": (64.874, -147.860),
    "SJG": (18.111, -66.150), "HON": (21.316, -158.000), "DED": (70.356, -148.793), "SHU": (55.348, -160.462),
}


def magnetometer_anomaly(t_utc: pd.Timestamp, lat, lon):
    """Return (obs, ratio, flag): ratio = max |dH/dt| within +-30 min divided by the median of the
    same-window maxima on days -7..-1 and +1..+7; flag = event window exceeds all control windows
    (exact rank test; null rate 1/15). Global storms affect all observatories, so a flag does
    NOT imply a local anomaly - see Kp at the time."""
    if t_utc.year < 1991:
        return None, np.nan, np.nan
    best = min(USGS_OBS, key=lambda k: haversine_km(lat, lon, *USGS_OBS[k]))
    if haversine_km(lat, lon, *USGS_OBS[best]) > 1500:
        return best, np.nan, np.nan
    start = (t_utc - pd.Timedelta(days=7, hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (t_utc + pd.Timedelta(days=7, hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    dest = RAW / "geomag" / f"{best}_{t_utc.strftime('%Y%m%dT%H%M')}.json"
    p = fetch("https://geomag.usgs.gov/ws/data/", dest, f"USGS geomag 1-min H {best} around event",
              params=dict(id=best, starttime=start, endtime=end, elements="H", sampling_period=60, format="json"))
    if p is None:
        return best, np.nan, np.nan
    try:
        d = json.load(open(p))
        times = pd.to_datetime(d["times"], utc=True)
        h = np.array([np.nan if v is None else v for v in d["values"][0]["values"]], float)
    except Exception:
        return best, np.nan, np.nan
    s = pd.Series(h, index=times)
    dh = s.diff().abs()

    def wmax(center):
        w = dh[(dh.index >= center - pd.Timedelta(minutes=30)) & (dh.index <= center + pd.Timedelta(minutes=30))]
        return np.nanmax(w) if w.notna().sum() >= 30 else np.nan

    ev_max = wmax(t_utc)
    ctl = np.array([wmax(t_utc + pd.Timedelta(days=k)) for k in list(range(-7, 0)) + list(range(1, 8))])
    ctl = ctl[np.isfinite(ctl)]
    if not np.isfinite(ev_max) or len(ctl) < 10:
        return best, np.nan, np.nan
    p_rank = (1 + np.sum(ctl >= ev_max)) / (len(ctl) + 1)
    ratio = float(ev_max / np.median(ctl)) if np.median(ctl) > 0 else np.nan
    # flag = event window is the most disturbed of all same-hour windows (null rate 1/(n+1) ~ 0.067)
    return best, ratio, float(p_rank <= 1 / (len(ctl) + 1) + 1e-12)


def main(top_n_mag=120):
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    pts = pd.read_parquet(PROCESSED / ("points.parquet" if (PROCESSED / "points.parquet").exists() else "points_core.parquet"),
                          columns=None)
    case = pts[pts.strategy == "CASE"].set_index("EVENT_ID")
    e = ev.set_index("EVENT_ID")
    f = lambda c: e[c].fillna(False).astype(bool) if c in e else pd.Series(False, index=e.index)
    I = pd.DataFrame(index=e.index)
    I["I1_multi_witness"] = ((e.REPORT_COUNT >= 2) & e.SEPARATED_WITNESSES.fillna(False)) | (e.INDEPENDENT_SOURCE_COUNT >= 2) | \
        (e.NUMBER_OF_WITNESSES.fillna(0) >= 3) & (e.NUMBER_OF_WITNESSES_BASIS == "structured")
    I["I2_radar"] = e.RADAR_INVOLVEMENT != "NOT_REPORTED"
    I["I3_photo_video"] = (e.PHOTOGRAPHIC_EVIDENCE != "NOT_REPORTED") | (e.VIDEO_EVIDENCE != "NOT_REPORTED")
    I["I4_trained_observer"] = e.WITNESS_TYPE.fillna("").str.contains("pilot|air_traffic|astronomer|police|military")
    ctx_flag = pd.Series(False, index=e.index)
    for c in ["n_launch_lpost3h_1500km", "n_fb_pm30m_1000km", "iss_visible_win"]:
        if c in case:
            ctx_flag = ctx_flag | (case[c].reindex(e.index).fillna(0) > 0)
    if "days_since_starlink_launch" in case:
        ctx_flag = ctx_flag | (case.days_since_starlink_launch.reindex(e.index) <= 10)
    gclass = e.GEIPAN_CLASS.fillna("")
    I["I5_no_conventional_match"] = np.where(e.SOURCE == "GEIPAN", gclass == "D", (e.P_EXPLAINED < 0.1) & ~ctx_flag)
    I["I6_precise"] = (e.TIME_UNCERTAINTY_MIN <= 15) & (e.LOCATION_UNCERTAINTY_RADIUS_KM <= 10)
    if "wx_dslp_24h" in case:
        wx = case.reindex(e.index)
        I["I8_unusual_atmosphere"] = ((wx.wx_dslp_24h.abs() >= 8) | ((wx.wx_inversion_proxy > 0) & (wx.sun_alt < -6)))
        I.loc[wx.wx_sky_oktas.isna(), "I8_unusual_atmosphere"] = np.nan
    else:
        I["I8_unusual_atmosphere"] = np.nan
    I["I7_magnetometer"] = np.nan
    I["score_without_mag"] = I.drop(columns=["I7_magnetometer"]).fillna(0).astype(float).sum(axis=1)
    out = e[["SOURCE", "DATE", "LOCAL_TIME", "LATITUDE", "LONGITUDE", "LOCATION_TEXT", "SHAPE", "EVENT_QUALITY",
             "P_EXPLAINED", "TOP_EXPLANATION", "GEIPAN_CLASS", "REPORT_COUNT", "SPLIT"]].join(I)
    out = out.sort_values(["score_without_mag", "EVENT_QUALITY"], ascending=False)
    # magnetometer only for top candidates with precise UTC time in the USGS era, US locations
    cand = out[(out.score_without_mag >= 4)].copy()
    cand = cand[cand.index.isin(e[e.utc_ts.notna() & (e.YEAR >= 1991) & e.COUNTRY.isin(["US"])].index)].head(top_n_mag)
    for eid in cand.index:
        obs, ratio, flag = magnetometer_anomaly(e.at[eid, "utc_ts"], e.at[eid, "LATITUDE"], e.at[eid, "LONGITUDE"])
        out.at[eid, "I7_magnetometer"] = flag
        out.at[eid, "mag_obs"] = obs
        out.at[eid, "mag_ratio"] = ratio
        time.sleep(0.3)
    out["score"] = out[[c for c in out.columns if c.startswith("I")]].fillna(0).astype(float).sum(axis=1)
    out = out.sort_values(["score", "EVENT_QUALITY"], ascending=False)
    out.head(500).to_csv(RESULTS / "convergence_events.csv")
    dist = out.groupby("SOURCE").score_without_mag.value_counts().unstack(fill_value=0)
    dist.to_csv(RESULTS / "convergence_score_distribution.csv")
    print(dist)
    print(out.head(25)[["SOURCE", "DATE", "LOCATION_TEXT", "score", "score_without_mag", "mag_ratio", "P_EXPLAINED"]].to_string())
    # magnetometer anomaly base rate check among queried candidates vs. expectation (1% by construction)
    q = out.I7_magnetometer.dropna()
    print(f"magnetometer queried {len(q)}; anomaly rate {q.mean():.3f} (exact null expectation 1/15 = 0.067)")


if __name__ == "__main__":
    main()
