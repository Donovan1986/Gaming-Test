"""Registered discovery-phase hypothesis tests (NUFORC DISCOVERY split only).

Families
  A  global temporal exposures (space weather, solar, lunar, showers, media) - CS1, CS2
  B  local temporal exposures (quakes, storms, weather, ISS, fireballs, launches) - CS1, CS2, CS4
  C  spatial exposures (infrastructure / geography) - CS4 (population-matched, same instant)
  G  window x radius grid for catalog exposures (+-1 min .. +-30 d; 1 .. 500 km) - CS1/CS4
Subsets: ALL, HQ, HQ_UNEXPLAINED, MULTI_SENSOR, UNEXPLAINED, EXPLAINED (contrast).
Events with local time 00:00 (time uncertainty 720 min) are excluded from
temporal tests (sensitivity analyses re-include them).
All results -> results/hypothesis_registry.csv and results/discovery_results.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cc
import features as F
import covariates as C
import weather
import stats as S
from common import RESULTS


def subsets(ev):
    return {
        "ALL": ev.EVENT_ID,
        "HQ": ev.loc[ev.HIGH_QUALITY, "EVENT_ID"],
        "HQ_UNEXPLAINED": ev.loc[ev.HQ_UNEXPLAINED, "EVENT_ID"],
        "MULTI_SENSOR": ev.loc[ev.MULTI_SENSOR, "EVENT_ID"],
        "UNEXPLAINED": ev.loc[ev.UNEXPLAINED, "EVENT_ID"],
        "EXPLAINED": ev.loc[ev.P_EXPLAINED >= 0.6, "EVENT_ID"],
    }


def derive(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    b = lambda cond, base: cond.astype("float32").where(base.notna())
    p["kp_storm5_prior24h"] = b(p.kp_max_prior24h >= 5, p.kp_max_prior24h)
    p["kp_storm7_prior24h"] = b(p.kp_max_prior24h >= 7, p.kp_max_prior24h)
    p["dst_storm50_prior24h"] = b(p.dst_min_prior24h <= -50, p.dst_min_prior24h)
    p["dst_storm100_prior72h"] = b(p.dst_min_prior72h <= -100, p.dst_min_prior72h)
    p["ae_substorm500_prior6h"] = b(p.ae_max_prior6h >= 500, p.ae_max_prior6h)
    p["bz_south10_prior3h"] = b(p.bz_min_prior3h <= -10, p.bz_min_prior3h)
    p["hss600_prior24h"] = b(p.v_max_prior24h >= 600, p.v_max_prior24h)
    p["proton_event_prior24h"] = b(p.pflux10_max_prior24h >= 10, p.pflux10_max_prior24h)
    p["flareMX_any_prior24h"] = b(p.n_flareMX_prior24h > 0, p.n_flareMX_prior24h)
    p["log_wiki_views"] = np.log10(p.wiki_ufo_views)
    p["moon_up_illum"] = ((p.moon_alt > 0) * p.moon_illum).where(p.moon_alt.notna())
    p["moon_full_up"] = b((p.moon_alt > 0) & (p.moon_illum > 0.9), p.moon_alt)
    p["venus_vis"] = b((p.venus_alt > 3) & (p.sun_alt < -3) & (p.venus_elong > 15), p.venus_alt)
    p["shower50"] = b(p.shower_zhr_total >= 50, p.shower_zhr_total)
    p["starlink_10d"] = ((p.days_since_starlink_launch <= 10) & (p.starlink_era > 0)).astype(float).where(p.starlink_era > 0)
    p["fireball_30m_1000"] = b(p.n_fb_pm30m_1000km > 0, p.n_fb_pm30m_1000km)
    p["launch_3h_1500"] = b(p.n_launch_lpost3h_1500km > 0, p.n_launch_lpost3h_1500km)
    if "wx_sky_oktas" in p:
        p["wx_clear"] = b(p.wx_sky_oktas <= 2, p.wx_sky_oktas)
        p["wx_overcast"] = b(p.wx_sky_oktas >= 7, p.wx_sky_oktas)
        p["wx_front_pressure_fall"] = b(p.wx_dslp_24h <= -8, p.wx_dslp_24h)
        p["wx_cold_front_proxy"] = b((p.wx_dtemp_24h <= -8) & (p.wx_dslp_24h > 0), p.wx_dtemp_24h)
        p["wx_clearing_6h"] = b(p.wx_dsky_6h <= -4, p.wx_dsky_6h)
        p["wx_inversion_night"] = (p.wx_inversion_proxy * (p.sun_alt < -6)).where(p.wx_inversion_proxy.notna())
        p["wx_precip_any"] = b(p.wx_precip_1h_mm > 0, p.wx_precip_1h_mm)
    for c in [c for c in p.columns if c.startswith("n_eq_") or c.startswith("n_eq4_") or c.startswith("n_storm_")]:
        p[c.replace("n_", "any_", 1)] = b(p[c] > 0, p[c])
    for c in [c for c in p.columns if c.startswith("dist_") and c.endswith("_km")]:
        p["log_" + c] = np.log10(p[c].clip(lower=0.5)).astype("float32")
        for r in (1, 5, 10, 25, 50, 100, 250, 500):
            p[f"within{r}_{c[5:-3]}"] = b(p[c] <= r, p[c])
    return p.copy()


FAM_A = [  # (variable, scale, label)
    ("kp", 1, "Kp at time (per unit)"), ("kp_max_prior24h", 1, "max Kp prior 24h"), ("kp_max_prior72h", 1, "max Kp prior 72h"),
    ("kp_change_prior24h", 1, "Kp change over 24h"), ("kp_storm5_prior24h", 1, "Kp>=5 prior 24h"),
    ("kp_storm7_prior24h", 1, "Kp>=7 prior 24h"), ("dst", 10, "Dst per 10 nT"), ("dst_min_prior24h", 10, "min Dst prior 24h per 10 nT"),
    ("dst_storm50_prior24h", 1, "Dst<=-50 prior 24h"), ("dst_storm100_prior72h", 1, "Dst<=-100 prior 72h"),
    ("dst_change_prior6h", 10, "Dst change 6h per 10 nT"), ("ae", 100, "AE per 100 nT"), ("ae_max_prior6h", 100, "max AE prior 6h per 100 nT"),
    ("ae_substorm500_prior6h", 1, "AE>=500 prior 6h"), ("bz_gsm", 1, "IMF Bz (nT)"), ("bz_min_prior3h", 1, "min Bz prior 3h"),
    ("bz_south10_prior3h", 1, "Bz<=-10 prior 3h"), ("v", 100, "solar wind speed per 100 km/s"),
    ("hss600_prior24h", 1, "solar wind >=600 km/s prior 24h"), ("np", 1, "proton density"), ("pdyn", 1, "dynamic pressure (nPa)"),
    ("efield", 1, "interplanetary E field"), ("pflux10", 1, "proton flux >10 MeV"), ("proton_event_prior24h", 1, ">=10 pfu prior 24h"),
    ("F107", 10, "F10.7 per 10 sfu"), ("SN", 10, "sunspot number per 10"), ("Ap", 10, "daily Ap per 10"),
    ("flareMX_any_prior24h", 1, "M/X flare prior 24h"), ("n_flareMX_prior72h", 1, "M/X flares prior 72h (count)"),
    ("moon_illum", 0.1, "moon illuminated fraction per 0.1"), ("moon_up_illum", 0.1, "moon up x illumination per 0.1"),
    ("moon_full_up", 1, "full moon above horizon"), ("venus_vis", 1, "Venus conspicuous"), ("shower50", 1, "meteor shower ZHR>=50"),
    ("media_event_within7d", 1, "major UAP media event in prior 7 d"), ("media_event_within30d", 1, "major UAP media event in prior 30 d"),
    ("log_wiki_views", 1, "log10 Wikipedia UFO pageviews"), ("jupiter_alt", 10, "Jupiter altitude per 10 deg"),
]
FAM_B_WX = [("wx_sky_oktas", 1), ("wx_clear", 1), ("wx_overcast", 1), ("wx_temp_c", 5), ("wx_dewpt_spread_c", 5), ("wx_wind_ms", 1),
            ("wx_slp_hpa", 5), ("wx_dslp_3h", 1), ("wx_dslp_24h", 5), ("wx_dtemp_24h", 5), ("wx_front_pressure_fall", 1),
            ("wx_cold_front_proxy", 1), ("wx_clearing_6h", 1), ("wx_inversion_night", 1), ("wx_inversion_proxy", 1),
            ("wx_precip_any", 1), ("wx_sky_mean_prior6h", 1)]
FAM_B_OTHER = [("iss_visible_win", 1), ("fireball_30m_1000", 1), ("launch_3h_1500", 1)]


def run_family(pts, ev_sub, variables, strategies, family, split="discovery", window_of=None, radius_of=None):
    rows = []
    for sname, ids in ev_sub.items():
        for v, scale, *lab in variables:
            if v not in pts:
                continue
            for st in strategies:
                r = cc.run(pts, ids, v, st, family=family, hypothesis=(lab[0] if lab else v), var_a=v,
                           subset=sname, split=split, scale=scale,
                           window=(window_of(v) if window_of else ""), radius=(radius_of(v) if radius_of else ""))
                rows.append(r)
    return rows


def win_of(v):
    import re
    m = re.search(r"_(pre\w+?|post\w+?|pm\w+?)_", v + "_")
    return m.group(1) if m else ""


def rad_of(v):
    import re
    m = re.search(r"(\d+)km", v)
    if m:
        return m.group(1)
    m = re.match(r"within(\d+)_", v)
    return m.group(1) if m else ""


GRID_WINDOWS = {"pm1m": (-60, 60), "pm5m": (-300, 300), "pm15m": (-900, 900), "pm30m": (-1800, 1800), "pm1h": (-3600, 3600),
                "pm3h": (-3 * 3600, 3 * 3600), "pm6h": (-6 * 3600, 6 * 3600), "pm24h": (-86400, 86400),
                "pm3d": (-3 * 86400, 3 * 86400), "pm7d": (-7 * 86400, 7 * 86400), "pm30d": (-30 * 86400, 30 * 86400)}
GRID_RADII = [1, 5, 10, 25, 50, 100, 250, 500]


def grid_tests(pts, ev_sub, disc_ids):
    """Window x radius grid for quakes (M>=2.5), storm reports and launches."""
    d = pts[pts.EVENT_ID.isin(set(disc_ids)) & pts.strategy.isin(["CASE", "CS1", "CS4"])].copy()
    t = F.to_sec(d.utc_ts)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    q = C.quakes().rename(columns={"latitude": "lat", "longitude": "lon"})
    L = C.launches()
    L = L[L.time_has_clock & L.lat.notna()]
    cats = {"quake": (q, (pd.Timestamp("1973-01-01", tz="UTC"), q.time.max())),
            "storm": (weather.storm_events(), (pd.Timestamp("1996-01-01", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC"))),
            "launch": (L, (pd.Timestamp("1957-01-01", tz="UTC"), L.time.max()))}
    rows = []
    for cname, (cat, cov) in cats.items():
        w = F.window_counts(t, la, lo, cat, GRID_WINDOWS, GRID_RADII, prefix=f"g{cname}_", coverage=cov)
        w.index = d.index
        for col in w.columns:
            d[col.replace("n_", "any_", 1)] = (w[col] > 0).astype(float).where(w[col].notna())
        for sname in ("ALL", "HQ_UNEXPLAINED"):
            ids = ev_sub[sname]
            for col in w.columns:
                v = col.replace("n_", "any_", 1)
                for st in ("CS1", "CS4"):
                    r = cc.run(d, ids, v, st, family=f"G_{cname}_grid", hypothesis=f"{cname} within window/radius",
                               var_a=v, subset=sname, split="discovery", window=v.split("_")[2],
                               radius=v.split("_")[-1].replace("km", ""))
                    rows.append(r)
    return rows


def main():
    pts = derive(cc.points())
    ev = cc.events()
    disc = ev[(ev.SOURCE == "NUFORC") & (ev.SPLIT == "discovery") & ev.utc_ts.notna()]
    disc_t = disc[disc.TIME_UNCERTAINTY_MIN < 720]
    sub_t = subsets(disc_t)
    sub_all = subsets(disc)
    print({k: len(v) for k, v in sub_t.items()})
    rows = []
    rows += run_family(pts, sub_t, FAM_A, ["CS1", "CS2"], "A_global_temporal")
    print("A done", len(rows), flush=True)
    quake_vars = [(c, 1) for c in pts.columns if c.startswith("any_eq")]
    storm_vars = [(c, 1) for c in pts.columns if c.startswith("any_storm")]
    rows += run_family(pts, sub_t, quake_vars + storm_vars, ["CS1", "CS2", "CS4"], "B_local_catalog",
                       window_of=win_of, radius_of=rad_of)
    rows += run_family(pts, sub_t, [(v, s) for v, s in FAM_B_WX if v in pts], ["CS1", "CS2", "CS4"], "B_weather")
    rows += run_family(pts, sub_t, FAM_B_OTHER, ["CS1", "CS2", "CS4"], "B_local_other")
    print("B done", len(rows), flush=True)
    sp_vars = [(c, 1) for c in pts.columns if c.startswith("within") or c.startswith("log_dist_")]
    sp_vars += [("in_icbm_field_active", 1), ("n_airports_lm_25km", 1), ("n_dod_sites_50km", 1)]
    rows += run_family(pts, sub_all, sp_vars, ["CS4"], "C_spatial", radius_of=rad_of)
    print("C done", len(rows), flush=True)
    rows += grid_tests(pts, sub_t, disc_t.EVENT_ID)
    print("G done", len(rows), flush=True)
    out = pd.DataFrame(rows)
    out["q_bh_all"] = S.bh(out.p_value.to_numpy())
    out["p_holm_all"] = S.holm(out.p_value.to_numpy())
    out.to_csv(RESULTS / "discovery_results.csv", index=False)
    print("tests:", len(out), "finite p:", out.p_value.notna().sum(), "BH q<0.05:", (out.q_bh_all < 0.05).sum())


if __name__ == "__main__":
    main()
