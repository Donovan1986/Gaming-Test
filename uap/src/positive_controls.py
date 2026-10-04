"""Positive controls (known relationships the pipeline MUST rediscover),
placebo exposures (pipeline false-positive rate) and detection power.

Run on the DISCOVERY split only (NUFORC, pre-2016). Holdouts untouched.
Outputs: results/positive_controls.csv, results/placebo_calibration.csv,
         results/detection_power.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cc
import features as F
import covariates as C
import stats as S
from common import RESULTS, MASTER_SEED


def subsets(ev):
    f = lambda c: ev[c].fillna(False).astype(bool)
    dur = ev.OBSERVATION_DURATION_S
    meteor_like = f("d_meteor_word_txt") | (ev.SHAPE == "fireball") | f("d_fireball_shape_txt") | (dur <= 10)
    line = f("d_line_formation_txt")
    planet_like = f("d_planet_word_txt") | (f("d_hover_txt") & (dur >= 600))
    sat_like = f("d_satellite_word_txt") | f("d_iss_word_txt") | ((dur >= 60) & (dur <= 420) & ~f("d_blinking_txt")
                                                                  & ~f("d_hover_txt") & ~f("d_sound_txt"))
    orange = f("d_orange_txt") | (ev.SHAPE == "fireball")
    aircraft_like = (f("d_blinking_txt") & f("d_red_green_txt")) | f("d_aircraft_word_txt") | (f("d_blinking_txt") & f("d_sound_txt"))
    return dict(meteor_like=meteor_like, non_meteor_like=~meteor_like, line_formation=line, planet_like=planet_like,
                satellite_like=sat_like, orange_or_fireball=orange, moon_word=f("d_moon_word_txt"),
                aircraft_like=aircraft_like, non_aircraft_like=~aircraft_like, all=pd.Series(True, index=ev.index))


def add_exposures(p):
    p = p.copy()
    dark = p.sun_alt < -12  # nautical darkness at the observer (meteor visibility); radiant altitude not modelled
    p["x_shower50"] = ((p.shower_zhr_total >= 50) & dark).astype(float).where(p.shower_zhr_total.notna() & p.sun_alt.notna())
    p["x_shower_zhr"] = p.shower_zhr_total
    p["x_fireball_30m_1000"] = (p.n_fb_pm30m_1000km > 0).astype(float).where(p.n_fb_pm30m_1000km.notna())
    p["x_fireball_30m_500"] = (p.n_fb_pm30m_500km > 0).astype(float).where(p.n_fb_pm30m_500km.notna())
    p["x_launch_3h_1500"] = (p.n_launch_lpost3h_1500km > 0).astype(float).where(p.n_launch_lpost3h_1500km.notna())
    tw = (p.sun_alt < -3) & (p.sun_alt > -20)
    p["x_launch_twilight"] = (p.x_launch_3h_1500 * tw).where(p.x_launch_3h_1500.notna())
    p["x_launch_dark_or_day"] = (p.x_launch_3h_1500 * ~tw).where(p.x_launch_3h_1500.notna())
    p["x_starlink_10d"] = ((p.days_since_starlink_launch <= 10) & (p.starlink_era > 0)).astype(float)
    p.loc[p.starlink_era.fillna(0) == 0, "x_starlink_10d"] = np.nan
    p["x_july4"] = p.hol_independence_day
    p["x_nye"] = ((p.mmdd == "12-31") | (p.mmdd == "01-01")).astype(float).where(p.mmdd.notna())
    p["x_venus_vis"] = ((p.venus_alt > 3) & (p.sun_alt < -3) & (p.venus_elong > 15)).astype(float).where(p.venus_alt.notna())
    p["x_venus_alt"] = p.venus_alt.clip(lower=-5)
    p["x_iss"] = p.iss_visible_win
    p["x_moon_up"] = (p.moon_alt > 0).astype(float).where(p.moon_alt.notna())
    p["x_moon_bright"] = ((p.moon_alt > 0) * p.moon_illum).where(p.moon_alt.notna())
    p["x_outburst"] = (p.meteor_outburst_pm1d * dark).where(p.meteor_outburst_pm1d.notna() & p.sun_alt.notna())
    p["x_large_airport_25km"] = (p.dist_large_airport_km <= 25).astype(float).where(p.dist_large_airport_km.notna())
    if "wx_sky_oktas" in p:
        p["x_sky_oktas"] = p.wx_sky_oktas
        p["x_clear"] = (p.wx_sky_oktas <= 2).astype(float).where(p.wx_sky_oktas.notna())
    return p


PCS = [
    # name, exposure, subset, strategies, expected direction, scale
    ("meteor shower ZHR>=50 & sun < -12 deg", "x_shower50", "meteor_like", ["CS1", "CS2"], "+", 1),
    ("meteor shower ZHR>=50 & sun < -12 deg (specificity: non-meteor reports)", "x_shower50", "non_meteor_like", ["CS1", "CS2"], "0", 1),
    ("meteor outburst/storm +-1 d & sun < -12 deg", "x_outburst", "meteor_like", ["CS1", "CS2"], "+", 1),
    ("CNEOS bolide within 30 min & 1000 km", "x_fireball_30m_1000", "all", ["CS1", "CS2"], "+", 1),
    ("CNEOS bolide within 30 min & 500 km", "x_fireball_30m_500", "meteor_like", ["CS1", "CS2"], "+", 1),
    ("rocket launch 0-3 h before, <=1500 km", "x_launch_3h_1500", "all", ["CS1", "CS2"], "+", 1),
    ("rocket launch in observer twilight", "x_launch_twilight", "all", ["CS1", "CS2"], "+", 1),
    ("rocket launch not in twilight", "x_launch_dark_or_day", "all", ["CS1", "CS2"], "+", 1),
    ("Starlink launch within 10 d (line-formation reports)", "x_starlink_10d", "line_formation", ["CS1", "CS2"], "+", 1),
    ("Starlink launch within 10 d (all reports)", "x_starlink_10d", "all", ["CS1", "CS2"], "+", 1),
    ("July 4th", "x_july4", "all", ["CS1"], "+", 1),
    ("July 4th (orange/fireball reports)", "x_july4", "orange_or_fireball", ["CS1"], "+", 1),
    ("New Year's Eve/Day", "x_nye", "orange_or_fireball", ["CS1"], "+", 1),
    ("Venus conspicuous (planet-like reports)", "x_venus_vis", "planet_like", ["CS1", "CS2"], "+", 1),
    ("Venus altitude per 10 deg (planet-like reports)", "x_venus_alt", "planet_like", ["CS1", "CS2"], "+", 10),
    ("ISS visible pass +-10 min (satellite-like reports)", "x_iss", "satellite_like", ["CS1", "CS2"], "+", 1),
    ("Moon above horizon (moon-word reports)", "x_moon_up", "moon_word", ["CS1", "CS2"], "+", 1),
    ("clear sky (<=2 oktas)", "x_clear", "all", ["CS1", "CS2"], "+", 1),
    ("sky cover per okta", "x_sky_oktas", "all", ["CS1", "CS2"], "-", 1),
    ("large airport within 25 km (aircraft-like reports; spatial controls)", "x_large_airport_25km", "aircraft_like", ["CS4"], "+", 1),
    ("large airport within 25 km (non-aircraft-like reports; spatial controls)", "x_large_airport_25km", "non_aircraft_like", ["CS4"], "info", 1),
]
EQUIV = (0.8, 1.25)  # predeclared equivalence margin for expected-null (specificity) controls


def classify(r, expected):
    """PASSED / FAILED / INCONCLUSIVE / NOT_TESTABLE (audit item 5)."""
    p, OR, lo, hi = (r.get(k, np.nan) for k in ("p_value", "effect", "ci_low", "ci_high"))
    if not np.isfinite(p) or str(r.get("status", "")).startswith(("NOT_", "TOO_")):
        return "NOT_TESTABLE"
    if expected == "+":
        if p < 0.05 and lo > 1:
            return "PASSED"
        if (p < 0.05 and OR < 1) or hi < 1.10:
            return "FAILED"
        return "INCONCLUSIVE"
    if expected == "-":
        if p < 0.05 and hi < 1:
            return "PASSED"
        if (p < 0.05 and OR > 1) or lo > 1 / 1.10:
            return "FAILED"
        return "INCONCLUSIVE"
    if expected == "0":
        if lo >= EQUIV[0] and hi <= EQUIV[1]:
            return "PASSED_EQUIVALENT"
        if p < 0.05:
            return "FAILED_ASSOCIATED"
        return "INCONCLUSIVE"
    return "INFORMATIONAL"


def main():
    pts = add_exposures(cc.points())
    ev = cc.events()
    disc = ev[(ev.SOURCE == "NUFORC") & (ev.SPLIT == "discovery")]
    sub = subsets(disc)
    rows = []
    for name, x, sname, strats, exp, scale in PCS:
        if x not in pts:
            continue
        ids = disc.loc[sub[sname], "EVENT_ID"].to_numpy()
        for st in strats:
            r = cc.run(pts, ids, x, st, family="positive_control", hypothesis=name, var_a=x, subset=sname,
                       split="discovery", phase="positive_control", scale=scale)
            r["expected"] = exp
            rows.append(r)
            r["pc_status"] = classify(r, exp)
            print(f"{name:72s} {st} n={r.get('n_cases')} exp_cases={r.get('n_exposed_cases')} OR={r.get('effect'):.3f} "
                  f"[{r.get('ci_low'):.3f},{r.get('ci_high'):.3f}] p={r.get('p_value'):.2e} {r.get('inference_method')} -> {r['pc_status']}")
    out = pd.DataFrame(rows)
    # aircraft-like vs non-aircraft-like contrast (independent subsets): ratio of ORs
    a = out[out.hypothesis.str.startswith("large airport") & (out.subset == "aircraft_like")]
    b = out[out.hypothesis.str.startswith("large airport") & (out.subset == "non_aircraft_like")]
    if len(a) and len(b) and np.isfinite(a.effect.iloc[0]) and np.isfinite(b.effect.iloc[0]):
        la, lb = np.log(a.effect.iloc[0]), np.log(b.effect.iloc[0])
        sa = (np.log(a.ci_high.iloc[0]) - np.log(a.ci_low.iloc[0])) / 3.92
        sb = (np.log(b.ci_high.iloc[0]) - np.log(b.ci_low.iloc[0])) / 3.92
        z = (la - lb) / np.sqrt(sa ** 2 + sb ** 2)
        from scipy import stats as sps
        pz = 2 * sps.norm.sf(abs(z))
        row = dict(hypothesis="large airport within 25 km: ratio of ORs aircraft-like / non-aircraft-like", subset="contrast",
                   control_strategy="CS4", effect=np.exp(la - lb), ci_low=np.exp(la - lb - 1.96 * np.sqrt(sa ** 2 + sb ** 2)),
                   ci_high=np.exp(la - lb + 1.96 * np.sqrt(sa ** 2 + sb ** 2)), p_value=pz, expected="+")
        row["pc_status"] = classify(row, "+")
        rows.append(row)
        S.register(family="positive_control", phase="positive_control", hypothesis=row["hypothesis"], variable_a="x_large_airport_25km",
                   model="ratio_of_conditional_ORs", subset="aircraft_like vs non_aircraft_like", split="discovery",
                   control_strategy="CS4", effect_measure="ROR", effect=row["effect"], ci_low=row["ci_low"],
                   ci_high=row["ci_high"], p_value=pz, status="OK", inference_method="z_test_log_ROR")
        print(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "positive_controls.csv", index=False)
    print(out.pc_status.value_counts().to_string())
    placebo(pts, disc)
    power(pts, disc, sub)


def placebo(pts, disc, n_noise=50):
    """False-positive calibration (audit item 3).
    Real catalogs shifted by EACH distinct k*364 d, k in +-3..+-15 (26 unique shifts, each used once).
    The catalog's documented coverage window shifts with it; exposure is NaN (not zero) where the
    shifted window is outside coverage or the point time is missing. A 364-d shift preserves weekday
    but lets calendar season drift by ~1.24 d/yr (<= 19 d at k=15) and does not preserve local clock
    across DST; it is a realistic null catalog, not a matched one. Noise: n_noise independent draws."""
    from scipy import stats as sps
    rng = np.random.default_rng(MASTER_SEED + 11)
    ids = disc.EVENT_ID.to_numpy()
    d = pts[pts.EVENT_ID.isin(set(ids))].copy()
    t = F.to_sec(d.utc_ts)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    rows = []
    q = C.quakes().rename(columns={"latitude": "lat", "longitude": "lon"})
    L = C.launches()
    L = L[L.time_has_clock & L.lat.notna()]
    cats = [("quake_pre1d_100km", q, {"w": (-F.D, 0)}, [100], (pd.Timestamp("1973-01-01", tz="UTC"), q.time.max())),
            ("launch_post3h_1500km", L, {"w": (-3 * F.H, 0.5 * F.H)}, [1500], (pd.Timestamp("1957-01-01", tz="UTC"), L.time.max()))]
    shifts = [sgn * 364 * k for k in range(3, 16) for sgn in (-1, 1)]
    for shift_days in shifts:
        for name, cat, win, rad, cov in cats:
            c2 = cat.copy()
            c2["time"] = c2["time"] + pd.Timedelta(days=shift_days)
            cov2 = (cov[0] + pd.Timedelta(days=shift_days), cov[1] + pd.Timedelta(days=shift_days))
            w = F.window_counts(t, la, lo, c2, win, rad, coverage=cov2)
            v = w.iloc[:, 0].to_numpy()
            d["x_placebo"] = np.where(np.isfinite(v), (v > 0).astype(float), np.nan)
            for st in ("CS1", "CS2"):
                r = cc.run(d, ids, "x_placebo", st, family="placebo", hypothesis=f"placebo {name}", var_a=name,
                           split="discovery", phase="placebo", register=False)
                r.update(shift_days=shift_days, catalog=name)
                rows.append(r)
    for i in range(n_noise):
        d["x_placebo"] = rng.normal(size=len(d))
        for st in ("CS1", "CS2", "CS4"):
            r = cc.run(d, ids, "x_placebo", st, family="placebo", hypothesis="gaussian noise exposure", var_a="noise",
                       split="discovery", phase="placebo", register=False)
            r.update(shift_days=np.nan, catalog="gaussian_noise", draw=i)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "placebo_calibration.csv", index=False)
    summ = []
    for (cat, st), g in out.groupby(["catalog", "control_strategy"]):
        p = g.p_value.dropna()
        k, n = int((p < 0.05).sum()), len(p)
        lo_ci = sps.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
        hi_ci = sps.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
        summ.append(dict(catalog=cat, control_strategy=st, n_tests=n, n_unique_shifts=g.shift_days.nunique(),
                         n_p_lt_05=k, fpr=k / n if n else np.nan, fpr_ci_low=lo_ci, fpr_ci_high=hi_ci,
                         median_n_exposed_cases=g.n_exposed_cases.median(), n_not_testable=int(g.p_value.isna().sum())))
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "placebo_calibration_summary.csv", index=False)
    print(summ.to_string())


def power(pts, disc, sub, sizes=(100, 300, 1000, 3000, 10000), reps=40):
    rng = np.random.default_rng(MASTER_SEED + 12)
    tests = [("CNEOS bolide", "x_fireball_30m_1000", "all"), ("launch 3h 1500km", "x_launch_3h_1500", "all"),
             ("July 4th", "x_july4", "all"), ("shower ZHR>=50 meteor-like", "x_shower50", "meteor_like"),
             ("clear sky", "x_clear", "all")]
    rows = []
    for name, x, sname in tests:
        if x not in pts:
            continue
        pool = disc.loc[sub[sname], "EVENT_ID"].to_numpy()
        for n in sizes:
            if n > len(pool):
                continue
            det = []
            for _ in range(reps):
                ids = rng.choice(pool, size=n, replace=False)
                r = cc.run(pts, ids, x, "CS1", family="power", hypothesis=name, var_a=x, register=False, min_cases=5)
                det.append(classify(r, "+") == "PASSED")
            rows.append(dict(control=name, n_events=n, power=np.mean(det), reps=reps))
            print(f"power {name} n={n}: {np.mean(det):.2f}")
    pd.DataFrame(rows).to_csv(RESULTS / "detection_power.csv", index=False)


if __name__ == "__main__":
    main()
