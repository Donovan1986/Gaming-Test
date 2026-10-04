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
    return dict(meteor_like=meteor_like, non_meteor_like=~meteor_like, line_formation=line, planet_like=planet_like,
                satellite_like=sat_like, orange_or_fireball=orange, moon_word=f("d_moon_word_txt"),
                all=pd.Series(True, index=ev.index))


def add_exposures(p):
    p = p.copy()
    p["x_shower50"] = (p.shower_zhr_total >= 50).astype(float).where(p.shower_zhr_total.notna())
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
    p["x_outburst"] = p.meteor_outburst_pm1d
    if "wx_sky_oktas" in p:
        p["x_sky_oktas"] = p.wx_sky_oktas
        p["x_clear"] = (p.wx_sky_oktas <= 2).astype(float).where(p.wx_sky_oktas.notna())
    return p


PCS = [
    # name, exposure, subset, strategies, expected direction, scale
    ("meteor shower ZHR>=50 night", "x_shower50", "meteor_like", ["CS1", "CS2"], "+", 1),
    ("meteor shower ZHR>=50 night (specificity: non-meteor reports)", "x_shower50", "non_meteor_like", ["CS1", "CS2"], "0", 1),
    ("meteor outburst/storm +-1 d", "x_outburst", "meteor_like", ["CS1", "CS2"], "+", 1),
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
]


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
            print(f"{name:60s} {st} n={r.get('n_cases')} OR={r.get('effect'):.3f} [{r.get('ci_low'):.3f},{r.get('ci_high'):.3f}] p={r.get('p_value'):.2e}")
    out = pd.DataFrame(rows)
    out["detected"] = np.where(out.expected == "+", (out.ci_low > 1) & (out.p_value < 0.05),
                               np.where(out.expected == "-", (out.ci_high < 1) & (out.p_value < 0.05),
                                        out.p_value >= 0.05))
    out.to_csv(RESULTS / "positive_controls.csv", index=False)
    print("detected", out.detected.sum(), "/", len(out))
    placebo(pts, disc)
    power(pts, disc, sub)


def placebo(pts, disc, n_shift=40):
    """False-positive calibration: real catalogs shifted by whole multiples of
    364 days (preserves season, weekday, time of day, geography) and pure noise."""
    rng = np.random.default_rng(MASTER_SEED + 11)
    ids = disc.EVENT_ID.to_numpy()
    d = pts[pts.EVENT_ID.isin(set(ids))].copy()
    t = F.to_sec(d.utc_ts)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    rows = []
    q = C.quakes().rename(columns={"latitude": "lat", "longitude": "lon"})
    L = C.launches()
    L = L[L.time_has_clock & L.lat.notna()]
    for k in range(n_shift):
        shift_days = int(rng.choice([-1, 1]) * 364 * rng.integers(3, 9))
        for name, cat, win, rad in [("quake_pre1d_100km", q, {"w": (-F.D, 0)}, [100]),
                                    ("launch_post3h_1500km", L, {"w": (-3 * F.H, 0.5 * F.H)}, [1500])]:
            c2 = cat.copy()
            c2["time"] = c2["time"] + pd.Timedelta(days=shift_days)
            w = F.window_counts(t, la, lo, c2, win, rad)
            d["x_placebo"] = (w.iloc[:, 0].to_numpy() > 0).astype(float)
            for st in ("CS1", "CS2"):
                r = cc.run(d, ids, "x_placebo", st, family="placebo", hypothesis=f"placebo {name} shifted {shift_days} d",
                           var_a=name, split="discovery", phase="placebo", register=False)
                rows.append(r)
        d["x_placebo"] = rng.normal(size=len(d))
        for st in ("CS1", "CS2", "CS4"):
            r = cc.run(d, ids, "x_placebo", st, family="placebo", hypothesis="gaussian noise exposure", var_a="noise",
                       split="discovery", phase="placebo", register=False)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "placebo_calibration.csv", index=False)
    p = out.p_value.dropna()
    print(f"placebo tests {len(p)}: frac p<0.05 = {(p < 0.05).mean():.3f}; p<0.01 = {(p < 0.01).mean():.3f}; "
          f"BH q<0.05 = {(S.bh(p.to_numpy()) < 0.05).mean():.3f}")
    for g, gg in out.groupby(out.hypothesis.str.extract(r"(placebo \w+|gaussian)")[0]):
        print("  ", g, len(gg), (gg.p_value < 0.05).mean().round(3))


def power(pts, disc, sub, sizes=(100, 300, 1000, 3000, 10000), reps=60):
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
                det.append((r["p_value"] < 0.05) and (r["effect"] > 1 if x != "x_sky_oktas" else r["effect"] < 1)
                           if np.isfinite(r.get("p_value", np.nan)) else False)
            rows.append(dict(control=name, n_events=n, power=np.mean(det), reps=reps))
            print(f"power {name} n={n}: {np.mean(det):.2f}")
    pd.DataFrame(rows).to_csv(RESULTS / "detection_power.csv", index=False)


if __name__ == "__main__":
    main()
