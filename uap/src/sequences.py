"""Ordered-sequence analyses (temporal ordering only; no causal claims).

1. Superposed-epoch analysis (SEA) of daily US NUFORC report counts around
   geomagnetic storm onsets (Kp >= 5 after >= 3 quiet days), split into
   flare-preceded (M/X flare 1-4 d before onset; A -> B -> UAP) and other
   storms. Expected counts from a Poisson model with year-month, weekday and
   fixed-holiday effects. Null: 1000 sets of random pseudo-onsets drawn from the
   same calendar months.
2. Pre/post asymmetry for local catalogs (A -> UAP vs UAP -> A): ratio of
   case-crossover ORs for the 'pre' and 'post' window of the same width.
3. Ordered pair search: indicators A in [t-72h, t-24h) and B in [t-24h, t] for
   pairs of candidate precursors, conditional logit vs CS1 referents, with
   the reversed order (B then A) as a contrast.
Discovery split only.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

import cc
import covariates as C
import features as F
import stats as S
from discovery_tests import derive
from common import RESULTS, MASTER_SEED
from cohorts import discovery_mask


def daily_counts(ev):
    e = ev[(ev.SOURCE == "NUFORC") & (ev.COUNTRY == "US") & ev.utc_ts.notna()]
    # assign to the UTC date of (time - 12 h) so one evening/night is one day
    d = (e.utc_ts - pd.Timedelta(hours=12)).dt.tz_convert(None).dt.floor("D")
    c = d.value_counts().sort_index()
    idx = pd.date_range("1995-01-01", "2015-12-31", freq="D")
    return c.reindex(idx, fill_value=0)


def expected_counts(c):
    df = pd.DataFrame({"n": c.values}, index=c.index)
    df["ym"] = df.index.strftime("%Y-%m")
    df["dow"] = df.index.dayofweek
    df["hol"] = df.index.strftime("%m-%d").isin(["07-04", "07-05", "12-31", "01-01", "10-31"]).astype(int) * \
        df.index.strftime("%m-%d").map({"07-04": 1, "07-05": 2, "12-31": 3, "01-01": 4, "10-31": 5}).fillna(0).astype(int)
    X = pd.get_dummies(df[["ym", "dow", "hol"]].astype(str), drop_first=True).astype(float)
    X = sm.add_constant(X)
    m = sm.GLM(df.n, X, family=sm.families.Poisson()).fit()
    return pd.Series(m.predict(X), index=c.index), m


def storm_onsets(quiet_days=3, thr=5.0):
    sd = C.sw_daily().set_index("date")
    km = sd.kp_max
    on = []
    for i in range(quiet_days, len(km)):
        if km.iloc[i] >= thr and (km.iloc[i - quiet_days:i] < 4).all():
            on.append(km.index[i])
    fl = C.flares()
    fmx = fl[fl.cls.isin(["M", "X"])].time.dt.tz_convert(None).dt.floor("D").unique()
    fmx = pd.DatetimeIndex(fmx)
    rows = []
    for d in on:
        pre = ((fmx >= d - pd.Timedelta(days=4)) & (fmx <= d - pd.Timedelta(days=1))).any()
        rows.append((d, bool(pre)))
    return pd.DataFrame(rows, columns=["onset", "flare_preceded"])


def sea(c, E, onsets, lags=range(-10, 11), n_null=1000, seed=MASTER_SEED + 41):
    ratio = (c / E)
    rows = {}
    on = [o for o in onsets if (o + pd.Timedelta(days=min(lags)) >= c.index[0]) and (o + pd.Timedelta(days=max(lags)) <= c.index[-1])]
    for L in lags:
        rows[L] = np.mean([ratio.get(o + pd.Timedelta(days=L), np.nan) for o in on])
    obs = pd.Series(rows)
    post = np.mean([obs[k] for k in (0, 1, 2)])
    pre = np.mean([obs[k] for k in (-5, -4, -3, -2, -1)])
    rng = np.random.default_rng(seed)
    idx_by_month = {m: c.index[c.index.month == m] for m in range(1, 13)}
    null = []
    for _ in range(n_null):
        po = [rng.choice(idx_by_month[o.month]) for o in on]
        r0 = np.mean([ratio.get(o + pd.Timedelta(days=k), np.nan) for o in po for k in (0, 1, 2)])
        r1 = np.mean([ratio.get(o + pd.Timedelta(days=k), np.nan) for o in po for k in (-5, -4, -3, -2, -1)])
        null.append(r0 - r1)
    null = np.array(null)
    diff = post - pre
    return obs, dict(n_onsets=len(on), post_ratio=post, pre_ratio=pre, diff=diff,
                     p_two_sided=(np.sum(np.abs(null) >= abs(diff)) + 1) / (n_null + 1))


def ordered_pairs(pts, ids, subset=""):
    d = cc.matched(pts, ids, "CS1").copy()
    t = F.to_sec(d.utc_ts)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    early, late = (-72 * 3600, -24 * 3600), (-24 * 3600, 0)
    kp = C.kp3h()
    kp = kp[kp.kp >= 5].assign(lat=0.0, lon=0.0)
    om = C.omni_hourly()
    ae = om[om.ae >= 500][["time"]].assign(lat=0.0, lon=0.0)
    fl = C.flares()
    fl = fl[fl.cls.isin(["M", "X"])][["time"]].assign(lat=0.0, lon=0.0)
    q = C.quakes().rename(columns={"latitude": "lat", "longitude": "lon"})
    q4 = q[q.mag >= 4.0]
    L = C.launches()
    L = L[L.time_has_clock & L.lat.notna()]
    import weather
    st = weather.storm_events()
    defs = {"kp_storm": (kp, False, 0), "substorm_AE500": (ae, False, 0), "flare_MX": (fl, False, 0),
            "quake_M4_500km": (q4, True, 500), "launch_1500km": (L, True, 1500), "convective_storm_50km": (st, True, 50)}
    ind = {}
    for name, (cat, loc, r) in defs.items():
        w = F.window_counts(t, la if loc else np.zeros(len(t)), lo if loc else np.zeros(len(t)), cat,
                            {"early": early, "late": late}, [r] if loc else [0], need_loc=loc)
        ind[name] = ((w.iloc[:, 0] > 0).to_numpy().astype(float), (w.iloc[:, 1] > 0).to_numpy().astype(float))
    rows = []
    for a, b in itertools.permutations(defs, 2):
        d["x_seq"] = ind[a][0] * ind[b][1]
        r = cc.run(d, ids, "x_seq", "CS1", family="SEQ_ordered_pair", hypothesis=f"{a} (72-24h before) then {b} (24-0h before)",
                   var_a=a, var_b=b, window="A:[-72h,-24h) B:[-24h,0]", split="discovery", subset=subset)
        rows.append(r)
    return rows


def main():
    ev = cc.events()
    disc_ev = ev[discovery_mask(ev)]
    c = daily_counts(disc_ev)
    E, m = expected_counts(c)
    on = storm_onsets()
    on = on[(on.onset >= "1995-01-15") & (on.onset <= "2015-12-15")]
    res = []
    for name, oo in [("all_storms", on.onset), ("flare_preceded", on[on.flare_preceded].onset),
                     ("not_flare_preceded", on[~on.flare_preceded].onset)]:
        obs, summ = sea(c, E, list(oo))
        summ["set"] = name
        res.append(summ)
        S.register(family="SEQ_superposed_epoch", phase="discovery", hypothesis=f"report rate days 0..+2 vs -5..-1 around Kp>=5 onsets ({name})",
                   variable_a="flare_MX (if flare_preceded)", variable_b="Kp>=5 onset", window="days 0..+2 vs -5..-1",
                   model="superposed_epoch_poisson_baseline", subset="ALL", split="discovery",
                   control_strategy="random month-matched pseudo-onsets", n_cases=summ["n_onsets"],
                   effect_measure="ratio_difference", effect=summ["diff"], p_value=summ["p_two_sided"])
        obs.to_csv(RESULTS / f"seq_sea_{name}.csv")
        print(name, summ)
    pd.DataFrame(res).to_csv(RESULTS / "seq_superposed_epoch_summary.csv", index=False)
    pts = derive(cc.points())
    disc = ev[discovery_mask(ev) & ev.utc_ts.notna() & (ev.TIME_UNCERTAINTY_MIN < 720)]
    asym = []
    for sname, ids in [("ALL", disc.EVENT_ID), ("HQ_UNEXPLAINED", disc.loc[disc.HQ_UNEXPLAINED, "EVENT_ID"])]:
        for w in ("1d", "7d"):
            for r_ in ("100", "250", "500"):
                pre = cc.run(pts, ids, f"any_eq_pre{w}_{r_}km", "CS1", family="SEQ_asymmetry", hypothesis="quake before report",
                             var_a=f"any_eq_pre{w}_{r_}km", subset=sname, split="discovery", register=False)
                post = cc.run(pts, ids, f"any_eq_post{w}_{r_}km", "CS1", family="SEQ_asymmetry", hypothesis="quake after report",
                              var_a=f"any_eq_post{w}_{r_}km", subset=sname, split="discovery", register=False)
                asym.append(dict(subset=sname, window=w, radius=r_, or_pre=pre["effect"], p_pre=pre["p_value"],
                                 or_post=post["effect"], p_post=post["p_value"]))
        print(pd.DataFrame(asym).tail(6).to_string())
    pd.DataFrame(asym).to_csv(RESULTS / "seq_quake_pre_post_asymmetry.csv", index=False)
    rows = []
    for sname, ids in [("ALL", disc.EVENT_ID), ("HQ_UNEXPLAINED", disc.loc[disc.HQ_UNEXPLAINED, "EVENT_ID"])]:
        rr = ordered_pairs(pts, ids, sname)
        for r in rr:
            r["subset"] = sname
        rows += rr
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "seq_ordered_pairs.csv", index=False)
    print(out.sort_values("p_value").head(10)[["hypothesis", "subset", "n_cases", "effect", "ci_low", "ci_high", "p_value"]].to_string())


if __name__ == "__main__":
    main()
