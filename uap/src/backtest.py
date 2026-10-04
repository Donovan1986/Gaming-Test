"""Rolling-origin prospective backtest for frozen candidate rules (no look-ahead).

Candidates are selected on the pre-2016 NUFORC discovery split, so the only test years
free of selection look-ahead are 2016+ (all NUFORC events from 2016 on form the temporal
holdout). For each test year Y in 2016..2023:
  * the rule's coefficient is estimated ONLY on NUFORC North-America events from years < Y
    (matched sets of the candidate's frozen control strategy and subset);
  * it is then applied, unchanged, to the matched sets of year Y.
Out-of-sample metrics per year:
  * conditional log-likelihood of picking the actual report time among the matched
    alternatives, minus the no-information baseline log(1/m) (gain over matched baseline);
  * OR in year Y; lift = share of cases among 'rule ON' points / share among all points
    ('rule ON' = exposure > 0 for binary, top decile for continuous).
Positive skill requires a summed out-of-sample log-likelihood gain > 0 with a bootstrap
(resampling test years) 95% CI excluding 0.
Which candidates: the declared two-stage set (single-exposure candidates nominally REPLICATED,
p<0.05 same direction, in >= 2 of E1-E3), read from validation_results.csv.
Outputs: results/backtest_summary.csv (+ backtest_<id>.csv per candidate).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import cc
import stats as S
from discovery_tests import derive
from common import RESULTS, MASTER_SEED
from validate import load_frozen, subset_ids, materialize_candidate

TEST_YEARS = range(2016, 2024)


def setwise_ll(beta, x, codes, case):
    df = pd.DataFrame({"c": codes, "eta": beta * x, "y": case})
    out = []
    for _, g in df.groupby("c"):
        e = g.eta.to_numpy()
        e = e - e.max()
        p = np.exp(e) / np.exp(e).sum()
        out.append((np.log(p[g.y.to_numpy() == 1][0]), np.log(1.0 / len(g))))
    a = np.array(out)
    return a[:, 0].sum(), a[:, 1].sum(), len(a)


def usable_sets(d, v):
    d = d[np.isfinite(d[v].to_numpy(float))]
    g = d.groupby("EVENT_ID").is_case.agg(["sum", "size"])
    keep = g.index[(g["sum"] == 1) & (g["size"] >= 2)]
    return d[d.EVENT_ID.isin(keep)]


def run(candidate: dict, pts, nuf, years=TEST_YEARS):
    v, st = candidate["variable"], candidate["control_strategy"]
    sc = float(candidate.get("scale", 1.0))
    p = materialize_candidate(pts, candidate)
    ids = set(subset_ids(nuf, candidate["subset"]))
    e = nuf[nuf.EVENT_ID.isin(ids)]
    rows = []
    for Y in years:
        tr = e[e.YEAR < Y].EVENT_ID
        te = e[e.YEAR == Y].EVENT_ID
        if len(te) < 30 or len(tr) < 200:
            continue
        dtr = usable_sets(cc.matched(p, tr, st), v)
        r = S.clogit(dtr[v].to_numpy(float) / sc, dtr.EVENT_ID.to_numpy(), dtr.is_case.to_numpy())
        if r.get("status") != "OK":
            continue
        beta = r["beta"][0]
        dte = usable_sets(cc.matched(p, te, st), v)
        if dte.EVENT_ID.nunique() < 20:
            continue
        x = dte[v].to_numpy(float) / sc
        codes = pd.factorize(dte.EVENT_ID)[0]
        ll, ll0, n = setwise_ll(beta, x, codes, dte.is_case.to_numpy())
        rte = S.clogit_or(x, dte.EVENT_ID.to_numpy(), dte.is_case.to_numpy())
        on = x > 0 if np.isin(np.unique(x), [0, 1]).all() else x >= np.nanquantile(x, 0.9)
        if beta < 0:  # rule ON = the exposure level the frozen direction favours
            on = ~on if np.isin(np.unique(x), [0, 1]).all() else x <= np.nanquantile(x, 0.1)
        case = dte.is_case.to_numpy() == 1
        rows.append(dict(candidate_id=candidate["candidate_id"], year=Y, n_train_sets=int(r.get("n_case", 0)),
                         n_test_sets=n, beta_train=beta, or_train=np.exp(beta), or_test=rte["OR"], p_test=rte["p"],
                         ll_gain=ll - ll0, ll_gain_per_1000_sets=1000 * (ll - ll0) / n,
                         lift_rule_on=case[on].mean() / case.mean() if on.any() else np.nan, n_rule_on=int(on.sum())))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"backtest_{candidate['candidate_id']}.csv", index=False)
    if out.empty:
        return dict(candidate_id=candidate["candidate_id"], variable=v, subset=candidate["subset"],
                    control_strategy=st, status="NOT_TESTABLE", test_years=0)
    rng = np.random.default_rng(MASTER_SEED + 71)
    g = out.ll_gain.to_numpy()
    boots = np.array([g[rng.integers(0, len(g), len(g))].sum() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(candidate_id=candidate["candidate_id"], variable=v, subset=candidate["subset"], control_strategy=st,
                status="OK", test_years=len(out), first_year=int(out.year.min()), last_year=int(out.year.max()),
                n_test_sets=int(out.n_test_sets.sum()), total_ll_gain=float(g.sum()),
                ll_gain_per_1000_sets=float(1000 * g.sum() / out.n_test_sets.sum()),
                ll_gain_ci_low=float(lo), ll_gain_ci_high=float(hi), years_positive_gain=int((g > 0).sum()),
                median_or_test=float(out.or_test.median()), median_lift=float(out.lift_rule_on.median()),
                prospective_skill="POSITIVE" if lo > 0 else ("NEGATIVE" if hi < 0 else "NONE_DETECTED"))


def eligible(spec):
    v = pd.read_csv(RESULTS / "validation_results.csv")
    rep = v[v.eval_set.isin(["E1_validation", "E2_holdout_random", "E3_holdout_temporal"]) &
            (v.verdict_nominal == "REPLICATED")].groupby("candidate_id").size()
    return [c for c in spec["candidates"] if c.get("kind", "single") == "single" and rep.get(c["candidate_id"], 0) >= 2]


def main():
    spec, h = load_frozen()
    pts = derive(cc.points())
    ev = cc.events()
    # training history = every North-America NUFORC event with a usable time, any split, strictly
    # earlier years only; test years 2016+ are the temporal holdout (selection-clean)
    t_ok = ev.utc_ts.notna() & (ev.TIME_UNCERTAINTY_MIN < 720)
    country = ev.COUNTRY.fillna("").astype(str).str.strip().str.upper()
    nuf = ev[(ev.SOURCE == "NUFORC") & t_ok & country.isin(["US", "USA", "CA", "CANADA"])].drop_duplicates("EVENT_ID")
    cands = eligible(spec)
    print(f"backtesting {len(cands)} eligible candidates", flush=True)
    rows = []
    for c in cands:
        r = run(c, pts, nuf)
        r["frozen_sha256"] = h
        rows.append(r)
        print(r, flush=True)
        if r.get("status") == "OK":
            S.register(family="BACKTEST", phase="prospective_backtest", hypothesis=f"[{c['candidate_id']}] {c['hypothesis']}",
                       variable_a=c["variable"], model="rolling_origin_clogit_ll_gain", subset=c["subset"],
                       split="NUFORC_2016_2023_rolling", control_strategy=c["control_strategy"],
                       n_cases=r["n_test_sets"], effect_measure="sum_ll_gain", effect=r["total_ll_gain"],
                       ci_low=r["ll_gain_ci_low"], ci_high=r["ll_gain_ci_high"], notes=r["prospective_skill"])
    cols = ["candidate_id", "variable", "subset", "control_strategy", "status", "test_years", "first_year", "last_year",
            "n_test_sets", "total_ll_gain", "ll_gain_per_1000_sets", "ll_gain_ci_low", "ll_gain_ci_high",
            "years_positive_gain", "median_or_test", "median_lift", "prospective_skill", "frozen_sha256"]
    pd.DataFrame(rows, columns=cols).to_csv(RESULTS / "backtest_summary.csv", index=False)


if __name__ == "__main__":
    main()
