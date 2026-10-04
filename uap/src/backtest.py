"""Rolling-origin prospective backtest for a frozen candidate rule (no look-ahead).

For each test year Y (2000..2023): the rule's coefficient is estimated ONLY on
NUFORC events from years < Y (matched sets of the candidate's control strategy),
then applied to the matched sets of year Y. Metrics (out-of-sample):
  * per-set conditional log-likelihood of identifying the actual report time
    among the matched alternatives, minus the no-information baseline log(1/m)
    (i.e. predictive improvement over matched baseline rates);
  * OR in year Y and lift = share of cases among exposed points / share among all;
  * 'rule ON' (binary) or top-decile exposure (continuous): observed vs expected
    number of cases under the baseline.
Positive skill requires the summed out-of-sample log-likelihood gain > 0 with a
bootstrap (by year) CI excluding 0.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import cc
import stats as S
from discovery_tests import derive
from common import RESULTS, MASTER_SEED


def setwise_ll(beta, x, codes, case):
    eta = beta * x
    out = []
    df = pd.DataFrame({"c": codes, "eta": eta, "y": case})
    for _, g in df.groupby("c"):
        e = g.eta.to_numpy()
        e = e - e.max()
        p = np.exp(e) / np.exp(e).sum()
        out.append((np.log(p[g.y.to_numpy() == 1][0]), np.log(1.0 / len(g))))
    a = np.array(out)
    return a[:, 0].sum(), a[:, 1].sum(), len(a)


def run(candidate: dict, years=range(2000, 2024)):
    pts = derive(cc.points())
    ev = cc.events()
    nuf = ev[(ev.SOURCE == "NUFORC") & ev.utc_ts.notna() & (ev.TIME_UNCERTAINTY_MIN < 720)]
    col = {"ALL": None, "HQ": "HIGH_QUALITY", "HQ_UNEXPLAINED": "HQ_UNEXPLAINED", "UNEXPLAINED": "UNEXPLAINED",
           "MULTI_SENSOR": "MULTI_SENSOR"}.get(candidate["subset"])
    if col:
        nuf = nuf[nuf[col].fillna(False).astype(bool)]
    v, st = candidate["variable"], candidate["control_strategy"]
    rows = []
    for Y in years:
        tr = nuf[nuf.YEAR < Y].EVENT_ID
        te = nuf[nuf.YEAR == Y].EVENT_ID
        if len(te) < 30 or len(tr) < 200:
            continue
        dtr = cc.matched(pts, tr, st)
        r = S.clogit(dtr[v].to_numpy(float), dtr.EVENT_ID.to_numpy(), dtr.is_case.to_numpy())
        if r.get("status") != "OK":
            continue
        beta = r["beta"][0]
        dte = cc.matched(pts, te, st)
        dte = dte[np.isfinite(dte[v].to_numpy(float))]
        g = dte.groupby("EVENT_ID").is_case.agg(["sum", "size"])
        keep = g.index[(g["sum"] == 1) & (g["size"] >= 2)]
        dte = dte[dte.EVENT_ID.isin(keep)]
        if dte.EVENT_ID.nunique() < 20:
            continue
        codes = pd.factorize(dte.EVENT_ID)[0]
        ll, ll0, n = setwise_ll(beta, dte[v].to_numpy(float), codes, dte.is_case.to_numpy())
        rte = S.clogit_or(dte[v].to_numpy(float), dte.EVENT_ID.to_numpy(), dte.is_case.to_numpy())
        x = dte[v].to_numpy(float)
        on = x > 0 if np.isin(np.unique(x), [0, 1]).all() else x >= np.nanquantile(x, 0.9)
        case = dte.is_case.to_numpy() == 1
        lift = case[on].mean() / case.mean() if on.any() else np.nan
        rows.append(dict(year=Y, n_sets=n, beta_train=beta, or_train=np.exp(beta), or_test=rte["OR"], p_test=rte["p"],
                         ll_gain=ll - ll0, ll_gain_per_1000_sets=1000 * (ll - ll0) / n, lift_rule_on=lift,
                         n_rule_on=int(on.sum())))
    out = pd.DataFrame(rows)
    rng = np.random.default_rng(MASTER_SEED + 71)
    boots = [out.sample(len(out), replace=True, random_state=int(rng.integers(1e9))).ll_gain.sum() for _ in range(2000)]
    summ = dict(candidate_id=candidate["candidate_id"], variable=v, test_years=len(out), total_ll_gain=out.ll_gain.sum(),
                ll_gain_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                years_positive_gain=int((out.ll_gain > 0).sum()), median_lift=float(out.lift_rule_on.median()))
    out.to_csv(RESULTS / f"backtest_{candidate['candidate_id']}.csv", index=False)
    json.dump(summ, open(RESULTS / f"backtest_{candidate['candidate_id']}_summary.json", "w"), indent=1, default=float)
    print(out.round(4).to_string())
    print(summ)
    return summ


if __name__ == "__main__":
    import sys
    spec = json.loads((RESULTS / "frozen_hypotheses.json").read_text())
    ids = sys.argv[1:]
    for c in spec["candidates"]:
        if not ids or c["candidate_id"] in ids:
            run(c)
