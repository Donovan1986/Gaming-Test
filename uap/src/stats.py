"""Statistical engine: matched-set conditional logistic regression, hypothesis
registry, multiple-testing correction, bootstrap/permutation helpers.

Every call to `register()` appends one row to results/hypothesis_registry.csv
so the complete testing history is preserved (including null results).
"""
from __future__ import annotations

import csv
import itertools
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import RESULTS

REGISTRY = RESULTS / "hypothesis_registry.csv"
FIELDS = ["test_id", "timestamp_utc", "family", "phase", "hypothesis", "variable_a", "variable_b", "variable_c",
          "window", "radius_km", "model", "subset", "split", "control_strategy", "n_cases", "n_controls",
          "n_strata", "effect_measure", "effect", "ci_low", "ci_high", "p_value", "direction", "notes"]
_counter = itertools.count(1)


def _next_id():
    if not REGISTRY.exists():
        return 1
    with open(REGISTRY) as f:
        return sum(1 for _ in f)


def register(**kw) -> dict:
    row = {k: kw.get(k, "") for k in FIELDS}
    row["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = not REGISTRY.exists()
    row["test_id"] = f"T{_next_id():06d}"
    with open(REGISTRY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


# ------------------------------------------------------------------ conditional logit
def clogit(x: np.ndarray, strata: np.ndarray, case: np.ndarray, max_iter=50):
    """Conditional logistic regression (1 case : M controls per stratum, variable M).
    x: (n,) or (n,k) covariates. Returns beta, se, loglik, n_strata used.
    Strata without both a case and >=1 control, or with NaN x, are dropped."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    ok = np.all(np.isfinite(x), axis=1)
    df = pd.DataFrame({"s": strata[ok], "c": case[ok]})
    xs = x[ok]
    g = df.groupby("s")["c"].agg(["sum", "size"])
    good = g.index[(g["sum"] == 1) & (g["size"] >= 2)]
    m = df["s"].isin(good).to_numpy()
    df, xs = df[m], xs[m]
    if len(good) < 10:
        return None
    codes, uniq = pd.factorize(df["s"])
    order = np.argsort(codes, kind="stable")
    codes, xs, cc = codes[order], xs[order], df["c"].to_numpy()[order]
    k = xs.shape[1]
    # centre x within strata for numerical stability (clogit is invariant to this)
    means = np.zeros((len(uniq), k))
    np.add.at(means, codes, xs)
    cnt = np.bincount(codes)
    means /= cnt[:, None]
    xc = xs - means[codes]
    beta = np.zeros(k)
    for it in range(max_iter):
        eta = xc @ beta
        mx = np.zeros(len(uniq)) - np.inf
        np.maximum.at(mx, codes, eta)
        w = np.exp(eta - mx[codes])
        den = np.bincount(codes, weights=w)
        p = w / den[codes]
        xbar = np.zeros((len(uniq), k))
        np.add.at(xbar, codes, p[:, None] * xc)
        grad = (xc[cc == 1] - xbar[codes[cc == 1]]).sum(0)
        d = xc - xbar[codes]
        H = -(p[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = xc @ beta
    ll = (eta[cc == 1]).sum() - np.log(np.bincount(codes, weights=np.exp(eta))).sum()
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return dict(beta=beta, se=se, ll=ll, n_strata=len(uniq), n_case=int(cc.sum()), n_ctrl=int((1 - cc).sum()))


def clogit_or(x, strata, case, scale=1.0):
    """Odds ratio per `scale` units of x (first covariate) with Wald 95% CI and p."""
    r = clogit(x, strata, case)
    if r is None or not np.isfinite(r["se"][0]) or r["se"][0] == 0:
        return None
    b, se = r["beta"][0] * scale, r["se"][0] * scale
    z = b / se
    p = 2 * sps.norm.sf(abs(z))
    return dict(OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=p, beta=b, se=se,
                n_strata=r["n_strata"], n_case=r["n_case"], n_ctrl=r["n_ctrl"])


def mh_or_binary(x, strata, case):
    """Mantel-Haenszel OR for binary exposure in matched sets + exact-ish CI (RGB variance)."""
    df = pd.DataFrame({"s": strata, "c": case, "x": x}).dropna()
    g = df.groupby("s")
    a = g.apply(lambda d: ((d.c == 1) & (d.x > 0)).sum(), include_groups=False)
    b = g.apply(lambda d: ((d.c == 1) & (d.x <= 0)).sum(), include_groups=False)
    c = g.apply(lambda d: ((d.c == 0) & (d.x > 0)).sum(), include_groups=False)
    d_ = g.apply(lambda d: ((d.c == 0) & (d.x <= 0)).sum(), include_groups=False)
    n = a + b + c + d_
    R = (a * d_ / n).sum()
    S = (b * c / n).sum()
    if R == 0 or S == 0:
        return None
    return R / S


# ------------------------------------------------------------------ multiple testing
def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    o = np.argsort(pv)
    ranked = pv[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[o] = np.minimum(q, 1)
    out[ok] = res
    return out


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    o = np.argsort(pv)
    adj = np.maximum.accumulate((n - np.arange(n)) * pv[o])
    res = np.empty(n)
    res[o] = np.minimum(adj, 1)
    out[ok] = res
    return out


def poisson_rr(cases_exp, cases_unexp, pt_exp, pt_unexp):
    """Rate ratio with exact (conditional binomial) CI."""
    a, b = cases_exp, cases_unexp
    if a + b == 0 or pt_exp <= 0 or pt_unexp <= 0:
        return None
    rr = (a / pt_exp) / (b / pt_unexp) if b > 0 else np.inf
    f = pt_exp / (pt_exp + pt_unexp)
    lo_p = sps.beta.ppf(0.025, a, b + 1) if a > 0 else 0
    hi_p = sps.beta.ppf(0.975, a + 1, b) if b > 0 else 1
    to_rr = lambda q: (q / (1 - q)) * ((1 - f) / f) if 0 < q < 1 else (0 if q <= 0 else np.inf)
    p = sps.binomtest(int(a), int(a + b), f).pvalue
    return dict(RR=rr, lo=to_rr(lo_p), hi=to_rr(hi_p), p=p)
