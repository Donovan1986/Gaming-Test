"""Statistical engine: matched-set conditional logistic regression, hypothesis
registry, multiple-testing correction, bootstrap/permutation helpers.

Every call to `register()` appends one row to results/hypothesis_registry.csv
so the complete testing history is preserved (including null results).
"""
from __future__ import annotations

import csv
import fcntl
import itertools
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import RESULTS, LOGS

REGISTRY = RESULTS / "hypothesis_registry.csv"
INFERENCE_VERSION = "v4_cluster_safe_scale_invariant"
FIELDS = ["test_id", "timestamp_utc", "family", "phase", "hypothesis", "variable_a", "variable_b", "variable_c",
          "window", "radius_km", "model", "subset", "split", "control_strategy", "n_cases", "n_controls",
          "n_strata", "effect_measure", "effect", "ci_low", "ci_high", "p_value", "direction", "notes",
          "inference_version", "status", "inference_method", "n_exposed_cases", "n_exposed_controls", "scale"]
_counter = itertools.count(1)
_registry_ids = {}
_registry_schemas = {}


def _migrate_registry_schema():
    """Normalize only proven historical layouts, preserving every parsed cell.

    The first source version had 24 columns; historical tagging added column
    25. Commit c793379 appended v3 rows with all 29 source-declared columns
    under that 25-column header. Wider rows are interpreted only when both
    their prefix/header and declared inference version prove that layout.
    Caller holds the registry lock. Unknown widths fail before replacement.
    """
    if not REGISTRY.exists() or REGISTRY.stat().st_size == 0:
        return list(FIELDS), 0
    with REGISTRY.open(newline="") as stream:
        old_fields = next(csv.reader(stream))
    fields = old_fields + [field for field in FIELDS if field not in old_fields]
    stat = REGISTRY.stat()
    stamp = (stat.st_size, stat.st_mtime_ns)
    cached = _registry_schemas.get(str(REGISTRY))
    if fields == old_fields and cached is not None and cached[0] == stamp:
        return fields, 0
    temporary = None
    total = 0
    changed = fields != old_fields
    try:
        with tempfile.NamedTemporaryFile("w", newline="", dir=REGISTRY.parent,
                                         prefix=".registry-schema-", delete=False) as target:
            temporary = target.name
            writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            with REGISTRY.open(newline="") as source:
                reader = csv.reader(source)
                next(reader)
                for record in reader:
                    total += 1
                    layout = old_fields
                    if len(record) != len(old_fields):
                        changed = True
                        prefix = old_fields == FIELDS[:len(old_fields)]
                        if prefix and len(record) in (24, 25):
                            layout = FIELDS[:len(record)]
                        elif prefix and len(record) == 29 and record[24] == "v3_dstfix_separation_exact":
                            layout = FIELDS[:29]
                        elif prefix and len(record) == len(FIELDS) and record[24] == INFERENCE_VERSION:
                            layout = FIELDS
                        else:
                            raise RuntimeError("Registry has an unproven row layout; preserve and investigate it")
                    writer.writerow(dict(zip(layout, record)))
        if changed:
            os.replace(temporary, REGISTRY)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    stat = REGISTRY.stat()
    _registry_schemas[str(REGISTRY)] = ((stat.st_size, stat.st_mtime_ns), fields)
    return fields, total if changed else 0


def migrate_registry_schema():
    """Explicit lossless migration without adding or modifying research rows."""
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "hypothesis_registry.lock.log").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fields, rows = _migrate_registry_schema()
    return {"fields": fields, "rows_migrated": rows}


def _next_id():
    if not REGISTRY.exists():
        return 1
    stat = REGISTRY.stat()
    stamp = (stat.st_size, stat.st_mtime_ns)
    cached = _registry_ids.get(str(REGISTRY))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    largest = count = 0
    with REGISTRY.open(newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            label = str(row.get("test_id", ""))
            if label.startswith("T") and label[1:].isdigit():
                largest = max(largest, int(label[1:]))
    next_id = max(largest, count) + 1
    _registry_ids[str(REGISTRY)] = (stamp, next_id)
    return next_id


def register(**kw) -> dict:
    row = {k: kw.get(k, "") for k in FIELDS}
    row["inference_version"] = kw.get("inference_version", INFERENCE_VERSION)
    row["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Serialize header migration and appends. Preserve every old cell and any
    # legacy columns, adding blank cells only for newly declared parameters.
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "hypothesis_registry.lock.log").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = not REGISTRY.exists() or REGISTRY.stat().st_size == 0
        fields, _ = _migrate_registry_schema()
        row["test_id"] = f"T{_next_id():06d}"
        with REGISTRY.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerow(row)
        stat = REGISTRY.stat()
        _registry_ids[str(REGISTRY)] = ((stat.st_size, stat.st_mtime_ns), int(row["test_id"][1:]) + 1)
        _registry_schemas[str(REGISTRY)] = ((stat.st_size, stat.st_mtime_ns), fields)
    return row


# ------------------------------------------------------------------ conditional logit
def _prepare(x, strata, case, clusters=None):
    """Drop rows with missing x and strata without exactly one case and >= 1 control.
    Returns arrays sorted by stratum code (the exact rows any estimator will use)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    ok = np.all(np.isfinite(x), axis=1) & pd.notna(np.asarray(strata)) & np.isin(np.asarray(case), [0, 1])
    s = np.asarray(strata)[ok]
    c = np.asarray(case)[ok].astype(int)
    xs = x[ok]
    karr = np.asarray(clusters, dtype=object)[ok] if clusters is not None else None
    codes, uniq = pd.factorize(s)
    ncase = np.bincount(codes, weights=c)
    size = np.bincount(codes)
    good = (ncase == 1) & (size >= 2)
    m = good[codes]
    codes, xs, c = codes[m], xs[m], c[m]
    karr = karr[m] if karr is not None else None
    codes = pd.factorize(codes)[0]
    order = np.argsort(codes, kind="stable")
    return codes[order], xs[order], c[order], (karr[order] if karr is not None else None)


def _ll_grad_hess(beta, codes, xc, cc, nstr):
    eta = xc @ beta
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    mx = np.maximum.reduceat(eta, starts)
    w = np.exp(eta - mx[codes])
    den = np.bincount(codes, weights=w, minlength=nstr)
    p = w / den[codes]
    k = xc.shape[1]
    if k == 1:
        xbar = np.bincount(codes, weights=p * xc[:, 0], minlength=nstr)[:, None]
    else:
        xbar = np.zeros((nstr, k))
        np.add.at(xbar, codes, p[:, None] * xc)
    grad = (xc[cc == 1] - xbar[codes[cc == 1]]).sum(0)
    d = xc - xbar[codes]
    H = -(p[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    ll = eta[cc == 1].sum() - (mx + np.log(den)).sum()
    return ll, grad, H, xbar


def clogit(x: np.ndarray, strata: np.ndarray, case: np.ndarray, max_iter=100, clusters=None, tol=1e-8):
    """Conditional logistic regression for matched sets (1 case : M controls).
    Returns dict with beta, se (cluster-robust if clusters given), se_model, ll,
    converged, separation flags, n_strata/n_case/n_ctrl and, for the first
    covariate if binary, exposed counts computed on the SAME retained rows.
    Inference must not be used unless converged and not separated."""
    codes, xs, cc, kk = _prepare(x, strata, case, clusters)
    nstr = int(codes.max() + 1) if len(codes) else 0
    out = dict(n_strata=nstr, n_case=int(cc.sum()), n_ctrl=int((1 - cc).sum()), converged=False, separation=False,
               n_exposed_cases=np.nan, n_exposed_controls=np.nan, beta=None, se=None, se_model=None, ll=np.nan,
               n_clusters=np.nan)
    x0 = xs[:, 0]
    if len(x0) and np.isin(np.unique(x0), [0.0, 1.0]).all():
        out["n_exposed_cases"] = int(((x0 > 0) & (cc == 1)).sum())
        out["n_exposed_controls"] = int(((x0 > 0) & (cc == 0)).sum())
    if nstr < 10:
        out["status"] = "TOO_FEW_STRATA"
        return out
    k = xs.shape[1]
    means = np.zeros((nstr, k))
    np.add.at(means, codes, xs)
    means /= np.bincount(codes)[:, None]
    xc = xs - means[codes]
    spread = np.sqrt(np.mean(xc ** 2, axis=0))
    if np.any(spread == 0):
        out["status"] = "NO_WITHIN_SET_VARIATION"
        return out
    # Fit in standardized units: coefficient magnitude and Newton tolerance
    # must not depend on whether an exposure is measured in metres or km.
    xc /= spread
    if k == 1:
        starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        minimum = np.minimum.reduceat(xc[:, 0], starts)
        maximum = np.maximum.reduceat(xc[:, 0], starts)
        case_x = xc[cc == 1, 0]
        separated = ((np.all(case_x == maximum) or np.all(case_x == minimum))
                     and np.any(maximum > minimum))
        if separated:
            out.update(status="SEPARATION", separation=True)
            return out
    beta = np.zeros(k)
    converged = False
    for it in range(max_iter):
        ll, grad, H, _ = _ll_grad_hess(beta, codes, xc, cc, nstr)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        # step-halving to guarantee likelihood increase
        t = 1.0
        accepted = False
        while t > 1e-4:
            nb = beta - t * step
            if _ll_grad_hess(nb, codes, xc, cc, nstr)[0] >= ll - 1e-10:
                accepted = True
                break
            t /= 2
        if not accepted:
            break
        beta = beta - t * step
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
    ll, grad, H, xbar = _ll_grad_hess(beta, codes, xc, cc, nstr)  # final Hessian AT the final estimate
    try:
        cov = np.linalg.inv(-H)
        cond = np.linalg.cond(-H)
    except np.linalg.LinAlgError:
        cov, cond = None, np.inf
    sep = (cond > 1e12) or (cov is None)
    if np.isfinite(out["n_exposed_cases"]):
        # binary first covariate: complete/quasi-complete separation patterns
        ec, ek = out["n_exposed_cases"], out["n_exposed_controls"]
        if ec == 0 or ek == 0 or ec == out["n_case"]:
            sep = True
    out.update(beta=beta / spread, ll=ll, converged=bool(converged and np.all(np.isfinite(beta))), separation=bool(sep))
    if cov is None:
        out["status"] = "SINGULAR_HESSIAN"
        return out
    se = np.sqrt(np.clip(np.diag(cov), 0, None)) / spread
    out.update(se=se, se_model=se.copy())
    if kk is not None:
        U = xc[cc == 1] - xbar[codes[cc == 1]]
        sk = kk[cc == 1]
        if sk.ndim == 1:
            sk = sk[:, None]
        if pd.isna(sk).any():
            out["status"] = "MISSING_CLUSTER"
            return out

        def meat_for(cols):
            if len(cols) > 1:
                a = pd.factorize(sk[:, cols[0]])[0].astype(np.int64)
                b = pd.factorize(sk[:, cols[1]])[0].astype(np.int64)
                kc = pd.factorize(a * (b.max() + 1) + b)[0]
            else:
                kc = pd.factorize(sk[:, cols[0]])[0]
            Uc = np.zeros((kc.max() + 1, k))
            np.add.at(Uc, kc, U)
            nc = Uc.shape[0]
            return (Uc.T @ Uc) * nc / max(nc - 1, 1), nc

        if sk.shape[1] == 1:
            meat, nc = meat_for([0])
        else:  # two-way clustering (Cameron, Gelbach & Miller 2011)
            mA, ncA = meat_for([0])
            mB, ncB = meat_for([1])
            mAB, _ = meat_for([0, 1])
            meat = mA + mB - mAB
            if np.any(np.diag(cov @ meat @ cov) <= 0):
                meat = mA if np.trace(cov @ mA @ cov) >= np.trace(cov @ mB @ cov) else mB
            nc = min(ncA, ncB)
        out["se"] = np.sqrt(np.clip(np.diag(cov @ meat @ cov), 0, None)) / spread
        out["n_clusters"] = nc
        if nc < 2:
            out["status"] = "TOO_FEW_CLUSTERS"
            return out
    out["status"] = "OK" if (out["converged"] and not out["separation"]) else (
        "SEPARATION" if out["separation"] else "NONCONVERGED")
    return out


def _poisson_binomial(p):
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    return dist


def exact_conditional_or(x, strata, case, max_dp=5000):
    """Exact conditional inference for a BINARY exposure in 1:M matched sets.
    In stratum s with n_s members of which e_s are exposed, P(case exposed | psi)
    = psi*e_s / (psi*e_s + n_s - e_s). T = number of exposed cases (Poisson-binomial).
    Returns median-unbiased OR, exact 95% CI (tail-inversion), two-sided p at psi=1
    (doubling of the smaller tail), and counts on the retained rows."""
    codes, xs, cc, _ = _prepare(x, strata, case)
    if not np.isin(xs[:, 0], [0, 1]).all():
        raise ValueError("Exact conditional inference requires a binary 0/1 exposure")
    x0 = xs[:, 0] > 0
    n = np.bincount(codes)
    e = np.bincount(codes, weights=x0)
    case_x = np.bincount(codes[cc == 1], weights=x0[cc == 1], minlength=len(n))
    inf = (e > 0) & (e < n)
    t = int(case_x[inf].sum())
    res = dict(n_strata=int(len(n)), n_informative=int(inf.sum()), n_exposed_cases=int(case_x.sum()),
               n_exposed_controls=int(x0[cc == 0].sum()), t=t)
    if inf.sum() == 0:
        res.update(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, method="exact_conditional(no informative sets)")
        return res
    ee, nn = e[inf], n[inf]
    m = len(ee)
    odds = ee / (nn - ee)
    equal_odds = np.all(odds == odds[0])
    if m > max_dp and not equal_odds:
        res.update(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, method="exact_conditional(too many sets; use Wald)")
        return res

    def tails(logpsi):
        psi = np.exp(logpsi)
        p = psi * ee / (psi * ee + nn - ee)
        if equal_odds:
            return sps.binom.sf(t - 1, m, p[0]), sps.binom.cdf(t, m, p[0])
        d = _poisson_binomial(p)
        return d[t:].sum(), d[:t + 1].sum()  # P(T>=t), P(T<=t)

    up0, lo0 = tails(0.0)
    p_two = min(1.0, 2 * min(up0, lo0))

    def solve(f, target, lo=-30.0, hi=30.0):
        flo, fhi = f(lo) - target, f(hi) - target
        if flo * fhi > 0:
            return np.nan
        for _ in range(80):
            mid = (lo + hi) / 2
            fm = f(mid) - target
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return (lo + hi) / 2

    up = lambda L: tails(L)[0]
    low = lambda L: tails(L)[1]
    l_lo = -np.inf if t == 0 else solve(up, 0.025)
    l_hi = np.inf if t == m else solve(low, 0.025)
    a = -np.inf if t == 0 else solve(up, 0.5)
    b = np.inf if t == m else solve(low, 0.5)
    mue = (a + b) / 2 if np.isfinite(a) and np.isfinite(b) else (a if np.isfinite(a) else b)
    res.update(OR=float(np.exp(mue)), lo=float(np.exp(l_lo)), hi=float(np.exp(l_hi)), p=float(p_two),
               method="exact_conditional_MUE")
    return res


def exact_binary_p(x, strata, case):
    r = exact_conditional_or(x, strata, case)
    return r["p"], r["n_exposed_cases"]


def clogit_or(x, strata, case, scale=1.0, clusters=None):
    """OR per `scale` units of x (first covariate). Wald inference (cluster-robust if
    clusters) ONLY when the fit converged without separation; otherwise status says why
    and OR/CI/p are NaN. Exposed counts always refer to the retained rows."""
    r = clogit(x, strata, case, clusters=clusters)
    base = dict(status=r.get("status"), n_strata=r["n_strata"], n_case=r["n_case"], n_ctrl=r["n_ctrl"],
                n_exposed_cases=r["n_exposed_cases"], n_exposed_controls=r["n_exposed_controls"],
                n_clusters=r["n_clusters"], OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, beta=np.nan, se=np.nan,
                se_model=np.nan)
    if r.get("status") != "OK":
        return base
    if r["se"] is None or not np.isfinite(r["se"][0]) or r["se"][0] == 0:
        base["status"] = "INVALID_STANDARD_ERROR"
        return base
    b, se = r["beta"][0] * scale, r["se"][0] * scale
    if np.isfinite(r["n_clusters"]):
        # Small-cluster uncertainty uses the limiting clustering dimension.
        law = sps.t(df=r["n_clusters"] - 1)
    else:
        law = sps.norm
    p = 2 * law.sf(abs(b / se))
    critical = law.ppf(0.975)
    with np.errstate(over="ignore"):
        base.update(OR=np.exp(b), lo=np.exp(b - critical * se), hi=np.exp(b + critical * se), p=p, beta=b, se=se,
                    se_model=r["se_model"][0] * scale)
    return base


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
