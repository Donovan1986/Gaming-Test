"""Case-crossover / matched-control helpers shared by all analysis scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

import stats as S
from common import PROCESSED

_cache = {}


def points(weather=True) -> pd.DataFrame:
    key = "w" if weather else "c"
    if key not in _cache:
        f = PROCESSED / ("points.parquet" if weather and (PROCESSED / "points.parquet").exists() else "points_core.parquet")
        p = pd.read_parquet(f)
        # cluster for robust SEs: ~0.5 deg cell of the CASE location (shared by all its controls)
        cl = p[p.strategy == "CASE"].set_index("EVENT_ID")
        cell = (np.floor(cl.lat / 0.5).astype("Int64").astype(str) + "_" + np.floor(cl.lon / 0.5).astype("Int64").astype(str))
        p["clu"] = p.EVENT_ID.map(cell)
        night = (cl.utc_ts - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
        p["clu_night"] = p.EVENT_ID.map(night)
        p["clu_i"] = pd.factorize(p["clu"])[0]
        p["clu_night_i"] = pd.factorize(p["clu_night"])[0]
        _cache[key] = p
    return _cache[key]


def events() -> pd.DataFrame:
    if "ev" not in _cache:
        _cache["ev"] = pd.read_parquet(PROCESSED / "events.parquet")
    return _cache["ev"]


def matched(pts: pd.DataFrame, event_ids, strategy: str) -> pd.DataFrame:
    """Rows for the given events: the case row + controls of `strategy`."""
    m = pts.EVENT_ID.isin(set(event_ids)) & pts.strategy.isin(["CASE", strategy])
    return pts[m]


def run(pts, event_ids, x, strategy, *, family, hypothesis, var_a, var_b="", var_c="", window="", radius="",
        subset="", split="", phase="discovery", scale=1.0, notes="", register=True, min_cases=20):
    """Matched-set test of case status on exposure x (column name in pts).

    Predeclared inference rule (audit fixes, v4):
      * continuous x: conditional logit, two-way cluster-robust Student-t inference, ONLY if the
        fit converged without separation; otherwise status NOT_ESTIMABLE (no p-value).
      * binary x: exact conditional inference (median-unbiased OR, exact CI, two-sided exact p)
        when the fit separated / did not converge, or exposed cases < 10, or exposed controls < 10;
        otherwise cluster-robust Wald. Independence-based exact inference is descriptive
        only when retained cases share a supplied cluster. Exposed counts always come
        from the retained rows.
      * fewer than `min_cases` retained cases -> status NOT_TESTABLE (no p-value)."""
    d = matched(pts, event_ids, strategy)
    xv = d[x].to_numpy(float)
    clu = np.c_[d["clu_night_i"].to_numpy(), d["clu_i"].to_numpy()] if "clu_i" in d else None  # two-way: night x place
    strata, case = d.EVENT_ID.to_numpy(), d.is_case.to_numpy()
    res = S.clogit_or(xv, strata, case, scale=scale, clusters=clu)
    is_bin = len(xv) > 0 and np.isin(np.unique(xv[np.isfinite(xv)]), [0, 1]).all()
    row = dict(family=family, phase=phase, hypothesis=hypothesis, variable_a=var_a, variable_b=var_b,
               variable_c=var_c, window=window, radius_km=radius, model="conditional_logit", subset=subset,
               split=split, control_strategy=strategy, scale=scale, inference_version=S.INFERENCE_VERSION,
               notes=notes, n_cases=res["n_case"], n_controls=res["n_ctrl"],
               n_strata=res["n_strata"], n_exposed_cases=res["n_exposed_cases"],
               n_exposed_controls=res["n_exposed_controls"], status=res["status"], effect_measure="OR",
               effect=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=np.nan, inference_method="")
    if res["n_case"] < min_cases:
        row["status"] = "NOT_TESTABLE"
    elif is_bin and (res["status"] != "OK" or res["n_exposed_cases"] < 10 or res["n_exposed_controls"] < 10):
        dependent = False
        if clu is not None:
            _, _, kept_case, kept_clusters = S._prepare(xv, strata, case, clu)
            case_clusters = kept_clusters[kept_case == 1]
            dependent = pd.isna(case_clusters).any() or any(
                len(pd.unique(case_clusters[:, j])) < len(case_clusters) for j in range(case_clusters.shape[1]))
        # Keep an affordable unadjusted diagnostic, but never promote its
        # independence assumption to valid inference on dependent cases.
        ex = S.exact_conditional_or(xv, strata, case, max_dp=100 if dependent else 5000)
        if dependent:
            row.update(status="UNADJUSTED_EXACT_CLUSTER_DEPENDENCE", inference_method="unadjusted_exact_descriptive")
            row["notes"] = (notes + f" | clogit status {res['status']}; informative sets {ex['n_informative']};"
                            f" unadjusted exact OR={ex['OR']:.3g} p={ex['p']:.3g};"
                            " shared clusters require dependence-preserving inference").strip(" |")
            if register:
                S.register(**row)
            return row
        row.update(effect=ex["OR"], ci_low=ex["lo"], ci_high=ex["hi"], p_value=ex["p"], effect_measure="OR_exact_MUE",
                   inference_method=ex["method"], model="exact_conditional_matched_sets",
                   status=("OK_EXACT" if np.isfinite(ex["p"]) else "NOT_TESTABLE"))
        row["notes"] = (notes + f" | clogit status {res['status']}; informative sets {ex['n_informative']}").strip(" |")
    elif res["status"] == "OK":
        row.update(effect=res["OR"], ci_low=res["lo"], ci_high=res["hi"], p_value=res["p"],
                   inference_method="wald_twoway_cluster_robust_t" if clu is not None else "wald_model")
        row["notes"] = (notes + f" | model-SE p={2 * S.sps.norm.sf(abs(res['beta'] / res['se_model'])):.3g};"
                        f" clusters={res['n_clusters']}").strip(" |")
    else:
        row["status"] = "NOT_ESTIMABLE_" + str(res["status"])
    if np.isfinite(row["effect"]):
        row["direction"] = "+" if row["effect"] > 1 else "-"
    if register:
        S.register(**row)
    return row


def ev_ids(ev: pd.DataFrame, mask) -> np.ndarray:
    return ev.loc[mask, "EVENT_ID"].to_numpy()
