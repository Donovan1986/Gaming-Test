"""Machine-learning DISCOVERY tools (never used as evidence by themselves).

Design: within-matched-set classification. Cases vs CS1 referents (temporal
features: location and clock time are matched, so only time-varying
exposures can discriminate) and cases vs CS4 spatial controls (same instant;
spatial + local features can discriminate). Models: LightGBM, random forest,
L1 logistic. Grouped 5-fold CV by EVENT_ID inside the discovery split.
Interactions ranked by mean |SHAP interaction value| (TreeExplainer).
Top pairwise interactions are then tested with transparent conditional
logistic models (main effects + product) on the discovery data and
registered as exploratory results. The current selector does not promote
interactions or splines; validating them needs a separately frozen design.
Nonlinearity: conditional logit with natural-cubic-spline basis (GAM-like).
BIC comparisons are descriptive likelihood heuristics; shared clusters do
not support calibrated Bayes factors or likelihood-ratio chi-square tests.
Outputs: results/ml_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import cc
import stats as S
from discovery_tests import derive
from common import RESULTS, MASTER_SEED
from cohorts import discovery_mask

TEMPORAL_FEATS = [
    "kp", "kp_max_prior24h", "kp_max_prior72h", "kp_change_prior24h", "dst", "dst_min_prior24h", "dst_change_prior6h",
    "ae", "ae_max_prior6h", "bz_gsm", "bz_min_prior3h", "v", "np", "pdyn", "efield", "pflux10", "F107", "SN", "Ap",
    "n_flareMX_prior24h", "moon_alt", "moon_illum", "venus_alt", "jupiter_alt", "shower_zhr_total", "meteor_outburst_pm1d",
    "n_eq_pre1d_100km", "n_eq_pre7d_250km", "n_eq_post1d_100km", "n_eq4_pre7d_500km", "n_storm_pm3h_50km", "n_storm_pm12h_100km",
    "n_fb_pm30m_1000km", "n_launch_lpost3h_1500km", "n_launch_lpm24h_1000km", "iss_visible_win", "media_event_within30d",
    "wx_sky_oktas", "wx_temp_c", "wx_dewpt_spread_c", "wx_wind_ms", "wx_slp_hpa", "wx_dslp_3h", "wx_dslp_24h", "wx_dtemp_24h",
    "wx_dsky_6h", "wx_precip_1h_mm", "wx_inversion_proxy", "hol_independence_day", "hol_new_years_eve", "hol_halloween",
]
KNOWN_STIMULI = ["shower_zhr_total", "meteor_outburst_pm1d", "n_fb_pm30m_1000km", "n_launch_lpost3h_1500km",
                 "n_launch_lpm24h_1000km", "iss_visible_win", "hol_independence_day", "hol_new_years_eve", "hol_halloween",
                 "venus_alt", "jupiter_alt", "moon_alt"]
SPATIAL_FEATS = ["dist_large_airport_km", "dist_medium_airport_km", "dist_small_airport_km", "n_airports_lm_25km",
                 "dist_dod_site_km", "n_dod_sites_50km", "dist_dod_air_naval_km", "dist_nuclear_operating_km",
                 "dist_nuclear_any_active_km", "dist_icbm_field_active_km", "dist_doe_weapons_active_km",
                 "dist_nuclear_test_site_km", "dist_launch_site_km", "dist_sua_moa_km", "dist_sua_restricted_km",
                 "dist_sua_alert_km", "dist_dod_boundary_km", "dist_coast_km", "lat", "lon",
                 "wx_sky_oktas", "wx_temp_c", "wx_wind_ms", "n_storm_pm3h_50km", "n_eq_pre7d_250km"]


def fit_models(df, feats, label, tag, seed=MASTER_SEED + 31):
    X = df[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    y = df["is_case"].to_numpy()
    groups = df["EVENT_ID"].to_numpy()
    n_splits = min(5, len(pd.unique(groups)))
    if n_splits < 2:
        raise ValueError("Grouped exploratory CV requires at least two matched sets")
    gkf = GroupKFold(n_splits=n_splits)
    oof = {k: np.full(len(df), np.nan) for k in ("lgb", "rf", "l1")}
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed, num_threads=4)
    for tr, te in gkf.split(X, y, groups):
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=300)
        oof["lgb"][te] = m.predict(X.iloc[te])
        # Held-out feature values must not determine imputation or scaling.
        # A wholly unobserved training column receives the fixed value zero.
        Xf = X.fillna(X.iloc[tr].median().fillna(0.0))
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=100, n_jobs=4, random_state=seed)
        rf.fit(Xf.iloc[tr], y[tr])
        oof["rf"][te] = rf.predict_proba(Xf.iloc[te])[:, 1]
        spread = Xf.iloc[tr].std().replace(0, np.nan).fillna(1.0)
        Z = (Xf - Xf.iloc[tr].mean()) / spread
        l1 = LogisticRegression(penalty="l1", C=0.05, solver="liblinear")
        l1.fit(Z.iloc[tr], y[tr])
        oof["l1"][te] = l1.predict_proba(Z.iloc[te])[:, 1]
    # conditional (within-set) AUC: fraction of sets where case outranks controls
    res = {}
    for k, s in oof.items():
        d = pd.DataFrame({"g": groups, "y": y, "s": s})
        cs = d[d.y == 1].set_index("g")["s"]
        ctl = d[d.y == 0].join(cs.rename("cs"), on="g")
        cond_auc = ((ctl.cs > ctl.s).mean() + 0.5 * (ctl.cs == ctl.s).mean())
        res[k] = dict(model=k, tag=tag, label=label, auc=roc_auc_score(y, s), conditional_auc=cond_auc, n=len(df),
                      n_cases=int(y.sum()), evidence_role="exploratory_discovery_cv")
    full = lgb.train(params, lgb.Dataset(X, y), num_boost_round=300)
    imp = pd.DataFrame({"feature": feats, "gain": full.feature_importance("gain")}).sort_values("gain", ascending=False)
    return res, full, imp


def shap_interactions(model, X, feats, n=6000, seed=MASTER_SEED + 32):
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    ex = shap.TreeExplainer(model)
    iv = ex.shap_interaction_values(X.iloc[idx].astype(float).replace([np.inf, -np.inf], np.nan))
    if isinstance(iv, list):
        iv = iv[1]
    m = np.abs(iv).mean(0)
    rows = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            rows.append(dict(a=feats[i], b=feats[j], mean_abs_interaction=m[i, j] * 2, main_a=m[i, i], main_b=m[j, j]))
    return pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False)


def _eligible_rows(d, x):
    """Use identical finite rows and valid matched sets in nested models."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    good = (np.isfinite(x).all(axis=1) & d.EVENT_ID.notna().to_numpy()
            & d.is_case.isin([0, 1]).to_numpy())
    d, x = d.loc[good], x[good]
    counts = d.groupby("EVENT_ID").is_case.agg(["sum", "size"])
    ids = counts.index[(counts["sum"] == 1) & (counts["size"] >= 2)]
    good = d.EVENT_ID.isin(ids).to_numpy()
    return d.loc[good], x[good]


def _clusters(d):
    # cc.points() supplies integer aliases for the case's night and cell.
    for pair in (("clu_night_i", "clu_i"), ("night_cluster", "cell_cluster")):
        columns = [column for column in pair if column in d]
        if columns:
            return d[columns].to_numpy()
    return None


def _valid_fit(r):
    return (r is not None and r.get("status") == "OK"
            and np.isfinite(r.get("ll", np.nan))
            and r.get("beta") is not None and r.get("se") is not None
            and np.isfinite(r["beta"]).all() and np.isfinite(r["se"]).all()
            and (np.asarray(r["se"]) > 0).all())


def test_interaction(pts, ids, a, b, strategy, subset, phase="discovery", split="discovery", register=True):
    d = cc.matched(pts, ids, strategy)
    d, raw = _eligible_rows(d, d[[a, b]].to_numpy(float))
    if not len(d) or (raw.std(axis=0) == 0).any():
        return None
    za, zb = ((raw - raw.mean(axis=0)) / raw.std(axis=0)).T
    X = np.c_[za, zb, za * zb]
    clusters = _clusters(d)
    r = S.clogit(X, d.EVENT_ID.to_numpy(), d.is_case.to_numpy(), clusters=clusters)
    if not _valid_fit(r):
        return None
    r0 = S.clogit(X[:, :2], d.EVENT_ID.to_numpy(), d.is_case.to_numpy(), clusters=clusters)
    if not _valid_fit(r0):
        return None
    b3, se3 = r["beta"][2], r["se"][2]
    from scipy import stats as sps
    if clusters is not None:
        df = r.get("n_clusters", np.nan) - 1
        if not np.isfinite(df) or df < 1:
            return None
        p = 2 * sps.t.sf(abs(b3 / se3), df)
        critical = sps.t.ppf(0.975, df)
    else:
        p, critical = 2 * sps.norm.sf(abs(b3 / se3)), sps.norm.ppf(0.975)
    log_bf_heuristic = r["ll"] - r0["ll"] - 0.5 * np.log(r["n_strata"])
    row = dict(family="ML_interaction", phase=phase, hypothesis=f"{a} x {b} interaction (per-SD product term)",
               variable_a=a, variable_b=b, model="conditional_logit_interaction", subset=subset, split=split,
               control_strategy=strategy, n_cases=r["n_case"], n_controls=r["n_ctrl"], n_strata=r["n_strata"],
               effect_measure="OR_per_SD_product", effect=np.exp(b3), ci_low=np.exp(b3 - critical * se3),
               ci_high=np.exp(b3 + critical * se3), p_value=p, direction="+" if b3 > 0 else "-", status="OK",
               inference_method="wald_cluster_robust_t" if clusters is not None else "wald_model",
               notes=f"exploratory discovery interaction; main OR/SD a={np.exp(r['beta'][0]):.3f}"
                     f" b={np.exp(r['beta'][1]):.3f}; log_BF10(BIC heuristic)={log_bf_heuristic:.3g};"
                     " BIC heuristic is not calibrated for shared clusters")
    if register:
        S.register(**row)
    return row


def spline_clogit(pts, ids, var, strategy, knots=4):
    """Nonlinear (GAM-like) dose-response: natural cubic spline basis in clogit."""
    from patsy import build_design_matrices, dmatrix
    d = cc.matched(pts, ids, strategy)
    d, x = _eligible_rows(d, d[var].to_numpy(float))
    x = x[:, 0]
    if not len(x):
        return None
    qs = np.quantile(x, np.linspace(0.05, 0.95, knots))
    qs = np.unique(qs)
    if len(qs) < 3:
        return None
    design = dmatrix(f"cr(x, knots={qs[1:-1].tolist()}, lower_bound={x.min()}, upper_bound={x.max()}) - 1",
                     {"x": x})
    # Natural-cubic cardinal columns sum to a constant. Conditional logit
    # removes that constant, so one reference column must be omitted.
    B = np.asarray(design)[:, 1:]
    clusters = _clusters(d)
    r = S.clogit(B, d.EVENT_ID.to_numpy(), d.is_case.to_numpy(), clusters=clusters)
    r_lin = S.clogit(x, d.EVENT_ID.to_numpy(), d.is_case.to_numpy(), clusters=clusters)
    if not _valid_fit(r) or not _valid_fit(r_lin):
        return None
    lr = max(0.0, 2 * (r["ll"] - r_lin["ll"]))
    from scipy import stats as sps
    p_nonlin = sps.chi2.sf(lr, B.shape[1] - 1) if clusters is None else np.nan
    grid = np.quantile(x, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    Bg = np.asarray(build_design_matrices([design.design_info], {"x": grid})[0])[:, 1:]
    eta = Bg @ r["beta"]
    eta -= eta[3]
    return dict(var=var, strategy=strategy, p_nonlinear=p_nonlin, likelihood_ratio_descriptive=lr,
                n_strata=r["n_strata"], grid=list(np.round(grid, 3)),
                or_vs_median=list(np.round(np.exp(eta), 3)),
                inference_method="descriptive_clustered_fit" if clusters is not None else "model_likelihood_ratio",
                notes="exploratory discovery fit; shared clusters invalidate ordinary chi-square LR calibration")


def main():
    pts = derive(cc.points())
    ev = cc.events()
    disc = ev[discovery_mask(ev) & ev.utc_ts.notna() & (ev.TIME_UNCERTAINTY_MIN < 720)]
    perf, imps, inter_rows, tests, splines = [], [], [], [], []
    tfe = [f for f in TEMPORAL_FEATS if f in pts]
    novel = [f for f in tfe if f not in KNOWN_STIMULI]
    sfe = [f for f in SPATIAL_FEATS if f in pts]
    for subset, ids in [("ALL", disc.EVENT_ID), ("HQ_UNEXPLAINED", disc.loc[disc.HQ_UNEXPLAINED, "EVENT_ID"])]:
        for tag, strat, feats in [("temporal_all", "CS1", tfe), ("temporal_novel_only", "CS1", novel),
                                  ("spatial", "CS4", sfe)]:
            d = cc.matched(pts, ids, strat)
            if subset == "ALL" and len(d) > 150000:
                keep = np.random.default_rng(MASTER_SEED + 33).choice(d.EVENT_ID.unique(), size=30000, replace=False)
                d = d[d.EVENT_ID.isin(keep)]
            res, model, imp = fit_models(d, feats, subset, tag)
            perf += list(res.values())
            imp["subset"], imp["tag"] = subset, tag
            imps.append(imp)
            print(subset, tag, {k: round(v["conditional_auc"], 3) for k, v in res.items()}, flush=True)
            print(imp.head(8).to_string(index=False))
            it = shap_interactions(model, d[feats], feats)
            it["subset"], it["tag"] = subset, tag
            inter_rows.append(it.head(40))
            for _, row in it.head(10).iterrows():
                t = test_interaction(pts, ids, row.a, row.b, strat, subset)
                if t:
                    t["ml_rank_tag"] = tag
                    tests.append(t)
    for v in ["kp_max_prior24h", "dst_min_prior24h", "ae_max_prior6h", "wx_sky_oktas", "wx_temp_c", "moon_illum",
              "wx_dslp_24h", "v", "bz_min_prior3h"]:
        if v in pts:
            s = spline_clogit(pts, disc.EVENT_ID, v, "CS1")
            if s:
                splines.append(s)
                print("spline", s)
    pd.DataFrame(perf).to_csv(RESULTS / "ml_performance.csv", index=False)
    pd.concat(imps).to_csv(RESULTS / "ml_feature_importance.csv", index=False)
    pd.concat(inter_rows).to_csv(RESULTS / "ml_shap_interactions.csv", index=False)
    pd.DataFrame(tests).to_csv(RESULTS / "ml_interaction_tests.csv", index=False)
    pd.DataFrame(splines).to_csv(RESULTS / "ml_spline_doseresponse.csv", index=False)


if __name__ == "__main__":
    main()
