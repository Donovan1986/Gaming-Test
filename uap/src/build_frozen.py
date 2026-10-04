"""Assemble and FREEZE everything that will be evaluated on unseen data.

1. Rule-promoted candidates (select_candidates rules R0-R4; reconstructed from the audit and
   discovery results without running new tests).
2. ML interaction candidates: BH q < 0.05 within the ML_interaction family (v3 rows), |log OR| >=
   log(1.15) per SD-product, and not a matcher input inside a matcher-defined subset.
3. Ordered-sequence candidates: BH q < 0.05 within SEQ_ordered_pair (v3), effect >= 1.15x.
4. SECONDARY hypotheses (literature-driven or flagged during discovery but NOT meeting the
   promotion rules). They are evaluated for transparency and can never be graded above C.
The JSON is written with sorted keys and its SHA-256 stored alongside; validate.py refuses to run
if the file changes afterwards.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import stats as S
from select_candidates import KNOWN, MATCHER_INPUTS, MATCHER_SUBSETS, freeze
from common import RESULTS


def rule_candidates():
    a = pd.read_csv(RESULTS / "candidate_selection_audit.csv")
    d = pd.read_csv(RESULTS / "discovery_results.csv")
    out = []
    for _, r in a[a.promoted].iterrows():
        g = d[(d.variable_a == r.variable) & (d.subset == r.subset) & (d.family == r.family) &
              d.status.isin(["OK", "OK_EXACT"]) & d.p_value.notna()].sort_values("p_value")
        b = g.iloc[0]
        out.append(dict(family=b.family, variable=b.variable_a, subset=b.subset, control_strategy=b.control_strategy,
                        window=str(b.window) if pd.notna(b.window) else "", radius_km=str(b.radius_km) if pd.notna(b.radius_km) else "",
                        direction=b.direction, discovery_or=float(b.effect), discovery_ci=[float(b.ci_low), float(b.ci_high)],
                        discovery_p=float(b.p_value), discovery_q=float(b.q_bh_all), discovery_n_cases=int(b.n_cases),
                        discovery_n_exposed_cases=None if pd.isna(b.n_exposed_cases) else int(b.n_exposed_cases),
                        strategies_supporting=str(r.strategies_supporting), kind="single",
                        label="CONVENTIONAL_STIMULUS" if KNOWN.search(b.variable_a) else "NOVEL_OR_CONFOUNDER",
                        tier="PRIMARY", hypothesis=b.hypothesis))
    return out


def ml_candidates():
    r = S.read_registry()
    m = r[(r.family == "ML_interaction") & r.inference_version.astype(str).str.startswith("v3")].copy()
    m = m.drop_duplicates(["variable_a", "variable_b", "subset", "control_strategy"])
    if m.empty:
        return [], m
    m["q"] = S.bh(m.p_value.to_numpy())
    keep = m[(m.q < 0.05) & (np.abs(np.log(m.effect)) >= np.log(1.15))]
    out = []
    for _, b in keep.iterrows():
        if b.subset in MATCHER_SUBSETS and (MATCHER_INPUTS.search(b.variable_a) or MATCHER_INPUTS.search(b.variable_b)):
            continue
        out.append(dict(family="ML_interaction", variable=b.variable_a, variable_b=b.variable_b, subset=b.subset,
                        control_strategy=b.control_strategy, window="", radius_km="", direction=b.direction,
                        discovery_or=float(b.effect), discovery_ci=[float(b.ci_low), float(b.ci_high)],
                        discovery_p=float(b.p_value), discovery_q=float(b.q), discovery_n_cases=int(b.n_cases),
                        kind="interaction", label="NOVEL_OR_CONFOUNDER", tier="PRIMARY",
                        hypothesis=f"{b.variable_a} x {b.variable_b} interaction (per-SD product, main effects adjusted)"))
    return out, m


def seq_candidates():
    r = S.read_registry()
    s = r[(r.family == "SEQ_ordered_pair") & r.inference_version.astype(str).str.startswith("v3")].copy()
    s = s.drop_duplicates(["variable_a", "variable_b", "subset"])
    if s.empty:
        return [], s
    s["q"] = S.bh(s.p_value.to_numpy())
    keep = s[(s.q < 0.05) & (np.abs(np.log(s.effect)) >= np.log(1.15))]
    out = [dict(family="SEQ_ordered_pair", variable=b.variable_a, variable_b=b.variable_b, subset=b.subset,
                control_strategy="CS1", window=b.window, radius_km="", direction=b.direction,
                discovery_or=float(b.effect), discovery_ci=[float(b.ci_low), float(b.ci_high)], discovery_p=float(b.p_value),
                discovery_q=float(b.q), discovery_n_cases=int(b.n_cases), kind="sequence", label="NOVEL_OR_CONFOUNDER",
                tier="PRIMARY", hypothesis=b.hypothesis) for _, b in keep.iterrows()]
    return out, s


SECONDARY = [
    dict(variable="moon_up_illum", subset="ALL", control_strategy="CS1", direction="-", family="A_global_temporal",
         hypothesis="moonlight (Moon above horizon x illuminated fraction) suppresses reports [failed R2 effect-size rule in discovery]"),
    dict(variable="moon_up_illum", subset="HQ_UNEXPLAINED", control_strategy="CS1", direction="-", family="A_global_temporal",
         hypothesis="moonlight suppression in high-quality unexplained events [secondary]"),
    dict(variable="kp_storm5_prior24h", subset="ALL", control_strategy="CS1", direction="+", family="A_global_temporal",
         hypothesis="LITERATURE (Persinger/Poher daily geomagnetic): Kp>=5 within prior 24 h increases reports"),
    dict(variable="kp_storm5_prior24h", subset="HQ_UNEXPLAINED", control_strategy="CS1", direction="+", family="A_global_temporal",
         hypothesis="LITERATURE: Kp>=5 within prior 24 h increases high-quality unexplained reports"),
    dict(variable="dst_storm50_prior24h", subset="HQ_UNEXPLAINED", control_strategy="CS1", direction="+", family="A_global_temporal",
         hypothesis="LITERATURE: Dst<=-50 nT storm within prior 24 h increases high-quality unexplained reports"),
    dict(variable="any_eq4_pre7d_250km", subset="ALL", control_strategy="CS1", direction="+", family="B_local_catalog",
         hypothesis="LITERATURE (Tectonic Strain Theory): M>=4 earthquake within 250 km in prior 7 d increases reports"),
    dict(variable="any_eq4_pre7d_250km", subset="HQ_UNEXPLAINED", control_strategy="CS1", direction="+", family="B_local_catalog",
         hypothesis="LITERATURE (TST): M>=4 quake within 250 km in prior 7 d increases high-quality unexplained reports"),
    dict(variable="within50_nuclear_any_active", subset="HQ_UNEXPLAINED", control_strategy="CS4", direction="+", family="C_spatial",
         hypothesis="LITERATURE (nuclear sites): operating/active nuclear plant within 50 km increases HQ unexplained reports"),
    dict(variable="in_icbm_field_active", subset="HQ_UNEXPLAINED", control_strategy="CS4", direction="+", family="C_spatial",
         hypothesis="LITERATURE (Hastings): location inside an active ICBM field increases HQ unexplained reports"),
    dict(variable="within50_doe_weapons_active", subset="HQ_UNEXPLAINED", control_strategy="CS4", direction="+", family="C_spatial",
         hypothesis="LITERATURE: active DOE nuclear-weapons complex site within 50 km increases HQ unexplained reports"),
    dict(variable="f107_ge172", subset="ALL", control_strategy="CS1", direction="+", family="A_global_temporal",
         hypothesis="EXPLORATORY (spline): F10.7 >= 172 sfu (discovery top decile) increases reports within month"),
    dict(variable="bz_south10_prior3h", subset="ALL", control_strategy="CS1", direction="+", family="A_global_temporal",
         hypothesis="EXPLORATORY (spline): IMF Bz <= -10 nT in prior 3 h increases reports"),
    dict(variable="hss600_prior24h", subset="ALL", control_strategy="CS1", direction="+", family="A_global_temporal",
         hypothesis="EXPLORATORY (spline): solar wind >= 600 km/s in prior 24 h increases reports"),
]

ADJUDICATION = [
    "H_SOLAR: annual detrended report counts track sunspot number in independent systems (solar_cycle_systems.py)",
    "H_SCAN_WA: the overdispersion-robust Washington-State cluster reappears in held-out data (spatial_analysis.py holdout)",
    "H_KNOX: space-time interaction of HQ unexplained events exceeds that of explained events in held-out data",
    "H_ICBM_DID: ICBM-field deactivation DiD (validation + random holdout events)",
    "H_URBAN: spatial candidates shrink to OR~1 after urbanisation adjustment (adjudicate.py A1)",
    "H_MECH: candidate ORs are equal in unexplained and explained events (adjudicate.py A2)",
]


def main():
    cands = rule_candidates()
    mlc, mltab = ml_candidates()
    sqc, sqtab = seq_candidates()
    all_c = cands + mlc + sqc
    for i, c in enumerate(all_c, 1):
        c["candidate_id"] = f"C{i:03d}"
    sec = []
    for i, s in enumerate(SECONDARY, 1):
        sec.append(dict(s, candidate_id=f"S{i:02d}", window="", radius_km="", kind="single", label="SECONDARY", tier="SECONDARY"))
    print(f"primary {len(all_c)} (rule {len(cands)}, ML {len(mlc)}, SEQ {len(sqc)}); secondary {len(sec)}")
    h = freeze(all_c + sec, {"adjudication_hypotheses": ADJUDICATION,
                              "two_stage_permutation_rule": "permutation only for candidates REPLICATED (p<0.05, same direction) in >=2 of E1-E3",
                              "grading_rules": "see report/METHODS.md section 11"})
    return h


if __name__ == "__main__":
    main()
