"""Pre-declared rules for promoting discovery results to candidates, then
FREEZING them (variables, thresholds, windows, radii, subset, direction)
before any validation / holdout data are read.

Rules (written before the discovery results were inspected):
 R1  BH q (across ALL discovery tests) < 0.05.
 R2  practical effect: OR >= 1.15 or OR <= 1/1.15.
 R3  >= 30 exposed cases for binary exposures (explicit integer count from the
     rows retained by the model; v3 audit fix - originally inferred from a note).
 R4  same direction with p < 0.05 under the strategies required by the family
     design (A: CS1 AND CS2; B: >= 2 of CS1/CS2/CS4; G grid: CS1 AND CS4).
     Untestable strategies count as NOT supporting (v3 audit fix). CS4-only
     spatial families must replicate in both non-overlapping discovery periods
     1995-2005 and 2006-2015 (v3 audit fix; the original 'two subsets' rule used
     overlapping subsets). Changes made 2026-10-04 in response to an external
     audit; at that time only the family-A discovery output had been viewed.
 Only results with status OK / OK_EXACT (converged, non-separated or exact) are eligible.
 R0  (added before families C/G were inspected) exposures that are inputs to the known-object
     matcher are ineligible within matcher-defined subsets (selection on a function of the exposure).
 R5  one representative per (variable family x subset) is promoted - the
     smallest p - but ALL its parameters are frozen as found.
 Known-stimulus exposures (fireballs, launches, ISS, Venus, showers,
 holidays, Starlink, moon) are frozen too but labelled CONVENTIONAL so they
 serve as holdout integrity checks rather than as discoveries.
Output: results/frozen_hypotheses.json + .sha256
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from common import RESULTS
from cohorts import discovery_mask
from stats import INFERENCE_VERSION

KNOWN = re.compile(r"fireball|launch|iss_|venus|shower|hol_|july|starlink|moon|outburst|jupiter")


def family_key(v):
    v = re.sub(r"_(pre|post|pm)\w*?_\d+km$", "", v)
    v = re.sub(r"^within\d+_", "within_", v)
    v = re.sub(r"_\d+km$", "", v)
    return v


REQUIRED = {  # predeclared design: strategies each family was run with, and how many must support
    "A_global_temporal": (["CS1", "CS2"], 2),
    "B_local_catalog": (["CS1", "CS2", "CS4"], 2),
    "B_weather": (["CS1", "CS2", "CS4"], 2),
    "B_local_other": (["CS1", "CS2", "CS4"], 2),
    "G_quake_grid": (["CS1", "CS4"], 2), "G_storm_grid": (["CS1", "CS4"], 2), "G_launch_grid": (["CS1", "CS4"], 2),
}
VALID = {"OK", "OK_EXACT"}
# Exposures that are INPUTS to the known-object matcher. Within matcher-defined subsets
# (UNEXPLAINED, HQ_UNEXPLAINED, EXPLAINED) their association is biased by construction
# (conditioning on a function of the exposure), e.g. launches get OR ~0.003 among 'unexplained'.
MATCHER_INPUTS = re.compile(r"launch|fireball|n_fb|iss_|venus|jupiter|shower|outburst|starlink|storm|hol_|mmdd|"
                            r"large_airport|reentry|moon_up|moon_full")
MATCHER_SUBSETS = {"UNEXPLAINED", "HQ_UNEXPLAINED", "EXPLAINED"}


def supports(rows, direction_sign):
    ok = rows[rows.status.isin(VALID) & rows.p_value.notna()]
    return set(ok[(np.sign(np.log(ok.effect)) == direction_sign) & (ok.p_value < 0.05)].control_strategy)


def halves_check(variable, subset, strategy, direction_sign):
    """Spatial (CS4-only) families: the effect must replicate in BOTH non-overlapping
    discovery periods 1995-2005 and 2006-2015 (replaces the overlapping-subset rule; audit item 7)."""
    import cc
    from discovery_tests import derive, subsets
    import re as _re
    m = _re.match(r"(?:within\d+_|log_dist_)(.+?)(?:_km)?$", variable)
    src_cols = [f"dist_{m.group(1)}_km"] if m else [variable]
    key = tuple(src_cols)
    if key not in _pts_cache:
        # memory-light: load only the needed raw column(s); derive() supports partial frames
        _pts_cache[key] = derive(cc.points_light(src_cols))
    pts = _pts_cache[key]
    ev = cc.events()
    disc = ev[discovery_mask(ev)]
    res = []
    for a, b in [(1995, 2005), (2006, 2015)]:
        e = disc[disc.YEAR.between(a, b)]
        ids = subsets(e)[subset]
        r = cc.run(pts, ids, variable, strategy, family="C_spatial", hypothesis=f"selection check {a}-{b}", var_a=variable,
                   subset=subset, split=f"discovery_{a}_{b}", phase="selection_check")
        res.append(r)
    ok = all(r["status"] in VALID and np.isfinite(r["p_value"]) and r["p_value"] < 0.05 and
             np.sign(np.log(r["effect"])) == direction_sign for r in res)
    return ok, res


_pts_cache = {}


def main():
    d = pd.read_csv(RESULTS / "discovery_results.csv")
    required = {"scale", "inference_version"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"Legacy discovery results lack {missing}; rerun discovery_tests.py "
                           f"with {INFERENCE_VERSION} before selecting or freezing hypotheses")
    if not d.inference_version.eq(INFERENCE_VERSION).all():
        raise RuntimeError(f"Discovery results use a superseded inference version; rerun discovery_tests.py "
                           f"with {INFERENCE_VERSION} before selecting or freezing hypotheses")
    scale = pd.to_numeric(d.scale, errors="coerce")
    if not (np.isfinite(scale) & (scale > 0)).all():
        raise RuntimeError("Discovery results have a missing or invalid effect scale; rerun discovery_tests.py "
                           "before selecting or freezing hypotheses")
    d["scale"] = scale
    d["fam"] = d.variable_a.map(family_key)
    d["logor"] = np.log(d.effect)
    cands, audit = [], []
    for (family, fam, subset), g in d.groupby(["family", "fam", "subset"]):
        gv = g[g.status.isin(VALID) & g.p_value.notna()]
        if gv.empty:
            continue
        best = gv.sort_values("p_value").iloc[0]
        reasons = []
        if subset in MATCHER_SUBSETS and MATCHER_INPUTS.search(str(best.variable_a)):
            reasons.append("R0 subset-definition artifact (exposure is a matcher input)")
        if not (best.q_bh_all < 0.05):
            reasons.append("R1 q>=0.05")
        if not (best.effect >= 1.15 or best.effect <= 1 / 1.15):
            reasons.append("R2 effect<1.15x")
        if pd.notna(best.n_exposed_cases) and best.n_exposed_cases < 30:
            reasons.append("R3 exposed cases<30")
        sign = np.sign(best.logor)
        same_var = d[(d.family == family) & (d.variable_a == best.variable_a) & (d.subset == subset)]
        sup = supports(same_var, sign)
        if family in REQUIRED:
            req, need = REQUIRED[family]
            n_sup = len(sup & set(req))
            if n_sup < need:
                reasons.append(f"R4 strategies supporting {sorted(sup)} < {need} of {req}")
        elif family == "C_spatial":
            if not reasons:  # only spend tests on otherwise-qualifying spatial results
                ok, _ = halves_check(best.variable_a, subset, best.control_strategy, sign)
                if not ok:
                    reasons.append("R4 not replicated in both discovery periods 1995-2005 / 2006-2015")
            n_sup = np.nan
        else:
            reasons.append("R4 no predeclared strategy design for family")
        audit.append(dict(family=family, fam=fam, subset=subset, variable=best.variable_a, OR=best.effect,
                          p=best.p_value, q=best.q_bh_all, n_exposed_cases=best.n_exposed_cases,
                          strategies_supporting=";".join(sorted(sup)), promoted=not reasons, reasons="; ".join(reasons)))
        if reasons:
            continue
        cands.append(dict(candidate_id=f"C{len(cands) + 1:02d}", family=best.family, variable=best.variable_a,
                          subset=subset, control_strategy=best.control_strategy, window=best.window,
                          radius_km=best.radius_km, scale=float(best.scale),
                          direction=best.direction, discovery_or=best.effect,
                          discovery_ci=[best.ci_low, best.ci_high], discovery_p=best.p_value, discovery_q=best.q_bh_all,
                          discovery_n_cases=int(best.n_cases), discovery_n_exposed_cases=best.n_exposed_cases,
                          strategies_supporting=sorted(sup),
                          label="CONVENTIONAL_STIMULUS" if KNOWN.search(best.variable_a) else "NOVEL_OR_CONFOUNDER",
                          hypothesis=best.hypothesis))
    pd.DataFrame(audit).to_csv(RESULTS / "candidate_selection_audit.csv", index=False)
    c = pd.DataFrame(cands)
    print(len(c), "candidates")
    if len(c):
        print(c[["candidate_id", "variable", "subset", "control_strategy", "discovery_or", "discovery_p", "label"]].to_string())
    return c


def freeze(cands: list[dict], extra: dict):
    if {"frozen_utc", "candidates"} & extra.keys():
        raise ValueError("Freeze metadata cannot override candidates or frozen_utc")
    spec_path = RESULTS / "frozen_hypotheses.json"
    hash_path = RESULTS / "frozen_hypotheses.sha256"

    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value.item() if isinstance(value, np.generic) else value

    payload = finite_json({"candidates": cands, **extra})
    payload = json.loads(json.dumps(payload, sort_keys=True, default=float, allow_nan=False))
    if spec_path.exists() or hash_path.exists():
        if not (spec_path.exists() and hash_path.exists()):
            raise RuntimeError("Incomplete existing hypothesis freeze; preserve and investigate it")
        existing = spec_path.read_text()
        h = hashlib.sha256(existing.encode()).hexdigest()
        if hash_path.read_text().strip() != h:
            raise RuntimeError("Existing hypothesis freeze does not match its checksum")
        old = json.loads(existing)
        old.pop("frozen_utc", None)
        if old != payload:
            raise RuntimeError("Hypotheses are already frozen; refusing to change the frozen design")
        return h
    spec = {"frozen_utc": pd.Timestamp.now("UTC").isoformat(), **payload}
    txt = json.dumps(spec, indent=1, sort_keys=True, default=float, allow_nan=False)
    h = hashlib.sha256(txt.encode()).hexdigest()
    # Exclusive creation preserves an existing freeze even if another process
    # starts freezing after the check above. Partial freezes must fail closed.
    with spec_path.open("x") as stream:
        stream.write(txt)
    with hash_path.open("x") as stream:
        stream.write(h + "\n")
    print("FROZEN", h)
    return h


if __name__ == "__main__":
    main()
