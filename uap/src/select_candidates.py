"""Pre-declared rules for promoting discovery results to candidates, then
FREEZING them (variables, thresholds, windows, radii, subset, direction)
before any validation / holdout data are read.

Rules (written before the discovery results were inspected):
 R1  BH q (across ALL discovery tests) < 0.05.
 R2  practical effect: OR >= 1.15 or OR <= 1/1.15.
 R3  >= 30 exposed cases (from the registry note) for binary exposures.
 R4  same direction with p < 0.05 under >= 2 control strategies where the
     family has more than one strategy (temporal: CS1, CS2; local: + CS4);
     spatial exposures (CS4 only) instead need the same direction with
     p < 0.05 in >= 2 subsets.
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

KNOWN = re.compile(r"fireball|launch|iss_|venus|shower|hol_|july|starlink|moon|outburst|jupiter")


def family_key(v):
    v = re.sub(r"_(pre|post|pm)\w*?_\d+km$", "", v)
    v = re.sub(r"^within\d+_", "within_", v)
    v = re.sub(r"_\d+km$", "", v)
    return v


def exposed_cases(note):
    m = re.search(r"exposed: cases ([\d.]+)", str(note))
    return float(m.group(1)) if m else np.nan


def main():
    d = pd.read_csv(RESULTS / "discovery_results.csv")
    d = d[d.p_value.notna()].copy()
    d["exp_frac_cases"] = d.notes.map(exposed_cases)
    d["exp_n_cases"] = d.exp_frac_cases * d.n_cases
    d["fam"] = d.variable_a.map(family_key)
    d["logor"] = np.log(d.effect)
    cands = []
    for (fam, subset), g in d.groupby(["fam", "subset"]):
        best = g.sort_values("p_value").iloc[0]
        if not (best.q_bh_all < 0.05):
            continue
        if not (best.effect >= 1.15 or best.effect <= 1 / 1.15):
            continue
        if np.isfinite(best.exp_n_cases) and best.exp_n_cases < 30:
            continue
        same_var = d[(d.variable_a == best.variable_a) & (d.subset == subset)]
        strat_ok = same_var[(np.sign(same_var.logor) == np.sign(best.logor)) & (same_var.p_value < 0.05)].control_strategy.nunique()
        n_strat = same_var.control_strategy.nunique()
        if n_strat > 1:
            robust = strat_ok >= 2
        else:
            other = d[(d.variable_a == best.variable_a) & (d.control_strategy == best.control_strategy)]
            robust = other[(np.sign(other.logor) == np.sign(best.logor)) & (other.p_value < 0.05)].subset.nunique() >= 2
        if not robust:
            continue
        cands.append(dict(candidate_id=f"C{len(cands) + 1:02d}", family=best.family, variable=best.variable_a,
                          subset=subset, control_strategy=best.control_strategy, window=best.window,
                          radius_km=best.radius_km, direction=best.direction, discovery_or=best.effect,
                          discovery_ci=[best.ci_low, best.ci_high], discovery_p=best.p_value, discovery_q=best.q_bh_all,
                          discovery_n_cases=int(best.n_cases), strategies_supporting=int(strat_ok),
                          label="CONVENTIONAL_STIMULUS" if KNOWN.search(best.variable_a) else "NOVEL_OR_CONFOUNDER",
                          hypothesis=best.hypothesis))
    c = pd.DataFrame(cands)
    print(len(c), "candidates")
    print(c[["candidate_id", "variable", "subset", "control_strategy", "discovery_or", "discovery_p", "label"]].to_string())
    return c


def freeze(cands: list[dict], extra: dict):
    spec = {"frozen_utc": pd.Timestamp.now("UTC").isoformat(), "candidates": cands, **extra}
    txt = json.dumps(spec, indent=1, sort_keys=True, default=float)
    (RESULTS / "frozen_hypotheses.json").write_text(txt)
    h = hashlib.sha256(txt.encode()).hexdigest()
    (RESULTS / "frozen_hypotheses.sha256").write_text(h + "\n")
    print("FROZEN", h)
    return h


if __name__ == "__main__":
    main()
