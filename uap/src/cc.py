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
    """Conditional logit of case status on exposure x within matched sets.
    x: column name in pts."""
    d = matched(pts, event_ids, strategy)
    xv = d[x].to_numpy(float)  # derived exposures must be added to pts as a column first
    res = S.clogit_or(xv, d.EVENT_ID.to_numpy(), d.is_case.to_numpy(), scale=scale)
    row = dict(family=family, phase=phase, hypothesis=hypothesis, variable_a=var_a, variable_b=var_b,
               variable_c=var_c, window=window, radius_km=radius, model="conditional_logit", subset=subset,
               split=split, control_strategy=strategy, notes=notes)
    if res is None or res["n_case"] < min_cases:
        row.update(n_cases=0 if res is None else res["n_case"], effect_measure="OR", effect=np.nan, ci_low=np.nan,
                   ci_high=np.nan, p_value=np.nan)
    else:
        row.update(n_cases=res["n_case"], n_controls=res["n_ctrl"], n_strata=res["n_strata"], effect_measure="OR",
                   effect=res["OR"], ci_low=res["lo"], ci_high=res["hi"], p_value=res["p"],
                   direction="+" if res["OR"] > 1 else "-")
        # exposure prevalence for interpretability
        if np.isin(np.unique(xv[np.isfinite(xv)]), [0, 1]).all():
            ca = d.is_case.to_numpy() == 1
            row["notes"] = (notes + f" | exposed: cases {np.nanmean(xv[ca]):.4f}, controls {np.nanmean(xv[~ca]):.4f}").strip(" |")
    if register:
        S.register(**row)
    return row


def ev_ids(ev: pd.DataFrame, mask) -> np.ndarray:
    return ev.loc[mask, "EVENT_ID"].to_numpy()
