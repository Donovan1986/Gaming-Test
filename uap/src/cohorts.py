"""Predeclared discovery eligibility without changing frozen split labels.

Discovery uses NUFORC in the US and Canada. European and independent-source
observations remain available for unchanged-hypothesis validation only.
"""
from __future__ import annotations

import pandas as pd


def discovery_mask(events: pd.DataFrame) -> pd.Series:
    """Return eligibility; callers may add their time/quality requirements."""
    country = events["COUNTRY"].fillna("").astype(str).str.strip().str.upper()
    mask = (events["SOURCE"] == "NUFORC") & (events["SPLIT"] == "discovery")
    mask &= country.isin(["US", "USA", "CA", "CANADA"])
    if "GEO_REGION" in events:
        mask &= ~events["GEO_REGION"].fillna("").astype(str).str.startswith("europe_")
    return mask.fillna(False)
