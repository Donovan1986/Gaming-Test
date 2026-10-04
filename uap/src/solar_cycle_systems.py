"""Adjudicate solar-cycle associations across INDEPENDENT reporting systems.

A slowly varying global exposure (sunspot number, F10.7) cannot be separated
from multi-year reporting-system trends within one database. If the
association were physical it should appear, with the same sign, in every
system after removing each system's own smooth trend.

For each system: annual event counts (deduplicated events, year range where
the system is active), log(count+1) detrended by a centred 9-year running
median + 5-year running mean; same detrending for annual mean SN. Statistic:
Pearson r of the residuals. Null: circular shifts of the SN residual series
by 3..(n-3) years (preserves autocorrelation). Systems: NUFORC discovery-era
(1995-2015, discovery split only), HATCH (1947-2001), GEIPAN (1977-2024),
BLUEBOOK_UNK (1947-1969), NICAP (1947-1990).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import covariates as C
import stats as S
from common import PROCESSED, RESULTS

SYSTEMS = {"NUFORC_discovery": ("NUFORC", (1995, 2015)), "HATCH": ("HATCH", (1947, 2001)),
           "GEIPAN": ("GEIPAN", (1977, 2024)), "BLUEBOOK_UNK": ("BLUEBOOK_UNK", (1947, 1969)),
           "NICAP": ("NICAP", (1947, 1990))}


def detrend(s: pd.Series):
    tr = s.rolling(9, center=True, min_periods=5).median().rolling(5, center=True, min_periods=3).mean()
    return s - tr


def main():
    from cohorts import discovery_mask
    ev = pd.read_parquet(PROCESSED / "events.parquet", columns=["SOURCE", "SPLIT", "YEAR", "COUNTRY", "GEO_REGION", "HQ_UNEXPLAINED", "HIGH_QUALITY"])
    sd = C.sw_daily()
    sn = sd.groupby(pd.to_datetime(sd.date).dt.year).SN.mean()
    rows = []
    for name, (src, (a, b)) in SYSTEMS.items():
        e = ev[(ev.SOURCE == src) & ev.YEAR.between(a, b)]
        if name == "NUFORC_discovery":
            e = e[discovery_mask(e)]
        for subset, ee in [("ALL", e), ("HIGH_QUALITY", e[e.HIGH_QUALITY])]:
            yrs = np.arange(a, b + 1)
            cnt = ee.groupby("YEAR").size().reindex(yrs, fill_value=0)
            y = detrend(np.log1p(cnt.astype(float)))
            x = detrend(sn.reindex(yrs).astype(float))
            ok = y.notna() & x.notna()
            yv, xv = y[ok].to_numpy(), x[ok].to_numpy()
            if len(yv) < 10:
                continue
            r = np.corrcoef(xv, yv)[0, 1]
            null = [np.corrcoef(np.roll(xv, k), yv)[0, 1] for k in range(3, len(xv) - 2)]
            p = (np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)
            # elasticity: change in log count per 100 SN
            beta = np.polyfit(xv, yv, 1)[0] * 100
            rows.append(dict(system=name, subset=subset, years=f"{a}-{b}", n_years=len(yv), n_events=int(cnt.sum()),
                             r_detrended=r, p_circular_shift=p, dlogcount_per_100SN=beta))
            S.register(family="SOLAR_cycle_systems", phase="adjudication", hypothesis="annual detrended report counts track sunspot number",
                       variable_a="SN_annual_detrended", model="pearson_r_detrended_circular_shift", subset=subset,
                       split=name, n_cases=int(cnt.sum()), effect_measure="r", effect=r, p_value=p,
                       notes=f"dlog(count) per 100 SN = {beta:.3f}")
    out = pd.DataFrame(rows)
    print(out.to_string())
    out.to_csv(RESULTS / "solar_cycle_across_systems.csv", index=False)


if __name__ == "__main__":
    main()
