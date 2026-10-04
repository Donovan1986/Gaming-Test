"""Natural experiments (difference-in-differences, Poisson with offsets).

NE1  ICBM field deactivation: Ellsworth (1994), Whiteman (1997), Grand Forks
     (1998) vs still-active Malmstrom, Minot, F.E. Warren. Outcome: events
     located inside each field disk per year (by EVENT year). Offset: log
     population inside the disk (GeoNames places). Year fixed effects absorb
     national reporting trends (internet, media). Sources: NUFORC + HATCH
     (deduplicated events). Window: 10 years before / after.
NE2  Nuclear power plant permanent shutdowns vs operating plants (25 km and
     50 km disks), years 1990-2015 (discovery-era), same DiD design.
Wild-cluster bootstrap over units for CIs (few units -> wide CIs reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import geo
import spatial
from common import PROCESSED, RESULTS, MASTER_SEED, haversine_km
import stats as S


def unit_population(lat, lon, r):
    G = geo.geonames()
    G = G[G.country == "US"]
    d = haversine_km(lat, lon, G.lat.to_numpy(), G.lon.to_numpy())
    return float(G.population.to_numpy()[d <= r].sum())


def did(panel, label, n_boot=199, seed=MASTER_SEED + 51):
    panel = panel[panel.pop_ > 0].copy()
    X = pd.get_dummies(panel[["unit", "year"]].astype(str), drop_first=True).astype(float)
    X["treat_post"] = panel["treat_post"].astype(float)
    X = sm.add_constant(X)
    m = sm.GLM(panel.n, X, family=sm.families.Poisson(), offset=np.log(panel.pop_)).fit(cov_type="cluster",
                                                                                         cov_kwds={"groups": pd.factorize(panel.unit)[0]})
    b = m.params["treat_post"]
    se = m.bse["treat_post"]
    rng = np.random.default_rng(seed)
    units = panel.unit.unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        bp = pd.concat([panel[panel.unit == u].assign(unit=f"{u}_{k}") for k, u in enumerate(pick)])
        if bp.treat_post.nunique() < 2:
            continue
        Xb = pd.get_dummies(bp[["unit", "year"]].astype(str), drop_first=True).astype(float)
        Xb["treat_post"] = bp["treat_post"].astype(float)
        Xb = sm.add_constant(Xb)
        try:
            mb = sm.GLM(bp.n, Xb, family=sm.families.Poisson(), offset=np.log(bp.pop_)).fit()
            boots.append(mb.params["treat_post"])
        except Exception:
            pass
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) > 50 else (np.nan, np.nan)
    res = dict(experiment=label, irr=np.exp(b), ci_low_cluster=np.exp(b - 1.96 * se), ci_high_cluster=np.exp(b + 1.96 * se),
               p_cluster=m.pvalues["treat_post"], ci_low_boot=np.exp(lo), ci_high_boot=np.exp(hi),
               n_units=len(units), n_events=int(panel.n.sum()))
    S.register(family="NE_did", phase="discovery", hypothesis=label, variable_a="treat_post", model="poisson_DiD_unit_year_FE",
               subset="ALL", split="all_years_event_level", control_strategy="untreated units + year FE",
               n_cases=int(panel.n.sum()), effect_measure="IRR", effect=res["irr"], ci_low=res["ci_low_boot"],
               ci_high=res["ci_high_boot"], p_value=res["p_cluster"], notes="cluster-robust p; bootstrap CI")
    return res


def ne1_icbm(ev):
    cur = spatial.curated()
    f = cur[cur.layer == "icbm_field"].dropna(subset=["lat"])
    mm = f[f.name.str.contains("Malmstrom|Minot|Warren|Ellsworth|Whiteman|Grand Forks")]
    e = ev[ev.SOURCE.isin(["NUFORC", "HATCH"]) & ev.LATITUDE.notna() & ev.YEAR.between(1975, 2015)]
    rows = []
    for _, u in mm.iterrows():
        d = haversine_km(e.LATITUDE.to_numpy(), e.LONGITUDE.to_numpy(), u.lat, u.lon)
        inside = e[d <= u.radius_km]
        pop_ = unit_population(u.lat, u.lon, u.radius_km)
        for y in range(1975, 2016):
            n = int((inside.YEAR == y).sum())
            post = (pd.notna(u.end) and y > u.end)
            rows.append(dict(unit=u["name"], year=y, n=n, pop_=pop_, treat_post=int(post),
                             deactivated=pd.notna(u.end)))
    panel = pd.DataFrame(rows)
    # restrict to +-10 years around each deactivation for treated units; all years for controls
    keep = []
    for _, u in mm.iterrows():
        p = panel[panel.unit == u["name"]]
        if pd.notna(u.end):
            p = p[(p.year >= u.end - 10) & (p.year <= u.end + 10)]
        else:
            p = p[(p.year >= 1984) & (p.year <= 2008)]
        keep.append(p)
    panel = pd.concat(keep)
    panel.to_csv(RESULTS / "ne1_icbm_panel.csv", index=False)
    return did(panel, "ICBM field deactivation (Ellsworth 1994, Whiteman 1997, Grand Forks 1998) vs active fields")


def ne2_nuclear(ev, radius=50):
    cur = spatial.curated()
    closed = cur[(cur.layer == "nuclear_plant_closed") & cur.end.between(1992, 2013)].dropna(subset=["lat"])
    op = spatial.nuclear_operating()
    e = ev[(ev.SOURCE == "NUFORC") & ev.LATITUDE.notna() & ev.YEAR.between(1985, 2015)]
    rows = []
    units = [(r["name"], r.lat, r.lon, r.end) for _, r in closed.iterrows()] + \
            [(r["name"], r.lat, r.lon, np.nan) for _, r in op.iterrows()]
    for name, la, lo, end in units:
        d = haversine_km(e.LATITUDE.to_numpy(), e.LONGITUDE.to_numpy(), la, lo)
        inside = e[d <= radius]
        pop_ = unit_population(la, lo, radius)
        for y in range(1985, 2016):
            if pd.notna(end) and not (end - 8 <= y <= end + 8):
                continue
            rows.append(dict(unit=name, year=y, n=int((inside.YEAR == y).sum()), pop_=pop_,
                             treat_post=int(pd.notna(end) and y > end)))
    panel = pd.DataFrame(rows)
    panel.to_csv(RESULTS / f"ne2_nuclear_panel_{radius}km.csv", index=False)
    return did(panel, f"Nuclear plant permanent shutdown (1992-2013) vs operating plants, {radius} km")


def main(splits=("discovery",)):
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[ev.SPLIT.isin(splits)]
    res = [ne1_icbm(ev), ne2_nuclear(ev, 25), ne2_nuclear(ev, 50)]
    out = pd.DataFrame(res)
    print(out.to_string())
    out.to_csv(RESULTS / "natural_experiments.csv", index=False)


if __name__ == "__main__":
    main()
