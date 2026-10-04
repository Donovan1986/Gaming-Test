"""Spatial / space-time clustering with population baselines.

1. County-level rates (events per 100k person-years), negative-binomial GLM
   with log population offset, density and Census-division fixed effects;
   residual Moran's I (k=8 nearest neighbours) to quantify autocorrelation.
2. Kulldorff circular Poisson scan statistic on county centroids; expected
   counts from (a) population only, (b) covariate-adjusted NB model. Monte
   Carlo p-values (999 multinomial replicates).
3. Knox space-time interaction test, permutation of event times (preserves
   purely spatial and purely temporal structure, including media waves).
Outputs: results/spatial_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import BallTree

import covariates as C
from common import RAW, RESULTS, PROCESSED, MASTER_SEED, haversine_km

DIVISION = {  # Census divisions
    **{s: "NewEngland" for s in "CT ME MA NH RI VT".split()}, **{s: "MidAtlantic" for s in "NJ NY PA".split()},
    **{s: "ENCentral" for s in "IL IN MI OH WI".split()}, **{s: "WNCentral" for s in "IA KS MN MO NE ND SD".split()},
    **{s: "SAtlantic" for s in "DE DC FL GA MD NC SC VA WV".split()}, **{s: "ESCentral" for s in "AL KY MS TN".split()},
    **{s: "WSCentral" for s in "AR LA OK TX".split()}, **{s: "Mountain" for s in "AZ CO ID MT NV NM UT WY".split()},
    **{s: "Pacific" for s in "CA OR WA".split()},
}


def counties():
    import geopandas as gpd
    c = gpd.read_file("zip://" + str(RAW / "population" / "cb_2020_us_county_20m.zip"))
    c["fips"] = c.STATEFP + c.COUNTYFP
    st = pd.read_csv(RAW / "population" / "2020_Gaz_counties_national.txt", sep="\t", dtype=str)
    st.columns = [x.strip() for x in st.columns]
    st["fips"] = st.GEOID
    st["lat"] = st.INTPTLAT.astype(float)
    st["lon"] = st.INTPTLONG.astype(float)
    st["aland_km2"] = st.ALAND.astype(float) / 1e6
    c = c.merge(st[["fips", "USPS", "lat", "lon", "aland_km2"]], on="fips")
    c = c[~c.USPS.isin(["AK", "HI", "PR"])].reset_index(drop=True)
    return c


def assign_county(lat, lon, cty):
    import geopandas as gpd
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    j = gpd.sjoin(pts, cty[["fips", "geometry"]].to_crs(4326), how="left", predicate="within")
    j = j[~j.index.duplicated()]
    return j["fips"].to_numpy()


def county_table(ev, years, cty):
    pop = C.county_pop()
    pop = pop[pop.year.between(years[0], years[1])].groupby("fips")["pop"].mean()
    t = cty[["fips", "USPS", "lat", "lon", "aland_km2"]].copy()
    t["pop"] = t.fips.map(pop)
    t = t.dropna(subset=["pop"])
    t["py"] = t["pop"] * (years[1] - years[0] + 1)
    t["density"] = t["pop"] / t["aland_km2"]
    t["division"] = t.USPS.map(DIVISION)
    e = ev[ev.YEAR.between(years[0], years[1])]
    f = assign_county(e.LATITUDE.to_numpy(), e.LONGITUDE.to_numpy(), cty)
    cnt = pd.Series(f).value_counts()
    t["n"] = t.fips.map(cnt).fillna(0).astype(int)
    return t.reset_index(drop=True)


def morans_i(values, lat, lon, k=8):
    tree = BallTree(np.radians(np.c_[lat, lon]), metric="haversine")
    _, idx = tree.query(np.radians(np.c_[lat, lon]), k=k + 1)
    idx = idx[:, 1:]
    z = values - values.mean()
    num = (z[:, None] * z[idx]).sum()
    I = (len(z) / (len(z) * k)) * num / (z ** 2).sum()
    rng = np.random.default_rng(MASTER_SEED + 21)
    null = []
    for _ in range(199):
        zp = rng.permutation(z)
        null.append((len(zp) / (len(zp) * k)) * (zp[:, None] * zp[idx]).sum() / (zp ** 2).sum())
    return I, (np.sum(np.array(null) >= I) + 1) / 200


def nb_expected(t, extra_cols=()):
    X = pd.get_dummies(t[["division"]], drop_first=True).astype(float)
    X["log_density"] = np.log10(t["density"].clip(lower=0.1))
    X["log_density2"] = X["log_density"] ** 2
    X["lat"] = t["lat"]
    for c in extra_cols:
        X[c] = t[c]
    X = sm.add_constant(X)
    m = sm.GLM(t["n"], X, family=sm.families.NegativeBinomial(alpha=0.5), offset=np.log(t["py"])).fit()
    return m, m.predict(X, offset=np.log(t["py"]))


def kulldorff(t, expected, max_pop_frac=0.05, max_km=300, n_sim=999, seed=MASTER_SEED + 22, top=10):
    lat, lon = t.lat.to_numpy(), t.lon.to_numpy()
    n = t.n.to_numpy().astype(float)
    E = np.asarray(expected, float)
    E = E * n.sum() / E.sum()
    tree = BallTree(np.radians(np.c_[lat, lon]), metric="haversine")
    K = 120
    dist, idx = tree.query(np.radians(np.c_[lat, lon]), k=K)
    dist *= 6371.0
    Ecum = np.cumsum(E[idx], axis=1)
    valid = (Ecum <= max_pop_frac * E.sum()) & (dist <= max_km)
    Ntot = n.sum()

    def llr_all(counts):
        cc_ = np.cumsum(counts[idx], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inside = np.where(cc_ > 0, cc_ * np.log(cc_ / Ecum), 0)
            out_c = Ntot - cc_
            out_e = Ntot - Ecum
            outside = np.where(out_c > 0, out_c * np.log(out_c / out_e), 0)
            l = inside + outside
        l[(cc_ <= Ecum) | ~valid] = 0
        return l

    L = llr_all(n)
    best = L.max()
    rng = np.random.default_rng(seed)
    p = E / E.sum()
    sims = np.array([llr_all(rng.multinomial(int(Ntot), p).astype(float)).max() for _ in range(n_sim)])
    # Overdispersed null: county-level extra-Poisson variation (moment estimate of NB alpha).
    alpha = max(1e-6, float(np.sum((n - E) ** 2 - n) / np.sum(E ** 2)))
    sims_nb = []
    for _ in range(n_sim):
        lam = E * rng.gamma(1 / alpha, alpha, size=len(E))
        sims_nb.append(llr_all(rng.poisson(lam).astype(float) * 1.0).max())
    sims_nb = np.array(sims_nb)
    # non-overlapping top clusters
    clusters = []
    used = np.zeros(len(n), bool)
    order = np.dstack(np.unravel_index(np.argsort(-L, axis=None), L.shape))[0]
    for c, k in order:
        if L[c, k] <= 0 or len(clusters) >= top:
            break
        members = idx[c, :k + 1]
        if used[members].any():
            continue
        used[members] = True
        obs, exp = n[members].sum(), E[members].sum()
        clusters.append(dict(center_fips=t.fips.iloc[c], center_lat=lat[c], center_lon=lon[c], radius_km=dist[c, k],
                             n_counties=k + 1, observed=obs, expected=exp, rr=obs / exp, llr=L[c, k],
                             p_mc=(np.sum(sims >= L[c, k]) + 1) / (n_sim + 1),
                             p_mc_overdispersed=(np.sum(sims_nb >= L[c, k]) + 1) / (n_sim + 1), nb_alpha=alpha,
                             counties=";".join(t.fips.iloc[members])))
    return pd.DataFrame(clusters)


def knox(ev, d_km=25, dt_days=7, n_perm=499, seed=MASTER_SEED + 23):
    lat, lon = ev.LATITUDE.to_numpy(), ev.LONGITUDE.to_numpy()
    t = (ev.utc_ts.dt.tz_convert(None).astype("datetime64[s]").astype("int64") / 86400.0).to_numpy()
    tree = BallTree(np.radians(np.c_[lat, lon]), metric="haversine")
    nb = tree.query_radius(np.radians(np.c_[lat, lon]), r=d_km / 6371.0)
    pairs_i = np.concatenate([np.full(len(v), i) for i, v in enumerate(nb)])
    pairs_j = np.concatenate(nb)
    m = pairs_i < pairs_j
    pi, pj = pairs_i[m], pairs_j[m]
    obs = int((np.abs(t[pi] - t[pj]) <= dt_days).sum())
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        tp = rng.permutation(t)
        null.append(int((np.abs(tp[pi] - tp[pj]) <= dt_days).sum()))
    null = np.array(null)
    return dict(close_space_pairs=len(pi), observed=obs, expected=null.mean(), ratio=obs / max(null.mean(), 1e-9),
                p_perm=(np.sum(null >= obs) + 1) / (n_perm + 1))


def main():
    from cohorts import discovery_mask
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    cty = counties()
    country = ev.COUNTRY.fillna("").astype(str).str.strip().str.upper()
    conus = ev[(ev.SOURCE == "NUFORC") & country.isin(["US", "USA"])
               & ev.LATITUDE.between(24, 50) & ev.LONGITUDE.between(-125, -66)]
    out_rows, scan_rows, knox_rows = [], [], []
    for split_name, sub, years in [("discovery", conus[discovery_mask(conus)], (1995, 2015))]:
        for subset, mask in [("ALL", np.ones(len(sub), bool)), ("HQ", sub.HIGH_QUALITY.to_numpy()),
                             ("HQ_UNEXPLAINED", sub.HQ_UNEXPLAINED.to_numpy()), ("EXPLAINED", (sub.P_EXPLAINED >= 0.6).to_numpy())]:
            t = county_table(sub[mask], years, cty)
            m, Eadj = nb_expected(t)
            resid = (t.n - Eadj) / np.sqrt(Eadj + 0.5 * Eadj ** 2)
            I, pI = morans_i(resid.to_numpy(), t.lat.to_numpy(), t.lon.to_numpy())
            Ipop, pIpop = morans_i(((t.n - t.py * t.n.sum() / t.py.sum()) / np.sqrt(t.py * t.n.sum() / t.py.sum())).to_numpy(),
                                   t.lat.to_numpy(), t.lon.to_numpy())
            out_rows.append(dict(split=split_name, subset=subset, n_events=int(t.n.sum()), n_counties=len(t),
                                 rate_per_100k_py=t.n.sum() / t.py.sum() * 1e5, moran_I_pop_only=Ipop, p_moran_pop=pIpop,
                                 moran_I_adjusted=I, p_moran_adj=pI,
                                 coef_log_density=m.params.get("log_density"), coef_lat=m.params.get("lat")))
            for base, E in [("population", t.py * t.n.sum() / t.py.sum()), ("covariate_adjusted", Eadj)]:
                cl = kulldorff(t, E)
                cl["split"], cl["subset"], cl["baseline"] = split_name, subset, base
                scan_rows.append(cl)
                print(subset, base, cl.head(5)[["center_fips", "radius_km", "observed", "expected", "rr", "p_mc", "p_mc_overdispersed"]].to_string())
            t.to_csv(RESULTS / f"spatial_county_table_{split_name}_{subset}.csv", index=False)
            # Knox on regional-cluster-deduplicated events
            s2 = sub[mask & sub.utc_ts.notna().to_numpy()].drop_duplicates("REGIONAL_CLUSTER")
            for dk, dd in [(10, 1), (25, 7), (50, 30)]:
                k = knox(s2, dk, dd)
                k.update(split=split_name, subset=subset, d_km=dk, dt_days=dd, n=len(s2))
                knox_rows.append(k)
                print("knox", subset, dk, dd, k)
    pd.DataFrame(out_rows).to_csv(RESULTS / "spatial_autocorrelation.csv", index=False)
    pd.concat(scan_rows).to_csv(RESULTS / "spatial_scan_clusters.csv", index=False)
    pd.DataFrame(knox_rows).to_csv(RESULTS / "spatial_knox.csv", index=False)


def main_holdout():
    """Frozen adjudication (run only after freezing): (a) does the discovery scan's top
    overdispersion-robust cluster (Washington State, NUFORC's home region) reappear in held-out
    data? (b) Knox space-time interaction in held-out data."""
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    cty = counties()
    conus = ev[(ev.SOURCE == "NUFORC") & ev.LATITUDE.between(24, 50) & ev.LONGITUDE.between(-125, -66)]
    scan_rows, knox_rows = [], []
    for split_name, years in [("validation", (1995, 2015)), ("holdout_random", (1995, 2015)), ("holdout_temporal", (2016, 2023))]:
        sub = conus[conus.SPLIT == split_name]
        for subset, mask in [("ALL", np.ones(len(sub), bool)), ("HQ_UNEXPLAINED", sub.HQ_UNEXPLAINED.to_numpy()),
                             ("EXPLAINED", (sub.P_EXPLAINED >= 0.6).to_numpy())]:
            t = county_table(sub[mask], years, cty)
            E = t.py * t.n.sum() / t.py.sum()
            cl = kulldorff(t, E, n_sim=499)
            cl["split"], cl["subset"], cl["baseline"] = split_name, subset, "population"
            cl["center_state"] = cl.center_fips.str[:2]
            scan_rows.append(cl)
            print(split_name, subset, cl.head(3)[["center_fips", "radius_km", "observed", "expected", "rr", "p_mc_overdispersed"]].to_string())
            s2 = sub[mask & sub.utc_ts.notna().to_numpy()].drop_duplicates("REGIONAL_CLUSTER")
            for dk, dd in [(10, 1), (25, 7), (50, 30)]:
                k = knox(s2, dk, dd, n_perm=199)
                k.update(split=split_name, subset=subset, d_km=dk, dt_days=dd, n=len(s2))
                knox_rows.append(k)
    pd.concat(scan_rows).to_csv(RESULTS / "spatial_scan_clusters_holdout.csv", index=False)
    pd.DataFrame(knox_rows).to_csv(RESULTS / "spatial_knox_holdout.csv", index=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "holdout":
        main_holdout()
    else:
        main()
