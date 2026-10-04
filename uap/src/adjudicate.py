"""Adversarial adjudication of frozen candidates (declared before validation data were read).

A1  Urbanisation / sky-brightness adjustment for SPATIAL candidates: conditional logit
    with the exposure plus log10(GeoNames population within 25 km) and log10(distance to
    the nearest Natural Earth urban-area polygon). A spatial association that is really
    the urban-core / sky-view gradient shrinks toward OR = 1 after adjustment.
A2  Same-mechanism test for EVERY candidate: ratio of ORs between unexplained events
    (matcher P<0.3 / GEIPAN D) and explained events (P>=0.6). ROR ~ 1 with a tight CI
    means 'unexplained' reports respond to the variable exactly like reports of
    identified conventional objects -> observation-opportunity / reporting mechanism.
Both are run on the pooled NON-discovery NUFORC data (validation + both holdouts), only
after the frozen specification hash has been verified.
Outputs: results/adjudication_urban.csv, results/adjudication_same_mechanism.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

import cc
import geo
import spatial
import stats as S
from common import RESULTS, INTERIM
from validate import load_frozen, eval_sets, subset_ids
from discovery_tests import derive


def urban_covariates(pts: pd.DataFrame) -> pd.DataFrame:
    p = INTERIM / "points_urban_covariates.parquet"
    if p.exists():
        u = pd.read_parquet(p)
        if len(u) == len(pts):
            return u
    G = geo.geonames()
    G = G[G.population > 0]
    tree = BallTree(np.radians(G[["lat", "lon"]].to_numpy()), metric="haversine")
    loc = pts[["lat", "lon"]].round(4).drop_duplicates().dropna().reset_index(drop=True)
    idx = tree.query_radius(np.radians(loc[["lat", "lon"]].to_numpy()), r=25 / 6371.0)
    popv = G.population.to_numpy(float)
    loc["pop25"] = [popv[i].sum() for i in idx]
    import geopandas as gpd
    from shapely import STRtree
    ua = gpd.read_file("zip://" + str(spatial.RAW / "infrastructure" / "ne_10m_urban_areas.zip")).set_crs(4326, allow_override=True)
    ua = ua.to_crs(5070).geometry.make_valid().values
    tr = STRtree(ua)
    pt = gpd.GeoSeries(gpd.points_from_xy(loc.lon, loc.lat), crs=4326).to_crs(5070).values
    (ip, ig), dist = tr.query_nearest(pt, all_matches=False, return_distance=True)
    d = np.full(len(loc), np.nan)
    d[ip] = dist / 1000.0
    loc["dist_urban_km"] = d
    m = pts[["lat", "lon"]].round(4).merge(loc, on=["lat", "lon"], how="left")
    out = pd.DataFrame({"log_pop25": np.log10(m.pop25.clip(lower=100)).to_numpy(),
                        "log_dist_urban": np.log10(m.dist_urban_km.clip(lower=0.5)).to_numpy()}, index=pts.index)
    out.to_parquet(p)
    return out


def adjusted(pts, ids, var, strategy):
    d = cc.matched(pts, ids, strategy)
    X = d[[var, "log_pop25", "log_dist_urban"]].to_numpy(float)
    clu = np.c_[d.clu_night_i.to_numpy(), d.clu_i.to_numpy()]
    r = S.clogit(X, d.EVENT_ID.to_numpy(), d.is_case.to_numpy(), clusters=clu)
    if r.get("status") != "OK":
        return dict(status=r.get("status"))
    b, se = r["beta"], r["se"]
    from scipy import stats as sps
    return dict(status="OK", OR_adj=np.exp(b[0]), lo_adj=np.exp(b[0] - 1.96 * se[0]), hi_adj=np.exp(b[0] + 1.96 * se[0]),
                p_adj=2 * sps.norm.sf(abs(b[0] / se[0])), OR_pop25=np.exp(b[1]), OR_dist_urban=np.exp(b[2]),
                n_cases=r["n_case"])


def main():
    spec, h = load_frozen()
    pts = derive(cc.points())
    ev = cc.events()
    sets = eval_sets(ev)
    pool = pd.concat([sets["E1_validation"], sets["E2_holdout_random"], sets["E3_holdout_temporal"]])
    rows_u, rows_m = [], []
    spatial_cands = [c for c in spec["candidates"] if c["family"] == "C_spatial" or c["control_strategy"] == "CS4"]
    if spatial_cands:
        u = urban_covariates(pts)
        for c_ in u.columns:
            pts[c_] = u[c_].to_numpy()
    for c in spec["candidates"]:
        ids = subset_ids(pool, c["subset"])
        if c.get("kind", "single") != "single":
            continue
        from validate import materialize_candidate
        pts_c = materialize_candidate(pts, c)
        sc = float(c.get("scale", 1.0))
        base = cc.run(pts_c, ids, c["variable"], c["control_strategy"], family="ADJ", hypothesis=c["candidate_id"],
                      var_a=c["variable"], register=False, min_cases=10, scale=sc)
        if c in spatial_cands:
            a = adjusted(pts_c, ids, c["variable"], c["control_strategy"])
            rows_u.append(dict(candidate_id=c["candidate_id"], variable=c["variable"], subset=c["subset"],
                               OR_unadjusted=base.get("effect"), p_unadjusted=base.get("p_value"), **a))
        un = cc.run(pts_c, subset_ids(pool, "UNEXPLAINED"), c["variable"], c["control_strategy"], family="ADJ",
                    hypothesis="", var_a=c["variable"], register=False, min_cases=10, scale=sc)
        ex_ids = pool.loc[pool.P_EXPLAINED >= 0.6, "EVENT_ID"].to_numpy()
        ex = cc.run(pts_c, ex_ids, c["variable"], c["control_strategy"], family="ADJ", hypothesis="", var_a=c["variable"],
                    register=False, min_cases=10, scale=sc)
        row = dict(candidate_id=c["candidate_id"], variable=c["variable"], OR_unexplained=un.get("effect"),
                   ci_unexplained=[un.get("ci_low"), un.get("ci_high")], OR_explained=ex.get("effect"),
                   ci_explained=[ex.get("ci_low"), ex.get("ci_high")])
        if all(np.isfinite([un.get("effect", np.nan), ex.get("effect", np.nan), un.get("ci_low", np.nan), ex.get("ci_low", np.nan)])):
            lu, le = np.log(un["effect"]), np.log(ex["effect"])
            su = (np.log(un["ci_high"]) - np.log(un["ci_low"])) / 3.92
            se_ = (np.log(ex["ci_high"]) - np.log(ex["ci_low"])) / 3.92
            z = (lu - le) / np.sqrt(su ** 2 + se_ ** 2)
            from scipy import stats as sps
            row.update(ROR=np.exp(lu - le), ROR_lo=np.exp(lu - le - 1.96 * np.sqrt(su ** 2 + se_ ** 2)),
                       ROR_hi=np.exp(lu - le + 1.96 * np.sqrt(su ** 2 + se_ ** 2)), p_ROR=2 * sps.norm.sf(abs(z)))
        rows_m.append(row)
        print(row, flush=True)
    pd.DataFrame(rows_u).to_csv(RESULTS / "adjudication_urban.csv", index=False)
    pd.DataFrame(rows_m).to_csv(RESULTS / "adjudication_same_mechanism.csv", index=False)


if __name__ == "__main__":
    main()
