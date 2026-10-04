"""Validation of FROZEN candidates on unseen data.

Refuses to run unless results/frozen_hypotheses.json matches its recorded
sha256 (written by select_candidates.freeze before this script first ran).

For each candidate (variable, subset, control strategy, direction frozen):
  E1  validation split (NUFORC)                     E2  locked random holdout (NUFORC)
  E3  locked temporal holdout (NUFORC >= 2016)       E4  geographic: GEIPAN (France)
  E5  geographic: NUFORC outside US                  E6  cross-source: HATCH
  E7  cross-source: Blue Book unknowns              E8  cross-source: NICAP
  LOSO pooled (all non-discovery sets) minus one source at a time
  Sensitivity grid (on validation + holdouts pooled)
  Structure-preserving permutation (E1+E2+E3 pooled)
Outputs: results/validation_*.csv
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

import cc
import features as F
import spatial
import stats as S
import weather
from discovery_tests import derive
from common import RESULTS, MASTER_SEED

SUBSET_COLS = {"ALL": None, "HQ": "HIGH_QUALITY", "HQ_UNEXPLAINED": "HQ_UNEXPLAINED", "MULTI_SENSOR": "MULTI_SENSOR",
               "UNEXPLAINED": "UNEXPLAINED"}


def load_frozen():
    txt = (RESULTS / "frozen_hypotheses.json").read_text()
    h = (RESULTS / "frozen_hypotheses.sha256").read_text().strip()
    assert hashlib.sha256(txt.encode()).hexdigest() == h, "frozen spec modified after freezing - refusing to validate"
    return json.loads(txt), h


def eval_sets(ev):
    t_ok = ev.utc_ts.notna() & (ev.TIME_UNCERTAINTY_MIN < 720)
    n = ev.SOURCE == "NUFORC"
    us = ev.COUNTRY.isin(["US"])
    return {
        "E1_validation": ev[n & (ev.SPLIT == "validation") & t_ok],
        "E2_holdout_random": ev[n & (ev.SPLIT == "holdout_random") & t_ok],
        "E3_holdout_temporal": ev[n & (ev.SPLIT == "holdout_temporal") & t_ok],
        "E4_geo_GEIPAN": ev[(ev.SOURCE == "GEIPAN") & t_ok],
        "E5_geo_NUFORC_nonUS": ev[n & ~us & ev.SPLIT.isin(["validation", "holdout_random", "holdout_temporal"]) & t_ok],
        "E6_src_HATCH": ev[(ev.SOURCE == "HATCH") & t_ok],
        "E7_src_BLUEBOOK": ev[(ev.SOURCE == "BLUEBOOK_UNK") & t_ok],
        "E8_src_NICAP": ev[(ev.SOURCE == "NICAP") & t_ok],
    }


def subset_ids(e, subset):
    col = SUBSET_COLS.get(subset)
    if col is None:
        return e.EVENT_ID.to_numpy()
    return e.loc[e[col].fillna(False).astype(bool), "EVENT_ID"].to_numpy()


def seq_indicator(pts, ids, a, b, strategy="CS1"):
    """Recompute the ordered-pair indicator A in [t-72h,t-24h) and B in [t-24h,t] for given events."""
    import covariates as C
    d = cc.matched(pts, ids, strategy).copy()
    t = F.to_sec(d.utc_ts)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    kp = C.kp3h()
    om = C.omni_hourly()
    fl = C.flares()
    q = C.quakes().rename(columns={"latitude": "lat", "longitude": "lon"})
    L = C.launches()
    defs = {"kp_storm": (kp[kp.kp >= 5].assign(lat=0.0, lon=0.0), False, 0),
            "substorm_AE500": (om[om.ae >= 500][["time"]].assign(lat=0.0, lon=0.0), False, 0),
            "flare_MX": (fl[fl.cls.isin(["M", "X"])][["time"]].assign(lat=0.0, lon=0.0), False, 0),
            "quake_M4_500km": (q[q.mag >= 4.0], True, 500), "launch_1500km": (L[L.time_has_clock & L.lat.notna()], True, 1500),
            "convective_storm_50km": (weather.storm_events(), True, 50)}
    ind = []
    for name, win in ((a, (-72 * 3600, -24 * 3600)), (b, (-24 * 3600, 0))):
        cat, loc, r = defs[name]
        w = F.window_counts(t, la if loc else np.zeros(len(t)), lo if loc else np.zeros(len(t)), cat, {"w": win},
                            [r] if loc else [0], need_loc=loc)
        ind.append((w.iloc[:, 0] > 0).to_numpy().astype(float))
    d["x_seq"] = ind[0] * ind[1]
    return d


def run_one(pts, ids, c, label, phase, register=True, strategy=None):
    st = strategy or c["control_strategy"]
    if c.get("kind") == "interaction":
        from ml_discovery import test_interaction
        r = test_interaction(pts, ids, c["variable"], c["variable_b"], st, c["subset"], phase=phase, split=label,
                             register=register)
        return r or dict(status="NOT_TESTABLE", effect=np.nan, p_value=np.nan, ci_low=np.nan, ci_high=np.nan, n_cases=0)
    if c.get("kind") == "sequence":
        d = seq_indicator(pts, ids, c["variable"], c["variable_b"], st)
        return cc.run(d, ids, "x_seq", st, family=c["family"], hypothesis=f"[{c['candidate_id']}] {c['hypothesis']}",
                      var_a=c["variable"], var_b=c["variable_b"], window=c.get("window", ""), subset=c["subset"],
                      split=label, phase=phase, register=register, min_cases=10)
    return cc.run(pts, ids, c["variable"], st, family=c["family"], hypothesis=f"[{c['candidate_id']}] {c['hypothesis']}",
                  var_a=c["variable"], window=c.get("window", ""), radius=c.get("radius_km", ""), subset=c["subset"],
                  split=label, phase=phase, register=register, min_cases=10)


def verdict(row, direction):
    if not np.isfinite(row.get("p_value", np.nan)):
        return "NOT_TESTABLE"
    same = (row["effect"] > 1) == (direction == "+")
    if same and row["p_value"] < 0.05:
        return "REPLICATED"
    if same:
        return "SAME_DIRECTION_NS"
    if row["p_value"] < 0.05:
        return "OPPOSITE_SIGNIFICANT"
    return "OPPOSITE_NS"


def sensitivity(pts, ev_pool, c):
    out = []
    base = ev_pool
    variants = {
        "quality>=6": base[base.EVENT_QUALITY >= 6], "quality>=7": base[base.EVENT_QUALITY >= 7],
        "quality>=8": base[base.EVENT_QUALITY >= 8],
        "exclude_mass_sightings(report_count>=10)": base[base.REPORT_COUNT < 10],
        "exclude_military_witness": base[~base.WITNESS_TYPE.fillna("").str.contains("military")],
        "only_trained_witness": base[base.WITNESS_TYPE.fillna("").str.contains("pilot|police|military|astronomer|air_traffic")],
        "without_quality_score(ALL)": base,
    }
    case_ap = pts[(pts.strategy == "CASE")].set_index("EVENT_ID")["dist_large_airport_km"]
    variants["exclude_within25km_large_airport"] = base[base.EVENT_ID.map(case_ap).fillna(1e9) > 25]
    for reg in ["NE", "MW", "S", "W"]:
        import analysis_data as A
        variants[f"drop_region_{reg}"] = base[base.STATE.map(A.CENSUS_REGION).fillna("X") != reg]
    for dec in [1990, 2000, 2010, 2020]:
        variants[f"drop_decade_{dec}s"] = base[(base.YEAR // 10 * 10) != dec]
    for name, e in variants.items():
        ids = e.EVENT_ID.to_numpy() if "ALL" in name or "quality" in name else subset_ids(e, c["subset"])
        r = run_one(pts, ids, c, "pooled_nondiscovery", "sensitivity", register=False)
        out.append(dict(candidate_id=c["candidate_id"], variant=name, n=r.get("n_cases"), OR=r.get("effect"),
                        lo=r.get("ci_low"), hi=r.get("ci_high"), p=r.get("p_value")))
    for st in ("CS1", "CS2", "CS4"):
        r = run_one(pts, subset_ids(base, c["subset"]), c, "pooled_nondiscovery", "sensitivity", register=False, strategy=st)
        out.append(dict(candidate_id=c["candidate_id"], variant=f"control_strategy_{st}", n=r.get("n_cases"),
                        OR=r.get("effect"), lo=r.get("ci_low"), hi=r.get("ci_high"), p=r.get("p_value")))
    return out


TEMPORAL_RECOMPUTE = {
    "space_weather": lambda t, la, lo, tz: F.space_weather_features(t),
    "astro": lambda t, la, lo, tz: F.astro_calendar_features(t, la, lo, tz),
    "quake": lambda t, la, lo, tz: F.quake_features(t, la, lo),
    "storm": lambda t, la, lo, tz: F.storm_event_features(t, la, lo, weather.storm_events()),
    "launch": lambda t, la, lo, tz: F.launch_features(t, la, lo),
    "fireball": lambda t, la, lo, tz: F.fireball_features(t, la, lo),
    "weather": lambda t, la, lo, tz: weather.weather_features(t, la, lo),
    "media": lambda t, la, lo, tz: F.media_features(t),
}


def var_kind(v):
    if v.startswith("wx_"):
        return "weather"
    if re.match(r"(any|n)_eq", v):
        return "quake"
    if re.match(r"(any|n)_storm", v):
        return "storm"
    if "launch" in v or "starlink" in v:
        return "launch"
    if "fireball" in v or v.startswith("n_fb"):
        return "fireball"
    if v.startswith(("media", "log_wiki")):
        return "media"
    if v.startswith(("moon", "venus", "shower", "jupiter", "hol_")):
        return "astro"
    if v.startswith(("within", "log_dist", "in_icbm", "n_airports", "n_dod")):
        return "spatial"
    return "space_weather"


def permutation_temporal(pts, ids, c, n_perm=200, seed=MASTER_SEED + 61):
    """Shift exposure time by k*364 d (k in +-1..+-15) for all points; keeps season,
    weekday, time of day and geography; destroys the exact temporal alignment."""
    kind = var_kind(c["variable"])
    d = cc.matched(pts, ids, c["control_strategy"]).copy()
    t = F.to_sec(d.utc_ts)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    tz = d.TIMEZONE.to_numpy()
    obs = run_one(d, ids, c, "perm_observed", "permutation", register=False)
    rng = np.random.default_rng(seed)
    ks = [k for k in range(-15, 16) if k != 0]
    nulls = []
    for i in range(n_perm):
        k = ks[i % len(ks)] if i < len(ks) else int(rng.choice(ks))
        f = TEMPORAL_RECOMPUTE[kind](t + k * 364 * 86400.0, la, lo, tz)
        f.index = d.index
        dd = d.drop(columns=[x for x in f.columns if x in d.columns]).join(f)
        dd = derive(dd)
        if c["variable"] not in dd:
            continue
        r = run_one(dd, ids, c, "perm_null", "permutation", register=False)
        if np.isfinite(r.get("effect", np.nan)):
            nulls.append(np.log(r["effect"]))
    nulls = np.array(nulls)
    lo_ = np.log(obs["effect"])
    p = (np.sum(np.abs(nulls) >= abs(lo_)) + 1) / (len(nulls) + 1)
    return dict(candidate_id=c["candidate_id"], kind=kind, observed_or=obs["effect"], n_null=len(nulls),
                null_median_or=float(np.exp(np.median(nulls))) if len(nulls) else np.nan,
                null_95pct_or=float(np.exp(np.percentile(nulls, 97.5))) if len(nulls) else np.nan, p_perm=p)


def permutation_spatial(pts, ids, c, n_perm=200, seed=MASTER_SEED + 62):
    """Translate the facility layer by a random offset (+-1..4 deg lat/lon) and
    recompute distances: preserves the layer's internal geometry and the
    events' spatial distribution, breaks their alignment."""
    d = cc.matched(pts, ids, c["control_strategy"]).copy()
    obs = run_one(d, ids, c, "perm_observed", "permutation", register=False)
    v = c["variable"]
    m = re.match(r"(?:within(\d+)_|log_dist_)(\w+?)(?:_km)?$", v)
    layer = m.group(2) if m else None
    rng = np.random.default_rng(seed)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    yr = d.utc_ts.dt.year.to_numpy()
    nulls = []
    for _ in range(n_perm):
        dlat = rng.uniform(1, 4) * rng.choice([-1, 1])
        dlon = rng.uniform(1, 4) * rng.choice([-1, 1])
        dist = shifted_layer_distance(layer, la, lo, yr, dlat, dlon)
        if dist is None:
            return dict(candidate_id=c["candidate_id"], kind="spatial", observed_or=obs["effect"], n_null=0, p_perm=np.nan,
                        note=f"layer {layer} not supported")
        if v.startswith("within"):
            d["x_perm"] = (dist <= float(m.group(1))).astype(float)
        else:
            d["x_perm"] = np.log10(np.clip(dist, 0.5, None))
        cc2 = dict(c, variable="x_perm")
        r = run_one(d, ids, cc2, "perm_null", "permutation", register=False)
        if np.isfinite(r.get("effect", np.nan)):
            nulls.append(np.log(r["effect"]))
    nulls = np.array(nulls)
    lo_ = np.log(obs["effect"])
    return dict(candidate_id=c["candidate_id"], kind="spatial", observed_or=obs["effect"], n_null=len(nulls),
                null_median_or=float(np.exp(np.median(nulls))), null_95pct_or=float(np.exp(np.percentile(nulls, 97.5))),
                p_perm=(np.sum(np.abs(nulls) >= abs(lo_)) + 1) / (len(nulls) + 1))


def shifted_layer_distance(layer, la, lo, yr, dlat, dlon):
    A = spatial.airports()
    A = A[A.iso_country.isin(["US", "CA", "MX"])]
    pt_layers = {
        "large_airport": A[A.type == "large_airport"], "medium_airport": A[A.type == "medium_airport"],
        "small_airport": A[A.type == "small_airport"], "dod_site": spatial.mirta_points(),
        "nuclear_operating": spatial.nuclear_operating(), "launch_site": spatial.launch_sites_us(),
    }
    if layer in pt_layers:
        L = pt_layers[layer]
        return spatial.nearest_km(la, lo, L.lat.to_numpy() + dlat, L.lon.to_numpy() + dlon)
    if layer in ("icbm_field_active", "doe_weapons_active", "nuclear_closed_active", "nuclear_test_site"):
        cur = spatial.curated()
        name = {"icbm_field_active": "icbm_field", "doe_weapons_active": "doe_weapons_site",
                "nuclear_closed_active": "nuclear_plant_closed", "nuclear_test_site": "nuclear_test_site"}[layer]
        sub = cur[cur.layer == name].dropna(subset=["lat"])
        best = np.full(len(la), np.inf)
        from common import haversine_km
        for _, s in sub.iterrows():
            active = (yr >= s.start) & ((yr <= s.end) if pd.notna(s.end) else True)
            dd = haversine_km(la, lo, s.lat + dlat, s.lon + dlon)
            best = np.where(active & (dd < best), dd, best)
        return best
    if layer in ("sua_moa", "sua_restricted", "sua_alert", "sua_prohibited", "dod_boundary", "coast"):
        return spatial.polygon_distance_km(la - dlat, lo - dlon, layer)  # equivalent to moving the layer by +d
    return None


# ---------------------------------------------------------------- fast weather permutation
_WX = {}
WX_BASE = {"wx_clear": ["wx_sky_oktas"], "wx_overcast": ["wx_sky_oktas"], "wx_sky_oktas": ["wx_sky_oktas"],
           "wx_front_pressure_fall": ["wx_dslp_24h"], "wx_cold_front_proxy": ["wx_dtemp_24h", "wx_dslp_24h"],
           "wx_clearing_6h": ["wx_dsky_6h"], "wx_inversion_night": ["wx_inversion_proxy"],
           "wx_precip_any": ["wx_precip_1h_mm"]}


def _wx_matrix(cols):
    """Station x hour matrices (float32) for the requested ISD-derived columns, 1995-2023."""
    key = tuple(sorted(cols))
    if key in _WX:
        return _WX[key]
    st = weather.stations().sid.tolist()
    t0 = pd.Timestamp("1995-01-01", tz="UTC")
    nh = int((pd.Timestamp("2024-01-01", tz="UTC") - t0) / pd.Timedelta(hours=1))
    M = {c: np.full((len(st), nh), np.nan, dtype=np.float32) for c in cols}
    for i, sid in enumerate(st):
        df = weather.load_station(sid)
        if df.empty:
            continue
        hi = ((df.index - t0) / pd.Timedelta(hours=1)).astype(int)
        ok = (hi >= 0) & (hi < nh)
        for c in cols:
            M[c][i, hi[ok]] = df[c].to_numpy(np.float32)[ok]
    _WX[key] = (st, t0, nh, M)
    return _WX[key]


def _wx_value(var, base):
    b = base
    if var == "wx_clear":
        v = (b["wx_sky_oktas"] <= 2).astype(float)
        v[np.isnan(b["wx_sky_oktas"])] = np.nan
    elif var == "wx_overcast":
        v = (b["wx_sky_oktas"] >= 7).astype(float)
        v[np.isnan(b["wx_sky_oktas"])] = np.nan
    elif var == "wx_front_pressure_fall":
        v = (b["wx_dslp_24h"] <= -8).astype(float)
        v[np.isnan(b["wx_dslp_24h"])] = np.nan
    elif var == "wx_cold_front_proxy":
        v = ((b["wx_dtemp_24h"] <= -8) & (b["wx_dslp_24h"] > 0)).astype(float)
        v[np.isnan(b["wx_dtemp_24h"])] = np.nan
    elif var == "wx_clearing_6h":
        v = (b["wx_dsky_6h"] <= -4).astype(float)
        v[np.isnan(b["wx_dsky_6h"])] = np.nan
    elif var == "wx_precip_any":
        v = (b["wx_precip_1h_mm"] > 0).astype(float)
        v[np.isnan(b["wx_precip_1h_mm"])] = np.nan
    else:
        v = b[var].astype(float)
    return v


def permutation_weather(pts, ids, c, n_perm=200, seed=MASTER_SEED + 63):
    var = c["variable"]
    cols = WX_BASE.get(var, [var])
    if var == "wx_inversion_night":
        cols = ["wx_inversion_proxy"]
    d = cc.matched(pts, ids, c["control_strategy"]).copy()
    obs = run_one(d, ids, c, "perm_observed", "permutation", register=False)
    st, t0, nh, M = _wx_matrix(cols)
    sidx = {s: i for i, s in enumerate(st)}
    s_i = np.array([sidx.get(x, -1) for x in d.wx_station.astype(object).where(d.wx_station.notna(), None)])
    h = np.round((F.to_sec(d.utc_ts) - t0.value / 1e9) / 3600.0).astype(np.int64)
    ks = [k for k in range(-15, 16) if abs(k) >= 1]
    rng = np.random.default_rng(seed)
    nulls = []
    for i in range(n_perm):
        k = ks[i % len(ks)] if i < len(ks) else int(rng.choice(ks))
        hh = h + k * 364 * 24
        ok = (s_i >= 0) & (hh >= 0) & (hh < nh)
        base = {}
        for col in cols:
            v = np.full(len(d), np.nan)
            v[ok] = M[col][s_i[ok], hh[ok]]
            base[col] = v
        x = _wx_value(var if var != "wx_inversion_night" else "wx_inversion_proxy", base)
        if var == "wx_inversion_night":
            x = x * (d.sun_alt.to_numpy() < -6)
        d["x_perm"] = x
        r = run_one(d, ids, dict(c, variable="x_perm"), "perm_null", "permutation", register=False)
        if np.isfinite(r.get("effect", np.nan)):
            nulls.append(np.log(r["effect"]))
    nulls = np.array(nulls)
    lo_ = np.log(obs["effect"])
    return dict(candidate_id=c["candidate_id"], kind="weather", observed_or=obs["effect"], n_null=len(nulls),
                null_median_or=float(np.exp(np.median(nulls))) if len(nulls) else np.nan,
                null_95pct_or=float(np.exp(np.percentile(nulls, 97.5))) if len(nulls) else np.nan,
                p_perm=(np.sum(np.abs(nulls) >= abs(lo_)) + 1) / (len(nulls) + 1))


def main(do_perm=True, n_perm=200):
    spec, h = load_frozen()
    pts = derive(cc.points())
    ev = cc.events()
    sets = eval_sets(ev)
    rows, sens, perms = [], [], []
    for c in spec["candidates"]:
        for label, e in sets.items():
            ids = subset_ids(e, c["subset"])
            r = run_one(pts, ids, c, label, "validation")
            r["candidate_id"] = c["candidate_id"]
            r["eval_set"] = label
            r["verdict"] = verdict(r, c["direction"])
            rows.append(r)
        # LOSO: pooled non-discovery sources, drop one source at a time
        pool = pd.concat([sets[k] for k in sets])
        for src in ["NUFORC", "GEIPAN", "HATCH", "BLUEBOOK_UNK", "NICAP"]:
            e = pool[pool.SOURCE != src]
            r = run_one(pts, subset_ids(e, c["subset"]), c, f"LOSO_drop_{src}", "validation")
            r.update(candidate_id=c["candidate_id"], eval_set=f"LOSO_drop_{src}", verdict=verdict(r, c["direction"]))
            rows.append(r)
        nuf_pool = pd.concat([sets["E1_validation"], sets["E2_holdout_random"], sets["E3_holdout_temporal"]])
        if c.get("kind", "single") == "single":
            sens += sensitivity(pts, nuf_pool, c)
        # Two-stage rule (declared before validation): permutation only if REPLICATED in >= 2 of E1-E3
        n_rep = sum(1 for r in rows if r["candidate_id"] == c["candidate_id"] and r["eval_set"] in
                    ("E1_validation", "E2_holdout_random", "E3_holdout_temporal") and r["verdict"] == "REPLICATED")
        if do_perm and n_rep >= 2 and c.get("kind", "single") == "single":
            ids = subset_ids(nuf_pool, c["subset"])
            kind = var_kind(c["variable"])
            if kind == "spatial":
                pr = permutation_spatial(pts, ids, c, n_perm)
            elif kind == "weather":
                pr = permutation_weather(pts, ids, c, n_perm)
            else:
                pr = permutation_temporal(pts, ids, c, n_perm if kind not in ("astro",) else 100)
            perms.append(pr)
            print(pr, flush=True)
        print(c["candidate_id"], c["variable"], [(r["eval_set"], round(r.get("effect", np.nan), 3), r["verdict"])
                                                 for r in rows if r["candidate_id"] == c["candidate_id"]], flush=True)
    out = pd.DataFrame(rows)
    out["frozen_sha256"] = h
    out.to_csv(RESULTS / "validation_results.csv", index=False)
    pd.DataFrame(sens).to_csv(RESULTS / "validation_sensitivity.csv", index=False)
    pd.DataFrame(perms).to_csv(RESULTS / "validation_permutation.csv", index=False)


if __name__ == "__main__":
    import sys
    main(do_perm="--no-perm" not in sys.argv)
