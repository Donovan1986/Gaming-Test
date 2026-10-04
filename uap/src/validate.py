"""Validation of FROZEN candidates on unseen data.

Refuses to run unless results/frozen_hypotheses.json matches its recorded
sha256 (written by select_candidates.freeze before this script first ran).

For each candidate (variable, subset, control strategy, direction frozen):
  E1  validation split (NUFORC North America)       E2  locked random holdout (NUFORC North America)
  E3  temporal holdout (NUFORC North America)       E4  geographic: GEIPAN (France)
  E5  geographic: NUFORC outside North America      E6  cross-source: HATCH
  E7  cross-source: Blue Book unknowns              E8  cross-source: NICAP
  LOSO pooled (all non-discovery sets) minus one source at a time
  Sensitivity grid (on validation + holdouts pooled)
  Exposure-shift diagnostics (E1+E2+E3 pooled; empirical ranks, not calibrated permutation inference)
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
from discovery_tests import derive, derived_exposure, FAM_A, GRID_WINDOWS, GRID_RADII
from common import RESULTS, MASTER_SEED

SUBSET_COLS = {"ALL": None, "HQ": "HIGH_QUALITY", "HQ_UNEXPLAINED": "HQ_UNEXPLAINED", "MULTI_SENSOR": "MULTI_SENSOR",
               "UNEXPLAINED": "UNEXPLAINED", "EXPLAINED": "P_EXPLAINED"}
VALID_TEST_STATUSES = {"OK", "OK_EXACT"}
MIN_VALID_NULLS = 20
TEMPORAL_SHIFT_LIMIT = 30


def validate_spec(spec):
    """Reject ambiguous legacy specifications before loading validation data."""
    if not isinstance(spec, dict) or not isinstance(spec.get("candidates"), list):
        raise ValueError("frozen spec must contain a candidates list")
    ids = set()
    for c in spec["candidates"]:
        if not isinstance(c, dict):
            raise ValueError("each frozen candidate must be an object")
        for key in ("candidate_id", "family", "variable", "hypothesis"):
            if not isinstance(c.get(key), str) or not c[key].strip():
                raise ValueError(f"frozen candidate requires a nonempty {key}")
        if c["candidate_id"] in ids:
            raise ValueError("duplicate frozen candidate_id")
        ids.add(c["candidate_id"])
        if c.get("subset") not in SUBSET_COLS:
            raise ValueError("unknown frozen subset")
        if c.get("control_strategy") not in {"CS1", "CS2", "CS4"}:
            raise ValueError("unknown frozen control strategy")
        if c.get("direction") not in {"+", "-"}:
            raise ValueError("frozen direction must be + or -")
        scale = c.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("frozen candidate requires a finite positive scale; legacy scales cannot be guessed")
        if var_kind(c["variable"]).startswith("grid_"):
            grid_cell(c)
    return spec


def load_frozen():
    txt = (RESULTS / "frozen_hypotheses.json").read_text()
    h = (RESULTS / "frozen_hypotheses.sha256").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", h) or hashlib.sha256(txt.encode()).hexdigest() != h:
        raise ValueError("frozen spec modified after freezing - refusing to validate")
    return validate_spec(json.loads(txt)), h


def eval_sets(ev):
    t_ok = ev.utc_ts.notna() & (ev.TIME_UNCERTAINTY_MIN < 720)
    n = ev.SOURCE == "NUFORC"
    country = ev.COUNTRY.fillna("").astype(str).str.strip().str.upper()
    north_america = country.isin(["US", "USA", "CA", "CANADA"])
    known_country = ~country.isin(["", "UNKNOWN", "UNK", "NOT_REPORTED", "N/A", "NA"])
    return {
        "E1_validation": ev[n & north_america & (ev.SPLIT == "validation") & t_ok],
        "E2_holdout_random": ev[n & north_america & (ev.SPLIT == "holdout_random") & t_ok],
        "E3_holdout_temporal": ev[n & north_america & (ev.SPLIT == "holdout_temporal") & t_ok],
        "E4_geo_GEIPAN": ev[(ev.SOURCE == "GEIPAN") & t_ok],
        "E5_geo_NUFORC_nonUS": ev[n & known_country & ~north_america & t_ok],
        "E6_src_HATCH": ev[(ev.SOURCE == "HATCH") & t_ok],
        "E7_src_BLUEBOOK": ev[(ev.SOURCE == "BLUEBOOK_UNK") & t_ok],
        "E8_src_NICAP": ev[(ev.SOURCE == "NICAP") & t_ok],
    }


def subset_ids(e, subset):
    if subset not in SUBSET_COLS:
        raise ValueError(f"unknown subset {subset}")
    if subset == "EXPLAINED":
        return e.loc[e.P_EXPLAINED >= 0.6, "EVENT_ID"].to_numpy()
    col = SUBSET_COLS[subset]
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
                  split=label, phase=phase, register=register, min_cases=10, scale=c["scale"])


def verdict(row, direction):
    if not valid_fit(row):
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
    base = ev_pool.drop_duplicates("EVENT_ID")
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
                        lo=r.get("ci_low"), hi=r.get("ci_high"), p=r.get("p_value"), status=r.get("status"),
                        inference_method=r.get("inference_method"), notes=r.get("notes"), analysis="exploratory"))
    for st in ("CS1", "CS2", "CS4"):
        r = run_one(pts, subset_ids(base, c["subset"]), c, "pooled_nondiscovery", "sensitivity", register=False, strategy=st)
        out.append(dict(candidate_id=c["candidate_id"], variant=f"control_strategy_{st}", n=r.get("n_cases"),
                        OR=r.get("effect"), lo=r.get("ci_low"), hi=r.get("ci_high"), p=r.get("p_value"), status=r.get("status"),
                        inference_method=r.get("inference_method"), notes=r.get("notes"), analysis="exploratory"))
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
    grid = re.fullmatch(r"any_g(quake|storm|launch)_(\w+)_(\d+)km", v)
    if grid:
        return "grid_" + grid.group(1)
    if v == "iss_visible_win":
        return "iss"
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
    if v.startswith(("moon", "venus", "shower", "jupiter", "hol_", "meteor_outburst")):
        return "astro"
    if v.startswith(("within", "log_dist", "in_icbm", "n_airports", "n_dod")):
        return "spatial"
    if v in {x[0] for x in FAM_A}:
        return "space_weather"
    return "unsupported"


class UnsupportedExposure(ValueError):
    pass


def grid_cell(c):
    """Check frozen grid metadata without opening catalogs or outcomes."""
    m = re.fullmatch(r"any_g(quake|storm|launch)_(\w+)_(\d+)km", c["variable"])
    if m is None:
        raise UnsupportedExposure("unrecognized grid variable")
    kind, window, radius = m.group(1), m.group(2), int(m.group(3))
    if window not in GRID_WINDOWS or radius not in GRID_RADII:
        raise UnsupportedExposure("grid window/radius is outside the declared design")
    if c.get("window") != window or str(c.get("radius_km")) not in {str(radius), str(float(radius))}:
        raise ValueError("frozen grid metadata disagrees with its variable")
    return kind, window, radius


def grid_exposure(t, la, lo, c, countries=None):
    """Materialize exactly the frozen grid cell, with its catalog coverage."""
    kind, window, radius = grid_cell(c)
    if kind == "quake":
        cat = F.C.quakes().rename(columns={"latitude": "lat", "longitude": "lon"})
        coverage = (pd.Timestamp("1973-01-01", tz="UTC"), cat.time.max())
    elif kind == "storm":
        cat = weather.storm_events()
        coverage = (pd.Timestamp("1996-01-01", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC"))
    else:
        cat = F.C.launches()
        cat = cat[cat.time_has_clock & cat.lat.notna()]
        coverage = (pd.Timestamp("1957-01-01", tz="UTC"), cat.time.max())
    if cat.empty or pd.isna(coverage[1]):
        return pd.Series(np.nan, index=range(len(t)))
    counts = F.window_counts(t, la, lo, cat, {window: GRID_WINDOWS[window]}, [radius],
                             prefix=f"g{kind}_", coverage=coverage).iloc[:, 0]
    if kind in {"quake", "storm"}:
        counts = counts.where(F.catalog_location_mask(la, lo, radius, kind, countries=countries))
    return (counts > 0).astype(float).where(counts.notna())


def iss_exposure(t, la, lo):
    import iss
    # Same fixed 10-minute half-window used to construct the original points.
    return iss.iss_visible_window(t, la, lo, np.full(len(t), 10.0))


def candidate_from_fresh(f, v):
    """Derive a candidate using fresh dependencies only; never fall back to old columns."""
    try:
        return derived_exposure(f, v)
    except KeyError as e:
        raise UnsupportedExposure(f"fresh dependencies unavailable for {v}: {e}") from e


def recompute_candidate(d, c, t=None):
    """Return only the frozen exposure; absent targets explicitly fail."""
    v, kind = c["variable"], var_kind(c["variable"])
    t = F.to_sec(d.utc_ts) if t is None else np.asarray(t, float)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    countries = d.COUNTRY.to_numpy() if "COUNTRY" in d else None
    if kind.startswith("grid_"):
        x = grid_exposure(t, la, lo, c, countries=countries)
    elif kind == "iss":
        x = iss_exposure(t, la, lo)
    else:
        if kind not in TEMPORAL_RECOMPUTE:
            raise UnsupportedExposure(f"temporal recomputation for {v} is not supported")
        if kind == "storm":
            f = F.storm_event_features(t, la, lo, weather.storm_events(), countries=countries)
        else:
            f = TEMPORAL_RECOMPUTE[kind](t, la, lo, d.TIMEZONE.to_numpy())
        if v == "wx_inversion_night":
            astro = F.astro_calendar_features(t, la, lo, d.TIMEZONE.to_numpy())
            if "sun_alt" in astro:
                f = f.copy()
                f["sun_alt"] = astro.sun_alt.to_numpy()
        x = candidate_from_fresh(f, v)
    return pd.Series(np.asarray(x, float), index=d.index, name=v)


def materialize_candidate(pts, c):
    """Grid discovery columns are transient and must be constructed for validation."""
    if c.get("kind") in ("interaction", "sequence"):
        return pts  # constructed inside run_one (product term / ordered-pair indicator)
    if c["variable"] in pts and not var_kind(c["variable"]).startswith("grid_"):
        return pts
    out = pts.copy()
    out[c["variable"]] = recompute_candidate(pts, c).to_numpy()
    return out


def valid_fit(row):
    effect, p = row.get("effect", np.nan), row.get("p_value", np.nan)
    return row.get("status") in VALID_TEST_STATUSES and np.isfinite(effect) and effect > 0 and np.isfinite(p) and 0 <= p <= 1


def retained_sample(d, variable):
    """Fix the exact observed rows; every null must cover this same population."""
    d = d[np.isfinite(d[variable].to_numpy(float))].copy()
    g = d.groupby("EVENT_ID", sort=False).is_case.agg(["sum", "size"])
    ids = g.index[(g["sum"] == 1) & (g["size"] >= 2)]
    return d[d.EVENT_ID.isin(ids)]


def diagnostic_result(c, kind, obs, nulls=(), *, status, n_unique=0, n_requested=0,
                      permutation_limit=None, n_coverage_excluded=0, n_fit_excluded=0, note=""):
    nulls = np.asarray(nulls, float)
    enough = len(nulls) >= MIN_VALID_NULLS and valid_fit(obs)
    p = (np.count_nonzero(np.abs(nulls) >= abs(np.log(obs["effect"]))) + 1) / (len(nulls) + 1) if enough else np.nan
    return dict(candidate_id=c["candidate_id"], kind=kind, status=status, observed_or=obs.get("effect", np.nan),
                n_null=len(nulls), n_unique=n_unique, n_requested=n_requested, permutation_limit=permutation_limit,
                min_valid_nulls=MIN_VALID_NULLS, n_coverage_excluded=n_coverage_excluded, n_fit_excluded=n_fit_excluded,
                null_median_or=float(np.exp(np.median(nulls))) if len(nulls) else np.nan,
                null_95pct_or=float(np.exp(np.percentile(nulls, 97.5))) if len(nulls) else np.nan,
                p_perm=p, rank_resolution=1 / (len(nulls) + 1) if enough else np.nan,
                inference_method="exposure_shift_diagnostic", note=note)


def permutation_temporal(pts, ids, c, n_perm=200, seed=MASTER_SEED + 61):
    """Evaluate each of the 30 possible 364-d shifts once, on fixed observed rows.

    Weekday and UTC hour are preserved; season drifts by up to ~19 days and
    local hour can change across DST. Ranks are sensitivity diagnostics, not
    calibrated randomization p-values: exchangeability is not established.
    """
    kind = var_kind(c["variable"])
    try:
        d = cc.matched(materialize_candidate(pts, c), ids, c["control_strategy"]).copy()
    except UnsupportedExposure as e:
        return diagnostic_result(c, kind, {}, status="NOT_SUPPORTED", n_requested=n_perm,
                                 permutation_limit=TEMPORAL_SHIFT_LIMIT, note=str(e))
    d = retained_sample(d, c["variable"])
    obs = run_one(d, ids, c, "perm_observed", "permutation", register=False)
    if not valid_fit(obs):
        return diagnostic_result(c, kind, obs, status="INVALID_OBSERVED", n_requested=n_perm,
                                 permutation_limit=TEMPORAL_SHIFT_LIMIT)
    t = F.to_sec(d.utc_ts)
    rng = np.random.default_rng(seed)
    ks = rng.permutation([k for k in range(-15, 16) if k != 0])[:max(0, min(n_perm, TEMPORAL_SHIFT_LIMIT))]
    nulls, coverage_excluded, fit_excluded = [], 0, 0
    for k in ks:
        try:
            x = recompute_candidate(d, c, t + k * 364 * 86400.0)
        except UnsupportedExposure as e:
            return diagnostic_result(c, kind, obs, status="NOT_SUPPORTED", n_requested=n_perm,
                                     permutation_limit=TEMPORAL_SHIFT_LIMIT, note=str(e))
        if not np.isfinite(x.to_numpy()).all():
            coverage_excluded += 1
            continue
        dd = d.copy()
        dd[c["variable"]] = x.to_numpy()
        r = run_one(dd, ids, c, "perm_null", "permutation", register=False)
        if valid_fit(r):
            nulls.append(np.log(r["effect"]))
        else:
            fit_excluded += 1
    return diagnostic_result(c, kind, obs, nulls, status="OK_DIAGNOSTIC" if len(nulls) >= MIN_VALID_NULLS else "INSUFFICIENT_NULLS",
                             n_unique=len(ks), n_requested=n_perm, permutation_limit=TEMPORAL_SHIFT_LIMIT,
                             n_coverage_excluded=coverage_excluded, n_fit_excluded=fit_excluded,
                             note="At most 30 distinct shifts; empirical ranks are not calibrated permutation inference.")


def permutation_spatial(pts, ids, c, n_perm=200, seed=MASTER_SEED + 62):
    """Translate the facility layer by a random offset (+-1..4 deg lat/lon) and
    recompute distances: preserves the layer's internal geometry and the
    events' spatial distribution, breaks their alignment."""
    d = retained_sample(cc.matched(pts, ids, c["control_strategy"]).copy(), c["variable"])
    obs = run_one(d, ids, c, "perm_observed", "permutation", register=False)
    if not valid_fit(obs):
        return diagnostic_result(c, "spatial", obs, status="INVALID_OBSERVED", n_requested=n_perm)
    v = c["variable"]
    m = re.match(r"(?:within(\d+)_|log_dist_)(\w+?)(?:_km)?$", v)
    layer = m.group(2) if m else None
    supported = {"large_airport", "medium_airport", "small_airport", "dod_site", "nuclear_operating", "launch_site",
                 "icbm_field_active", "doe_weapons_active", "nuclear_closed_active", "nuclear_test_site",
                 "sua_moa", "sua_restricted", "sua_alert", "sua_prohibited", "dod_boundary", "coast"}
    if layer not in supported:
        return diagnostic_result(c, "spatial", obs, status="NOT_SUPPORTED", n_requested=n_perm,
                                 note=f"layer {layer} is not supported")
    rng = np.random.default_rng(seed)
    la, lo = d.lat.to_numpy(float), d.lon.to_numpy(float)
    yr = d.utc_ts.dt.year.to_numpy()
    nulls, offsets, coverage_excluded, fit_excluded = [], set(), 0, 0
    if not (np.isfinite(la) & np.isfinite(lo)).all():
        return diagnostic_result(c, "spatial", obs, status="INSUFFICIENT_COVERAGE", n_requested=n_perm,
                                 note="Observed matched rows require finite latitude and longitude.")
    for _ in range(max(0, n_perm)):
        dlat = rng.uniform(1, 4) * rng.choice([-1, 1])
        dlon = rng.uniform(1, 4) * rng.choice([-1, 1])
        if (dlat, dlon) in offsets:
            continue
        offsets.add((dlat, dlon))
        dist = shifted_layer_distance(layer, la, lo, yr, dlat, dlon)
        if dist is None:
            return diagnostic_result(c, "spatial", obs, status="NOT_SUPPORTED", n_requested=n_perm,
                                     note=f"layer {layer} is not supported")
        dist = np.asarray(dist, float)
        if np.isnan(dist).any():
            coverage_excluded += 1
            continue
        if v.startswith("within"):
            d["x_perm"] = (dist <= float(m.group(1))).astype(float)
        else:
            d["x_perm"] = np.log10(np.clip(dist, 0.5, None))
        if not np.isfinite(d.x_perm.to_numpy()).all():
            coverage_excluded += 1
            continue
        cc2 = dict(c, variable="x_perm")
        r = run_one(d, ids, cc2, "perm_null", "permutation", register=False)
        if valid_fit(r):
            nulls.append(np.log(r["effect"]))
        else:
            fit_excluded += 1
    return diagnostic_result(c, "spatial", obs, nulls, status="OK_DIAGNOSTIC" if len(nulls) >= MIN_VALID_NULLS else "INSUFFICIENT_NULLS",
                             n_unique=len(offsets), n_requested=n_perm, n_coverage_excluded=coverage_excluded,
                             n_fit_excluded=fit_excluded,
                             note="Translated layers provide empirical sensitivity ranks, not calibrated permutation inference.")


def shifted_layer_distance(layer, la, lo, yr, dlat, dlon):
    if layer in {"large_airport", "medium_airport", "small_airport"}:
        A = spatial.airports()
        L = A[A.iso_country.isin(["US", "CA", "MX"]) & (A.type == layer)]
        return spatial.nearest_km(la, lo, L.lat.to_numpy() + dlat, L.lon.to_numpy() + dlon)
    pt_layers = {"dod_site": spatial.mirta_points, "nuclear_operating": spatial.nuclear_operating,
                 "launch_site": spatial.launch_sites_us}
    if layer in pt_layers:
        L = pt_layers[layer]()
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


def adjust_validation(rows, candidates):
    """Correct all finite scheduled validation tests, including LOSO, together."""
    out = pd.DataFrame(rows)
    directions = {c["candidate_id"]: c["direction"] for c in candidates}
    records = out.to_dict("records")
    p = np.array([r["p_value"] if valid_fit(r) else np.nan for r in records], float)
    out["q_bh_validation"] = S.bh(p)
    out["p_holm_validation"] = S.holm(p)
    out["verdict_nominal"] = [verdict(r, directions[r["candidate_id"]]) for r in records]
    out["verdict"] = [verdict(dict(r, p_value=adjusted), directions[r["candidate_id"]])
                      for r, adjusted in zip(records, out.p_holm_validation)]
    out["validation_multiplicity"] = "Holm across all finite scheduled validation and LOSO tests"
    return out


def main(do_perm=True, n_perm=200):
    spec, h = load_frozen()
    pts = derive(cc.points())
    ev = cc.events()
    sets = eval_sets(ev)
    rows, sens, perms = [], [], []
    for c in spec["candidates"]:
        try:
            candidate_pts = materialize_candidate(pts, c)
        except UnsupportedExposure as e:
            for label in sets:
                rows.append(dict(candidate_id=c["candidate_id"], variable_a=c["variable"], eval_set=label,
                                 status="NOT_SUPPORTED", effect=np.nan, p_value=np.nan, verdict="NOT_TESTABLE", notes=str(e)))
            if do_perm:
                perms.append(diagnostic_result(c, var_kind(c["variable"]), {}, status="NOT_SUPPORTED", n_requested=n_perm, note=str(e)))
            continue
        for label, e in sets.items():
            ids = subset_ids(e, c["subset"])
            r = run_one(candidate_pts, ids, c, label, "validation")
            r["candidate_id"] = c["candidate_id"]
            r["eval_set"] = label
            r["verdict"] = verdict(r, c["direction"])
            rows.append(r)
        # LOSO: pooled non-discovery sources, drop one source at a time
        pool = pd.concat(list(sets.values())).drop_duplicates("EVENT_ID")
        for src in ["NUFORC", "GEIPAN", "HATCH", "BLUEBOOK_UNK", "NICAP"]:
            e = pool[pool.SOURCE != src]
            r = run_one(candidate_pts, subset_ids(e, c["subset"]), c, f"LOSO_drop_{src}", "validation")
            r.update(candidate_id=c["candidate_id"], eval_set=f"LOSO_drop_{src}", verdict=verdict(r, c["direction"]))
            rows.append(r)
        nuf_pool = pd.concat([sets["E1_validation"], sets["E2_holdout_random"], sets["E3_holdout_temporal"]]).drop_duplicates("EVENT_ID")
        if c.get("kind", "single") == "single":
            sens += sensitivity(candidate_pts, nuf_pool, c)
        # Two-stage rule (declared before validation): permutation only if nominally REPLICATED in >= 2 of E1-E3
        n_rep = sum(1 for r in rows if r["candidate_id"] == c["candidate_id"] and r["eval_set"] in
                    ("E1_validation", "E2_holdout_random", "E3_holdout_temporal") and r["verdict"] == "REPLICATED")
        if do_perm and n_rep >= 2 and c.get("kind", "single") == "single":
            ids = subset_ids(nuf_pool, c["subset"])
            kind = var_kind(c["variable"])
            if kind == "spatial":
                pr = permutation_spatial(candidate_pts, ids, c, n_perm)
            elif kind == "weather":
                pr = permutation_weather(candidate_pts, ids, c, n_perm)
            else:
                pr = permutation_temporal(candidate_pts, ids, c, n_perm if kind != "astro" else 100)
            perms.append(pr)
            print(pr, flush=True)
    out = adjust_validation(rows, spec["candidates"])
    for c in spec["candidates"]:
        selected = out[out.candidate_id == c["candidate_id"]]
        print(c["candidate_id"], c["variable"], [(r["eval_set"], round(r.get("effect", np.nan), 3), r["verdict"])
                                                 for r in selected.to_dict("records")], flush=True)
    out["frozen_sha256"] = h
    out.to_csv(RESULTS / "validation_results.csv", index=False)
    pd.DataFrame(sens).to_csv(RESULTS / "validation_sensitivity.csv", index=False)
    pd.DataFrame(perms).to_csv(RESULTS / "validation_permutation.csv", index=False)


if __name__ == "__main__":
    import sys
    main(do_perm="--no-perm" not in sys.argv)
