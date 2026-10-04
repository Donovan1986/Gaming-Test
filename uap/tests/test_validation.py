"""Synthetic validation regressions; no repository datasets or outcomes are read."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import validate as V


def candidate(**changes):
    c = dict(candidate_id="C01", family="A_global_temporal", variable="kp", hypothesis="synthetic hypothesis",
             subset="ALL", control_strategy="CS1", direction="+", scale=1.0)
    c.update(changes)
    return c


def points(n=12, variable="kp"):
    return pd.DataFrame({"EVENT_ID": np.repeat([f"s{i}" for i in range(n)], 2),
                         "strategy": ["CASE", "CS1"] * n, "is_case": [1, 0] * n,
                         "utc_ts": pd.to_datetime(["2000-01-01", "2000-01-08"] * n, utc=True),
                         "lat": [40.] * (2 * n), "lon": [-100.] * (2 * n), "TIMEZONE": ["UTC"] * (2 * n),
                         variable: [1., 0.] * n})


def fitted(d, ids, c, label, phase, **kwargs):
    return dict(effect=2. if label == "perm_observed" else 1.5, p_value=0.01, status="OK")


class FrozenSpecificationTests(unittest.TestCase):
    def test_hash_is_enforced_in_optimized_python(self):
        with tempfile.TemporaryDirectory(prefix="uap-validation-test-") as tmp:
            root = Path(tmp)
            (root / "frozen_hypotheses.json").write_text(json.dumps({"candidates": [candidate()]}))
            (root / "frozen_hypotheses.sha256").write_text("0" * 64)
            code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import validate; from pathlib import Path; "
                    f"validate.RESULTS = Path({tmp!r}); validate.load_frozen()")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run([sys.executable, "-O", "-c", code], env=env, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("frozen spec modified", proc.stderr)

    def test_valid_hash_and_scale_load(self):
        with tempfile.TemporaryDirectory(prefix="uap-validation-test-") as tmp:
            txt = json.dumps({"candidates": [candidate(scale=10.)]})
            root = Path(tmp)
            (root / "frozen_hypotheses.json").write_text(txt)
            (root / "frozen_hypotheses.sha256").write_text(hashlib.sha256(txt.encode()).hexdigest())
            with patch.object(V, "RESULTS", root):
                spec, _ = V.load_frozen()
            self.assertEqual(spec["candidates"][0]["scale"], 10.)

    def test_ambiguous_specifications_are_rejected(self):
        missing = candidate()
        del missing["scale"]
        bad = [missing, candidate(scale=0), candidate(scale=np.nan), candidate(scale=True),
               candidate(subset="TYPO"), candidate(direction="either"), candidate(control_strategy="CS3")]
        for c in bad:
            with self.subTest(candidate=c), self.assertRaises(ValueError):
                V.validate_spec({"candidates": [c]})
        with self.assertRaises(ValueError):
            V.validate_spec({"candidates": [candidate(), candidate()]})
        grid = candidate(variable="any_gquake_pm1h_100km", window="pm3h", radius_km="100")
        with patch.object(V.F.C, "quakes") as catalog, self.assertRaises(ValueError):
            V.validate_spec({"candidates": [grid]})
        catalog.assert_not_called()

    def test_validation_passes_frozen_effect_scale(self):
        with patch.object(V.cc, "run", return_value={}) as run:
            V.run_one(pd.DataFrame(), [], candidate(scale=100.), "synthetic", "validation", register=False)
        self.assertEqual(run.call_args.kwargs["scale"], 100.)


class EvaluationCohortTests(unittest.TestCase):
    def test_geographic_and_north_american_sets_are_disjoint_without_relabeling(self):
        countries = ["US", "USA", "CA", "Canada", "FR", "FR", "GB", None, "UNKNOWN", ""]
        splits = ["validation"] * 5 + ["discovery", "holdout_temporal"] + ["validation"] * 3
        ev = pd.DataFrame({"EVENT_ID": [f"e{i}" for i in range(len(countries))], "SOURCE": "NUFORC",
                           "COUNTRY": countries, "SPLIT": splits, "utc_ts": pd.Timestamp("2000-01-01", tz="UTC"),
                           "TIME_UNCERTAINTY_MIN": 1})
        original = ev.SPLIT.copy()
        sets = V.eval_sets(ev)
        self.assertEqual(set(sets["E1_validation"].EVENT_ID), {"e0", "e1", "e2", "e3"})
        self.assertEqual(set(sets["E5_geo_NUFORC_nonUS"].EVENT_ID), {"e4", "e5", "e6"})
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertFalse(set(sets[a].EVENT_ID) & set(sets[b].EVENT_ID))
        pd.testing.assert_series_equal(ev.SPLIT, original)

    def test_explained_subset_is_not_all_events(self):
        ev = pd.DataFrame({"EVENT_ID": ["a", "b", "c"], "P_EXPLAINED": [0.6, 0.2, np.nan]})
        self.assertEqual(V.subset_ids(ev, "EXPLAINED").tolist(), ["a"])
        with self.assertRaises(ValueError):
            V.subset_ids(ev, "TYPO")

    def test_replication_rejects_invalid_status_and_effect(self):
        for status, effect in [("NOT_TESTABLE", 2.), ("UNADJUSTED_EXACT_CLUSTER_DEPENDENCE", 2.),
                               ("OK", np.nan), ("OK", 0.), ("OK", np.inf)]:
            self.assertEqual(V.verdict(dict(status=status, effect=effect, p_value=0.001), "+"), "NOT_TESTABLE")
        self.assertEqual(V.verdict(dict(status="OK_EXACT", effect=2., p_value=0.001), "+"), "REPLICATED")

    def test_final_replication_uses_holm_across_validation_and_loso(self):
        rows = [dict(candidate_id="C01", eval_set=label, status="OK", effect=2., p_value=p)
                for label, p in [("E1_validation", 0.04), ("E2_holdout_random", 0.04), ("LOSO_drop_GEIPAN", 0.001)]]
        rows.append(dict(candidate_id="C01", eval_set="E3_holdout_temporal", status="NOT_TESTABLE", effect=2., p_value=0.0001))
        out = V.adjust_validation(rows, [candidate()])
        np.testing.assert_allclose(out.p_holm_validation, [0.08, 0.08, 0.003, np.nan], equal_nan=True)
        np.testing.assert_allclose(out.q_bh_validation, [0.04, 0.04, 0.003, np.nan], equal_nan=True)
        self.assertEqual(out.verdict_nominal.tolist(), ["REPLICATED", "REPLICATED", "REPLICATED", "NOT_TESTABLE"])
        self.assertEqual(out.verdict.tolist(), ["SAME_DIRECTION_NS", "SAME_DIRECTION_NS", "REPLICATED", "NOT_TESTABLE"])


class ShiftDiagnosticTests(unittest.TestCase):
    def test_invalid_observed_fit_never_gets_small_rank(self):
        for status, effect in [("NOT_TESTABLE", np.nan), ("OK", 0.), ("OK", np.inf),
                               ("UNADJUSTED_EXACT_CLUSTER_DEPENDENCE", 2.)]:
            row = dict(status=status, effect=effect, p_value=0.001)
            with self.subTest(status=status, effect=effect), patch.object(V, "run_one", return_value=row), \
                    patch.object(V, "recompute_candidate") as recompute:
                out = V.permutation_temporal(points(), points().EVENT_ID, candidate())
            self.assertEqual(out["status"], "INVALID_OBSERVED")
            self.assertTrue(np.isnan(out["p_perm"]))
            recompute.assert_not_called()
        p = points(variable="within25_large_airport")
        with patch.object(V, "run_one", return_value=dict(status="NOT_TESTABLE", effect=np.nan, p_value=np.nan)), \
                patch.object(V, "shifted_layer_distance") as shifted:
            out = V.permutation_spatial(p, p.EVENT_ID, candidate(variable="within25_large_airport"))
        self.assertTrue(np.isnan(out["p_perm"]))
        shifted.assert_not_called()

    def test_temporal_shifts_are_unique_and_resolution_is_honest(self):
        p = points()
        seen = []
        def fresh(d, c, t):
            seen.append(tuple(t))
            return pd.Series([0., 1.] * 12, index=d.index)
        with patch.object(V, "run_one", side_effect=fitted), patch.object(V, "recompute_candidate", side_effect=fresh):
            out = V.permutation_temporal(p, p.EVENT_ID, candidate(), n_perm=200)
        self.assertEqual(len(seen), 30)
        self.assertEqual(len(set(seen)), 30)
        self.assertEqual(out["n_unique"], 30)
        self.assertEqual(out["n_requested"], 200)
        self.assertEqual(out["permutation_limit"], 30)
        self.assertAlmostEqual(out["rank_resolution"], 1 / 31)
        self.assertAlmostEqual(out["p_perm"], 1 / 31)

    def test_unknown_coverage_never_changes_null_population(self):
        p = points()
        calls = []
        def run(d, *args, **kwargs):
            calls.append(d.index.tolist())
            return fitted(d, *args, **kwargs)
        with patch.object(V, "run_one", side_effect=run), \
                patch.object(V, "recompute_candidate", return_value=pd.Series([np.nan] + [0.] * 23)):
            out = V.permutation_temporal(p, p.EVENT_ID, candidate())
        self.assertEqual(calls, [p.index.tolist()])
        self.assertEqual(out["n_coverage_excluded"], 30)
        self.assertEqual(out["status"], "INSUFFICIENT_NULLS")
        self.assertTrue(np.isnan(out["p_perm"]))

    def test_invalid_null_fit_is_excluded_even_with_finite_effect(self):
        p = points()
        def run(d, ids, c, label, phase, **kwargs):
            row = fitted(d, ids, c, label, phase, **kwargs)
            if label == "perm_null":
                row["status"] = "UNADJUSTED_EXACT_CLUSTER_DEPENDENCE"
            return row
        with patch.object(V, "run_one", side_effect=run), \
                patch.object(V, "recompute_candidate", return_value=p.kp):
            out = V.permutation_temporal(p, p.EVENT_ID, candidate())
        self.assertEqual(out["n_null"], 0)
        self.assertEqual(out["n_fit_excluded"], 30)
        self.assertTrue(np.isnan(out["p_perm"]))

    def test_too_few_distinct_valid_nulls_have_no_rank(self):
        p = points()
        with patch.object(V, "run_one", side_effect=fitted), patch.object(V, "recompute_candidate", return_value=p.kp):
            out = V.permutation_temporal(p, p.EVENT_ID, candidate(), n_perm=10)
        self.assertEqual(out["n_null"], 10)
        self.assertEqual(out["status"], "INSUFFICIENT_NULLS")
        self.assertTrue(np.isnan(out["p_perm"]))

    def test_missing_fresh_candidate_cannot_reuse_original_column(self):
        p = points(variable="kp_storm5_prior24h")
        with patch.object(V, "run_one", side_effect=fitted), \
                patch.dict(V.TEMPORAL_RECOMPUTE, space_weather=lambda *args: pd.DataFrame({"wrong": [0.] * 24})):
            out = V.permutation_temporal(p, p.EVENT_ID, candidate(variable="kp_storm5_prior24h"))
        self.assertEqual(out["status"], "NOT_SUPPORTED")
        self.assertEqual(out["n_null"], 0)
        self.assertTrue(np.isnan(out["p_perm"]))

    def test_spatial_empty_nulls_and_unsupported_layers_are_explicit(self):
        p = points(variable="within25_large_airport")
        with patch.object(V, "run_one", side_effect=fitted), \
                patch.object(V, "shifted_layer_distance", return_value=np.full(len(p), np.nan)):
            out = V.permutation_spatial(p, p.EVENT_ID, candidate(variable="within25_large_airport"))
        self.assertEqual(out["status"], "INSUFFICIENT_NULLS")
        self.assertEqual(out["n_coverage_excluded"], 200)
        self.assertTrue(np.isnan(out["null_95pct_or"]))
        p = points(variable="in_icbm_field_active")
        with patch.object(V, "run_one", side_effect=fitted), patch.object(V, "shifted_layer_distance") as shifted:
            out = V.permutation_spatial(p, p.EVENT_ID, candidate(variable="in_icbm_field_active"))
        self.assertEqual(out["status"], "NOT_SUPPORTED")
        shifted.assert_not_called()

    def test_retained_sample_removes_invalid_sets_and_missing_rows_once(self):
        p = points(3)
        p.loc[0, "kp"] = np.nan  # Missing case removes the whole first set.
        p.loc[3, "kp"] = np.nan  # Missing only control removes the second set.
        out = V.retained_sample(p, "kp")
        self.assertEqual(out.EVENT_ID.tolist(), ["s2", "s2"])


class FreshExposureTests(unittest.TestCase):
    def test_iss_recomputation_replaces_the_original_exposure(self):
        p = points(1, variable="iss_visible_win")
        with patch.object(V, "iss_exposure", return_value=np.array([0., 1.])):
            out = V.recompute_candidate(p, candidate(variable="iss_visible_win"))
        self.assertEqual(out.tolist(), [0., 1.])

    def test_grid_cell_is_materialized_and_missing_coverage_stays_unknown(self):
        p = points(1, variable="any_gquake_pm1h_100km")
        p["any_gquake_pm1h_100km"] = [0., 1.]
        c = candidate(variable="any_gquake_pm1h_100km", family="G_quake_grid", window="pm1h", radius_km="100")
        q = pd.DataFrame({"time": pd.to_datetime(["2000-01-01", "2020-01-01"], utc=True),
                          "latitude": [40., 40.], "longitude": [-100., -100.], "mag": [3., 3.]})
        with patch.object(V.F.C, "quakes", return_value=q):
            out = V.materialize_candidate(p, c)
            past = p.copy()
            past["utc_ts"] = pd.to_datetime(["1970-01-01", "1970-01-08"], utc=True)
            missing = V.materialize_candidate(past, c)
        self.assertEqual(out[c["variable"]].tolist(), [1., 0.])
        self.assertTrue(missing[c["variable"]].isna().all())
        with patch.object(V.F.C, "quakes", return_value=q), self.assertRaises(ValueError):
            V.materialize_candidate(p, dict(c, window="pm3h"))

    def test_grid_geographic_coverage_cannot_be_assumed_from_temporal_coverage(self):
        p = points(1, variable="any_gquake_pm1h_100km")
        p["lat"], p["lon"] = -33., 151.
        c = candidate(variable="any_gquake_pm1h_100km", family="G_quake_grid", window="pm1h", radius_km="100")
        q = pd.DataFrame({"time": pd.to_datetime(["2000-01-01", "2020-01-01"], utc=True),
                          "latitude": [40., 40.], "longitude": [-100., -100.], "mag": [3., 3.]})
        with patch.object(V.F.C, "quakes", return_value=q):
            out = V.materialize_candidate(p, c)
        self.assertTrue(out[c["variable"]].isna().all())

    def test_storm_recomputation_passes_matched_country_metadata(self):
        p = points(1, variable="any_storm_pm1h_25km").assign(COUNTRY=["US", "CA"])
        fresh = pd.DataFrame({"n_storm_pm1h_25km": [1., np.nan]})
        with patch.object(V.weather, "storm_events", return_value=pd.DataFrame()), \
                patch.object(V.F, "storm_event_features", return_value=fresh) as storm:
            out = V.recompute_candidate(p, candidate(variable="any_storm_pm1h_25km"))
        self.assertEqual(storm.call_args.kwargs["countries"].tolist(), ["US", "CA"])
        np.testing.assert_allclose(out, [1., np.nan], equal_nan=True)

    def test_fresh_binary_derivation_preserves_missing_dependencies(self):
        f = pd.DataFrame({"n_eq_pre1d_100km": [0., np.nan, 1.]})
        out = V.candidate_from_fresh(f, "any_eq_pre1d_100km")
        np.testing.assert_allclose(out, [0., np.nan, 1.], equal_nan=True)

    def test_main_materializes_grid_before_fitting_and_deduplicates_pool(self):
        c = candidate(variable="any_gquake_pm1h_100km", family="G_quake_grid", window="pm1h", radius_km="100")
        ev = pd.DataFrame({"EVENT_ID": ["s0"], "SOURCE": ["NUFORC"]})
        sets = {"E1_validation": ev, "E2_holdout_random": ev.iloc[:0], "E3_holdout_temporal": ev.iloc[:0],
                "E5_geo_NUFORC_nonUS": ev.copy()}
        p = points(1)
        materialized = p.assign(**{c["variable"]: [1., 0.]})
        seen = []
        def run(d, ids, candidate, label, phase, **kwargs):
            self.assertIn(c["variable"], d)
            seen.append((label, list(ids)))
            return dict(effect=2., p_value=0.01, status="OK")
        with patch.object(V, "load_frozen", return_value=({"candidates": [c]}, "synthetic")), \
                patch.object(V.cc, "points", return_value=p), patch.object(V.cc, "events", return_value=ev), \
                patch.object(V, "derive", side_effect=lambda d: d), patch.object(V, "eval_sets", return_value=sets), \
                patch.object(V, "materialize_candidate", return_value=materialized), patch.object(V, "run_one", side_effect=run), \
                patch.object(V, "sensitivity", return_value=[]), patch.object(pd.DataFrame, "to_csv"), patch("builtins.print"):
            V.main(do_perm=False)
        pooled = [ids for label, ids in seen if label == "LOSO_drop_GEIPAN"]
        self.assertEqual(pooled, [["s0"]])


if __name__ == "__main__":
    unittest.main()
