"""Synthetic inference regressions; never load research data or held-out outcomes."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import cc
import select_candidates as selection
import stats


def pairs(positive=20, negative=10):
    x = np.array([[1.0, 0.0]] * positive + [[0.0, 1.0]] * negative).ravel()
    n = positive + negative
    return x, np.repeat(np.arange(n), 2), np.tile([1, 0], n)


def points(x, strata, case):
    return pd.DataFrame({"EVENT_ID": strata, "strategy": np.where(case, "CASE", "CS1"),
                         "is_case": case, "x": x})


def candidate_row(strategy="CS1", subset="ALL", **updates):
    row = dict(variable_a="synthetic_solar", family="A_global_temporal", subset=subset,
               control_strategy=strategy, p_value=0.001, q_bh_all=0.01, status="OK",
               inference_version=stats.INFERENCE_VERSION, phase="discovery",
               effect=2.0, ci_low=1.5, ci_high=2.5, n_cases=1000, n_exposed_cases=np.nan,
               direction="+", scale=10.0, window="pre1d", radius_km="", hypothesis="synthetic exposure")
    row.update(updates)
    return row


class InferenceTests(unittest.TestCase):
    def test_conditional_logit_matches_exact_pair_mle(self):
        fitted = stats.clogit_or(*pairs())
        self.assertEqual(fitted["status"], "OK")
        self.assertAlmostEqual(fitted["OR"], 2.0, places=8)
        self.assertAlmostEqual(fitted["se"], np.sqrt(1 / 20 + 1 / 10), places=8)

    def test_changing_exposure_units_preserves_inference(self):
        x, strata, case = pairs()
        reference = stats.clogit_or(x, strata, case)
        for unit in (0.001, 1e7):
            fitted = stats.clogit_or(x * unit, strata, case, scale=unit)
            self.assertEqual(fitted["status"], "OK")
            for field in ("OR", "lo", "hi", "p", "beta", "se"):
                self.assertAlmostEqual(fitted[field], reference[field], places=8)

    def test_separated_pairs_have_no_wald_and_correct_exact_p(self):
        data = pairs(100, 0)
        fitted = stats.clogit_or(*data)
        self.assertEqual(fitted["status"], "SEPARATION")
        self.assertTrue(np.isnan(fitted["p"]))
        exact = stats.exact_conditional_or(*data)
        self.assertAlmostEqual(exact["p"] / (2 * 0.5 ** 100), 1.0, places=10)
        self.assertGreater(exact["lo"], 1)
        self.assertTrue(np.isinf(exact["hi"]))

    def test_heterogeneous_exact_p_matches_enumeration(self):
        xs, strata, cases = [], [], []
        for index, (n, exposed, case_exposed) in enumerate([(2, 1, 1), (3, 2, 0), (4, 1, 1)]):
            controls = [1] * (exposed - case_exposed) + [0] * (n - exposed - (1 - case_exposed))
            xs.extend([case_exposed] + controls)
            strata.extend([index] * n)
            cases.extend([1] + [0] * (n - 1))
        upper = lower = 0.0
        probabilities = (1 / 2, 2 / 3, 1 / 4)
        for outcome in itertools.product((0, 1), repeat=3):
            probability = np.prod([p if bit else 1 - p for p, bit in zip(probabilities, outcome)])
            upper += probability * (sum(outcome) >= 2)
            lower += probability * (sum(outcome) <= 2)
        exact = stats.exact_conditional_or(xs, strata, cases)
        self.assertAlmostEqual(exact["p"], min(1.0, 2 * min(upper, lower)), places=12)

    def test_equal_probability_exact_uses_binomial_without_quadratic_dp(self):
        data = pairs(4000, 2000)
        with patch.object(stats, "_poisson_binomial", side_effect=AssertionError("Unnecessary quadratic DP")):
            exact = stats.exact_conditional_or(*data)
        expected = 2 * scipy_stats.binom.sf(3999, 6000, 0.5)
        self.assertAlmostEqual(exact["p"] / expected, 1.0, places=8)

    def test_exact_rejects_continuous_exposure(self):
        x, strata, case = pairs()
        with self.assertRaises(ValueError):
            stats.exact_conditional_or(x * 0.2, strata, case)

    def test_no_variation_and_nonconvergence_never_emit_p(self):
        x, strata, case = pairs()
        no_variation = stats.clogit_or(np.ones_like(x), strata, case)
        self.assertEqual(no_variation["status"], "NO_WITHIN_SET_VARIATION")
        self.assertTrue(np.isnan(no_variation["p"]))
        unfinished = stats.clogit(x, strata, case, max_iter=1)
        self.assertEqual(unfinished["status"], "NONCONVERGED")

    def test_exposure_counts_use_retained_strata(self):
        x, strata, case = pairs(2, 20)
        extra_x = np.tile([1.0, np.nan], 1000)
        data = points(np.r_[x, extra_x], np.r_[strata, np.repeat(np.arange(22, 1022), 2)],
                      np.r_[case, np.tile([1, 0], 1000)])
        row = cc.run(data, data.EVENT_ID.unique(), "x", "CS1", family="synthetic",
                     hypothesis="synthetic", var_a="x", register=False)
        self.assertEqual(row["n_cases"], 22)
        self.assertEqual(row["n_exposed_cases"], 2)
        self.assertEqual(row["n_exposed_controls"], 20)

    def test_one_dependence_cluster_cannot_claim_exact_significance(self):
        data = points(*pairs(100, 0))
        data["clu_i"] = 0
        data["clu_night_i"] = 0
        row = cc.run(data, data.EVENT_ID.unique(), "x", "CS1", family="synthetic",
                     hypothesis="synthetic", var_a="x", register=False)
        self.assertEqual(row["status"], "UNADJUSTED_EXACT_CLUSTER_DEPENDENCE")
        self.assertTrue(np.isnan(row["p_value"]))
        self.assertNotIn(row["status"], selection.VALID)

    def test_independent_binary_fallback_remains_exact(self):
        data = points(*pairs(100, 0))
        row = cc.run(data, data.EVENT_ID.unique(), "x", "CS1", family="synthetic",
                     hypothesis="synthetic", var_a="x", register=False)
        self.assertEqual(row["status"], "OK_EXACT")
        self.assertAlmostEqual(row["p_value"] / (2 * 0.5 ** 100), 1.0, places=10)

    def test_cluster_inference_uses_limiting_dimension_t_distribution(self):
        x, strata, case = pairs(20, 10)
        clusters = np.c_[np.repeat(np.arange(10), 6), np.tile(np.repeat(np.arange(5), 2), 6)]
        fitted = stats.clogit_or(x, strata, case, clusters=clusters)
        self.assertEqual(fitted["status"], "OK")
        self.assertEqual(fitted["n_clusters"], 5)
        expected = 2 * scipy_stats.t.sf(abs(fitted["beta"] / fitted["se"]), df=4)
        self.assertAlmostEqual(fitted["p"], expected, places=12)

    def test_regular_binary_wald_skips_exact_diagnostic(self):
        data = points(*pairs())
        with patch.object(stats, "exact_conditional_or", side_effect=AssertionError("Unneeded exact diagnostic")):
            row = cc.run(data, data.EVENT_ID.unique(), "x", "CS1", family="synthetic",
                         hypothesis="synthetic", var_a="x", register=False)
        self.assertEqual(row["status"], "OK")
        self.assertEqual(row["inference_method"], "wald_model")


class CandidateTests(unittest.TestCase):
    def select(self, rows):
        with patch.object(pd, "read_csv", return_value=pd.DataFrame(rows)), \
                patch.object(pd.DataFrame, "to_csv"), contextlib.redirect_stdout(io.StringIO()):
            return selection.main()

    def test_untestable_required_strategy_cannot_be_replaced_by_overlapping_subsets(self):
        rows = [candidate_row(strategy, subset, p_value=np.nan if strategy == "CS2" else 0.001,
                              status="NOT_TESTABLE" if strategy == "CS2" else "OK")
                for subset in ("ALL", "HQ") for strategy in ("CS1", "CS2")]
        self.assertTrue(self.select(rows).empty)

    def test_two_exposed_cases_cannot_pass_minimum_thirty(self):
        rows = [candidate_row(strategy, n_exposed_cases=2) for strategy in ("CS1", "CS2")]
        self.assertTrue(self.select(rows).empty)

    def test_supported_candidate_preserves_scale(self):
        selected = self.select([candidate_row(strategy) for strategy in ("CS1", "CS2")])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0].scale, 10.0)

    def test_legacy_results_missing_scale_fail_before_other_work(self):
        row = candidate_row(family="C_spatial")
        row.pop("scale")
        with patch.object(selection, "halves_check") as halves, \
                patch.object(pd, "read_csv", return_value=pd.DataFrame([row])), \
                patch.object(pd.DataFrame, "to_csv") as writes:
            with self.assertRaisesRegex(RuntimeError, "rerun discovery_tests.py"):
                selection.main()
        halves.assert_not_called()
        writes.assert_not_called()

    def test_invalid_scale_and_superseded_versions_require_fresh_results(self):
        for update in ({"scale": np.nan}, {"scale": 0}, {"scale": -1}, {"scale": "bad"},
                       {"inference_version": "v3_dstfix_separation_exact"}, {"inference_version": None}):
            with self.subTest(update=update), self.assertRaisesRegex(RuntimeError, "rerun discovery_tests.py"):
                self.select([candidate_row(**update)])

    def test_legacy_results_missing_version_require_fresh_results(self):
        row = candidate_row()
        row.pop("inference_version")
        with self.assertRaisesRegex(RuntimeError, "rerun discovery_tests.py"):
            self.select([row])

    def test_undeclared_family_does_not_bypass_strategy_rule(self):
        self.assertTrue(self.select([candidate_row(family="not_predeclared")]).empty)

    def test_spatial_replication_checks_only_discovery_eligible_countries(self):
        import discovery_tests

        events = pd.DataFrame({"EVENT_ID": ["us-old", "fr-old", "us-new", "fr-new"],
                               "SOURCE": "NUFORC", "SPLIT": "discovery", "COUNTRY": ["US", "FR", "US", "FR"],
                               "YEAR": [1999, 1999, 2010, 2010], "HIGH_QUALITY": False, "HQ_UNEXPLAINED": False,
                               "MULTI_SENSOR": False, "UNEXPLAINED": False, "P_EXPLAINED": 0.0})
        calls = []

        def fit(pts, ids, *args, **kwargs):
            calls.append(list(ids))
            return {"status": "OK", "p_value": 0.001, "effect": 2.0}

        with patch.dict(selection._pts_cache, {}, clear=True), patch.object(cc, "points", return_value=pd.DataFrame()), \
                patch.object(cc, "events", return_value=events), patch.object(cc, "run", side_effect=fit), \
                patch.object(discovery_tests, "derive", side_effect=lambda frame: frame):
            supported, _ = selection.halves_check("synthetic", "ALL", "CS4", 1)
        self.assertTrue(supported)
        self.assertEqual(calls, [["us-old"], ["us-new"]])

    def test_freeze_is_immutable_idempotent_and_rejects_reserved_metadata(self):
        with tempfile.TemporaryDirectory() as scratch, patch.object(selection, "RESULTS", Path(scratch)):
            candidate = [{"candidate_id": "C01", "variable": "synthetic", "scale": 10.0,
                          "discovery_n_exposed_cases": np.nan, "discovery_ci": [1.5, np.inf]}]
            with contextlib.redirect_stdout(io.StringIO()):
                first = selection.freeze(candidate, {"design": "synthetic"})
            before = (Path(scratch) / "frozen_hypotheses.json").read_text()
            self.assertIsNone(json.loads(before)["candidates"][0]["discovery_n_exposed_cases"])
            self.assertEqual(selection.freeze(candidate, {"design": "synthetic"}), first)
            self.assertEqual((Path(scratch) / "frozen_hypotheses.json").read_text(), before)
            with self.assertRaises(RuntimeError):
                selection.freeze([{**candidate[0], "scale": 1.0}], {"design": "synthetic"})
            with self.assertRaises(ValueError):
                selection.freeze(candidate, {"candidates": []})
            self.assertEqual(hashlib.sha256(before.encode()).hexdigest(), first)

    def test_freeze_rejects_partial_or_corrupted_existing_record(self):
        with tempfile.TemporaryDirectory() as scratch, patch.object(selection, "RESULTS", Path(scratch)):
            path = Path(scratch) / "frozen_hypotheses.json"
            path.write_text(json.dumps({"candidates": []}))
            with self.assertRaises(RuntimeError):
                selection.freeze([], {})
            (Path(scratch) / "frozen_hypotheses.sha256").write_text("incorrect")
            with self.assertRaises(RuntimeError):
                selection.freeze([], {})

    def test_registry_migration_preserves_legacy_rows_and_adds_scale(self):
        with tempfile.TemporaryDirectory() as scratch:
            registry = Path(scratch) / "registry.csv"
            fields = ["test_id", "notes", "status", "inference_version", "legacy_extra"]
            legacy = dict(test_id="T000007", notes="historical\r\nmultiline evidence", status="SUPERSEDED",
                          inference_version="v1", legacy_extra="retain this")
            with registry.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(legacy)
            with patch.object(stats, "REGISTRY", registry), patch.object(stats, "LOGS", Path(scratch) / "logs"):
                new = stats.register(scale=10, notes="synthetic new fit", status="OK")
                self.assertEqual(new["test_id"], "T000008")
                with registry.open(newline="") as stream:
                    records = list(csv.DictReader(stream))
                for field, value in legacy.items():
                    self.assertEqual(records[0][field], value)
                self.assertEqual(records[0]["scale"], "")
                self.assertEqual(records[1]["scale"], "10")
                self.assertEqual(records[1]["inference_version"], "v4_cluster_safe_scale_invariant")
                self.assertFalse(registry.read_bytes().split(b"\n", 1)[0].endswith(b"\r"))

    def test_mixed_v2_header_and_v3_rows_migrate_without_losing_any_cell(self):
        with tempfile.TemporaryDirectory() as scratch:
            registry = Path(scratch) / "registry.csv"
            old = [f"legacy-{index}" for index in range(25)]
            old[0] = "T000001"
            old[23] = "historical, quoted\nmultiline note"
            newer = [f"newer-{index}" for index in range(29)]
            newer[0] = "T000980"
            newer[24] = "v3_dstfix_separation_exact"
            with registry.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(stats.FIELDS[:25])
                writer.writerow(old)
                writer.writerow(newer)
            with patch.object(stats, "REGISTRY", registry), patch.object(stats, "LOGS", Path(scratch) / "logs"):
                result = stats.migrate_registry_schema()
                self.assertEqual(result["rows_migrated"], 2)
                with registry.open(newline="") as stream:
                    records = list(csv.reader(stream))
                self.assertEqual(records[0], stats.FIELDS)
                self.assertEqual(records[1][:25], old)
                self.assertEqual(records[1][25:], [""] * 5)
                self.assertEqual(records[2][:29], newer)
                self.assertEqual(records[2][29:], [""])
                self.assertEqual(stats._next_id(), 981)

    def test_unproven_wider_registry_layout_fails_without_replacement(self):
        with tempfile.TemporaryDirectory() as scratch:
            registry = Path(scratch) / "registry.csv"
            with registry.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(stats.FIELDS[:25])
                writer.writerow(["unproven"] * 29)
            before = registry.read_bytes()
            with patch.object(stats, "REGISTRY", registry), patch.object(stats, "LOGS", Path(scratch) / "logs"):
                with self.assertRaises(RuntimeError):
                    stats.migrate_registry_schema()
            self.assertEqual(registry.read_bytes(), before)

    def test_canonical_header_does_not_hide_unproven_extra_cells(self):
        with tempfile.TemporaryDirectory() as scratch:
            registry = Path(scratch) / "registry.csv"
            with registry.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(stats.FIELDS)
                writer.writerow(["unproven"] * (len(stats.FIELDS) + 1))
            before = registry.read_bytes()
            with patch.object(stats, "REGISTRY", registry), patch.object(stats, "LOGS", Path(scratch) / "logs"):
                with self.assertRaises(RuntimeError):
                    stats.migrate_registry_schema()
            self.assertEqual(registry.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
