"""Synthetic exploratory-model regressions; no scientific data or registry writes."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.model_selection import GroupKFold

import ml_discovery as ml


def matched_points(n=120, seed=32, clustered=True):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"EVENT_ID": np.repeat(np.arange(n), 3),
                      "is_case": np.tile([1, 0, 0], n),
                      "strategy": np.tile(["CASE", "CS1", "CS1"], n),
                      "a": rng.normal(size=n * 3), "b": rng.normal(size=n * 3)})
    if clustered:
        d["clu_night_i"] = d.EVENT_ID % 20
        d["clu_i"] = d.EVENT_ID % 17
    return d


class ModelTests(unittest.TestCase):
    def test_fold_imputation_and_scaling_use_only_training_data(self):
        d = pd.DataFrame({"EVENT_ID": np.repeat(np.arange(5), 2), "is_case": np.tile([1, 0], 5),
                          "x": [np.nan, 1000, 1, 3, 7, 9, 11, 13, 15, 17],
                          "empty": [np.nan, np.inf] * 5})
        seen_rf, seen_l1, seen_params = [], [], []

        class Predictor:
            def __init__(self, records):
                self.records = records

            def fit(self, x, y):
                self.records.append([x.copy()])

            def predict_proba(self, x):
                self.records[-1].append(x.copy())
                return np.full((len(x), 2), 0.5)

        class Tree:
            def predict(self, x):
                self_x = np.asarray(x)
                if np.isinf(self_x).any():
                    raise AssertionError("Infinite LightGBM feature")
                return np.full(len(x), 0.5)

            def feature_importance(self, kind):
                return np.ones(2)

        def train(params, data, **kwargs):
            seen_params.append(params)
            return Tree()

        with patch.object(ml.lgb, "train", side_effect=train), \
                patch.object(ml.lgb, "Dataset", side_effect=lambda x, y: (x, y)), \
                patch.object(ml, "RandomForestClassifier", side_effect=lambda **kw: Predictor(seen_rf)), \
                patch.object(ml, "LogisticRegression", side_effect=lambda **kw: Predictor(seen_l1)):
            results, _, _ = ml.fit_models(d, ["x", "empty"], "synthetic", "synthetic")
        x = d[["x", "empty"]].replace([np.inf, -np.inf], np.nan)
        for index, (tr, te) in enumerate(GroupKFold(5).split(x, d.is_case, d.EVENT_ID)):
            filled = x.fillna(x.iloc[tr].median().fillna(0))
            pd.testing.assert_frame_equal(seen_rf[index][0], filled.iloc[tr])
            pd.testing.assert_frame_equal(seen_rf[index][1], filled.iloc[te])
            spread = filled.iloc[tr].std().replace(0, np.nan).fillna(1)
            scaled = (filled - filled.iloc[tr].mean()) / spread
            pd.testing.assert_frame_equal(seen_l1[index][0], scaled.iloc[tr])
            pd.testing.assert_frame_equal(seen_l1[index][1], scaled.iloc[te])
            self.assertTrue(np.isfinite(scaled).all().all())
        self.assertTrue(all(params["num_threads"] == 4 for params in seen_params))
        self.assertTrue(all(row["evidence_role"] == "exploratory_discovery_cv" for row in results.values()))

    def test_insufficient_cv_groups_fail_explicitly(self):
        d = matched_points(1)
        with self.assertRaisesRegex(ValueError, "at least two matched sets"):
            ml.fit_models(d, ["a"], "synthetic", "synthetic")


class MatchedExplorationTests(unittest.TestCase):
    def test_interaction_uses_cluster_t_and_identical_nested_rows(self):
        d = matched_points(12)
        d.loc[0, "a"] = np.inf  # Missing case drops the entire first set.
        d.loc[4:5, "b"] = np.nan  # No controls drops the second set.
        calls = []

        def fit(x, strata, case, clusters=None):
            calls.append((x.copy(), strata.copy(), case.copy(), clusters.copy()))
            k = np.asarray(x).shape[1]
            return dict(status="OK", beta=np.array([0.1, 0.2, 0.8])[:k], se=np.full(k, 0.4),
                        ll=-10 if k == 3 else -11, n_strata=10, n_case=10, n_ctrl=20, n_clusters=3)

        with patch.object(ml.S, "clogit", side_effect=fit), patch.object(ml.S, "register") as registry:
            row = ml.test_interaction(d, d.EVENT_ID.unique(), "a", "b", "CS1", "SYNTHETIC", register=False)
        self.assertAlmostEqual(row["p_value"], 2 * sps.t.sf(2, 2))
        self.assertAlmostEqual(row["ci_high"], np.exp(0.8 + sps.t.ppf(0.975, 2) * 0.4))
        self.assertEqual(row["inference_method"], "wald_cluster_robust_t")
        self.assertIn("not calibrated", row["notes"])
        self.assertTrue(np.isfinite(calls[0][0]).all())
        self.assertFalse(np.isin(calls[0][1], [0, 1]).any())
        for full, reduced in zip(calls[0][1:], calls[1][1:]):
            np.testing.assert_array_equal(full, reduced)
        registry.assert_not_called()

    def test_invalid_interaction_fits_never_produce_inference(self):
        d = matched_points()
        invalid = [dict(status="SEPARATION", beta=None, se=None, ll=np.nan),
                   dict(status="NONCONVERGED", beta=np.ones(3), se=np.ones(3), ll=-10),
                   dict(status="OK", beta=np.ones(3), se=np.zeros(3), ll=-10),
                   dict(status="OK", beta=np.ones(3), se=np.ones(3), ll=-10, n_clusters=1)]
        for result in invalid:
            with self.subTest(status=result["status"], clusters=result.get("n_clusters")), \
                    patch.object(ml.S, "clogit", return_value=result), patch.object(ml.S, "register") as registry:
                self.assertIsNone(ml.test_interaction(d, d.EVENT_ID.unique(), "a", "b", "CS1", "SYNTHETIC"))
                registry.assert_not_called()

    def test_spline_reference_column_is_estimable_and_cluster_lr_descriptive(self):
        d = matched_points(400)
        d.loc[0, "a"] = np.inf
        calls = []
        original_fit = ml.S.clogit

        def record_fit(x, strata, case, **kwargs):
            result = original_fit(x, strata, case, **kwargs)
            calls.append((np.asarray(x).copy(), strata.copy(), result))
            return result

        import patsy
        with patch.object(ml.S, "clogit", side_effect=record_fit), \
                patch.object(patsy, "dmatrix", wraps=patsy.dmatrix) as design, \
                patch.object(patsy, "build_design_matrices", wraps=patsy.build_design_matrices) as reuse:
            row = ml.spline_clogit(d, d.EVENT_ID.unique(), "a", "CS1")
        self.assertIsNotNone(row)
        self.assertEqual(calls[0][2]["status"], "OK")
        self.assertEqual(calls[0][0].shape[1], 3)
        np.testing.assert_array_equal(calls[0][1], calls[1][1])
        self.assertFalse((calls[0][1] == 0).any())
        self.assertTrue(np.isnan(row["p_nonlinear"]))
        self.assertEqual(row["or_vs_median"][3], 1.0)
        self.assertEqual(row["inference_method"], "descriptive_clustered_fit")
        self.assertEqual(design.call_count, 1)
        self.assertEqual(reuse.call_count, 1)

    def test_spline_invalid_fit_returns_no_result(self):
        d = matched_points()
        with patch.object(ml.S, "clogit", return_value=dict(status="NONCONVERGED", beta=None, se=None)):
            self.assertIsNone(ml.spline_clogit(d, d.EVENT_ID.unique(), "a", "CS1"))


if __name__ == "__main__":
    unittest.main()
