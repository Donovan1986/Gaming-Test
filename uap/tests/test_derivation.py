"""Pure synthetic discovery/validation derivation tests; no datasets are read."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import discovery_tests as D
import validate as V


class SharedDerivationTests(unittest.TestCase):
    def test_partial_frame_derives_available_targets_without_unrelated_columns(self):
        p = pd.DataFrame({"kp_max_prior24h": [4., 5., np.nan, np.inf]})
        out = D.derive(p)
        np.testing.assert_allclose(out.kp_storm5_prior24h, [0., 1., np.nan, np.nan], equal_nan=True)
        self.assertNotIn("venus_vis", out)
        self.assertNotIn("wx_clear", out)
        self.assertEqual(list(p.columns), ["kp_max_prior24h"])

    def test_joint_missingness_never_becomes_false_zero(self):
        frames = {
            "moon_full_up": {"moon_alt": [30., 30., np.nan], "moon_illum": [0.95, np.nan, 0.95]},
            "venus_vis": {"venus_alt": [10., 10., 10.], "sun_alt": [-10., np.nan, -10.],
                          "venus_elong": [20., 20., np.nan]},
            "wx_cold_front_proxy": {"wx_dtemp_24h": [-10., -10., np.nan], "wx_dslp_24h": [2., np.nan, 2.]},
            "wx_inversion_night": {"wx_inversion_proxy": [1., 1., np.nan], "sun_alt": [-10., np.nan, -10.]},
        }
        for v, values in frames.items():
            p = pd.DataFrame(values)
            with self.subTest(variable=v):
                np.testing.assert_allclose(D.derived_exposure(p, v), [1., np.nan, np.nan], equal_nan=True)

    def test_discovery_and_validation_use_identical_derivations_on_partial_inputs(self):
        p = pd.DataFrame({"moon_alt": [20., np.nan, -20.], "moon_illum": [0.95, 0.95, 0.95],
                          "venus_alt": [10., 10., 10.], "sun_alt": [-10., np.nan, 1.], "venus_elong": [20., 20., 20.],
                          "wx_dtemp_24h": [-10., -10., np.nan], "wx_dslp_24h": [2., np.nan, 2.],
                          "wiki_ufo_views": [100., 0., np.nan], "n_eq_pre1d_100km": [1., np.nan, 0.],
                          "dist_large_airport_km": [20., np.nan, 100.]})
        discovered = D.derive(p)
        for v in set(discovered) - set(p):
            with self.subTest(variable=v):
                pd.testing.assert_series_equal(discovered[v], V.candidate_from_fresh(p, v))

    def test_existing_derived_values_cannot_replace_missing_raw_dependencies(self):
        p = pd.DataFrame({"moon_full_up": [1., 0.], "any_eq_pre1d_100km": [1., 0.]})
        with self.assertRaises(KeyError):
            D.derived_exposure(p, "moon_full_up")
        with self.assertRaises(V.UnsupportedExposure):
            V.candidate_from_fresh(p, "any_eq_pre1d_100km")
        self.assertNotIn("moon_full_up", D.derive(p))
        self.assertNotIn("any_eq_pre1d_100km", D.derive(p))

    def test_log_units_and_spatial_thresholds_are_preserved(self):
        p = pd.DataFrame({"wiki_ufo_views": [100., 0., np.inf, np.nan],
                          "dist_large_airport_km": [25., 25.01, np.inf, np.nan]})
        out = D.derive(p)
        np.testing.assert_allclose(out.log_wiki_views, [2., np.nan, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(out.within25_large_airport, [1., 0., np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(out.log_dist_large_airport_km,
                                   [np.log10(25), np.log10(25.01), np.nan, np.nan], equal_nan=True)

    def test_missing_starlink_history_is_unknown(self):
        p = pd.DataFrame({"days_since_starlink_launch": [1., np.nan, 1.], "starlink_era": [1., 1., 0.]})
        np.testing.assert_allclose(D.derived_exposure(p, "starlink_10d"), [1., np.nan, np.nan], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
