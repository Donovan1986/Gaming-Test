"""Positive-control safeguards; synthetic inputs and no reserved outcomes."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import positive_controls as controls
import validate


def fixture():
    columns = ["sun_alt", "shower_zhr_total", "n_fb_pm30m_1000km", "n_fb_pm30m_500km",
               "n_launch_lpost3h_1500km", "days_since_starlink_launch", "starlink_era",
               "hol_independence_day", "venus_alt", "venus_elong", "iss_visible_win",
               "moon_alt", "moon_illum", "meteor_outburst_pm1d", "dist_large_airport_km"]
    result = pd.DataFrame({name: [1.] for name in columns})
    result["mmdd"] = "01-01"
    return result


class ControlTests(unittest.TestCase):
    def test_joint_visibility_exposures_preserve_missingness(self):
        points = fixture()
        points["sun_alt"] = np.nan
        points["days_since_starlink_launch"] = np.nan
        result = controls.add_exposures(points)
        for exposure in ("x_launch_twilight", "x_launch_dark_or_day", "x_venus_vis", "x_starlink_10d"):
            self.assertTrue(np.isnan(result[exposure].iloc[0]), exposure)

    def test_missing_elongation_is_not_venus_absence(self):
        points = fixture()
        points["sun_alt"] = -10.
        points["venus_alt"] = 30.
        points["venus_elong"] = np.nan
        self.assertTrue(np.isnan(controls.add_exposures(points).x_venus_vis.iloc[0]))

    def test_invalid_fit_is_not_a_successful_control(self):
        row = dict(p_value=0.001, effect=2., ci_low=1.5, ci_high=3., status="NONCONVERGED")
        self.assertEqual(controls.classify(row, "+"), "NOT_TESTABLE")
        row.update(status="OK", effect=np.nan)
        self.assertEqual(controls.classify(row, "+"), "NOT_TESTABLE")

    def test_starlink_integrity_verifies_freeze_before_reserved_reads(self):
        with patch.object(validate, "load_frozen", side_effect=ValueError("Tampered freeze")), \
             patch.object(controls.cc, "events", side_effect=AssertionError("Reserved outcomes read")):
            with self.assertRaisesRegex(ValueError, "Tampered"):
                controls.post_freeze_starlink_integrity()


if __name__ == "__main__":
    unittest.main()
