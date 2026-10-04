"""Geometry and propagation regressions using synthetic observations only."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import covariates
import iss


class Satellite:
    def __init__(self, error=0, position=(-7000., 0., 0.)):
        self.error = error
        self.position = position
        self.samples = []

    def sgp4_array(self, jd, fraction):
        self.samples.append(len(jd))
        return (np.full(len(jd), self.error),
                np.tile(self.position, (len(jd), 1)), np.zeros((len(jd), 3)))


class AstronomyTests(unittest.TestCase):
    def test_same_orbital_bin_retains_different_observation_rotation(self):
        times = pd.to_datetime(["2020-01-01T00:01:00Z", "2020-01-01T00:09:00Z"])

        def rotation(t):
            return np.asarray((pd.DatetimeIndex(t) - times[0]).total_seconds()) / 86400 * 2 * np.pi

        def positions(t, body):
            return np.zeros(len(t)), np.zeros(len(t)), np.ones(len(t))

        with patch.object(covariates, "gmst_rad", side_effect=rotation), \
             patch.object(covariates, "body_radec", side_effect=positions):
            result = covariates.astro_features(times, np.zeros(2), np.full(2, 90.))
        self.assertAlmostEqual(result.sun_alt.iloc[0], 0., places=8)
        self.assertAlmostEqual(result.sun_alt.iloc[1], -2., places=8)

    def visible(self, satellite, **kwargs):
        t = pd.Timestamp("2020-01-01T00:00:00Z").timestamp()
        jd = t / 86400 + 2440587.5
        options = dict(t_sec=[t], lat=[0.], lon=[180.], half_window_min=[10])
        options.update(kwargs)
        sun = np.array([0.1, np.sqrt(0.99), 0.])
        with patch.object(iss, "tles", return_value=([satellite, satellite], np.array([jd-1, jd+1]))), \
             patch.object(iss, "_gmst", side_effect=lambda x: np.zeros_like(x)), \
             patch.object(iss, "_sun_eci", side_effect=lambda x: np.broadcast_to(sun, x.shape+(3,))):
            return iss.iss_visible_window(**options)

    def test_valid_visible_pass_and_requested_window_only(self):
        satellite = Satellite()
        self.assertEqual(self.visible(satellite)[0], 1.)
        self.assertEqual(satellite.samples, [21])

    def test_failed_propagation_cannot_establish_absence(self):
        self.assertTrue(np.isnan(self.visible(Satellite(error=6))[0]))

    def test_nonfinite_position_cannot_establish_absence(self):
        self.assertTrue(np.isnan(self.visible(Satellite(position=(np.nan, 0., 0.)))[0]))

    def test_successfully_propagated_invisible_pass_is_zero(self):
        self.assertEqual(self.visible(Satellite(position=(7000., 0., 0.)))[0], 0.)

    def test_missing_longitude_does_not_load_archive_or_become_zero(self):
        with patch.object(iss, "tles", side_effect=AssertionError("Archive must not be loaded")):
            value = iss.iss_visible_window([0.], [0.], [np.nan], [10])
        self.assertTrue(np.isnan(value[0]))

    def test_empty_and_single_epoch_archives_are_safe(self):
        with patch.object(iss, "tles", return_value=([], np.array([]))):
            self.assertTrue(np.isnan(iss.iss_visible_window([0.], [0.], [0.], [10])[0]))
        satellite = Satellite()
        with patch.object(iss, "tles", return_value=([satellite], np.array([2440587.5]))):
            result = iss.iss_visible_window([86400.], [0.], [0.], [10])
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(satellite.samples, [])


if __name__ == "__main__":
    unittest.main()
