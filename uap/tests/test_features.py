"""Synthetic regression tests; no research observations or locked outcomes."""
import gzip
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "uap-test-numba-cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import analysis_data as A
import cohorts
import features as F
import weather


class FeatureIntegrity(unittest.TestCase):
    def test_quake_geography_requires_entire_radius_in_fetched_box(self):
        lat = np.array([40., 40., 71.9, 50., 50., 0.])
        lon = np.array([-105., -169.9, -105., 0., 19.9, 0.])
        np.testing.assert_array_equal(F.catalog_location_mask(lat, lon, 100, "quake"),
                                      [True, False, False, True, False, False])
        self.assertTrue(F.catalog_location_mask([41.], [-165.], 1, "quake")[0])
        self.assertFalse(F.catalog_location_mask([41.], [-165.], 500, "quake")[0])

    def test_quake_features_do_not_encode_unfetched_regions_as_zero(self):
        cat = pd.DataFrame({"time": pd.to_datetime(["2010-06-01", "2015-01-01"], utc=True),
                            "latitude": [40., 50.], "longitude": [-105., 0.], "mag": [3., 3.]})
        with patch.object(F.C, "quakes", return_value=cat):
            result = F.quake_features(F.to_sec(pd.to_datetime(["2010-06-01"] * 2, utc=True)),
                                      np.array([40., 80.]), np.array([-105., -105.]))
        self.assertEqual(result.n_eq_pre1d_100km.iloc[0], 1.)
        self.assertTrue(result.iloc[1].isna().all())

    def test_storm_geography_requires_authoritative_complete_circle(self):
        import shapely
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Placeholder is never read: the synthetic boundary supplies all
            # geometry for this test; no real geographic/source data are used.
            path = base / "population" / "cb_2020_us_state_20m.zip"
            path.parent.mkdir()
            path.touch()
            boundary = shapely.box(-124., 25., -66., 49.)
            with patch.object(F, "RAW", base), patch.object(F, "_storm_us_boundary", return_value=boundary):
                result = F.catalog_location_mask([40., 40., 40., 60.], [-105., -66.1, -105., -105.],
                                                   100, "storm", countries=["US", "US", "CA", "US"])
                np.testing.assert_array_equal(result, [True, False, False, False])
                self.assertTrue(F.catalog_location_mask([40.], [-105.], 0, "storm", countries=["USA"])[0])
            path.unlink()
            with patch.object(F, "RAW", base):
                self.assertFalse(F.catalog_location_mask([40.], [-105.], 100, "storm", countries=["US"])[0])

    def test_shower_dedup_preserves_values_and_repeated_row_order(self):
        dates = pd.to_datetime(["2000-01-01T00:00Z", "2000-01-01T00:00Z", "2000-01-01T01:00Z"], utc=True)
        seen = []
        def astronomy(t, lat, lon):
            return pd.DataFrame({"sun_alt": np.zeros(len(t))})
        def showers(t):
            seen.append(len(t))
            return pd.DataFrame({"shower_zhr_total": np.arange(len(t), dtype=float)})
        with patch.object(F.C, "astro_features", side_effect=astronomy), patch.object(F.C, "shower_activity", side_effect=showers):
            result = F.astro_calendar_features(F.to_sec(dates), np.zeros(3), np.zeros(3), ["UTC"] * 3)
        self.assertEqual(seen, [2])
        np.testing.assert_array_equal(result.shower_zhr_total, [0., 0., 1.])

    def test_spatial_missing_and_window_specific_coverage(self):
        cat = pd.DataFrame({"time": pd.to_datetime(["2000-01-01T01:00Z"], utc=True), "lat": [0.], "lon": [0.]})
        t = F.to_sec(pd.to_datetime(["2000-01-01T01:00Z"] * 3, utc=True))
        result = F.window_counts(t, np.array([0., np.nan, 0.]), np.array([0., 0., np.nan]), cat,
                                 {"short": (-60., 60.), "long": (-86400., 86400.)}, [100],
                                 coverage=("2000-01-01", "2000-01-03"))
        np.testing.assert_allclose(result.n_short_100km, [1., np.nan, np.nan], equal_nan=True)
        self.assertTrue(result.n_long_100km.isna().all())

    def test_unknown_empty_catalog_is_unavailable(self):
        cat = pd.DataFrame({"time": pd.to_datetime([], utc=True), "lat": [], "lon": []})
        result = F.window_counts([0.], [0.], [0.], cat, {"w": (-60., 60.)}, [100], coverage=(pd.NaT, pd.NaT))
        self.assertTrue(result.isna().all().all())

    def test_launch_subtypes_and_absent_starlink(self):
        launches = pd.DataFrame({"time": pd.to_datetime(["2000-01-01"], utc=True), "lat": [0.], "lon": [0.],
                                 "time_has_clock": [True], "orbital": [True], "is_starlink": [False]})
        with patch.object(F.C, "launches", return_value=launches):
            result = F.launch_features(F.to_sec(pd.to_datetime(["1940-01-01", "2005-01-01"], utc=True)),
                                       np.zeros(2), np.zeros(2))
        self.assertTrue(result.filter(regex="^n_launch").isna().all().all())
        self.assertTrue(result.days_since_starlink_launch.isna().all())

    def test_space_weather_never_clamps_outside_catalog(self):
        times = pd.to_datetime(["2000-01-01T00:00Z", "2000-01-01T03:00Z"], utc=True)
        kp = pd.DataFrame({"time": times, "kp": [1., 2.]})
        omni = pd.DataFrame({"time": times})
        for name in ("dst", "ae", "bz_gsm", "v", "np", "pdyn", "pflux10", "efield"):
            omni[name] = [1., 2.]
        daily = pd.DataFrame({"date": pd.to_datetime(["2000-01-01"]), "F107": [100.], "SN": [20.], "Ap": [2.]})
        flares = pd.DataFrame({"time": times, "cls": ["M", "X"], "mag": [1., 1.]})
        with patch.object(F.C, "kp3h", return_value=kp), patch.object(F.C, "omni_hourly", return_value=omni), \
             patch.object(F.C, "sw_daily", return_value=daily), patch.object(F.C, "flares", return_value=flares):
            result = F.space_weather_features(F.to_sec(pd.to_datetime(["1900-01-01", "2020-01-01"], utc=True)))
        self.assertTrue(result[["kp", "F107", "SN", "Ap", "dst"]].isna().all().all())

    def test_weather_source_missing_not_overcast_or_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            target = base / "isd_lite" / "2015" / "000001-00001-2015.gz"
            target.parent.mkdir(parents=True)
            with gzip.open(target, "wt") as handle:
                for hour, sky in [(0, 0), (1, -9999), (3, 9), (4, 10)]:
                    handle.write(f"2015 1 1 {hour} 100 50 10100 180 10 {sky} 0 0\n")
            with patch.object(weather, "W", base):
                result = weather.load_station("000001-00001", years=[2015])
        self.assertEqual(result.wx_sky_oktas.iloc[0], 0.)
        self.assertTrue(np.isnan(result.wx_sky_oktas.iloc[1]))
        self.assertTrue(np.isnan(result.wx_inversion_proxy.iloc[1]))
        np.testing.assert_array_equal(result.wx_sky_oktas.iloc[2:], [8., 8.])
        self.assertNotIn(pd.Timestamp("2015-01-01T02:00Z"), result.index)

    def test_weather_nearest_observation_age_and_empty_schema(self):
        observations = pd.DataFrame({"wx_sky_oktas": [np.nan, 0.]}, index=pd.to_datetime(["2015-01-01T00:00Z", "2015-01-01T03:00Z"]))
        dates = pd.to_datetime(["2015-01-01T00:10Z", "2015-01-01T02:50Z", "2015-01-01T10:00Z"], utc=True)
        with patch.object(weather, "assign_station", return_value=(np.array(["X"] * 3), np.zeros(3))), \
             patch.object(weather, "load_station", return_value=observations):
            result = weather.weather_features(F.to_sec(dates), np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(result.wx_observation_age_min, [10., -10., np.nan], equal_nan=True)
        self.assertTrue(np.isnan(result.wx_sky_oktas.iloc[0]))
        self.assertEqual(result.wx_sky_oktas.iloc[1], 0.)
        self.assertTrue(np.isnan(result.wx_sky_oktas.iloc[2]))
        with patch.object(weather, "assign_station", return_value=(np.array([None]), np.array([100.]))):
            empty = weather.weather_features([0.], [0.], [0.])
        self.assertTrue(set(weather.WEATHER_COLUMNS) <= set(empty.columns))
        self.assertTrue(empty[list(weather.WEATHER_COLUMNS)].isna().all().all())

    def test_discovery_geographic_and_source_isolation_without_resplitting(self):
        events = pd.DataFrame({"SOURCE": ["NUFORC", "NUFORC", "NUFORC", "GEIPAN", "HATCH", "NUFORC", "NUFORC"],
                               "COUNTRY": ["US", "Canada", "GB", "FR", "US", "US", "US"],
                               "SPLIT": ["discovery"] * 5 + ["holdout_temporal", "discovery"],
                               "GEO_REGION": ["north_america", "north_america", "europe_other", "europe_fr", "north_america", "north_america", "europe_other"]})
        before = events.copy(deep=True)
        np.testing.assert_array_equal(cohorts.discovery_mask(events), [True, True, False, False, False, False, False])
        pd.testing.assert_frame_equal(events, before)

    def test_dst_clock_and_weekday_preserved(self):
        events = pd.DataFrame({"EVENT_ID": ["A", "B"], "utc_ts": pd.to_datetime(["2015-03-15T04:30Z", "2015-11-08T05:30Z"], utc=True),
                               "TIMEZONE": "America/New_York", "LATITUDE": 40., "LONGITUDE": -74.})
        for function in (A.cs1_referents, A.cs2_referents):
            controls = function(events)
            result = A.check_referents(events, controls)
            self.assertEqual(result["clock_mismatch"], 0)
            self.assertEqual(result["weekday_mismatch"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
