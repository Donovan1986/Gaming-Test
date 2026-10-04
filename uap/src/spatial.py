"""Distances from points to infrastructure / geography layers.

Point layers -> BallTree (haversine) nearest distance and counts within radii.
Polygon layers (SUA, MIRTA boundaries, coastline) -> shapely STRtree in an
equal-area projection (EPSG:5070 for CONUS; EPSG:3035 for Europe).
Time-varying layers (ICBM fields, closed reactors, DOE sites) take a year
argument and only count sites operating in that year.
"""
from __future__ import annotations

import glob
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from common import RAW, INTERIM

R_EARTH = 6371.0088


@lru_cache(maxsize=1)
def airports():
    a = pd.read_csv(RAW / "infrastructure" / "ourairports_airports.csv",
                    usecols=["ident", "type", "name", "latitude_deg", "longitude_deg", "iso_country", "scheduled_service"])
    a = a[a.type.isin(["large_airport", "medium_airport", "small_airport", "heliport"])]
    return a.rename(columns={"latitude_deg": "lat", "longitude_deg": "lon"})


@lru_cache(maxsize=1)
def mirta_points():
    rows = []
    for f in sorted(glob.glob(str(RAW / "infrastructure" / "mirta_points_p*.geojson"))):
        for ft in json.load(open(f))["features"]:
            p = ft["properties"]
            lon, lat = ft["geometry"]["coordinates"][:2]
            rows.append((p.get("SITENAME"), (p.get("SITEREPORTINGCOMPONENT") or "").lower(), lat, lon,
                         (p.get("SITEOPERATIONALSTATUS") or "").lower()))
    return pd.DataFrame(rows, columns=["name", "component", "lat", "lon", "status"])


@lru_cache(maxsize=1)
def nuclear_operating():
    d = json.load(open(RAW / "infrastructure" / "fema_nuclear_plants_p000.geojson"))
    rows = [(f["properties"]["plant_name"], f["properties"]["latitude"], f["properties"]["longitude"]) for f in d["features"]]
    return pd.DataFrame(rows, columns=["name", "lat", "lon"])


@lru_cache(maxsize=1)
def curated():
    p = INTERIM / "curated_sites.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import curated_sites
    d = curated_sites.build()
    d.to_parquet(p, index=False)
    return d


@lru_cache(maxsize=1)
def launch_sites_us():
    import covariates as C
    L = C.launches()
    s = L[L.StateCode == "US"].dropna(subset=["lat"]).groupby("site").agg(lat=("lat", "first"), lon=("lon", "first"),
                                                                         n=("Launch_Tag", "size"))
    return s[s.n >= 5].reset_index()


def _tree(lat, lon):
    return BallTree(np.radians(np.c_[lat, lon]), metric="haversine")


def nearest_km(points_lat, points_lon, layer_lat, layer_lon):
    ok = np.isfinite(layer_lat) & np.isfinite(layer_lon)
    t = _tree(layer_lat[ok], layer_lon[ok])
    d, _ = t.query(np.radians(np.c_[points_lat, points_lon]), k=1)
    return d[:, 0] * R_EARTH


def count_within(points_lat, points_lon, layer_lat, layer_lon, radius_km):
    ok = np.isfinite(layer_lat) & np.isfinite(layer_lon)
    t = _tree(layer_lat[ok], layer_lon[ok])
    return t.query_radius(np.radians(np.c_[points_lat, points_lon]), r=radius_km / R_EARTH, count_only=True)


@lru_cache(maxsize=1)
def _polys():
    import geopandas as gpd
    out = {}
    sua = pd.concat([gpd.read_file(f) for f in sorted(glob.glob(str(RAW / "infrastructure" / "faa_sua_p*.geojson")))])
    sua = sua.set_crs(4326, allow_override=True).to_crs(5070)
    for code, name in [("MOA", "sua_moa"), ("R", "sua_restricted"), ("P", "sua_prohibited"), ("A", "sua_alert")]:
        out[name] = sua[sua.TYPE_CODE == code].geometry.make_valid().values
    mb = pd.concat([gpd.read_file(f) for f in sorted(glob.glob(str(RAW / "infrastructure" / "mirta_boundaries_p*.geojson")))])
    mb = mb.set_crs(4326, allow_override=True).to_crs(5070)
    out["dod_boundary"] = mb.geometry.make_valid().values
    import zipfile, tempfile, os
    coast = gpd.read_file("zip://" + str(RAW / "infrastructure" / "ne_50m_coastline.zip"))
    out["coast"] = coast.set_crs(4326, allow_override=True).to_crs(5070).geometry.values
    return out


def polygon_distance_km(points_lat, points_lon, layer: str, crs=5070):
    """Distance (km) to nearest polygon of `layer`; 0 if inside."""
    import geopandas as gpd
    from shapely import STRtree
    geoms = _polys()[layer]
    tree = STRtree(geoms)
    pts = gpd.GeoSeries(gpd.points_from_xy(points_lon, points_lat), crs=4326).to_crs(crs).values
    idx = tree.query_nearest(pts, all_matches=False, return_distance=True)
    (ip, ig), dist = idx[0], idx[1]
    out = np.full(len(pts), np.nan)
    out[ip] = dist / 1000.0
    return out


def time_varying_count(points_lat, points_lon, years, layer: str, radius_km: float, use_site_radius=False):
    """Number of curated sites of `layer` operating in each point's year within radius_km
    (or within the site's own field radius + radius_km if use_site_radius)."""
    c = curated()
    c = c[(c.layer == layer)].dropna(subset=["lat"])
    out = np.zeros(len(points_lat), dtype=int)
    years = np.asarray(years)
    for _, s in c.iterrows():
        active = (years >= s.start) & ((years <= s.end) if pd.notna(s.end) else True)
        if not active.any():
            continue
        from common import haversine_km
        d = haversine_km(points_lat, points_lon, s.lat, s.lon)
        lim = radius_km + (s.radius_km if use_site_radius else 0)
        out += (active & (d <= lim)).astype(int)
    return out


def time_varying_nearest(points_lat, points_lon, years, layer: str):
    from common import haversine_km
    c = curated()
    c = c[(c.layer == layer)].dropna(subset=["lat"])
    best = np.full(len(points_lat), np.inf)
    years = np.asarray(years)
    for _, s in c.iterrows():
        active = (years >= s.start) & ((years <= s.end) if pd.notna(s.end) else True)
        d = haversine_km(points_lat, points_lon, s.lat, s.lon)
        best = np.where(active & (d < best), d, best)
    return best


def spatial_features(lat, lon, year) -> pd.DataFrame:
    """Static + time-varying spatial covariates for CONUS-like points."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    year = np.asarray(year)
    A = airports()
    us_ap = A[A.iso_country.isin(["US", "CA", "MX"])]
    out = {}
    for t in ["large_airport", "medium_airport", "small_airport"]:
        s = us_ap[us_ap.type == t]
        out[f"dist_{t}_km"] = nearest_km(lat, lon, s.lat.values, s.lon.values)
    big = us_ap[us_ap.type.isin(["large_airport", "medium_airport"])]
    out["n_airports_lm_25km"] = count_within(lat, lon, big.lat.values, big.lon.values, 25)
    M = mirta_points()
    out["dist_dod_site_km"] = nearest_km(lat, lon, M.lat.values, M.lon.values)
    out["n_dod_sites_50km"] = count_within(lat, lon, M.lat.values, M.lon.values, 50)
    af = M[M.component.isin(["usaf", "usn", "usmc", "ang", "afr", "usnr"])]
    out["dist_dod_air_naval_km"] = nearest_km(lat, lon, af.lat.values, af.lon.values)
    N = nuclear_operating()
    out["dist_nuclear_operating_km"] = nearest_km(lat, lon, N.lat.values, N.lon.values)
    out["dist_nuclear_closed_active_km"] = time_varying_nearest(lat, lon, year, "nuclear_plant_closed")
    out["dist_nuclear_any_active_km"] = np.minimum(out["dist_nuclear_operating_km"], out["dist_nuclear_closed_active_km"])
    out["dist_icbm_field_active_km"] = time_varying_nearest(lat, lon, year, "icbm_field")
    out["in_icbm_field_active"] = time_varying_count(lat, lon, year, "icbm_field", 0, use_site_radius=True)
    out["dist_doe_weapons_active_km"] = time_varying_nearest(lat, lon, year, "doe_weapons_site")
    out["dist_nuclear_test_site_km"] = time_varying_nearest(lat, lon, year, "nuclear_test_site")
    LS = launch_sites_us()
    out["dist_launch_site_km"] = nearest_km(lat, lon, LS.lat.values, LS.lon.values)
    for layer in ["sua_moa", "sua_restricted", "sua_alert", "sua_prohibited", "dod_boundary", "coast"]:
        out[f"dist_{layer}_km"] = polygon_distance_km(lat, lon, layer)
    return pd.DataFrame(out)
