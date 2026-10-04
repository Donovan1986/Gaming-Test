"""Acquire environmental, astronomical, geophysical, infrastructure and
reporting-bias covariates. All downloads logged in data/manifest.jsonl.

Station-specific weather (ISD-Lite) and radiosonde (IGRA2-derived) files are
fetched by acquire_weather.py after event locations are known.
"""
import json
import sys
import time

import requests

from common import RAW, fetch, log_manifest, _now, UA

SW = RAW / "spaceweather"
GEO = RAW / "geophysical"
SPACE = RAW / "space"
POP = RAW / "population"
INFRA = RAW / "infrastructure"
MEDIA = RAW / "media"
ASTRO = RAW / "astro"


def space_weather():
    fetch("https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt",
          SW / "Kp_ap_Ap_SN_F107_since_1932.txt", "GFZ Kp/ap/Ap/SN/F10.7 daily+3h since 1932")
    fetch("https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat",
          SW / "omni2_all_years.dat", "NASA OMNI2 hourly (IMF, solar wind, Kp, Dst, AE) 1963-", timeout=900)
    fetch("https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text",
          SW / "omni2_format.txt", "OMNI2 format description")
    # GOES X-ray flare event lists (NGDC). Yearly files.
    base = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/"
    r = requests.get(base, timeout=60, headers={"User-Agent": UA})
    import re
    files = sorted(set(re.findall(r'href="(goes-xrs-report_\d{4}[^"]*\.txt)"', r.text)))
    for f in files:
        fetch(base + f, SW / "goes_xrs" / f, "NGDC GOES XRS flare report " + f)


def quakes():
    """USGS ComCat, M>=2.5, North America box and Western Europe box, yearly."""
    boxes = {"na": dict(minlatitude=15, maxlatitude=72, minlongitude=-170, maxlongitude=-50),
             "weu": dict(minlatitude=35, maxlatitude=60, minlongitude=-12, maxlongitude=20)}
    for name, box in boxes.items():
        for y in range(1940, 2025):
            for half, (a, b) in enumerate([("01-01", "07-01"), ("07-01", "01-01")]):
                y2 = y if half == 0 else y + 1
                params = dict(format="csv", starttime=f"{y}-{a}", endtime=f"{y2}-{b}",
                              minmagnitude=2.5, orderby="time-asc", **box)
                fetch("https://earthquake.usgs.gov/fdsnws/event/1/query",
                      GEO / "usgs" / f"{name}_{y}_{half}.csv",
                      f"USGS ComCat M>=2.5 {name} {y} H{half+1}", params=params,
                      sleep_between=0.3)


def space():
    fetch("https://planet4589.org/space/gcat/tsv/launch/launch.tsv", SPACE / "gcat_launch.tsv",
          "GCAT (McDowell) launch log incl. suborbital")
    fetch("https://planet4589.org/space/gcat/tsv/cat/satcat.tsv", SPACE / "gcat_satcat.tsv",
          "GCAT satellite catalog (launch/decay dates)")
    fetch("https://planet4589.org/space/gcat/tsv/tables/sites.tsv", SPACE / "gcat_sites.tsv",
          "GCAT launch sites with coordinates")
    fetch("https://planet4589.org/space/gcat/tsv/tables/platforms.tsv", SPACE / "gcat_platforms.tsv",
          "GCAT mobile launch platforms")
    fetch("https://ssd-api.jpl.nasa.gov/fireball.api", SPACE / "cneos_fireballs.json",
          "JPL CNEOS fireball (bolide) API, all records")
    fetch("https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp", ASTRO / "de421.bsp",
          "JPL DE421 ephemeris")
    fetch("https://raw.githubusercontent.com/skyfielders/python-skyfield/master/skyfield/data/hip_main.dat",
          ASTRO / "hip_main_probe.dat", "(probe) Hipparcos via skyfield repo")


def population():
    fetch("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip",
          POP / "2020_Gaz_counties_national.zip", "Census 2020 county gazetteer (centroids, land area)")
    fetch("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_place_national.zip",
          POP / "2020_Gaz_place_national.zip", "Census 2020 places gazetteer")
    fetch("https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv",
          POP / "co-est2020-alldata.csv", "Census county population estimates 2010-2020")
    fetch("https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv",
          POP / "co-est00int-tot.csv", "Census county intercensal estimates 2000-2010")
    fetch("https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv",
          POP / "co-est2023-alldata.csv", "Census county population estimates 2020-2023")
    fetch("https://www2.census.gov/programs-surveys/popest/tables/1990-2000/counties/totals/99c8_00.txt",
          POP / "county_1990s_99c8_00.txt", "Census county estimates 1990-1999")
    fetch("https://download.geonames.org/export/dump/cities1000.zip", POP / "geonames_cities1000.zip",
          "GeoNames cities with pop >= 1000")
    fetch("https://download.geonames.org/export/dump/admin1CodesASCII.txt", POP / "geonames_admin1.txt",
          "GeoNames admin1 codes")
    for c in ("USA", "FRA", "CAN", "GBR", "WLD"):
        fetch(f"https://api.worldbank.org/v2/country/{c}/indicator/IT.NET.USER.ZS",
              POP / f"worldbank_internet_{c}.json", f"World Bank internet users % {c}",
              params={"format": "json", "per_page": 200})
        fetch(f"https://api.worldbank.org/v2/country/{c}/indicator/SP.POP.TOTL",
              POP / f"worldbank_pop_{c}.json", f"World Bank population {c}",
              params={"format": "json", "per_page": 200})


def infrastructure():
    fetch("https://davidmegginson.github.io/ourairports-data/airports.csv", INFRA / "ourairports_airports.csv",
          "OurAirports airports")
    fetch("https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip", INFRA / "ne_50m_coastline.zip",
          "Natural Earth 1:50m coastline")
    fetch("https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_urban_areas.zip", INFRA / "ne_10m_urban_areas.zip",
          "Natural Earth urban areas")


def media():
    # Wikipedia daily pageviews (aggregate attention proxy; available from 2015-07)
    for art in ("Unidentified_flying_object", "Unidentified_anomalous_phenomena",
                "National_UFO_Reporting_Center", "Starlink"):
        fetch(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{art}/daily/20150701/20240101",
              MEDIA / f"wiki_pageviews_{art}.json", f"Wikipedia pageviews {art}")




def _arcgis_all(layer_url: str, dest, desc: str, page: int = 1000):
    """Page through an ArcGIS FeatureServer layer, saving GeoJSON pages."""
    offset, n = 0, 0
    while True:
        p = fetch(layer_url + "/query", dest.parent / f"{dest.stem}_p{n:03d}.geojson",
                  f"{desc} (offset {offset})",
                  params={"where": "1=1", "outFields": "*", "f": "geojson",
                          "resultOffset": offset, "resultRecordCount": page,
                          "outSR": 4326})
        if p is None:
            break
        feats = json.load(open(p)).get("features", [])
        if len(feats) < page:
            break
        offset += page
        n += 1


def facilities():
    _arcgis_all("https://services7.arcgis.com/n1YM8pTrFmm7L4hs/arcgis/rest/services/mirta/FeatureServer/0",
                INFRA / "mirta_points.geojson", "DoD MIRTA sites (points)")
    _arcgis_all("https://services7.arcgis.com/n1YM8pTrFmm7L4hs/arcgis/rest/services/mirta/FeatureServer/1",
                INFRA / "mirta_boundaries.geojson", "DoD MIRTA sites (boundaries)", page=200)
    _arcgis_all("https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/Special_Use_Airspace/FeatureServer/0",
                INFRA / "faa_sua.geojson", "FAA Special Use Airspace polygons", page=500)
    _arcgis_all("https://gis.fema.gov/arcgis/rest/services/Partner/Operating_Nuclear_Power_Plant_Sites/FeatureServer/0",
                INFRA / "fema_nuclear_plants.geojson", "FEMA operating nuclear power plant sites")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["space_weather", "space", "population", "infrastructure", "media", "quakes"]
    for s in steps:
        print("==", s, flush=True)
        globals()[s]()
