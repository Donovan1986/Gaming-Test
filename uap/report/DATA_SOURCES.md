# Data sources, retrieval and provenance

All retrievals are logged one-per-line in `uap/data/manifest.jsonl` (URL incl. query, UTC retrieval
timestamp, HTTP status, bytes, SHA-256). Raw files are not committed (licence/size); re-acquire with
`uap/src/acquire_uap.py`, `acquire_env.py`, `acquire_weather.py`, `iss.py` and verify hashes.

| Source | files OK | MB | failed/404 | retrieval date(s) | URL pattern(s) | notes |
|---|---|---|---|---|---|---|
| NUFORC (TidyTuesday 2023 release) | 3 | 23.8 | 0 | 2026-10-04 | https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-06-20/day_parts_map.csv<br>https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-06-20/places.csv<br>https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-06-20/ufo_sightings.csv | NUFORC terms forbid redistribution; raw text and per-record data are NOT committed. |
| NUFORC (planetsig scrape to 2014) | 1 | 13.9 | 0 | 2026-10-04 | https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-scrubbed-geocoded-time-standardized.csv |  |
| GEIPAN (CNES official export) | 1 | 1.8 | 2 | 2026-10-04 | https://www.cnes-geipan.fr/fr/cnes/export/cas | Testimony export (/fr/cnes/export/temoignages) repeatedly HTTP 429; case export used. |
| Larry Hatch *U* database (via R. Geldreich ufo_data) | 1 | 24.9 | 0 | 2026-10-04 | https://raw.githubusercontent.com/richgel999/ufo_data/main/bin/hatch_udb.json | Third-party compilation; not redistributed. Hatch 'Strangeness' never used. |
| NICAP chronology (via R. Geldreich ufo_data) | 1 | 2.6 | 0 | 2026-10-04 | https://raw.githubusercontent.com/richgel999/ufo_data/main/bin/nicap_db.json |  |
| Blue Book unknowns (Berliner / FUFOR) | 7 | 0.5 | 0 | 2026-10-04 | ...<br>https://mirror.cyberbits.eu/textfiles.com/ufo/bluebuk1<br>https://mirror.cyberbits.eu/textfiles.com/ufo/bluebuk2<br>https://raw.githubusercontent.com/richgel999/ufo_data/main/bin/bb_unknowns.json |  |
| GFZ Potsdam Kp/ap/Ap/SN/F10.7 | 1 | 5.5 | 0 | 2026-10-04 | https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt |  |
| NASA OMNI2 hourly (IMF, solar wind, Kp, Dst, AE, protons) | 2 | 184.0 | 0 | 2026-10-04 | https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text<br>https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat |  |
| NOAA NGDC GOES XRS flare lists | 45 | 7.8 | 0 | 2026-10-04 | ...<br>https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_1975.txt<br>https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_1976.txt<br>https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_1977.txt |  |
| USGS ComCat earthquakes | 340 | 73.9 | 0 | 2026-10-04 | https://earthquake.usgs.gov/fdsnws/event/1/query |  |
| JPL CNEOS fireballs | 1 | 0.1 | 0 | 2026-10-04 | https://ssd-api.jpl.nasa.gov/fireball.api |  |
| GCAT launch / satellite / site catalogs (J. McDowell) | 4 | 33.3 | 0 | 2026-10-04 | ...<br>https://planet4589.org/space/gcat/tsv/cat/satcat.tsv<br>https://planet4589.org/space/gcat/tsv/launch/launch.tsv<br>https://planet4589.org/space/gcat/tsv/tables/sites.tsv |  |
| ISS historical TLEs (J. McDowell archive) | 1 | 1.3 | 0 | 2026-10-04 | https://planet4589.org/space/elements/25500/S25544 |  |
| JPL DE421 ephemeris | 1 | 16.8 | 0 | 2026-10-04 | https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp |  |
| NOAA ISD-Lite hourly surface weather | 9430 | 779.5 | 779 | 2026-10-04 | https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv<br>https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/<year>/<file> |  |
| NOAA Storm Events database | 29 | 276.6 | 0 | 2026-10-04 | ...<br>https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d1995_c20260323.csv.gz<br>https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d1996_c20260323.csv.gz<br>https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d1997_c20260323.csv.gz |  |
| US Census population estimates / gazetteers / boundaries | 8 | 13.6 | 0 | 2026-10-04 | ...<br>https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip<br>https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_place_national.zip<br>https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv |  |
| GeoNames | 2 | 11.2 | 0 | 2026-10-04 | https://download.geonames.org/export/dump/admin1CodesASCII.txt<br>https://download.geonames.org/export/dump/cities1000.zip |  |
| World Bank indicators (population, internet users) | 10 | 0.1 | 0 | 2026-10-04 | ...<br>https://api.worldbank.org/v2/country/FRA/indicator/IT.NET.USER.ZS<br>https://api.worldbank.org/v2/country/USA/indicator/IT.NET.USER.ZS<br>https://api.worldbank.org/v2/country/USA/indicator/SP.POP.TOTL |  |
| OurAirports | 1 | 12.7 | 0 | 2026-10-04 | https://davidmegginson.github.io/ourairports-data/airports.csv |  |
| DoD MIRTA installations | 6 | 24.2 | 0 | 2026-10-04 | https://services7.arcgis.com/n1YM8pTrFmm7L4hs/arcgis/rest/services/mirta/FeatureServer/0/query<br>https://services7.arcgis.com/n1YM8pTrFmm7L4hs/arcgis/rest/services/mirta/FeatureServer/1/query |  |
| FAA Special Use Airspace | 4 | 22.3 | 0 | 2026-10-04 | https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/Special_Use_Airspace/FeatureServer/0/query |  |
| FEMA/EIA operating nuclear power plants | 1 | 0.1 | 0 | 2026-10-04 | https://gis.fema.gov/arcgis/rest/services/Partner/Operating_Nuclear_Power_Plant_Sites/FeatureServer/0/query |  |
| Natural Earth | 3 | 14.4 | 0 | 2026-10-04 | https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_urban_areas.zip<br>https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip<br>https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip |  |
| Wikimedia pageviews | 4 | 1.5 | 0 | 2026-10-04 | ...<br>https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/National_UFO_Reporting_Center/daily/20150701/20240101<br>https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Unidentified_anomalous_phenomena/daily/20150701/20240101<br>https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Unidentified_flying_object/daily/20150701/20240101 |  |

## Sources attempted but not accessible from this environment

* `https://nuforc.org/subndx/?id=all` - NUFORC primary site: Cloudflare bot challenge (HTTP 403, cf-mitigated). (attempted 2026-10-04)
* `https://www.aaro.mil/UAP-Cases/` - AARO case pages: Akamai HTTP 403 to non-browser clients. (attempted 2026-10-04)
* `https://huggingface.co/datasets/kcimc/NUFORC` - kcimc NUFORC scrape: HTTP 401 (gated / withdrawn). (attempted 2026-10-04)
* `https://www.bluebookarchive.org` - Blue Book archive site: CONNECT refused by gateway (502). (attempted 2026-10-04)
* GEIPAN testimony export `https://www.cnes-geipan.fr/fr/cnes/export/temoignages` - HTTP 429 on every attempt.
* AARO case resolutions, FAA pilot UAP reports, NARCAP pilot catalogue, UK MoD files, NASA UAP study data: no machine-readable public table reachable; PDFs only or blocked. Not used.
* OpenSky / ADS-B historical tracks: require registered account (Trino); not used -> aircraft presence is proxied by airport proximity (DATA_UNAVAILABLE for actual traffic).
* NEXRAD, lightning networks (NLDN/GLM), sprites/TLE catalogs, infrasound (IMS), ionospheric TEC, Schumann resonance, NOTAM/TFR/military-exercise schedules, smartphone adoption by county: not obtained at scale; all treated as DATA_UNAVAILABLE (never as zero). Storm Events (thunderstorm wind / hail / tornado / lightning damage reports) is used as a convective-activity proxy; an inversion PROXY (clear sky + wind <= 2 m/s at night) replaces radiosonde inversions (IGRA2 derived files are ~155 MB per station).

## Curated tables (hand-assembled from public sources)

* `uap/src/curated_sites.py`: ICBM wings and operating periods (Malmstrom, Minot, F.E. Warren, Ellsworth, Whiteman, Grand Forks, Titan II wings), DOE weapons-complex sites, nuclear test sites, permanently shut-down US power reactors. Coordinates are geocoded through the same gazetteer as UAP reports, except 8 entries hand-entered (flagged). ICBM fields are coarse disks (+-30 km); analyses using them are run at >= 25 km.
* `uap/src/covariates.py`: IMO meteor-shower parameters (solar longitude of maximum, ZHR, width) and documented outbursts; list of major UAP media events with dates.
