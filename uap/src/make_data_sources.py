"""Summarise data/manifest.jsonl into report/DATA_SOURCES.md."""
import json
import re
from collections import defaultdict

from common import ROOT, REPORT

GROUPS = [
    ("NUFORC (TidyTuesday 2023 release)", r"TidyTuesday|tt2023|NUFORC place|NUFORC day-part"),
    ("NUFORC (planetsig scrape to 2014)", r"planetsig"),
    ("GEIPAN (CNES official export)", r"GEIPAN"),
    ("Larry Hatch *U* database (via R. Geldreich ufo_data)", r"Hatch"),
    ("NICAP chronology (via R. Geldreich ufo_data)", r"NICAP"),
    ("Blue Book unknowns (Berliner / FUFOR)", r"Blue Book|Berliner|BlueBook"),
    ("GFZ Potsdam Kp/ap/Ap/SN/F10.7", r"GFZ"),
    ("NASA OMNI2 hourly (IMF, solar wind, Kp, Dst, AE, protons)", r"OMNI"),
    ("NOAA NGDC GOES XRS flare lists", r"GOES XRS"),
    ("USGS ComCat earthquakes", r"USGS ComCat"),
    ("JPL CNEOS fireballs", r"CNEOS"),
    ("GCAT launch / satellite / site catalogs (J. McDowell)", r"GCAT"),
    ("ISS historical TLEs (J. McDowell archive)", r"ISS \(25544\)"),
    ("JPL DE421 ephemeris", r"DE421"),
    ("NOAA ISD-Lite hourly surface weather", r"ISD-Lite|ISD station"),
    ("NOAA Storm Events database", r"Storm Events"),
    ("US Census population estimates / gazetteers / boundaries", r"Census"),
    ("GeoNames", r"GeoNames"),
    ("World Bank indicators (population, internet users)", r"World Bank"),
    ("OurAirports", r"OurAirports"),
    ("DoD MIRTA installations", r"MIRTA"),
    ("FAA Special Use Airspace", r"Special Use Airspace"),
    ("FEMA/EIA operating nuclear power plants", r"nuclear power plant"),
    ("Natural Earth", r"Natural Earth"),
    ("Wikimedia pageviews", r"Wikipedia pageviews"),
    ("USGS geomagnetism 1-min data (convergence checks)", r"USGS geomag"),
]

NOTES = {
    "NUFORC (TidyTuesday 2023 release)": "NUFORC terms forbid redistribution; raw text and per-record data are NOT committed.",
    "Larry Hatch *U* database (via R. Geldreich ufo_data)": "Third-party compilation; not redistributed. Hatch 'Strangeness' never used.",
    "GEIPAN (CNES official export)": "Testimony export (/fr/cnes/export/temoignages) repeatedly HTTP 429; case export used.",
}


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "manifest.jsonl")]
    agg = defaultdict(lambda: dict(n=0, bytes=0, urls=set(), dates=set(), fail=0))
    blocked = []
    for r in rows:
        d = r.get("description", "")
        if d.startswith("ACCESS BLOCKED"):
            blocked.append(r)
            continue
        for name, pat in GROUPS:
            if re.search(pat, d):
                g = agg[name]
                if r.get("ok"):
                    g["n"] += 1
                    g["bytes"] += r.get("bytes", 0) or 0
                    u = re.sub(r"/\d{4}/[^/]+$", "/<year>/<file>", r["url"].split("?")[0])
                    g["urls"].add(u if len(g["urls"]) < 3 else "...")
                    g["dates"].add(r["retrieved_utc"][:10])
                else:
                    g["fail"] += 1
                break
    out = ["# Data sources, retrieval and provenance", "",
           "All retrievals are logged one-per-line in `uap/data/manifest.jsonl` (URL incl. query, UTC retrieval",
           "timestamp, HTTP status, bytes, SHA-256). Raw files are not committed (licence/size); re-acquire with",
           "`uap/src/acquire_uap.py`, `acquire_env.py`, `acquire_weather.py`, `iss.py` and verify hashes.", "",
           "| Source | files OK | MB | failed/404 | retrieval date(s) | URL pattern(s) | notes |",
           "|---|---|---|---|---|---|---|"]
    for name, _ in GROUPS:
        g = agg.get(name)
        if not g:
            continue
        out.append(f"| {name} | {g['n']} | {g['bytes'] / 1e6:.1f} | {g['fail']} | {', '.join(sorted(g['dates']))} | "
                   f"{'<br>'.join(sorted(g['urls']))} | {NOTES.get(name, '')} |")
    out += ["", "## Sources attempted but not accessible from this environment", ""]
    for r in blocked:
        out.append(f"* `{r['url']}` - {r['description'].replace('ACCESS BLOCKED: ', '')} (attempted {r['retrieved_utc'][:10]})")
    out += ["* GEIPAN testimony export `https://www.cnes-geipan.fr/fr/cnes/export/temoignages` - HTTP 429 on every attempt.",
            "* AARO case resolutions, FAA pilot UAP reports, NARCAP pilot catalogue, UK MoD files, NASA UAP study data: "
            "no machine-readable public table reachable; PDFs only or blocked. Not used.",
            "* OpenSky / ADS-B historical tracks: require registered account (Trino); not used -> aircraft presence is "
            "proxied by airport proximity (DATA_UNAVAILABLE for actual traffic).",
            "* NEXRAD, lightning networks (NLDN/GLM), sprites/TLE catalogs, infrasound (IMS), ionospheric TEC, Schumann "
            "resonance, NOTAM/TFR/military-exercise schedules, smartphone adoption by county: not obtained at scale; "
            "all treated as DATA_UNAVAILABLE (never as zero). Storm Events (thunderstorm wind / hail / tornado / lightning "
            "damage reports) is used as a convective-activity proxy; an inversion PROXY (clear sky + wind <= 2 m/s at night) "
            "replaces radiosonde inversions (IGRA2 derived files are ~155 MB per station).",
            "", "## Curated tables (hand-assembled from public sources)", "",
            "* `uap/src/curated_sites.py`: ICBM wings and operating periods (Malmstrom, Minot, F.E. Warren, Ellsworth, "
            "Whiteman, Grand Forks, Titan II wings), DOE weapons-complex sites, nuclear test sites, permanently shut-down "
            "US power reactors. Coordinates are geocoded through the same gazetteer as UAP reports, except 8 entries "
            "hand-entered (flagged). ICBM fields are coarse disks (+-30 km); analyses using them are run at >= 25 km.",
            "* `uap/src/covariates.py`: IMO meteor-shower parameters (solar longitude of maximum, ZHR, width) and documented "
            "outbursts; list of major UAP media events with dates.", ""]
    (REPORT / "DATA_SOURCES.md").write_text("\n".join(out))
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
