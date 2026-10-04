"""Acquire UAP / anomalous-observation report datasets.

Sources actually reachable from this environment on the retrieval date.
Sources that were attempted but blocked are recorded in the manifest with
ok=false and documented in report/DATA_SOURCES.md.
"""
from common import RAW, fetch, log_manifest, _now

NUFORC = RAW / "nuforc"
OTHER = RAW / "catalogs"

TARGETS = [
    # NUFORC, cleaned + enriched by Jon Harmon for R4DS TidyTuesday 2023-06-20
    ("https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-06-20/ufo_sightings.csv",
     NUFORC / "tt2023_ufo_sightings.csv", "NUFORC reports (TidyTuesday 2023 release)"),
    ("https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-06-20/places.csv",
     NUFORC / "tt2023_places.csv", "NUFORC place gazetteer (geonames lat/lon, pop, tz)"),
    ("https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-06-20/day_parts_map.csv",
     NUFORC / "tt2023_day_parts_map.csv", "NUFORC day-part map (sunrise-sunset.org)"),
    # Independent older NUFORC scrape (to 2014) with full comment text + lat/lon
    ("https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-scrubbed-geocoded-time-standardized.csv",
     NUFORC / "planetsig_scrubbed.csv", "NUFORC scrape to 2014 (planetsig)"),
    # Historical catalogs compiled by R. Geldreich (Apache-2 tool; data credited)
    ("https://raw.githubusercontent.com/richgel999/ufo_data/main/bin/hatch_udb.json",
     OTHER / "hatch_udb.json", "Larry Hatch *U* database (~18k cases, credibility/strangeness)"),
    ("https://raw.githubusercontent.com/richgel999/ufo_data/main/bin/nicap_db.json",
     OTHER / "nicap_db.json", "NICAP case chronology"),
    ("https://raw.githubusercontent.com/richgel999/ufo_data/main/bin/bb_unknowns.json",
     OTHER / "bb_unknowns_geldreich.json", "Blue Book unknowns (Geldreich JSON)"),
]
for i in range(1, 7):
    TARGETS.append((f"https://mirror.cyberbits.eu/textfiles.com/ufo/bluebuk{i}",
                    OTHER / f"berliner_bluebook_unknowns_pt{i}.txt",
                    f"Berliner 'Blue Book Unknowns' (FUFOR) usenet repost part {i}"))

BLOCKED = [
    ("https://nuforc.org/subndx/?id=all", "NUFORC primary site: Cloudflare bot challenge (HTTP 403, cf-mitigated)."),
    ("https://www.aaro.mil/UAP-Cases/", "AARO case pages: Akamai HTTP 403 to non-browser clients."),
    ("https://huggingface.co/datasets/kcimc/NUFORC", "kcimc NUFORC scrape: HTTP 401 (gated / withdrawn)."),
    ("https://www.bluebookarchive.org", "Blue Book archive site: CONNECT refused by gateway (502)."),
]

if __name__ == "__main__":
    for url, dest, desc in TARGETS:
        p = fetch(url, dest, desc)
        print(("OK  " if p else "FAIL"), dest.name)
    for url, why in BLOCKED:
        log_manifest({"url": url, "ok": False, "retrieved_utc": _now(),
                      "description": "ACCESS BLOCKED: " + why})
