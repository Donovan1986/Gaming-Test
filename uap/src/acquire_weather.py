"""Hourly surface weather (NOAA ISD-Lite) for stations chosen by greedy coverage
of NUFORC event locations (see selected_isd_stations.csv), 1995-2023, plus
NOAA Storm Events details files.

ISD-Lite columns: year month day hour airT(0.1C) dewpt(0.1C) slp(0.1hPa)
wind_dir(deg) wind_speed(0.1 m/s) sky_cover(oktas code 0-10) precip_1h precip_6h
(-9999 = missing).
"""
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from common import RAW, fetch, UA

W = RAW / "weather"


def isd(years=range(1995, 2024)):
    st = pd.read_csv(W / "selected_isd_stations.csv", dtype=str)
    jobs = []
    for _, s in st.iterrows():
        for y in years:
            sid = f"{s.USAF}-{s.WBAN}"
            jobs.append((f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{y}/{sid}-{y}.gz",
                         W / "isd_lite" / str(y) / f"{sid}-{y}.gz", f"ISD-Lite {sid} {y}"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch(*j, retries=3), jobs))


def storm_events(years=range(1995, 2024)):
    base = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    html = requests.get(base, timeout=60, headers={"User-Agent": UA}).text
    for y in years:
        m = re.findall(rf'(StormEvents_details-ftp_v1\.0_d{y}_c\d+\.csv\.gz)', html)
        if m:
            fetch(base + sorted(set(m))[-1], W / "storm_events" / f"details_{y}.csv.gz", f"NOAA Storm Events details {y}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["storm_events", "isd"]):
        globals()[s]()
