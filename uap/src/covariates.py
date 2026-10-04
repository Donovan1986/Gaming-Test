"""Loaders for temporal / spatial covariates, cached to data/interim/*.parquet.

All times are UTC. Missing values in source files are converted to NaN and
are NOT filled; downstream code records DATA_UNAVAILABLE for those cells.
"""
from __future__ import annotations

import glob
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from common import RAW, INTERIM

SW = RAW / "spaceweather"
SPACE = RAW / "space"


def _cache(name):
    def deco(fn):
        def wrapper(*a, **k):
            p = INTERIM / f"{name}.parquet"
            if p.exists():
                return pd.read_parquet(p)
            df = fn(*a, **k)
            df.to_parquet(p, index=False)
            return df
        wrapper.__name__ = fn.__name__
        return lru_cache(maxsize=1)(wrapper)
    return deco


# ------------------------------------------------------------------ space weather
@_cache("kp3h")
def kp3h():
    rows = []
    for line in open(SW / "Kp_ap_Ap_SN_F107_since_1932.txt"):
        if line.startswith("#"):
            continue
        f = line.split()
        y, m, d = int(f[0]), int(f[1]), int(f[2])
        kps = [float(x) for x in f[7:15]]
        aps = [int(x) for x in f[15:23]]
        base = pd.Timestamp(year=y, month=m, day=d, tz="UTC")
        for i in range(8):
            rows.append((base + pd.Timedelta(hours=3 * i), kps[i] if kps[i] >= 0 else np.nan,
                         aps[i] if aps[i] >= 0 else np.nan))
    return pd.DataFrame(rows, columns=["time", "kp", "ap"])


@_cache("sw_daily")
def sw_daily():
    rows = []
    for line in open(SW / "Kp_ap_Ap_SN_F107_since_1932.txt"):
        if line.startswith("#"):
            continue
        f = line.split()
        kps = [float(x) for x in f[7:15]]
        kps = [k for k in kps if k >= 0]
        rows.append((pd.Timestamp(year=int(f[0]), month=int(f[1]), day=int(f[2])),
                     int(f[23]) if int(f[23]) >= 0 else np.nan,
                     int(f[24]) if int(f[24]) >= 0 else np.nan,
                     float(f[25]) if float(f[25]) > 0 else np.nan,
                     max(kps) if kps else np.nan, sum(kps) if kps else np.nan))
    return pd.DataFrame(rows, columns=["date", "Ap", "SN", "F107", "kp_max", "kp_sum"])


@_cache("omni_hourly")
def omni_hourly():
    cols = {0: "year", 1: "doy", 2: "hour", 16: "bz_gsm", 23: "np", 24: "v", 28: "pdyn",
            35: "efield", 38: "kp10", 39: "R", 40: "dst", 41: "ae", 45: "pflux10"}
    fill = {"bz_gsm": 999.9, "np": 999.9, "v": 9999.0, "pdyn": 99.99, "efield": 999.99, "kp10": 99,
            "R": 999, "dst": 99999, "ae": 9999, "pflux10": 99999.99}
    df = pd.read_csv(SW / "omni2_all_years.dat", sep=r"\s+", header=None, usecols=list(cols), engine="c")
    df = df.rename(columns=cols)
    for k, v in fill.items():
        df.loc[df[k] >= v * 0.999, k] = np.nan
    df["time"] = (pd.to_datetime(df.year.astype(str), format="%Y", utc=True)
                  + pd.to_timedelta(df.doy - 1, unit="D") + pd.to_timedelta(df.hour, unit="h"))
    df["kp_omni"] = df.kp10 / 10.0
    return df.drop(columns=["year", "doy", "hour", "kp10"])


@_cache("flares")
def flares():
    """GOES X-ray flare events 1975-2017 (NGDC). Columns: start time, class letter, magnitude."""
    rows = []
    for f in sorted(glob.glob(str(SW / "goes_xrs" / "goes-xrs-report_*.txt"))):
        if "input" in f:
            continue
        for line in open(f, errors="ignore"):
            if len(line) < 70:
                continue
            try:
                yy = int(line[5:7]); mm = int(line[7:9]); dd = int(line[9:11])
                hh = int(line[13:15]); mi = int(line[15:17])
                cls = line[59:60]; mag = float(line[60:63]) / 10.0
            except ValueError:
                continue
            year = 1900 + yy if yy >= 70 else 2000 + yy
            if cls not in "ABCMX":
                continue
            rows.append((pd.Timestamp(year=year, month=mm, day=dd, hour=hh, minute=mi, tz="UTC"), cls, mag))
    df = pd.DataFrame(rows, columns=["time", "cls", "mag"]).drop_duplicates()
    return df


# ------------------------------------------------------------------ geophysical / space
@_cache("quakes")
def quakes():
    fs = sorted(glob.glob(str(RAW / "geophysical" / "usgs" / "*.csv")))
    parts = [pd.read_csv(f, usecols=["time", "latitude", "longitude", "depth", "mag", "type"]) for f in fs]
    df = pd.concat(parts, ignore_index=True)
    df = df[df["type"].fillna("earthquake") == "earthquake"]
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df.drop_duplicates(["time", "latitude", "longitude"]).sort_values("time").reset_index(drop=True)


@_cache("fireballs")
def fireballs():
    d = json.load(open(SPACE / "cneos_fireballs.json"))
    df = pd.DataFrame(d["data"], columns=d["fields"])
    df["time"] = pd.to_datetime(df["date"], utc=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce") * np.where(df["lat-dir"] == "S", -1, 1)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce") * np.where(df["lon-dir"] == "W", -1, 1)
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df["impact_e_kt"] = pd.to_numeric(df["impact-e"], errors="coerce")
    df["alt"] = pd.to_numeric(df["alt"], errors="coerce")
    return df[["time", "lat", "lon", "energy", "impact_e_kt", "alt"]]


_MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _gcat_time(s: str):
    m = re.match(r"(\d{4})\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{2})(\d{2})(?::(\d{2}))?)?", str(s).strip())
    if not m:
        return pd.NaT, False
    y, mo, d = int(m.group(1)), _MON.get(m.group(2)), int(m.group(3))
    if mo is None:
        return pd.NaT, False
    if m.group(4):
        return pd.Timestamp(year=y, month=mo, day=d, hour=int(m.group(4)), minute=int(m.group(5)),
                            second=int(m.group(6) or 0), tz="UTC"), True
    return pd.Timestamp(year=y, month=mo, day=d, hour=12, tz="UTC"), False


@_cache("launches")
def launches():
    L = pd.read_csv(SPACE / "gcat_launch.tsv", sep="\t", dtype=str, comment=None, skiprows=[1])
    L.columns = [c.lstrip("#").strip() for c in L.columns]
    S = pd.read_csv(SPACE / "gcat_sites.tsv", sep="\t", dtype=str, skiprows=[1])
    S.columns = [c.lstrip("#").strip() for c in S.columns]
    S["lon"] = pd.to_numeric(S["Longitude"].str.strip(), errors="coerce")
    S["lat"] = pd.to_numeric(S["Latitude"].str.strip(), errors="coerce")
    site_xy = S.dropna(subset=["lat"]).drop_duplicates("Site").set_index("Site")[["lat", "lon", "StateCode", "Name"]]
    tt = [ _gcat_time(s) for s in L["Launch_Date"] ]
    L["time"] = [t for t, _ in tt]
    L["time_has_clock"] = [b for _, b in tt]
    L["site"] = L["Launch_Site"].str.strip()
    L = L.join(site_xy, on="site")
    L["orbital"] = L["Category"].fillna("").str.startswith("Sat") | L["Category"].fillna("").str.contains("Orb")
    L["mission"] = L["Mission"].fillna("") + " " + L["Flight"].fillna("")
    L["is_starlink"] = L["mission"].str.contains("Starlink", case=False)
    L["failed"] = L["LaunchCode"].fillna("").str.strip().str.upper().str.startswith("F")
    L["apogee_km"] = pd.to_numeric(L["Apogee"].str.strip(), errors="coerce")
    keep = ["Launch_Tag", "time", "time_has_clock", "LV_Type", "mission", "site", "lat", "lon", "StateCode",
            "Category", "orbital", "is_starlink", "failed", "apogee_km"]
    return L[keep].dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


@_cache("starlink_decays")
def satcat_reentries():
    C = pd.read_csv(SPACE / "gcat_satcat.tsv", sep="\t", dtype=str, skiprows=[1])
    C.columns = [c.lstrip("#").strip() for c in C.columns]
    tt = [_gcat_time(s) for s in C["DDate"]]
    C["decay_time"] = [t for t, _ in tt]
    C["mass"] = pd.to_numeric(C["Mass"].str.strip(), errors="coerce")
    C["is_starlink"] = C["Name"].fillna("").str.contains("Starlink", case=False)
    return C.dropna(subset=["decay_time"])[["JCAT", "Name", "Type", "decay_time", "mass", "is_starlink"]]


# ------------------------------------------------------------------ astronomy
@lru_cache(maxsize=1)
def _sky():
    from skyfield.api import load
    ts = load.timescale()
    eph = load(str(RAW / "astro" / "de421.bsp"))
    return ts, eph


def body_radec(times_utc: pd.DatetimeIndex, body: str):
    """Geocentric apparent RA/Dec (radians) and distance (au) for many times."""
    ts, eph = _sky()
    t = ts.from_datetimes(pd.DatetimeIndex(times_utc).to_pydatetime())
    names = {"sun": "sun", "moon": "moon", "venus": "venus", "jupiter": "jupiter barycenter",
             "mars": "mars", "saturn": "saturn barycenter"}
    ast = eph["earth"].at(t).observe(eph[names[body]]).apparent()
    ra, dec, dist = ast.radec("date")
    return ra.radians, dec.radians, dist.au


def gmst_rad(times_utc: pd.DatetimeIndex):
    ts, _ = _sky()
    t = ts.from_datetimes(pd.DatetimeIndex(times_utc).to_pydatetime())
    return t.gmst / 24.0 * 2 * np.pi


def altitude_deg(ra, dec, gmst, lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    H = gmst + np.radians(lon_deg) - ra
    s = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H)
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def azimuth_deg(ra, dec, gmst, lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    H = gmst + np.radians(lon_deg) - ra
    az = np.arctan2(-np.sin(H), np.tan(dec) * np.cos(lat) - np.sin(lat) * np.cos(H))
    return (np.degrees(az) + 360) % 360


def astro_features(times_utc, lats, lons) -> pd.DataFrame:
    """Geometry using 10-minute apparent-position bins and exact Earth rotation.

    Binning RA/Dec approximates slow orbital motion. Sidereal rotation must
    use each observation's actual timestamp; flooring it can shift altitude
    by up to 2.5 degrees and change a visibility threshold decision.
    """
    times_utc = pd.DatetimeIndex(pd.to_datetime(times_utc, utc=True))
    th = times_utc.floor("10min").as_unit("ns")
    uniq, inv = np.unique(th.asi8, return_inverse=True)
    ut = pd.DatetimeIndex(pd.to_datetime(uniq, unit="ns", utc=True))
    g = gmst_rad(times_utc)
    out = {}
    pos = {}
    for b in ("sun", "moon", "venus", "jupiter"):
        ra, dec, dist = body_radec(ut, b)
        pos[b] = (ra[inv], dec[inv], dist[inv])
        out[f"{b}_alt"] = altitude_deg(ra[inv], dec[inv], g, lats, lons)
    out["venus_az"] = azimuth_deg(*pos["venus"][:2], g, lats, lons)
    out["jupiter_az"] = azimuth_deg(*pos["jupiter"][:2], g, lats, lons)
    out["moon_az"] = azimuth_deg(*pos["moon"][:2], g, lats, lons)
    # Moon illuminated fraction from Sun-Moon elongation
    ras, decs, _ = pos["sun"]
    ram, decm, _ = pos["moon"]
    cos_el = np.sin(decs) * np.sin(decm) + np.cos(decs) * np.cos(decm) * np.cos(ras - ram)
    elong = np.arccos(np.clip(cos_el, -1, 1))
    out["moon_illum"] = (1 - np.cos(elong)) / 2
    # Venus elongation from Sun and apparent magnitude (Mallama & Hilton 2018 approx)
    rav, decv, dv = pos["venus"]
    cos_ev = np.sin(decs) * np.sin(decv) + np.cos(decs) * np.cos(decv) * np.cos(ras - rav)
    out["venus_elong"] = np.degrees(np.arccos(np.clip(cos_ev, -1, 1)))
    # phase angle via triangle Sun-Venus-Earth (r_sv ~ 0.723 au)
    r = 0.7233
    R = 1.0
    cos_i = np.clip((r ** 2 + dv ** 2 - R ** 2) / (2 * r * dv), -1, 1)
    i = np.degrees(np.arccos(cos_i))
    out["venus_mag"] = -4.384 + 5 * np.log10(r * dv) + 1.044e-3 * i + 3.687e-4 * i ** 2 - 2.814e-6 * i ** 3 + 8.938e-9 * i ** 4
    return pd.DataFrame(out)


def solar_longitude_deg(times_utc) -> np.ndarray:
    ts, eph = _sky()
    t = ts.from_datetimes(pd.DatetimeIndex(pd.to_datetime(times_utc, utc=True)).to_pydatetime())
    from skyfield.framelib import ecliptic_frame
    ast = eph["earth"].at(t).observe(eph["sun"]).apparent()
    lat, lon, _ = ast.frame_latlon(ecliptic_frame)
    return lon.degrees


# IMO meteor shower parameters (IMO Meteor Shower Calendar 2024 working list):
# name, solar longitude of max (J2000), ZHR, width parameter B (per degree, Jenniskens profile)
SHOWERS = [
    ("Quadrantids", 283.15, 110, 0.75), ("Lyrids", 32.32, 18, 0.22), ("Eta Aquariids", 45.5, 50, 0.08),
    ("S Delta Aquariids", 127.0, 25, 0.091), ("Perseids", 140.0, 100, 0.20), ("Draconids", 195.4, 10, 0.80),
    ("Orionids", 208.0, 20, 0.12), ("S Taurids", 197.0, 5, 0.026), ("N Taurids", 230.0, 5, 0.026),
    ("Leonids", 235.27, 15, 0.39), ("Geminids", 262.2, 150, 0.39), ("Ursids", 270.7, 10, 0.61),
]
# Documented outbursts / storms (date of peak, ZHR) — positive-control set, from IMO/literature
OUTBURSTS = [("1998-11-17", "Leonids", 300), ("1999-11-18", "Leonids", 3700), ("2001-11-18", "Leonids", 3000),
             ("2002-11-19", "Leonids", 3000), ("2011-10-08", "Draconids", 600), ("2012-10-08", "Draconids", 1000),
             ("2005-11-05", "Taurids", 15), ("2015-11-05", "Taurids", 15), ("2022-11-05", "Taurids", 10)]


def shower_activity(times_utc) -> pd.DataFrame:
    """ZHR-weighted activity: sum_i ZHR_i * 10^(-B_i |lambda - lambda_max_i|)."""
    lam = solar_longitude_deg(times_utc)
    out = {}
    tot = np.zeros(len(lam))
    for name, lmax, zhr, B in SHOWERS:
        dl = np.abs(((lam - lmax) + 180) % 360 - 180)
        a = zhr * 10 ** (-B * dl)
        out[f"shw_{name.replace(' ', '_')}"] = a
        tot += a
    out["shower_zhr_total"] = tot
    out["solar_longitude"] = lam
    return pd.DataFrame(out)


# ------------------------------------------------------------------ population / media
@_cache("county_pop")
def county_pop():
    P = RAW / "population"
    a = pd.read_csv(P / "co-est2020-alldata.csv", encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    a = a[a.SUMLEV == 50]
    a["fips"] = a.STATE.str.zfill(2) + a.COUNTY.str.zfill(3)
    long = []
    for y in range(2010, 2021):
        long.append(pd.DataFrame({"fips": a.fips, "year": y, "pop": a[f"POPESTIMATE{y}"]}))
    b = pd.read_csv(P / "co-est00int-tot.csv", encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    b = b[b.SUMLEV == 50]
    b["fips"] = b.STATE.str.zfill(2) + b.COUNTY.str.zfill(3)
    for y in range(2000, 2010):
        long.append(pd.DataFrame({"fips": b.fips, "year": y, "pop": b[f"POPESTIMATE{y}"]}))
    c = pd.read_csv(P / "co-est2023-alldata.csv", encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    c = c[c.SUMLEV == 50]
    c["fips"] = c.STATE.str.zfill(2) + c.COUNTY.str.zfill(3)
    for y in range(2021, 2024):
        long.append(pd.DataFrame({"fips": c.fips, "year": y, "pop": c[f"POPESTIMATE{y}"]}))
    rows = []
    for line in open(P / "county_1990s_99c8_00.txt", errors="ignore"):
        m = re.match(r"^1\s+(\d{5})\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", line)
        if m:
            vals = [int(v.replace(",", "")) for v in m.groups()[1:]]
            for k, y in enumerate(range(1999, 1989, -1)):
                rows.append((m.group(1), y, vals[k]))
    long.append(pd.DataFrame(rows, columns=["fips", "year", "pop"]))
    df = pd.concat(long, ignore_index=True)
    return df.drop_duplicates(["fips", "year"], keep="first")


@_cache("wiki_pageviews")
def wiki_pageviews():
    out = []
    for f in glob.glob(str(RAW / "media" / "wiki_pageviews_*.json")):
        art = f.split("wiki_pageviews_")[1][:-5]
        for it in json.load(open(f)).get("items", []):
            out.append((art, pd.Timestamp(it["timestamp"][:8]), it["views"]))
    return pd.DataFrame(out, columns=["article", "date", "views"])


# Major UAP media events, curated with citations in report/DATA_SOURCES.md.
MEDIA_EVENTS = [
    ("1947-07-08", "Roswell press release / 'flying disc' wave coverage"),
    ("1952-07-27", "Washington National radar-visual sightings front pages"),
    ("1966-03-25", "Michigan 'swamp gas' (Hynek) national coverage"),
    ("1969-12-17", "USAF announces end of Project Blue Book"),
    ("1977-11-16", "Close Encounters of the Third Kind release"),
    ("1993-09-10", "The X-Files premiere"),
    ("1996-07-03", "Independence Day release"),
    ("1997-03-13", "Phoenix Lights (event; coverage peaks June 1997)"),
    ("1997-06-18", "Phoenix Lights national coverage (USA Today, June 18)"),
    ("2002-07-15", "Signs (film) release window / NUFORC media"),
    ("2006-11-07", "O'Hare Airport UFO (coverage Jan 1, 2007)"),
    ("2008-01-08", "Stephenville, Texas sightings coverage"),
    ("2017-12-16", "NYT AATIP / Navy videos story"),
    ("2019-05-26", "NYT 'Wow, what is that?' Navy pilots story"),
    ("2020-04-27", "DoD officially releases three Navy UAP videos"),
    ("2021-05-16", "60 Minutes UAP segment"),
    ("2021-06-25", "ODNI Preliminary Assessment on UAP"),
    ("2022-05-17", "First Congressional UAP hearing in 50 years"),
    ("2023-02-04", "Chinese balloon shootdown and Feb 10-12 object shootdowns"),
    ("2023-07-26", "House Oversight UAP hearing (Grusch)"),
    ("2023-09-14", "NASA UAP Independent Study Team report"),
]

HOLIDAYS_FIXED = {"07-04": "independence_day", "12-31": "new_years_eve", "01-01": "new_years_day",
                  "10-31": "halloween", "07-05": "day_after_july4", "07-03": "day_before_july4"}
