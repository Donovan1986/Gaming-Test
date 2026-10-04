"""Gazetteer-based geocoding with explicit uncertainty.

Rules (documented in report/METHODS.md):
* A free-text location is resolved to a populated place only when the place
  name matches (exactly, or fuzzily with score >= 90 inside the stated
  state/country). The returned coordinate is the place's internal point.
* LOCATION_UNCERTAINTY_RADIUS_KM is never smaller than 5 km for a town-level
  match; it grows with place size (sqrt of land area or population).
* "N miles DIR of X" phrases are offset from X and the radius is
  max(10 km, 0.3 * offset distance).
* State-only or country-only locations are NOT geocoded to a centroid; they
  get precision='region' and NaN coordinates (excluded from spatial work).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from common import RAW

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico", "GU": "Guam",
}
STATE_BY_NAME = {v.lower(): k for k, v in US_STATES.items()}
STATE_ALIASES = {"calif": "CA", "cal": "CA", "fla": "FL", "mass": "MA", "penn": "PA",
                 "penna": "PA", "wash": "WA", "mich": "MI", "minn": "MN", "wis": "WI",
                 "wisc": "WI", "tex": "TX", "ariz": "AZ", "colo": "CO", "conn": "CT",
                 "ill": "IL", "ind": "IN", "kans": "KS", "kan": "KS", "nebr": "NE",
                 "neb": "NE", "nev": "NV", "okla": "OK", "ore": "OR", "oreg": "OR",
                 "tenn": "TN", "va": "VA", "wyo": "WY", "n.m": "NM", "n. mex": "NM",
                 "n.c": "NC", "s.c": "SC", "n.d": "ND", "s.d": "SD", "n.j": "NJ",
                 "n.y": "NY", "n.h": "NH", "w.va": "WV", "w. va": "WV", "d.c": "DC",
                 "ala": "AL", "miss": "MS", "mont": "MT", "ark": "AR", "del": "DE",
                 "md": "MD", "vt": "VT", "me": "ME", "ga": "GA", "ky": "KY", "la": "LA",
                 "mo": "MO", "oh": "OH", "ut": "UT", "id": "ID", "ia": "IA", "ri": "RI",
                 "alas": "AK", "hawaii": "HI"}
COUNTRY_ALIASES = {"uk": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
                   "great britain": "GB", "united kingdom": "GB", "n. ireland": "GB",
                   "northern ireland": "GB", "korea": "KR", "south korea": "KR",
                   "west germany": "DE", "germany": "DE", "japan": "JP", "canada": "CA",
                   "france": "FR", "mexico": "MX", "italy": "IT", "spain": "ES",
                   "australia": "AU", "brazil": "BR", "argentina": "AR", "chile": "CL",
                   "greenland": "GL", "iceland": "IS", "philippines": "PH", "okinawa": "JP",
                   "newfoundland": "CA", "labrador": "CA", "ontario": "CA", "quebec": "CA",
                   "british columbia": "CA", "alberta": "CA", "manitoba": "CA",
                   "saskatchewan": "CA", "nova scotia": "CA", "new brunswick": "CA",
                   "libya": "LY", "morocco": "MA", "french morocco": "MA", "saudi arabia": "SA",
                   "turkey": "TR", "iran": "IR", "netherlands": "NL", "holland": "NL",
                   "belgium": "BE", "portugal": "PT", "azores": "PT", "bermuda": "BM",
                   "puerto rico": "PR", "guam": "GU", "panama": "PA", "cuba": "CU",
                   "venezuela": "VE", "colombia": "CO", "peru": "PE", "new zealand": "NZ",
                   "south africa": "ZA", "india": "IN", "china": "CN", "russia": "RU",
                   "ussr": "RU", "sweden": "SE", "norway": "NO", "denmark": "DK",
                   "finland": "FI", "poland": "PL", "austria": "AT", "switzerland": "CH",
                   "ireland": "IE", "greece": "GR", "israel": "IL", "egypt": "EG",
                   "vietnam": "VN", "thailand": "TH", "taiwan": "TW", "formosa": "TW",
                   "ethiopia": "ET", "pakistan": "PK", "laos": "LA"}
DIRS = {"n": 0, "nne": 22.5, "ne": 45, "ene": 67.5, "e": 90, "ese": 112.5, "se": 135,
        "sse": 157.5, "s": 180, "ssw": 202.5, "sw": 225, "wsw": 247.5, "w": 270,
        "wnw": 292.5, "nw": 315, "nnw": 337.5, "north": 0, "northeast": 45, "east": 90,
        "southeast": 135, "south": 180, "southwest": 225, "west": 270, "northwest": 315}
LSAD_SUFFIX = re.compile(r"\s+(city|town|village|borough|cdp|municipality|city and borough|"
                         r"consolidated government.*|metro government.*|unified government.*|"
                         r"urban county|comunidad|zona urbana|plantation|corporation)$", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"\bst\.?\s", "saint ", s)
    s = re.sub(r"\bft\.?\s", "fort ", s)
    s = re.sub(r"\bmt\.?\s", "mount ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def us_places() -> pd.DataFrame:
    p = pd.read_csv(RAW / "population" / "2020_Gaz_place_national.txt", sep="\t", dtype=str)
    p.columns = [c.strip() for c in p.columns]
    p["name_clean"] = p["NAME"].str.replace(LSAD_SUFFIX, "", regex=True).map(norm)
    p["lat"] = p["INTPTLAT"].astype(float)
    p["lon"] = p["INTPTLONG"].astype(float)
    p["aland_km2"] = p["ALAND"].astype(float) / 1e6
    p["state"] = p["USPS"]
    # pop proxy: CDPs/cities without pop in gazetteer; use land area for radius
    return p[["name_clean", "NAME", "state", "lat", "lon", "aland_km2"]]


@lru_cache(maxsize=1)
def geonames() -> pd.DataFrame:
    cols = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass", "fcode",
            "country", "cc2", "admin1", "admin2", "admin3", "admin4", "population", "elevation",
            "dem", "tz", "moddate"]
    g = pd.read_csv(RAW / "population" / "cities1000.txt", sep="\t", names=cols, dtype=str,
                    quoting=3, keep_default_na=False, na_values=[""])
    g["name_clean"] = g["asciiname"].map(norm)
    g["lat"] = g["lat"].astype(float)
    g["lon"] = g["lon"].astype(float)
    g["population"] = pd.to_numeric(g["population"], errors="coerce").fillna(0)
    return g[["geonameid", "name", "name_clean", "alternatenames", "lat", "lon", "country", "admin1",
              "population", "tz"]]


@lru_cache(maxsize=1)
def dod_sites() -> pd.DataFrame:
    rows = []
    import glob
    for f in sorted(glob.glob(str(RAW / "infrastructure" / "mirta_points_p*.geojson"))):
        for ft in json.load(open(f))["features"]:
            pr = ft["properties"]
            lon, lat = ft["geometry"]["coordinates"][:2]
            name = pr.get("SITENAME") or pr.get("FEATURENAME") or ""
            st = (pr.get("STATENAMECODE") or "").upper()
            rows.append((name, st, lat, lon))
    d = pd.DataFrame(rows, columns=["name", "state", "lat", "lon"])
    d["name_clean"] = d["name"].map(norm).str.replace(r"\b(afb|air force base|air base|nas|naval air station|"
                                                      r"army airfield|aaf|field|base)\b", "", regex=True).str.strip()
    return d


def place_radius_km(aland_km2=None, population=None) -> float:
    if aland_km2 is not None and not math.isnan(aland_km2) and aland_km2 > 0:
        return float(np.clip(math.sqrt(aland_km2) / 1.5, 5, 40))
    if population is not None and population > 0:
        return float(np.clip(10 * math.sqrt(population / 1e5), 5, 40))
    return 10.0


def offset_point(lat, lon, dist_km, bearing_deg):
    R = 6371.0
    br = math.radians(bearing_deg)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1) * math.cos(dist_km / R) + math.cos(la1) * math.sin(dist_km / R) * math.cos(br))
    lo2 = lo1 + math.atan2(math.sin(br) * math.sin(dist_km / R) * math.cos(la1),
                           math.cos(dist_km / R) - math.sin(la1) * math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)


def parse_region(region: str):
    r = norm(region).replace(" ", "") if region else ""
    raw = (region or "").strip().strip(".").lower()
    if raw.upper() in US_STATES:
        return "US", raw.upper()
    if raw in STATE_BY_NAME:
        return "US", STATE_BY_NAME[raw]
    if raw in STATE_ALIASES:
        return "US", STATE_ALIASES[raw]
    if raw in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[raw], None
    for k, v in COUNTRY_ALIASES.items():
        if k in raw:
            return v, None
    for k, v in STATE_BY_NAME.items():
        if k in raw:
            return "US", v
    return None, None


_OFFSET = re.compile(r"^\s*(?:about\s+|approx\.?\s+)?([\d.l]+)\s*(?:-\s*[\d.]+\s*)?(mi|miles?|km|kilometers?|nm|nautical miles?)\.?\s+"
                     r"([nsew]{1,3}|north(?:east|west)?|south(?:east|west)?|east|west)\s+(?:of|from)\s+(.*)$", re.I)


def geocode(loc_text: str, default_country: str | None = None):
    """Return dict(lat, lon, radius_km, precision, matched, country, state)."""
    out = dict(lat=np.nan, lon=np.nan, radius_km=np.nan, precision="unresolved", matched=None,
               country=default_country, state=None)
    if not isinstance(loc_text, str) or not loc_text.strip():
        return out
    txt = re.sub(r"\([^)]*\)", " ", loc_text).strip()
    txt = re.sub(r"^(over|near|off|above|in|at|vicinity of|outside|north of|south of|east of|west of)\s+", "", txt, flags=re.I)
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    country, state = (None, None)
    if len(parts) >= 2:
        country, state = parse_region(parts[-1])
        place = ",".join(parts[:-1]) if country else ",".join(parts)
    else:
        place = parts[0] if parts else ""
        c2, s2 = parse_region(place)
        if c2:  # region only, no false precision
            out.update(precision="region", country=c2, state=s2)
            return out
    country = country or default_country
    out.update(country=country, state=state)
    off = None
    m = _OFFSET.match(place)
    if m:
        d = float(m.group(1).replace("l", "1"))
        unit = m.group(2).lower()
        d_km = d * (1.609 if unit.startswith("mi") else 1.852 if unit.startswith("n") else 1.0)
        off = (d_km, DIRS.get(m.group(3).lower(), None))
        place = m.group(4)
    place = place.split("/")[0].split(" and ")[0]
    pn = norm(place)
    pn_base = re.sub(r"\b(afb|air force base|air base|nas|naval air station|army airfield|aaf|airport|"
                     r"municipal airport|naval station|army base|field|base|area|vicinity)\b", "", pn).strip()
    hit = None
    maybe_base = bool(re.search(r"\b(fort|camp|afb|air force base|air base|nas|base|field)\b", pn))
    if country == "US" or (country is None and default_country == "US"):
        P = us_places()
        cand = P[P.state == state] if state else P
        is_base = bool(re.search(r"\b(afb|air force base|air base|nas|naval air station|army airfield|aaf|"
                                 r"naval station|army base|arsenal|proving ground|test range|air station)\b", pn))
        maybe_base = is_base or bool(re.search(r"\b(fort|camp)\b", pn))
        if is_base:  # military installation names are matched before towns
            D = dod_sites()
            dc = D[D.state == state] if state else D
            if len(dc):
                best = process.extractOne(pn_base, dc.name_clean.tolist(), scorer=fuzz.token_sort_ratio)
                if best and best[1] >= 85:
                    r = dc.iloc[best[2]]
                    hit = (r.lat, r.lon, 10.0, "dod_site", r["name"])
        if hit is None and not is_base:
            ex = cand[cand.name_clean == pn_base]
            if len(ex):
                r = ex.iloc[ex.aland_km2.values.argmax()]
                hit = (r.lat, r.lon, place_radius_km(r.aland_km2), "place_exact", r.NAME)
        if hit is None and is_base:  # base not in MIRTA (closed): fall back to town, wider radius
            ex = cand[cand.name_clean == pn_base]
            if len(ex):
                r = ex.iloc[ex.aland_km2.values.argmax()]
                hit = (r.lat, r.lon, max(15.0, place_radius_km(r.aland_km2)), "place_exact_basename", r.NAME)
        if hit is None and maybe_base and not is_base:
            D = dod_sites()
            dc = D[D.state == state] if state else D
            if len(dc):
                best = process.extractOne(pn, dc.name_clean.tolist(), scorer=fuzz.token_sort_ratio)
                if best and best[1] >= 90:
                    r = dc.iloc[best[2]]
                    hit = (r.lat, r.lon, 10.0, "dod_site", r["name"])
        if hit is None and state and len(pn_base) >= 4 and not maybe_base:
            names = cand.name_clean.tolist()
            best = process.extractOne(pn_base, names, scorer=fuzz.ratio)
            if best and best[1] >= 85:
                r = cand.iloc[best[2]]
                hit = (r.lat, r.lon, place_radius_km(r.aland_km2) * 1.5, "place_fuzzy", r.NAME)
    if hit is None and len(pn_base) >= 3:
        G = geonames()
        gc = G[G.country == country] if country and country != "US" else (G[G.country == "US"] if country == "US" else G)
        if country == "US" and state:
            gc = gc[gc.admin1 == state]
        ex = gc[gc.name_clean == pn_base]
        if len(ex):
            r = ex.iloc[ex.population.values.argmax()]
            hit = (r.lat, r.lon, place_radius_km(population=r.population), "geonames_exact", r["name"])
            out["country"] = r.country
        elif country and len(gc) and len(pn_base) >= 5 and not maybe_base:
            best = process.extractOne(pn_base, gc.name_clean.tolist(), scorer=fuzz.ratio)
            if best and best[1] >= 92:
                r = gc.iloc[best[2]]
                hit = (r.lat, r.lon, place_radius_km(population=r.population) * 1.5, "geonames_fuzzy", r["name"])
    if hit is None:
        return out
    lat, lon, rad, prec, name = hit
    if off and off[1] is not None:
        lat, lon = offset_point(lat, lon, off[0], off[1])
        rad = max(10.0, 0.3 * off[0], rad)
        prec += "+offset"
    elif off:
        rad = max(rad, off[0])
        prec += "+undirected_offset"
    out.update(lat=float(lat), lon=float(lon), radius_km=float(rad), precision=prec, matched=name)
    return out
