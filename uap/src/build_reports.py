"""Build the unified REPORT-level table from every UAP source.

Output: data/processed/reports.parquet (full, includes ORIGINAL_TEXT; git-ignored)
        data/processed/reports_public.parquet (ORIGINAL_TEXT dropped for NUFORC)
        results/geocoding_decisions.csv, results/excluded_records.csv

Uncertainty conventions
* TIME_UNCERTAINTY_MIN: 5 (minute given, not a multiple of 5), 10 (multiple of 5),
  15 (multiple of 15), 30 (on the hour/half hour), 720 (local 00:00 in NUFORC,
  which is the scraper default for blank times), NaN with TIME_PRECISION='none'
  when no time of day exists. Vague words ("evening", "daytime") are kept as
  TIME_TEXT and NEVER converted to a clock time.
* DATE_PRECISION: day / month / year.
* LOCATION_UNCERTAINTY_RADIUS_KM from geo.py rules.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

import geo
import textparse as T
from common import RAW, PROCESSED, RESULTS

RETRIEVAL = {}
for line in open(RAW.parent / "manifest.jsonl"):
    d = json.loads(line)
    if d.get("ok"):
        RETRIEVAL[d["dest"]] = (d["url"], d["retrieved_utc"][:10])

TF = TimezoneFinder()


def rhash(*vals) -> str:
    return hashlib.sha256("|".join(map(str, vals)).encode()).hexdigest()[:16]


def time_unc(minute: int, hour: int, source: str) -> float:
    if source == "NUFORC" and hour == 0 and minute == 0:
        return 720.0
    if minute % 30 == 0:
        return 30.0
    if minute % 15 == 0:
        return 15.0
    if minute % 5 == 0:
        return 10.0
    return 5.0


def status_bool(flag: bool, structured: bool = False) -> str:
    """Missing-data aware status for evidence fields."""
    if flag:
        return "YES_STRUCTURED" if structured else "YES_TEXT"
    return "NOT_REPORTED"


# ----------------------------------------------------------------------------- NUFORC
def nuforc() -> pd.DataFrame:
    t = pd.read_csv(RAW / "nuforc" / "tt2023_ufo_sightings.csv")
    p = pd.read_csv(RAW / "nuforc" / "tt2023_places.csv").drop_duplicates(["city", "state", "country_code"])
    t = t.merge(p, on=["city", "state", "country_code"], how="left")
    t["src_row"] = np.arange(len(t))
    utc = pd.to_datetime(t["reported_date_time_utc"], utc=True)
    local = [u.tz_convert(z) if isinstance(z, str) else pd.NaT for u, z in zip(utc, t["timezone"])]
    local = pd.Series(local, index=t.index)
    url, rdate = RETRIEVAL[str((RAW / "nuforc" / "tt2023_ufo_sightings.csv").relative_to(RAW.parents[1]))]
    rows = []
    for i, r in t.iterrows():
        lt = local[i]
        text = r["summary"] if isinstance(r["summary"], str) else ""
        dur = T.parse_duration_s(r["reported_duration"])
        tu = time_unc(lt.minute, lt.hour, "NUFORC") if pd.notna(lt) else np.nan
        rad = geo.place_radius_km(population=r["population"]) if pd.notna(r["latitude"]) else np.nan
        posted = pd.to_datetime(r["posted_date"], errors="coerce")
        rows.append(dict(
            SOURCE="NUFORC", SOURCE_RECORD_ID=f"tt2023:{r['src_row']}", ORIGINAL_TEXT=text,
            DATE=lt.strftime("%Y-%m-%d") if pd.notna(lt) else None, DATE_PRECISION="day",
            LOCAL_TIME=lt.strftime("%H:%M") if pd.notna(lt) else None,
            UTC_TIME=utc[i].strftime("%Y-%m-%dT%H:%M:%SZ"), TIMEZONE=r["timezone"],
            TIME_UNCERTAINTY_MIN=tu, TIME_PRECISION="minute" if tu and tu < 720 else "uncertain",
            TIME_TEXT=None,
            LATITUDE=r["latitude"], LONGITUDE=r["longitude"], LOCATION_UNCERTAINTY_RADIUS_KM=rad,
            LOCATION_PRECISION="place_geonames" if pd.notna(r["latitude"]) else "unresolved",
            LOCATION_TEXT=f"{r['city']}, {r['state']}, {r['country_code']}", COUNTRY=r["country_code"],
            STATE=r["state"], PLACE_POPULATION=r["population"], ELEVATION_M=r["elevation_m"],
            ALTITUDE=np.nan, ALTITUDE_UNCERTAINTY=np.nan, ALTITUDE_STATUS="DATA_UNAVAILABLE",
            OBSERVATION_DURATION_S=dur, DURATION_TEXT=r["reported_duration"],
            POSTED_DATE=posted.strftime("%Y-%m-%d") if pd.notna(posted) else None,
            SHAPE=T.norm_shape(r["shape"]), SHAPE_RAW=r["shape"], HAS_IMAGES_STRUCT=bool(r["has_images"]) if pd.notna(r["has_images"]) else None,
            DAY_PART_SOURCE=r["day_part"], SOURCE_URL=url, RETRIEVAL_DATE=rdate,
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Hatch
_HDATE = re.compile(r"^(-?\d+)/(-?\d+)/(-?\d+)")


def _parse_mdy(s):
    m = _HDATE.match(str(s).strip())
    if not m:
        return None, None, None
    mo, d, y = (int(x) for x in m.groups())
    return y, (mo if 1 <= mo <= 12 else None), (d if 1 <= d <= 31 else None)


def hatch() -> pd.DataFrame:
    H = json.load(open(RAW / "catalogs" / "hatch_udb.json", encoding="utf-8-sig"))["Hatch_UDB_Timeline"]
    url, rdate = RETRIEVAL["data/raw/catalogs/hatch_udb.json"]
    rows = []
    for x in H:
        kv = x.get("key_vals", {})
        y, mo, d = _parse_mdy(x.get("alt_basic_date") or x.get("basic_date") or x.get("date"))
        approx = "approximate" in str(x.get("date", "")).lower()
        attrs = {a.split(":")[0].strip() for a in x.get("attributes", [])}
        attr_text = {a.split(":")[0].strip(): a for a in x.get("attributes", [])}
        mil_obs = any("Observer(s) military" in a for a in x.get("attributes", []))
        tm = str(x.get("time", "") or "")
        mt = re.search(r"(\d{1,2}):(\d{2})", tm)
        lat = lon = np.nan
        try:
            lat, lon = (float(v) for v in kv.get("LatLong", "").split())
        except ValueError:
            pass
        if (lat == 0 and lon == 0) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            lat = lon = np.nan
        date_prec = "day" if (y and mo and d) else ("month" if (y and mo) else ("year" if y else None))
        tz = TF.timezone_at(lng=lon, lat=lat) if np.isfinite(lat) else None
        local_dt = utc_s = None
        tunc = np.nan
        if date_prec == "day" and mt and 1000 <= y <= 2100:
            hh, mm = int(mt.group(1)), int(mt.group(2))
            if hh < 24:
                tunc = (60.0 if "~" in tm else time_unc(mm, hh, "Hatch"))
                if tz and y >= 1900:
                    try:
                        loc = pd.Timestamp(datetime(y, mo, d, hh, mm)).tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
                        utc_s = loc.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        pass
                local_dt = f"{hh:02d}:{mm:02d}"
        wit = np.nan
        mw = re.match(r"\s*(\d+|many|several|thousands|hundreds|dozens|1000s|100s)\s*(?:/\s*)?(?:observer|obs|witness|military|mil|pilots?|crew|police|cops)", x.get("desc", ""), re.I)
        if mw:
            tok = mw.group(1).lower()
            wit = {"many": 10, "several": 4, "thousands": 1000, "hundreds": 100, "dozens": 24, "1000s": 1000, "100s": 100}.get(tok)
            wit = float(wit if wit else int(tok))
        dur_min = pd.to_numeric(kv.get("Duration"), errors="coerce")
        rows.append(dict(
            SOURCE="HATCH", SOURCE_RECORD_ID=x.get("source_id"), ORIGINAL_TEXT=x.get("desc"),
            DATE=(f"{y:04d}-{mo:02d}-{d:02d}" if date_prec == "day" and y and y > 0 else None),
            YEAR=y, MONTH=mo, DATE_PRECISION=date_prec if not approx else (date_prec or "") + "_approx",
            LOCAL_TIME=local_dt, UTC_TIME=utc_s, TIMEZONE=tz, TIME_UNCERTAINTY_MIN=tunc,
            TIME_PRECISION="minute" if local_dt else "none", TIME_TEXT=tm or None,
            LATITUDE=lat, LONGITUDE=lon,
            LOCATION_UNCERTAINTY_RADIUS_KM=10.0 if np.isfinite(lat) else np.nan,
            LOCATION_PRECISION="hatch_latlong" if np.isfinite(lat) else "unresolved",
            LOCATION_TEXT=x.get("location"), COUNTRY=kv.get("Country"), STATE=kv.get("State/Prov"),
            ALTITUDE=pd.to_numeric(kv.get("RelAlt"), errors="coerce"), ALTITUDE_UNCERTAINTY=np.nan,
            ALTITUDE_STATUS="REPORTED_RELATIVE" if kv.get("RelAlt") else "NOT_REPORTED",
            ELEVATION_M=pd.to_numeric(kv.get("Elev"), errors="coerce"),
            OBSERVATION_DURATION_S=float(dur_min) * 60 if pd.notna(dur_min) and dur_min > 0 else np.nan,
            DURATION_TEXT=kv.get("Duration"),
            HATCH_CREDIBILITY=pd.to_numeric(kv.get("Credibility"), errors="coerce"),
            HATCH_STRANGENESS=pd.to_numeric(kv.get("Strangeness"), errors="coerce"),
            HATCH_ATTRS=";".join(sorted(attrs)), HATCH_MIL_OBSERVER=mil_obs,
            HATCH_LOCALE=kv.get("Locale"), NUMBER_OF_WITNESSES_STRUCT=wit,
            HATCH_REF=x.get("ref"), SOURCE_URL=url, RETRIEVAL_DATE=rdate,
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Blue Book unknowns / NICAP
_TIME12 = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", re.I)
_TIME24 = re.compile(r"\b(\d{2})(\d{2})\s*(?:hrs?|h|z|local)?\b")
US_TZ_ABBR = {"EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
              "GMT": 0, "Z": 0, "UT": 0, "UTC": 0, "AST": -4, "HST": -10, "AKST": -9, "YST": -9}


def parse_time_text(tm: str):
    """Return (hh, mm, explicit_utc_offset or None) or None for vague text."""
    if not isinstance(tm, str):
        return None
    m = _TIME12.search(tm)
    if m:
        hh = int(m.group(1)) % 12 + (12 if m.group(3).lower().startswith("p") else 0)
        mm = int(m.group(2) or 0)
        off = None
        for ab, o in US_TZ_ABBR.items():
            if re.search(rf"\b{ab}\b", tm):
                off = o
                break
        return hh, mm, off
    m = re.search(r"\b([01]\d|2[0-3]):?([0-5]\d)\s*(z|gmt|utc)?\b", tm, re.I)
    if m and re.search(r"\d{4}|\d{1,2}:\d{2}", tm):
        return int(m.group(1)), int(m.group(2)), (0 if m.group(3) else None)
    return None


def geocoded_text_source(name: str, items: list, url: str, rdate: str, default_country="US") -> tuple[pd.DataFrame, list]:
    rows, decisions = [], []
    for x in items:
        y, mo, d = _parse_mdy(x.get("date", ""))
        if y is None:
            m = re.match(r"^\s*(\d{1,2})/(\d{4})", x.get("date", ""))
            if m:
                mo, y, d = int(m.group(1)), int(m.group(2)), None
            else:
                m = re.match(r"^\s*(\d{4})", x.get("date", ""))
                y = int(m.group(1)) if m else None
        date_prec = "day" if (y and mo and d) else ("month" if (y and mo) else ("year" if y else None))
        g = geo.geocode(x.get("location", ""), default_country=default_country)
        decisions.append(dict(SOURCE=name, SOURCE_RECORD_ID=x.get("source_id"), LOCATION_TEXT=x.get("location"),
                              **{f"geo_{k}": v for k, v in g.items()}))
        tm = x.get("time") or ""
        desc = x.get("desc", "")
        pt = parse_time_text(tm) if tm else None
        if pt is None and name == "NICAP":  # NICAP puts the time in the description
            mt = re.match(r"^\s*(\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?)", desc, re.I)
            if mt:
                pt = parse_time_text(mt.group(1))
                tm = mt.group(1)
        tz = TF.timezone_at(lng=g["lon"], lat=g["lat"]) if np.isfinite(g["lat"]) else None
        local_s = utc_s = None
        tunc = np.nan
        if pt and date_prec == "day":
            hh, mm, off = pt
            if hh < 24:
                local_s = f"{hh:02d}:{mm:02d}"
                tunc = time_unc(mm, hh, name)
                try:
                    if off is not None:
                        utc = pd.Timestamp(datetime(y, mo, d, hh, mm)) - pd.Timedelta(hours=off)
                        utc_s = utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                    elif tz:
                        loc = pd.Timestamp(datetime(y, mo, d, hh, mm)).tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
                        utc_s = loc.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass
        nicap_cat = re.findall(r"\(NICAP: ([^)]*)\)", desc)
        rows.append(dict(
            SOURCE=name, SOURCE_RECORD_ID=x.get("source_id"), ORIGINAL_TEXT=desc,
            DATE=(f"{y:04d}-{mo:02d}-{d:02d}" if date_prec == "day" else None), YEAR=y, MONTH=mo,
            DATE_PRECISION=date_prec, LOCAL_TIME=local_s, UTC_TIME=utc_s, TIMEZONE=tz,
            TIME_UNCERTAINTY_MIN=tunc, TIME_PRECISION="minute" if local_s else "none",
            TIME_TEXT=tm or None, LATITUDE=g["lat"], LONGITUDE=g["lon"],
            LOCATION_UNCERTAINTY_RADIUS_KM=g["radius_km"], LOCATION_PRECISION=g["precision"],
            LOCATION_TEXT=x.get("location"), COUNTRY=g["country"], STATE=g["state"],
            ALTITUDE=np.nan, ALTITUDE_UNCERTAINTY=np.nan, ALTITUDE_STATUS="NOT_REPORTED",
            OBSERVATION_DURATION_S=T.parse_duration_s(desc), DURATION_TEXT=None,
            NICAP_CATEGORY=nicap_cat[0] if nicap_cat else None,
            SOURCE_URL=url, RETRIEVAL_DATE=rdate,
        ))
    return pd.DataFrame(rows), decisions


def bluebook_unknowns():
    items = json.load(open(RAW / "catalogs" / "bb_unknowns_geldreich.json", encoding="utf-8-sig"))["BlueBookUnknowns Timeline"]
    url, rdate = RETRIEVAL["data/raw/catalogs/bb_unknowns_geldreich.json"]
    df, dec = geocoded_text_source("BLUEBOOK_UNK", items, url, rdate)
    df["EXPLANATION_STATUS_SOURCE"] = "UNEXPLAINED_PER_USAF"
    return df, dec


def nicap():
    items = json.load(open(RAW / "catalogs" / "nicap_db.json", encoding="utf-8-sig"))["NICAP Data"]
    url, rdate = RETRIEVAL["data/raw/catalogs/nicap_db.json"]
    return geocoded_text_source("NICAP", items, url, rdate)


# ----------------------------------------------------------------------------- GEIPAN
GEIPAN_STATUS = {"A": "EXPLAINED_CERTAIN", "B": "EXPLAINED_PROBABLE", "C": "UNIDENTIFIED_INSUFFICIENT_DATA",
                 "D": "UNIDENTIFIED_AFTER_INVESTIGATION"}


def geipan() -> pd.DataFrame:
    f = RAW / "geipan" / "geipan_export_cas.xlsx"
    if not f.exists():
        return pd.DataFrame()
    x = pd.read_excel(f)
    url, rdate = RETRIEVAL["data/raw/geipan/geipan_export_cas.xlsx"]
    rows = []
    for _, r in x.iterrows():
        ds = str(r["Date d'observation"])
        m = re.match(r"(\d{2}|--)/(\d{2}|--)/(\d{4})", ds)
        d = mo = None
        y = int(r["Année"]) if pd.notna(r["Année"]) else None
        if m:
            d = int(m.group(1)) if m.group(1) != "--" else None
            mo = int(m.group(2)) if m.group(2) != "--" else None
            y = int(m.group(3))
        date_prec = "day" if (y and mo and d) else ("month" if (y and mo) else ("year" if y else None))
        det = r["Détails"] if isinstance(r["Détails"], str) else ""
        lat, lon = r["Latitude"], r["Longitude"]
        tz = TF.timezone_at(lng=lon, lat=lat) if pd.notna(lat) else None
        pt = T.parse_time_fr(det)
        local_s = utc_s = None
        tunc = np.nan
        if pt and date_prec == "day":
            hh, mm, approx = pt
            local_s = f"{hh:02d}:{mm:02d}"
            tunc = 30.0 if approx else time_unc(mm, hh, "GEIPAN")
            if tz:
                try:
                    loc = pd.Timestamp(datetime(y, mo, d, hh, mm)).tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
                    utc_s = loc.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass
        rows.append(dict(
            SOURCE="GEIPAN", SOURCE_RECORD_ID=r["ID Etude de Cas"], ORIGINAL_TEXT=det,
            DATE=(f"{y:04d}-{mo:02d}-{d:02d}" if date_prec == "day" else None), YEAR=y, MONTH=mo,
            DATE_PRECISION=date_prec, LOCAL_TIME=local_s, UTC_TIME=utc_s, TIMEZONE=tz,
            TIME_UNCERTAINTY_MIN=tunc, TIME_PRECISION="minute" if local_s else "none", TIME_TEXT=None,
            LATITUDE=lat, LONGITUDE=lon, LOCATION_UNCERTAINTY_RADIUS_KM=8.0 if pd.notna(lat) else np.nan,
            LOCATION_PRECISION="geipan_commune" if pd.notna(lat) else "unresolved",
            LOCATION_TEXT=r["Titre du Cas"], COUNTRY="FR", STATE=r["Département"],
            ALTITUDE=np.nan, ALTITUDE_UNCERTAINTY=np.nan, ALTITUDE_STATUS="NOT_REPORTED",
            OBSERVATION_DURATION_S=T.parse_duration_fr(det), DURATION_TEXT=None,
            NUMBER_OF_WITNESSES_STRUCT=np.nan, GEIPAN_WITNESS_TXT=T.witness_count_fr(det),
            GEIPAN_CLASS=r["Classification"], GEIPAN_PHENOMENON=r["Phénomène"],
            GEIPAN_IDENTIFICATION=r["Identification"], GEIPAN_CASE_TYPE=r["Type de cas"],
            EXPLANATION_STATUS_SOURCE=GEIPAN_STATUS.get(r["Classification"]),
            TEXT_LANG="fr", SOURCE_URL=url, RETRIEVAL_DATE=rdate,
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- common derived fields
def add_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["ORIGINAL_TEXT"].fillna("")
    lang = df.get("TEXT_LANG", pd.Series("en", index=df.index)).fillna("en")
    obs = pd.DataFrame([T.observer_flags_fr(s) if l == "fr" else T.observer_flags(s) for s, l in zip(txt, lang)], index=df.index)
    des = pd.DataFrame([T.desc_flags_fr(s) if l == "fr" else T.desc_flags(s) for s, l in zip(txt, lang)], index=df.index)
    df = pd.concat([df, obs, des], axis=1)
    df["COLOR"] = [T.colors_txt(s) for s in txt]
    df["DIRECTION_OF_TRAVEL"] = [T.direction_txt(s) for s in txt]
    wt = pd.Series([T.witness_count_txt(s) if l != "fr" else np.nan for s, l in zip(txt, lang)], index=df.index)
    if "GEIPAN_WITNESS_TXT" in df:
        wt = wt.fillna(df["GEIPAN_WITNESS_TXT"])
    if "NUMBER_OF_WITNESSES_STRUCT" not in df:
        df["NUMBER_OF_WITNESSES_STRUCT"] = np.nan
    df["NUMBER_OF_WITNESSES"] = df["NUMBER_OF_WITNESSES_STRUCT"].fillna(wt)
    df["NUMBER_OF_WITNESSES_BASIS"] = np.where(df["NUMBER_OF_WITNESSES_STRUCT"].notna(), "structured",
                                              np.where(wt.notna(), "text_lower_bound", "UNKNOWN"))
    # Witness type
    hatch_attr = df.get("HATCH_ATTRS", pd.Series("", index=df.index)).fillna("")
    hmil = df.get("HATCH_MIL_OBSERVER", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    types = []
    for i in df.index:
        t = []
        if df.at[i, "obs_pilot_txt"] or "AIR" in hatch_attr[i].split(";"):
            t.append("pilot_or_aircrew")
        if df.at[i, "obs_military_txt"] or hmil[i]:
            t.append("military")
        if df.at[i, "obs_police_txt"]:
            t.append("police")
        if df.at[i, "obs_astronomer_txt"]:
            t.append("astronomer")
        if df.at[i, "obs_atc_txt"]:
            t.append("air_traffic_control")
        if df.at[i, "obs_scientist_engineer_txt"] or "SCI" in hatch_attr[i].split(";"):
            t.append("scientist_engineer")
        if "HQO" in hatch_attr[i].split(";"):
            t.append("high_quality_observer_hatch")
        types.append(";".join(t) if t else "civilian_or_unspecified")
    df["WITNESS_TYPE"] = types
    has = lambda code: hatch_attr.str.split(";").map(lambda s: code in s)
    df["RADAR_INVOLVEMENT"] = np.where(has("RDR"), "YES_STRUCTURED", np.where(df["obs_radar_txt"], "YES_TEXT", "NOT_REPORTED"))
    if "NICAP_CATEGORY" in df:
        df.loc[df["NICAP_CATEGORY"].fillna("").str.contains("RADAR", case=False), "RADAR_INVOLVEMENT"] = "YES_STRUCTURED"
    img = df.get("HAS_IMAGES_STRUCT", pd.Series(None, index=df.index))
    df["PHOTOGRAPHIC_EVIDENCE"] = np.where(img.fillna(False).astype(bool) | has("PHT"), "YES_STRUCTURED",
                                           np.where(df["d_photo_word_txt"], "YES_TEXT", "NOT_REPORTED"))
    df["VIDEO_EVIDENCE"] = np.where(txt.str.contains(r"\bvideo|filmed|footage|movie film", case=False, regex=True), "YES_TEXT", "NOT_REPORTED")
    df["REPORTED_EM_EFFECTS"] = np.where(has("EME") | has("VEH"), "YES_STRUCTURED", np.where(df["d_em_effects_txt"], "YES_TEXT", "NOT_REPORTED"))
    df["REPORTED_PHYSIOLOGICAL_EFFECTS"] = np.where(has("HUM") | has("INJ"), "YES_STRUCTURED", np.where(df["d_physio_effects_txt"], "YES_TEXT", "NOT_REPORTED"))
    n_mod = ((df["RADAR_INVOLVEMENT"] != "NOT_REPORTED").astype(int) + (df["PHOTOGRAPHIC_EVIDENCE"] != "NOT_REPORTED").astype(int)
             + (df["VIDEO_EVIDENCE"] != "NOT_REPORTED").astype(int) + 1)  # +1 = visual
    df["N_SENSOR_MODALITIES"] = n_mod
    df["MULTI_SENSOR_EVIDENCE"] = np.where(n_mod >= 2, "YES", "NO_VISUAL_ONLY_AS_REPORTED")
    df["OBSERVATION_TYPE"] = np.where(df["RADAR_INVOLVEMENT"] != "NOT_REPORTED", "visual+radar_or_radar", "visual")
    motion = []
    for i in df.index:
        m = [k for k in ("hover", "fast", "erratic", "silent", "sound", "blinking") if df.at[i, f"d_{k}_txt"]]
        motion.append(";".join(m))
    df["MOTION_DESCRIPTION"] = motion
    df["ANGULAR_SIZE"] = np.nan
    df["ANGULAR_SIZE_STATUS"] = "DATA_UNAVAILABLE"
    df["ANGULAR_VELOCITY"] = np.nan
    df["ANGULAR_VELOCITY_STATUS"] = "NOT_ESTIMABLE"
    df["ELEVATION_AZIMUTH"] = None
    df["ELEVATION_AZIMUTH_STATUS"] = "DATA_UNAVAILABLE"
    if "SHAPE" not in df or df["SHAPE"].isna().all():
        df["SHAPE"] = "unknown"
    return df


def main():
    parts, decisions = [], []
    print("NUFORC ...", flush=True)
    parts.append(nuforc())
    print("HATCH ...", flush=True)
    parts.append(hatch())
    print("BLUEBOOK unknowns ...", flush=True)
    bb, dec = bluebook_unknowns()
    parts.append(bb)
    decisions += dec
    print("NICAP ...", flush=True)
    nc, dec = nicap()
    parts.append(nc)
    decisions += dec
    print("GEIPAN ...", flush=True)
    parts.append(geipan())
    df = pd.concat(parts, ignore_index=True)
    df = add_text_fields(df)
    df["SHAPE"] = df["SHAPE"].fillna("unknown")
    df["RAW_RECORD_HASH"] = [rhash(a, b, c) for a, b, c in zip(df.SOURCE, df.SOURCE_RECORD_ID, df.ORIGINAL_TEXT)]
    df.insert(0, "REPORT_ID", [f"R{i:07d}" for i in range(len(df))])
    # Parse helpers
    df["utc_ts"] = pd.to_datetime(df["UTC_TIME"], utc=True, errors="coerce")
    df["date_ts"] = pd.to_datetime(df["DATE"], errors="coerce")
    if "YEAR" not in df:
        df["YEAR"] = np.nan
    df["YEAR"] = df["YEAR"].fillna(df["date_ts"].dt.year)
    # Retrospective delay (NUFORC only)
    df["REPORT_DELAY_DAYS"] = (pd.to_datetime(df.get("POSTED_DATE"), errors="coerce") - df["date_ts"]).dt.days
    # Exclusions (kept in file, flagged)
    excl = pd.Series("", index=df.index)
    excl[df["YEAR"].isna()] = "no_year"
    excl[(df["YEAR"] < 1900) & (excl == "")] = "pre_1900"
    excl[(df["YEAR"] > 2024) & (excl == "")] = "future_or_bad_year"
    df["EXCLUDE_REASON"] = excl
    df.to_parquet(PROCESSED / "reports.parquet", index=False)
    pub = df.copy()
    pub.loc[pub.SOURCE == "NUFORC", "ORIGINAL_TEXT"] = None  # NUFORC ToS: no redistribution of text
    pub.to_parquet(PROCESSED / "reports_public.parquet", index=False)
    pd.DataFrame(decisions).to_csv(RESULTS / "geocoding_decisions.csv", index=False)
    df.loc[excl != "", ["REPORT_ID", "SOURCE", "SOURCE_RECORD_ID", "EXCLUDE_REASON"]].to_csv(RESULTS / "excluded_records.csv", index=False)
    print(df.groupby("SOURCE").agg(n=("REPORT_ID", "size"), geocoded=("LATITUDE", lambda s: s.notna().mean()),
                                   utc=("UTC_TIME", lambda s: s.notna().mean()),
                                   day=("DATE", lambda s: s.notna().mean())))
    print(df.EXCLUDE_REASON.value_counts())


if __name__ == "__main__":
    main()
