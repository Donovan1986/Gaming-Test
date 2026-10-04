"""Report context -> known-object explanation scores -> quality score ->
duplicate clustering -> EVENT-level table -> frozen data splits.

Outputs
  data/interim/report_context.parquet  (context features per report)
  data/processed/reports_scored.parquet
  data/processed/events.parquet
  results/explanation_matcher_validation.csv (against GEIPAN labels & NUFORC notes)
  results/quality_score_formula.md
  results/splits_manifest.json  (split rule + hash of event ids per split)
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

import features as F
import spatial
import weather
import covariates as C
from common import PROCESSED, INTERIM, RESULTS, MASTER_SEED, haversine_km


# ---------------------------------------------------------------- context features
def report_context(r: pd.DataFrame) -> pd.DataFrame:
    p = INTERIM / "report_context.parquet"
    if p.exists():
        return pd.read_parquet(p)
    okc = r["LATITUDE"].between(-90, 90) & r["LONGITUDE"].between(-180, 180)
    ok = r["utc_ts"].notna() & okc
    t = F.to_sec(r["utc_ts"])
    t[~ok.to_numpy()] = np.nan
    la, lo = r["LATITUDE"].to_numpy(float), r["LONGITUDE"].to_numpy(float)
    parts = [F.astro_calendar_features(t, la, lo, r["TIMEZONE"].to_numpy()),
             F.launch_features(t, la, lo), F.fireball_features(t, la, lo)]
    st = weather.storm_events()
    parts.append(F.storm_event_features(t, la, lo, st, countries=r.COUNTRY.to_numpy()))
    # reentries of tracked objects (global, +-1 day; no location available)
    re_ = C.satcat_reentries()
    re_ = re_[(re_.mass >= 500)].rename(columns={"decay_time": "time"})
    re_["lat"] = 0.0; re_["lon"] = 0.0
    parts.append(F.window_counts(t, np.zeros(len(t)), np.zeros(len(t)), re_, {"reentry_pm1d": (-F.D, F.D)}, [0],
                                 need_loc=False))
    import iss
    iss_df = pd.DataFrame({"iss_visible_win": iss.iss_visible_window(t, la, lo, r["TIME_UNCERTAINTY_MIN"].to_numpy(float))})
    parts.append(iss_df)
    sp = pd.DataFrame(index=r.index)
    okc = okc.to_numpy()
    yr = r["YEAR"].fillna(2000).to_numpy()
    s = spatial.spatial_features(la[okc], lo[okc], yr[okc])
    for c in s.columns:
        sp.loc[okc, c] = s[c].to_numpy()
    ctx = pd.concat([x.reset_index(drop=True) for x in parts] + [sp.reset_index(drop=True)], axis=1)
    ctx.insert(0, "REPORT_ID", r["REPORT_ID"].to_numpy())
    ctx.to_parquet(p, index=False)
    return ctx


# ---------------------------------------------------------------- explanation matcher
def _comb(*ps):
    q = np.ones_like(np.asarray(ps[0], float))
    for p in ps:
        q *= 1 - np.clip(np.nan_to_num(np.asarray(p, float)), 0, 1)
    return 1 - q


def explanation_scores(r: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Heuristic, transparent known-object confidence scores in [0,1].
    'ctx' components use only time/place context (language independent);
    'txt' components use descriptive text. Both are reported separately."""
    f = lambda c: r[c].fillna(False).astype(bool).to_numpy() if c in r else np.zeros(len(r), bool)
    g = lambda c: x[c].to_numpy(float) if c in x else np.full(len(r), np.nan)
    dur = r["OBSERVATION_DURATION_S"].to_numpy(float)
    sun = g("sun_alt")
    night = sun < -6
    twilight = (sun < -3) & (sun > -20)
    year = r["YEAR"].to_numpy(float)
    mmdd = x["mmdd"].fillna("").to_numpy() if "mmdd" in x else np.array([""] * len(r))
    dow = g("dow")
    hover = f("d_hover_txt")
    out = {}
    # meteor / fireball
    fb = np.nan_to_num(g("n_fb_pm30m_1000km")) > 0
    shower = (np.nan_to_num(g("shower_zhr_total")) >= 50) & night
    outburst = np.nan_to_num(g("meteor_outburst_pm1d")) > 0
    short = np.nan_to_num(dur, nan=1e9) <= 10
    desc_met = f("d_meteor_word_txt") | (r["SHAPE"].to_numpy() == "fireball") | f("d_fireball_shape_txt")
    out["ctx_meteor"] = _comb(0.85 * fb, 0.25 * (shower & (short | desc_met)), 0.3 * (outburst & (short | desc_met)))
    out["txt_meteor"] = _comb(0.35 * desc_met, 0.30 * short, 0.25 * (desc_met & short), 0.9 * f("d_nuforc_note_meteor_txt"))
    # Starlink
    era = np.nan_to_num(g("starlink_era")) > 0
    dss = g("days_since_starlink_launch")
    line = f("d_line_formation_txt")
    sl_ctx = era * (0.45 * (np.nan_to_num(dss, nan=99) <= 10) + 0.15 * (np.nan_to_num(dss, nan=99) <= 30) + 0.15 * twilight)
    out["ctx_starlink"] = np.clip(sl_ctx * line, 0, 1) + 0 * sl_ctx
    out["txt_starlink"] = _comb(0.9 * f("d_starlink_word_txt"), 0.95 * f("d_nuforc_note_starlink_txt"),
                                0.35 * (line & era & ~hover))
    # rocket launch
    l15 = np.nan_to_num(g("n_launch_lpost3h_1500km")) > 0
    l25 = np.nan_to_num(g("n_launchorb_lpost3h_2500km")) > 0
    l10 = np.nan_to_num(g("n_launch_lpost1h_1000km")) > 0
    out["ctx_launch"] = _comb(0.45 * l15, 0.35 * (l15 & twilight), 0.2 * (l25 & twilight), 0.15 * l10)
    out["txt_launch"] = _comb(0.25 * (f("d_rocket_word_txt") & l15), 0.9 * f("d_nuforc_note_launch_txt"))
    # satellites / ISS (no historical TLEs: text-driven, weak)
    sat_like = (np.nan_to_num(dur, nan=0) >= 60) & (np.nan_to_num(dur, nan=1e9) <= 420) & ~f("d_blinking_txt") & ~f("d_sound_txt") & ~hover
    issv = np.nan_to_num(g("iss_visible_win")) > 0
    out["ctx_satellite"] = _comb(0.10 * (sat_like & (sun < -6) & (sun > -30)), 0.55 * (issv & sat_like), 0.25 * (issv & ~sat_like))
    out["txt_satellite"] = _comb(0.55 * f("d_satellite_word_txt"), 0.6 * f("d_iss_word_txt"),
                                 0.9 * f("d_nuforc_note_satellite_txt"), 0.9 * f("d_nuforc_note_iss_txt"))
    # planets / bright stars
    venus_vis = (g("venus_alt") > 3) & (sun < -3) & (g("venus_elong") > 15)
    jup_vis = (g("jupiter_alt") > 10) & (sun < -8)
    long_ = np.nan_to_num(dur, nan=0) >= 600
    bright_static = hover | f("d_planet_word_txt")
    out["ctx_planet"] = _comb(0.5 * (venus_vis & bright_static & long_), 0.25 * (jup_vis & bright_static & long_),
                              0.15 * (venus_vis & (long_ | bright_static)), 0.08 * (jup_vis & (long_ | bright_static)))
    out["txt_planet"] = _comb(0.4 * f("d_planet_word_txt"), 0.9 * f("d_nuforc_note_planet_txt"))
    out["txt_moon"] = 0.3 * (f("d_moon_word_txt") & (g("moon_alt") > 0))
    # aircraft / helicopters
    near_ap = np.nan_to_num(g("dist_large_airport_km"), nan=1e9) <= 25
    out["ctx_aircraft"] = 0.08 * near_ap
    out["txt_aircraft"] = _comb(0.55 * (f("d_blinking_txt") & f("d_red_green_txt")), 0.35 * f("d_aircraft_word_txt"),
                                0.15 * f("d_sound_txt"), 0.10 * (near_ap & f("d_blinking_txt")),
                                0.9 * f("d_nuforc_note_aircraft_txt"))
    out["txt_drone"] = 0.6 * (f("d_drone_word_txt") & (year >= 2010))
    out["txt_balloon"] = 0.6 * f("d_balloon_word_txt")
    holiday = np.isin(mmdd, ["07-04", "12-31", "01-01", "07-05", "07-14"])
    weekend_night = np.isin(dow, [4, 5]) & night
    orange_soft = f("d_orange_txt") & (f("d_silent_txt") | hover | f("d_lantern_word_txt"))
    out["ctx_lantern"] = _comb(0.30 * (f("d_orange_txt") & holiday), 0.10 * (f("d_orange_txt") & weekend_night))
    out["txt_lantern"] = _comb(0.7 * f("d_lantern_word_txt"), 0.35 * orange_soft, 0.9 * f("d_nuforc_note_lantern_txt"))
    out["txt_fireworks"] = _comb(0.6 * f("d_firework_word_txt"), 0.2 * (holiday & f("d_blinking_txt")))
    storm = np.nan_to_num(g("n_storm_pm1h_100km")) > 0
    out["ctx_lightning"] = 0.25 * (storm & (short | f("d_blinking_txt")))
    out["txt_lightning"] = 0.3 * f("d_lightning_word_txt")
    out["txt_searchlight"] = 0.6 * f("d_searchlight_word_txt")
    out["txt_flare"] = 0.5 * f("d_flare_word_txt")
    out["txt_cloud"] = 0.2 * (f("d_cloud_word_txt") & (sun > 0))
    reentry = np.nan_to_num(g("n_reentry_pm1d")) > 0
    out["ctx_reentry"] = 0.15 * (reentry & (desc_met | f("d_line_formation_txt")) & ~short)
    hatch_attr = r.get("HATCH_ATTRS", pd.Series("", index=r.index)).fillna("")
    out["src_hoax"] = _comb(0.9 * f("d_nuforc_note_hoax_txt"), 0.6 * hatch_attr.str.contains("HOX").to_numpy())
    out["src_misid"] = 0.6 * hatch_attr.str.contains("MID").to_numpy()
    df = pd.DataFrame(out, index=r.index)
    cats = sorted({c.split("_", 1)[1] for c in df.columns})
    for c in cats:
        cols = [k for k in df.columns if k.split("_", 1)[1] == c]
        df[f"P_{c}"] = _comb(*[df[k] for k in cols])
    pcols = [f"P_{c}" for c in cats]
    df["P_EXPLAINED"] = _comb(*[df[c] for c in pcols])
    df["P_EXPLAINED_CTX_ONLY"] = _comb(*[df[c] for c in df.columns if c.startswith("ctx_")])
    df["TOP_EXPLANATION"] = df[pcols].idxmax(axis=1).str[2:]
    df.loc[df[pcols].max(axis=1) < 0.05, "TOP_EXPLANATION"] = "none"
    return df


def known_object_label(p):
    return np.where(p >= 0.6, "LIKELY_EXPLAINED", np.where(p >= 0.3, "AMBIGUOUS", "UNEXPLAINED_BY_MATCHER"))


# ---------------------------------------------------------------- quality score
QUALITY_DOC = r"""
# CASE QUALITY SCORE (transparent, additive; clipped to [0, 10])

INFORMATION_QUALITY = clip(3.0 + Q_time + Q_date + Q_loc + Q_dur + Q_wit + Q_trained + Q_sensor
                           + Q_contemp + Q_source + Q_hoax, 0, 10)
CASE_QUALITY_SCORE  = clip(INFORMATION_QUALITY - 4.0 * P_EXPLAINED, 0, 10)
EVENT_QUALITY       = max(report INFORMATION_QUALITY in cluster) + 1.0 if the cluster has >= 2 reports
                      from locations >= 10 km apart within 60 min (simultaneous separated witnesses)

| component  | rule |
|------------|------|
| Q_time     | +1.0 if time uncertainty <= 15 min; +0.5 if <= 30 min; 0 if <= 60; -1.0 if no time / >= 720 min |
| Q_date     | 0 day precision; -1 month; -2 year; additional -0.5 if marked approximate |
| Q_loc      | +1.0 radius <= 10 km; +0.5 <= 25 km; 0 <= 50 km; -2.0 unresolved/region only |
| Q_dur      | +1.0 if 60 s <= duration <= 3600 s; +0.5 if 10-60 s or 1-3 h; 0 if unknown or < 10 s; -0.5 if > 3 h |
| Q_wit      | structured count n: +min(1.5, 0.75*log2(n)); text lower bound: half of that |
| Q_trained  | +1.5 pilot/aircrew, ATC or astronomer; +1.0 police, military, scientist/engineer, Hatch HQO; capped at 2.0 |
| Q_sensor   | radar +2.0 (structured) / +1.0 (text); photo or video +1.0 (structured) / +0.5 (text); capped at 2.5 |
| Q_contemp  | NUFORC: +1 if posted <= 30 d after event; 0 if <= 1 y; -1 if 1-10 y; -2 if > 10 y. Blue Book (contemporaneous official file) +1; GEIPAN (official investigation) +1 |
| Q_source   | Hatch / NICAP secondary compilations -0.5; Blue Book / GEIPAN official files +0.5; NUFORC 0 |
| Q_hoax     | -3.0 if NUFORC hoax note or Hatch HOX attribute |

Narrative strangeness (e.g. Hatch 'Strangeness', occupants, abductions) is NEVER used.
HIGH_QUALITY: INFORMATION_QUALITY >= 7 (sensitivity: 6, 8).
HIGH_QUALITY_UNEXPLAINED: HIGH_QUALITY and P_EXPLAINED < 0.3 and not GEIPAN class A/B.
MULTI_SENSOR: >= 2 sensor modalities reported (visual + radar/photo/video).
"""


def quality(r: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    tu = r["TIME_UNCERTAINTY_MIN"].to_numpy(float)
    q_time = np.select([tu <= 15, tu <= 30, tu <= 60], [1.0, 0.5, 0.0], -1.0)
    dp = r["DATE_PRECISION"].fillna("none").astype(str)
    q_date = np.select([dp.str.startswith("day"), dp.str.startswith("month"), dp.str.startswith("year")], [0, -1, -2], -2.5) \
        - 0.5 * dp.str.contains("approx").to_numpy()
    rad = r["LOCATION_UNCERTAINTY_RADIUS_KM"].to_numpy(float)
    q_loc = np.select([rad <= 10, rad <= 25, rad <= 50], [1.0, 0.5, 0.0], -2.0)
    dur = r["OBSERVATION_DURATION_S"].to_numpy(float)
    q_dur = np.select([(dur >= 60) & (dur <= 3600), ((dur >= 10) & (dur < 60)) | ((dur > 3600) & (dur <= 10800)), dur > 10800],
                      [1.0, 0.5, -0.5], 0.0)
    n = r["NUMBER_OF_WITNESSES"].to_numpy(float)
    base = np.minimum(1.5, 0.75 * np.log2(np.maximum(n, 1)))
    q_wit = np.where(r["NUMBER_OF_WITNESSES_BASIS"] == "structured", base,
                     np.where(r["NUMBER_OF_WITNESSES_BASIS"] == "text_lower_bound", base / 2, 0))
    wt = r["WITNESS_TYPE"].fillna("")
    q_tr = np.minimum(2.0, 1.5 * wt.str.contains("pilot|air_traffic|astronomer").to_numpy()
                      + 1.0 * wt.str.contains("police|military|scientist|high_quality").to_numpy())
    rd = r["RADAR_INVOLVEMENT"]
    ph = r["PHOTOGRAPHIC_EVIDENCE"]
    vd = r["VIDEO_EVIDENCE"]
    q_sen = np.minimum(2.5, np.where(rd == "YES_STRUCTURED", 2.0, np.where(rd == "YES_TEXT", 1.0, 0))
                       + np.where((ph == "YES_STRUCTURED"), 1.0, np.where((ph == "YES_TEXT") | (vd == "YES_TEXT"), 0.5, 0)))
    delay = r["REPORT_DELAY_DAYS"].to_numpy(float)
    src = r["SOURCE"].to_numpy()
    q_con = np.where(src == "NUFORC", np.select([delay <= 30, delay <= 365, delay <= 3650], [1, 0, -1], -2), 0)
    q_con = np.where(np.isin(src, ["BLUEBOOK_UNK", "GEIPAN"]), 1, q_con)
    q_src = np.select([np.isin(src, ["HATCH", "NICAP"]), np.isin(src, ["BLUEBOOK_UNK", "GEIPAN"])], [-0.5, 0.5], 0)
    q_hoax = -3.0 * (e["src_hoax"].to_numpy() >= 0.5)
    info = np.clip(3.0 + q_time + q_date + q_loc + q_dur + q_wit + q_tr + q_sen + q_con + q_src + q_hoax, 0, 10)
    out = pd.DataFrame(dict(Q_time=q_time, Q_date=q_date, Q_loc=q_loc, Q_dur=q_dur, Q_wit=q_wit, Q_trained=q_tr,
                            Q_sensor=q_sen, Q_contemp=q_con, Q_source=q_src, Q_hoax=q_hoax,
                            INFORMATION_QUALITY=info), index=r.index)
    out["CASE_QUALITY_SCORE"] = np.clip(info - 4.0 * e["P_EXPLAINED"].to_numpy(), 0, 10)
    return out


# ---------------------------------------------------------------- clustering
class DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def cluster(r: pd.DataFrame, dt_strict=3600, d_strict=50, dt_reg=7200, d_reg=400):
    """Union-find clustering. Strict = likely same event; regional = same widely
    visible stimulus (e.g. launch plume, fireball). Date-only records match on
    same local date and distance <= max(100 km, r1+r2)."""
    n = len(r)
    strict, reg = DSU(n), DSU(n)
    t = F.to_sec(r["utc_ts"])
    date = r["DATE"].fillna("").to_numpy()
    la, lo = r["LATITUDE"].to_numpy(float), r["LONGITUDE"].to_numpy(float)
    rad = np.nan_to_num(r["LOCATION_UNCERTAINTY_RADIUS_KM"].to_numpy(float), nan=50)
    tu = np.nan_to_num(r["TIME_UNCERTAINTY_MIN"].to_numpy(float), nan=720) * 60
    shape = r["SHAPE"].to_numpy()
    df = pd.DataFrame({"i": np.arange(n), "date": date})
    df = df[(df.date != "") & np.isfinite(la)]
    dnum = pd.to_datetime(df.date).values.astype("datetime64[D]").astype(int)
    df["d"] = dnum
    groups = {k: v.i.to_numpy() for k, v in df.groupby("d")}
    pairs_s = pairs_r = 0
    for k, idx in groups.items():
        nxt = groups.get(k + 1, np.array([], int))
        cand = np.concatenate([idx, nxt])
        if len(cand) < 2:
            continue
        A = idx[:, None]
        B = cand[None, :]
        dd = haversine_km(la[A], lo[A], la[B], lo[B])
        bothT = np.isfinite(t[A]) & np.isfinite(t[B])
        dtt = np.abs(t[A] - t[B])
        tol = np.minimum(np.maximum(dt_strict, tu[A] + tu[B]), 3 * 3600)
        same_date = date[A] == date[B]
        cond_s = np.where(bothT, (dtt <= tol) & (dd <= np.maximum(d_strict, rad[A] + rad[B])),
                          same_date & (dd <= np.maximum(100, rad[A] + rad[B])))
        cond_r = np.where(bothT, (dtt <= np.maximum(dt_reg, tol)) & (dd <= d_reg), same_date & (dd <= d_reg))
        ia, ib = np.nonzero(cond_s & (A != B))
        for a, b in zip(idx[ia], cand[ib]):
            strict.union(a, b); pairs_s += 1
        ia, ib = np.nonzero(cond_r & (A != B))
        for a, b in zip(idx[ia], cand[ib]):
            reg.union(a, b); pairs_r += 1
    cs = np.array([strict.find(i) for i in range(n)])
    cr = np.array([reg.find(i) for i in range(n)])
    print(f"strict pairs {pairs_s}, regional pairs {pairs_r}")
    return cs, cr


# ---------------------------------------------------------------- splits
def split_label(event_id: str, seed=MASTER_SEED) -> float:
    h = hashlib.sha256(f"{seed}:{event_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def main():
    r = pd.read_parquet(PROCESSED / "reports.parquet")
    r.loc[r.EXCLUDE_REASON == "future_or_bad_year", "EXCLUDE_REASON"] = np.where(
        r.loc[r.EXCLUDE_REASON == "future_or_bad_year", "YEAR"] <= 2026, "", "future_or_bad_year")
    r = r[r.EXCLUDE_REASON == ""].reset_index(drop=True)
    bad = r.DATE.notna() & pd.to_datetime(r.DATE, errors="coerce", format="%Y-%m-%d").isna()
    print("impossible calendar dates downgraded to month precision:", int(bad.sum()))
    r.loc[bad, ["DATE", "UTC_TIME", "LOCAL_TIME"]] = None
    r.loc[bad, "DATE_PRECISION"] = "month"
    r.loc[bad, "utc_ts"] = pd.NaT
    print("reports kept", len(r), flush=True)
    x = report_context(r)
    assert (x.REPORT_ID.values == r.REPORT_ID.values).all()
    e = explanation_scores(r, x)
    q = quality(r, e)
    rs = pd.concat([r, e, q], axis=1)
    rs["KNOWN_OBJECT_STATUS"] = known_object_label(rs["P_EXPLAINED"].to_numpy())
    rs["KNOWN_OBJECT_MATCHES"] = [";".join(f"{c[2:]}:{v:.2f}" for c, v in row.items() if v >= 0.3)
                                  for _, row in e[[c for c in e.columns if c.startswith("P_") and c not in
                                                   ("P_EXPLAINED", "P_EXPLAINED_CTX_ONLY")]].iterrows()]
    st = rs.get("EXPLANATION_STATUS_SOURCE").fillna("")
    rs["EXPLANATION_STATUS"] = np.where(st != "", st + "|" + rs["KNOWN_OBJECT_STATUS"], rs["KNOWN_OBJECT_STATUS"])
    # clusters
    cs, cr = cluster(rs)
    rs["LIKELY_SAME_EVENT_CLUSTER"] = ["E" + rs.REPORT_ID.iloc[c][1:] for c in cs]
    rs["REGIONAL_CLUSTER"] = ["G" + rs.REPORT_ID.iloc[c][1:] for c in cr]
    g = rs.groupby("LIKELY_SAME_EVENT_CLUSTER")
    rs["REPORT_COUNT"] = g["REPORT_ID"].transform("size")
    rs["INDEPENDENT_SOURCE_COUNT"] = g["SOURCE"].transform("nunique")
    # simultaneous geographically separated witnesses
    sep = g.apply(lambda d: (len(d) >= 2) and (haversine_km(d.LATITUDE.values[:, None], d.LONGITUDE.values[:, None],
                                                            d.LATITUDE.values[None, :], d.LONGITUDE.values[None, :]).max() >= 10),
                  include_groups=False)
    rs["SEPARATED_WITNESSES"] = rs["LIKELY_SAME_EVENT_CLUSTER"].map(sep).fillna(False)
    rs["EVENT_QUALITY"] = g["INFORMATION_QUALITY"].transform("max") + 1.0 * rs["SEPARATED_WITNESSES"]
    rs["EVENT_QUALITY"] = rs["EVENT_QUALITY"].clip(0, 10)
    rs.to_parquet(PROCESSED / "reports_scored.parquet", index=False)
    # event table: representative report = highest INFORMATION_QUALITY (ties: earliest)
    rs = rs.sort_values(["LIKELY_SAME_EVENT_CLUSTER", "INFORMATION_QUALITY", "utc_ts"], ascending=[True, False, True])
    agg = rs.groupby("LIKELY_SAME_EVENT_CLUSTER").agg(
        SOURCES=("SOURCE", lambda s: ";".join(sorted(set(s)))),
        P_EXPLAINED_MIN=("P_EXPLAINED", "min"), P_EXPLAINED_MEAN=("P_EXPLAINED", "mean"),
        ANY_GEIPAN_D=("GEIPAN_CLASS", lambda s: (s == "D").any()))
    ev = rs.drop_duplicates("LIKELY_SAME_EVENT_CLUSTER").set_index("LIKELY_SAME_EVENT_CLUSTER").join(agg).reset_index()
    ev = ev.rename(columns={"LIKELY_SAME_EVENT_CLUSTER": "EVENT_ID"})
    ev["HIGH_QUALITY"] = ev["EVENT_QUALITY"] >= 7
    # Investigator judgement supersedes the heuristic matcher where it exists (GEIPAN): D = unexplained.
    is_g = ev["SOURCE"] == "GEIPAN"
    ev["UNEXPLAINED"] = np.where(is_g, ev["GEIPAN_CLASS"] == "D", ev["P_EXPLAINED"] < 0.3)
    ev["HQ_UNEXPLAINED"] = ev["HIGH_QUALITY"] & ev["UNEXPLAINED"]
    ev["MULTI_SENSOR_RADAR"] = ev["RADAR_INVOLVEMENT"] != "NOT_REPORTED"
    ev["MULTI_SENSOR"] = ev["N_SENSOR_MODALITIES"] >= 2
    # ---- frozen splits (decided before any outcome analysis)
    u = ev["EVENT_ID"].map(split_label)
    yr = ev["YEAR"].to_numpy()
    nuforc_like = ev["SOURCE"] == "NUFORC"
    split = np.where(u < 0.6, "discovery", np.where(u < 0.8, "validation", "holdout_random"))
    split = np.where(nuforc_like & (yr >= 2016), "holdout_temporal", split)
    ev["SPLIT"] = split
    ev["GEO_REGION"] = np.where(ev["COUNTRY"].isin(["US", "USA"]), "north_america",
                                np.where(ev["COUNTRY"].isin(["CA", "Canada"]), "north_america",
                                         np.where(ev["SOURCE"] == "GEIPAN", "europe_fr", "other")))
    eu = ev["LATITUDE"].between(35, 72) & ev["LONGITUDE"].between(-11, 30)
    ev.loc[eu & (ev.GEO_REGION == "other"), "GEO_REGION"] = "europe_other"
    ev.to_parquet(PROCESSED / "events.parquet", index=False)
    man = {"rule": "u=sha256(f'{MASTER_SEED}:{EVENT_ID}')[:8]/0xFFFFFFFF; u<0.6 discovery; <0.8 validation; else holdout_random; "
                   "NUFORC events with YEAR>=2016 -> holdout_temporal (overrides). Geographic holdout = GEO_REGION europe_*.",
           "master_seed": MASTER_SEED, "frozen_utc": pd.Timestamp.now("UTC").isoformat()}
    for s in sorted(set(split)):
        ids = sorted(ev.loc[ev.SPLIT == s, "EVENT_ID"])
        man[s] = {"n": len(ids), "sha256_ids": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    json.dump(man, open(RESULTS / "splits_manifest.json", "w"), indent=1)
    open(RESULTS / "quality_score_formula.md", "w").write(QUALITY_DOC)
    print(ev.groupby("SOURCE").agg(events=("EVENT_ID", "size"), hq=("HIGH_QUALITY", "mean"),
                                   hqu=("HQ_UNEXPLAINED", "sum"), multi=("MULTI_SENSOR", "sum")))
    print(ev.SPLIT.value_counts())
    print("reports", len(rs), "events", len(ev))


if __name__ == "__main__":
    main()
