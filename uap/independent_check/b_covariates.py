"""Independent check B: covariate spot checks with my own code (no uap/src imports).

b1  Sun/Moon altitude + Moon illuminated fraction (skyfield topocentric) on 50 discovery CASE points
b2  ISS visibility (skyfield EarthSatellite, nearest-epoch TLE) on 20 x {0,1} discovery points 1999-2004
b3  Launch catalogue timing (GCAT) + window-count semantics recomputed on discovery points
b4  Kp 3-hour bin convention
b5  Storm Events local -> UTC conversion
b6  ISD-Lite hour convention / sky-cover code distribution
Outputs JSON/CSV under independent_check/out/.
"""
from __future__ import annotations

import gzip
import json
import re
import sys

import numpy as np
import pandas as pd

from ic_common import OUT, RAW, INTERIM, SEED, disc_points, to_sec, haversine_km, rss_mb

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ B1
def b1(n=50):
    from skyfield.api import load, wgs84
    from skyfield import almanac
    ts = load.timescale()
    eph = load(str(RAW / "astro" / "de421.bsp"))
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    p = disc_points(["utc_ts", "lat", "lon", "sun_alt", "moon_alt", "moon_illum"], strategies=["CASE"])
    p = p[p.utc_ts > pd.Timestamp("1900-01-02", tz="UTC")]
    s = p.sample(n=n, random_state=SEED).reset_index(drop=True)
    rows = []
    for r in s.itertuples():
        obs = earth + wgs84.latlon(r.lat, r.lon)
        res = {}
        for lab, tt in (("exact", r.utc_ts), ("floor10", r.utc_ts.floor("10min"))):
            t = ts.from_datetime(tt.to_pydatetime())
            a_s = obs.at(t).observe(sun).apparent().altaz()[0].degrees
            a_m = obs.at(t).observe(moon).apparent().altaz()[0].degrees
            # geocentric moon altitude (no parallax) for diagnosis
            res[lab] = (a_s, a_m, float(almanac.fraction_illuminated(eph, "moon", t)))
        rows.append(dict(EVENT_ID=r.EVENT_ID, utc_ts=r.utc_ts, lat=r.lat, lon=r.lon,
                         sun_alt_pipe=r.sun_alt, sun_alt_ind=res["exact"][0], sun_alt_ind_floor10=res["floor10"][0],
                         moon_alt_pipe=r.moon_alt, moon_alt_ind=res["exact"][1], moon_alt_ind_floor10=res["floor10"][1],
                         moon_illum_pipe=r.moon_illum, moon_illum_ind=res["exact"][2]))
    d = pd.DataFrame(rows)
    d["d_sun"] = d.sun_alt_pipe - d.sun_alt_ind
    d["d_sun_floor10"] = d.sun_alt_pipe - d.sun_alt_ind_floor10
    d["d_moon"] = d.moon_alt_pipe - d.moon_alt_ind
    d["d_moon_floor10"] = d.moon_alt_pipe - d.moon_alt_ind_floor10
    d["d_illum"] = d.moon_illum_pipe - d.moon_illum_ind
    d.to_csv(OUT / "b1_sun_moon.csv", index=False)
    summ = {}
    for c in ("d_sun", "d_sun_floor10", "d_moon", "d_moon_floor10", "d_illum"):
        v = d[c].abs()
        summ[c] = dict(max_abs=float(v.max()), median_abs=float(v.median()), mean=float(d[c].mean()),
                       frac_le_1p5=float((v <= 1.5).mean()))
    summ["n"] = len(d)
    json.dump(summ, open(OUT / "b1_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


# ------------------------------------------------------------------ B2
def _load_tles():
    lines = [l.rstrip("\n") for l in open(RAW / "space" / "iss_25544_tle_mcdowell.txt") if l.startswith(("1 ", "2 "))]
    pairs, i = [], 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            pairs.append((lines[i][:69].ljust(69), lines[i + 1][:69].ljust(69)))
            i += 2
        else:
            i += 1
    return pairs


def b2(n_each=20):
    from skyfield.api import load, wgs84, EarthSatellite
    ts = load.timescale()
    eph = load(str(RAW / "astro" / "de421.bsp"))
    earth, sun = eph["earth"], eph["sun"]
    pairs = _load_tles()
    sats = [EarthSatellite(a, b, "ISS", ts) for a, b in pairs]
    ep = np.array([s.model.jdsatepoch + s.model.jdsatepochF for s in sats])
    o = np.argsort(ep)
    sats = [sats[k] for k in o]
    ep = ep[o]
    p = disc_points(["utc_ts", "lat", "lon", "iss_visible_win", "sun_alt"])
    p = p[(p.utc_ts >= pd.Timestamp("1999-01-01", tz="UTC")) & (p.utc_ts < pd.Timestamp("2005-01-01", tz="UTC"))]
    ones = p[p.iss_visible_win == 1].sample(n=n_each, random_state=SEED)
    zeros_night = p[(p.iss_visible_win == 0) & (p.sun_alt < -4)].sample(n=n_each, random_state=SEED)
    zeros_any = p[(p.iss_visible_win == 0)].sample(n=10, random_state=SEED + 1)
    night_rand = p[(p.sun_alt < -4) & p.iss_visible_win.notna()].sample(n=400, random_state=SEED + 2)
    s = pd.concat([ones.assign(grp="pipe=1"), zeros_night.assign(grp="pipe=0 (night)"),
                   zeros_any.assign(grp="pipe=0 (any)"), night_rand.assign(grp="random night 400")]).reset_index(drop=True)
    rows = []
    offs = np.arange(-10, 11)
    for r in s.itertuples():
        jd0 = to_sec([r.utc_ts])[0] / 86400 + 2440587.5
        k = int(np.argmin(np.abs(ep - jd0)))
        sat = sats[k]
        obs_top = wgs84.latlon(r.lat, r.lon)
        tt = ts.from_datetimes([(r.utc_ts + pd.Timedelta(minutes=int(m))).to_pydatetime() for m in offs])
        alt = (sat - obs_top).at(tt).altaz()[0].degrees
        lit = sat.at(tt).is_sunlit(eph)
        salt = (earth + obs_top).at(tt).observe(sun).apparent().altaz()[0].degrees
        vis = (alt > 10) & lit & (salt < -4)
        rows.append(dict(EVENT_ID=r.EVENT_ID, strategy=r.strategy, utc_ts=r.utc_ts, lat=r.lat, lon=r.lon, grp=r.grp,
                         pipe=r.iss_visible_win, ind=float(vis.any()), tle_age_days=float(jd0 - ep[k]),
                         max_alt=float(alt.max()), n_vis_min=int(vis.sum()),
                         margin=float(np.max(np.minimum.reduce([alt - 10, np.where(lit, 99, -99), -4 - salt])))))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "b2_iss.csv", index=False)
    summ = {g: dict(n=len(x), agree=int((x["pipe"] == x["ind"]).sum()), n_pipe1=int((x["pipe"] == 1).sum()),
                    n_ind1=int((x["ind"] == 1).sum())) for g, x in d.groupby("grp")}
    summ["overall_agree_rate"] = float((d["pipe"] == d["ind"]).mean())
    summ["disagreements"] = d.loc[d["pipe"] != d["ind"], ["EVENT_ID", "strategy", "utc_ts", "grp", "pipe", "ind", "max_alt",
                                                   "n_vis_min", "margin", "tle_age_days"]].astype(str).to_dict("records")
    # Coverage note: TLE gaps
    gaps = np.diff(ep)
    big = np.where(gaps > 30)[0]
    summ["tle_coverage_gaps_gt30d"] = [(str(pd.Timestamp((ep[i] - 2440587.5) * 86400, unit="s")),
                                       str(pd.Timestamp((ep[i + 1] - 2440587.5) * 86400, unit="s"))) for i in big]
    json.dump(summ, open(OUT / "b2_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


# ------------------------------------------------------------------ B3 (own GCAT parser)
_MON = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
_RX = re.compile(r"^\s*(\d{4})\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{2})(\d{2})(?::(\d{2}))?")


def gcat_launches() -> pd.DataFrame:
    """Launches with a clock time and a site that has coordinates. Own parser."""
    L = pd.read_csv(RAW / "space" / "gcat_launch.tsv", sep="\t", dtype=str)
    L.columns = [c.lstrip("#").strip() for c in L.columns]
    L = L[~L["Launch_Tag"].fillna("#").str.startswith("#")]
    S = pd.read_csv(RAW / "space" / "gcat_sites.tsv", sep="\t", dtype=str)
    S.columns = [c.lstrip("#").strip() for c in S.columns]
    S = S[~S["Site"].fillna("#").str.startswith("#")]
    S["slat"] = pd.to_numeric(S["Latitude"].str.strip(), errors="coerce")
    S["slon"] = pd.to_numeric(S["Longitude"].str.strip(), errors="coerce")
    S = S.dropna(subset=["slat"])
    S["Site"] = S["Site"].str.strip()
    dup_sites = int(S.Site.duplicated().sum())
    S = S.drop_duplicates("Site")
    t = []
    for s in L["Launch_Date"].fillna(""):
        m = _RX.match(s)
        if not m or m.group(2) not in _MON:
            t.append(pd.NaT)
            continue
        t.append(pd.Timestamp(int(m.group(1)), _MON[m.group(2)], int(m.group(3)), int(m.group(4)), int(m.group(5)),
                              int(m.group(6) or 0), tz="UTC"))
    L["t"] = pd.DatetimeIndex(t)
    L["site"] = L["Launch_Site"].fillna("").str.strip()
    L = L.merge(S[["Site", "slat", "slon", "Name"]], left_on="site", right_on="Site", how="left")
    L.attrs["dup_sites"] = dup_sites
    return L


def window_count(pt_t, pt_lat, pt_lon, cat_t, cat_lat, cat_lon, lo, hi, radius):
    """# catalogue events with lo <= (cat_t - pt_t) <= hi and distance <= radius (vectorised by time-sorting)."""
    o = np.argsort(cat_t)
    cat_t, cat_lat, cat_lon = cat_t[o], cat_lat[o], cat_lon[o]
    j0 = np.searchsorted(cat_t, pt_t + lo, side="left")
    j1 = np.searchsorted(cat_t, pt_t + hi, side="right")
    n = np.zeros(len(pt_t))
    m = j1 - j0
    maxm = int(m.max()) if len(m) else 0
    for k in range(maxm):
        has = m > k
        idx = np.where(has)[0]
        jj = j0[idx] + k
        d = haversine_km(pt_lat[idx], pt_lon[idx], cat_lat[jj], cat_lon[jj])
        n[idx] += d <= radius
    n[~np.isfinite(pt_t)] = np.nan
    return n


def b3():
    L = gcat_launches()
    known = [
        ("Iridium-4 Falcon 9 VAFB (twilight 'LA UFO')", "2017-12-23 01:27", "V"),
        ("Minotaur V LADEE, Wallops", "2013-09-07 03:27", "WI"),
        ("Minotaur I ORS-3, Wallops", "2013-11-20 01:15", "WI"),
        ("STS-51L Challenger", "1986-01-28 16:38", "KSC"),
        ("Apollo 11", "1969-07-16 13:32", "KSC"),
        ("Falcon Heavy demo", "2018-02-06 20:45", "KSC"),
        ("Antares Orb-3 (failure)", "2014-10-28 22:22", "WI"),
    ]
    rows = []
    for name, when, site_hint in known:
        tt = pd.Timestamp(when, tz="UTC")
        m = L[(L.t - tt).abs() <= pd.Timedelta(minutes=5)]
        for r in m.itertuples():
            rows.append(dict(expected=name, expected_utc=when, gcat_tag=r.Launch_Tag, gcat_date=r.Launch_Date,
                             LV=r.LV_Type, mission=f"{r.Mission}".strip(), site=r.site, site_name=r.Name,
                             lat=r.slat, lon=r.slon, dt_min=(r.t - tt).total_seconds() / 60))
        if len(m) == 0:
            rows.append(dict(expected=name, expected_utc=when, gcat_tag="NOT FOUND"))
    known_df = pd.DataFrame(rows)
    known_df.to_csv(OUT / "b3_known_launches.csv", index=False)
    print(known_df.to_string())
    # pipeline interim launches vs my parse (catalogue-level, not holdout data)
    Lp = pd.read_parquet(INTERIM / "launches.parquet", columns=["Launch_Tag", "time", "time_has_clock", "lat", "lon", "site"])
    Lp = Lp[Lp.time_has_clock & Lp.lat.notna()]
    mine = L.dropna(subset=["t", "slat"])
    mm = mine.merge(Lp, on="Launch_Tag", how="outer", indicator=True)
    both = mm[mm._merge == "both"]
    cat_cmp = dict(n_mine=len(mine), n_pipeline=len(Lp), only_mine=int((mm._merge == "left_only").sum()),
                   only_pipeline=int((mm._merge == "right_only").sum()),
                   time_mismatch=int(((both.t - both.time).abs() > pd.Timedelta(seconds=1)).sum()),
                   coord_mismatch=int(((both.slat - both.lat).abs() > 1e-6).sum()), dup_site_codes=L.attrs["dup_sites"],
                   only_pipeline_examples=mm.loc[mm._merge == "right_only", ["Launch_Tag", "time", "site_y"]].head(5).astype(str).to_dict("records"),
                   only_mine_examples=mm.loc[mm._merge == "left_only", ["Launch_Tag", "t", "site_x"]].head(5).astype(str).to_dict("records"))
    print(cat_cmp)
    # Recompute n_launch_lpost3h_1500km for ALL discovery points with my parser + my window code
    p = disc_points(["utc_ts", "lat", "lon", "n_launch_lpost3h_1500km", "n_launch_lpost3h_500km"])
    t = to_sec(p.utc_ts)
    ct, cla, clo = to_sec(mine.t), mine.slat.to_numpy(float), mine.slon.to_numpy(float)
    H = 3600.0
    n1500 = window_count(t, p.lat.to_numpy(float), p.lon.to_numpy(float), ct, cla, clo, -3 * H, 0.5 * H, 1500)
    n500 = window_count(t, p.lat.to_numpy(float), p.lon.to_numpy(float), ct, cla, clo, -3 * H, 0.5 * H, 500)
    # semantic alternative (wrong sign) to show the column is not the "after" window
    n1500_flip = window_count(t, p.lat.to_numpy(float), p.lon.to_numpy(float), ct, cla, clo, -0.5 * H, 3 * H, 1500)
    pipe = p.n_launch_lpost3h_1500km.to_numpy(float)
    ok = np.isfinite(pipe)
    agree = dict(n_points=int(ok.sum()), exact_agree_1500=float((n1500[ok] == pipe[ok]).mean()),
                 exact_agree_500=float((n500[ok] == p.n_launch_lpost3h_500km.to_numpy(float)[ok]).mean()),
                 binary_agree_1500=float(((n1500[ok] > 0) == (pipe[ok] > 0)).mean()),
                 n_pipe_pos=int((pipe[ok] > 0).sum()), n_mine_pos=int((n1500[ok] > 0).sum()),
                 n_disagree=int((n1500[ok] != pipe[ok]).sum()),
                 flipped_window_binary_agree=float(((n1500_flip[ok] > 0) == (pipe[ok] > 0)).mean()))
    print(agree)
    dis = p[ok & (n1500 != pipe)].assign(mine=n1500[ok & (n1500 != pipe)])
    dis.head(20).to_csv(OUT / "b3_window_disagreements.csv", index=False)
    # hand check of 3 discovery CASE points with launches
    cases = p[(p.strategy == "CASE") & (pipe > 0)].sample(n=3, random_state=SEED)
    hand = []
    for r in cases.itertuples():
        dt_h = (mine.t - r.utc_ts).dt.total_seconds() / 3600
        dkm = haversine_km(r.lat, r.lon, mine.slat.to_numpy(), mine.slon.to_numpy())
        near = mine[(dt_h > -6) & (dt_h < 3) & (dkm < 2500)].copy()
        near["dt_h"] = dt_h[near.index]
        near["dist_km"] = dkm[near.index.to_numpy()] if False else haversine_km(r.lat, r.lon, near.slat, near.slon)
        for q in near.itertuples():
            hand.append(dict(EVENT_ID=r.EVENT_ID, point_utc=str(r.utc_ts), lat=r.lat, lon=r.lon,
                             pipe_n1500=r.n_launch_lpost3h_1500km, launch=q.Launch_Tag, LV=q.LV_Type,
                             launch_utc=str(q.t), site=q.site, dt_launch_minus_point_h=round(q.dt_h, 3),
                             dist_km=round(q.dist_km, 1),
                             counts=bool((-3 <= q.dt_h <= 0.5) and q.dist_km <= 1500)))
    hand = pd.DataFrame(hand)
    hand.to_csv(OUT / "b3_hand_check.csv", index=False)
    print(hand.to_string())
    json.dump(dict(catalog=cat_cmp, window=agree), open(OUT / "b3_summary.json", "w"), indent=1, default=str)
    print("rss MB", rss_mb())


# ------------------------------------------------------------------ B4
def b4(n=300):
    rows = []
    for line in open(RAW / "spaceweather" / "Kp_ap_Ap_SN_F107_since_1932.txt"):
        if line.startswith("#"):
            continue
        f = line.split()
        for i in range(8):
            rows.append((int(f[0]), int(f[1]), int(f[2]), i, float(f[7 + i])))
    K = pd.DataFrame(rows, columns=["y", "m", "d", "bin", "kp"])
    K["start"] = pd.to_datetime(dict(year=K.y, month=K.m, day=K.d)).dt.tz_localize("UTC") + pd.to_timedelta(3 * K.bin, unit="h")
    K = K.set_index("start")["kp"]
    p = disc_points(["utc_ts", "kp"]).sample(n=n, random_state=SEED)
    st = p.utc_ts.dt.floor("3h")
    mine = K.reindex(st).to_numpy()
    prev = K.reindex(st - pd.Timedelta(hours=3)).to_numpy()
    nxt = K.reindex(st + pd.Timedelta(hours=3)).to_numpy()
    pipe = p.kp.to_numpy(float)
    res = dict(n=n, agree_containing_bin=float(np.mean(np.abs(mine - pipe) < 1e-3)),
               agree_previous_bin=float(np.mean(np.abs(prev - pipe) < 1e-3)),
               agree_next_bin=float(np.mean(np.abs(nxt - pipe) < 1e-3)),
               examples=[dict(utc=str(a), pipe=float(b), mine=float(c)) for a, b, c in zip(p.utc_ts[:5], pipe[:5], mine[:5])])
    json.dump(res, open(OUT / "b4_kp.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ B5
def b5():
    f = RAW / "weather" / "storm_events" / "details_2005.csv.gz"
    d = pd.read_csv(f, usecols=["EVENT_ID", "BEGIN_DATE_TIME", "CZ_TIMEZONE", "EVENT_TYPE", "BEGIN_LAT", "BEGIN_LON",
                                "BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME", "STATE"], low_memory=False)
    tzvals = d.CZ_TIMEZONE.value_counts().to_dict()
    d = d[d.EVENT_TYPE.isin(["Lightning", "Thunderstorm Wind", "Hail", "Tornado"])].dropna(subset=["BEGIN_LAT"])
    pick = pd.concat([d[d.CZ_TIMEZONE.str.startswith(z)].sample(n=2, random_state=SEED)
                      for z in ("EST", "CST", "MST", "PST") if (d.CZ_TIMEZONE.str.startswith(z)).sum() >= 2])
    # by hand: NOAA Storm Events times are local *standard* time of CZ_TIMEZONE (e.g. 'CST-6' -> UTC = local + 6 h)
    off = {"EST": 5, "CST": 6, "MST": 7, "PST": 8}
    pick["utc_mine"] = [pd.Timestamp(year=int(str(ym)[:4]), month=int(str(ym)[4:]), day=int(dd), hour=int(bt) // 100,
                                     minute=int(bt) % 100, tz="UTC") + pd.Timedelta(hours=off[z[:3]])
                        for ym, dd, bt, z in zip(pick.BEGIN_YEARMONTH, pick.BEGIN_DAY, pick.BEGIN_TIME, pick.CZ_TIMEZONE)]
    S = pd.read_parquet(INTERIM / "storm_events.parquet")
    S = S[(S.time >= pd.Timestamp("2005-01-01", tz="UTC") - pd.Timedelta(days=1)) & (S.time < pd.Timestamp("2006-01-02", tz="UTC"))]
    out = []
    for r in pick.itertuples():
        m = S[(np.abs(S.lat - r.BEGIN_LAT) < 1e-4) & (np.abs(S.lon - r.BEGIN_LON) < 1e-4)]
        m = m.iloc[(m.time - r.utc_mine).abs().argsort()[:1]]
        out.append(dict(storm_event_id=r.EVENT_ID, state=r.STATE, begin_local=r.BEGIN_DATE_TIME, tz=r.CZ_TIMEZONE,
                         utc_mine=str(r.utc_mine), utc_pipeline=str(m.time.iloc[0]) if len(m) else "not found",
                         diff_h=(m.time.iloc[0] - r.utc_mine).total_seconds() / 3600 if len(m) else np.nan))
    out = pd.DataFrame(out)
    out.to_csv(OUT / "b5_storm.csv", index=False)
    print(out.to_string())
    # year range sanity of the interim file (2-digit year parsing)
    Sall = pd.read_parquet(INTERIM / "storm_events.parquet", columns=["time"])
    yr = Sall.time.dt.year
    res = dict(cz_timezone_values_2005=tzvals, interim_year_min=int(yr.min()), interim_year_max=int(yr.max()),
               n_after_2024=int((yr > 2024).sum()))
    # check across ALL files which timezone strings occur (DST-labelled ones?)
    allz = {}
    for fn in sorted((RAW / "weather" / "storm_events").glob("details_*.csv.gz")):
        z = pd.read_csv(fn, usecols=["CZ_TIMEZONE"]).CZ_TIMEZONE.value_counts()
        for k, v in z.items():
            allz[k] = allz.get(k, 0) + int(v)
    res["cz_timezone_values_all_files"] = allz
    json.dump(res, open(OUT / "b5_storm.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ B6
def b6():
    import glob
    fs = sorted(glob.glob(str(RAW / "weather" / "isd_lite" / "*" / "*.gz")))
    sel = [fs[i] for i in np.random.default_rng(SEED).choice(len(fs), size=min(60, len(fs)), replace=False)]
    codes = {}
    hours = {}
    minute_nonzero = 0
    for f in sel:
        a = np.loadtxt(gzip.open(f), dtype=float, ndmin=2)
        for v, c in zip(*np.unique(a[:, 9], return_counts=True)):
            codes[int(v)] = codes.get(int(v), 0) + int(c)
        for v, c in zip(*np.unique(a[:, 3], return_counts=True)):
            hours[int(v)] = hours.get(int(v), 0) + int(c)
    res = dict(n_files_total=len(fs), n_files_sampled=len(sel), sky_code_counts=dict(sorted(codes.items())),
               hour_counts=dict(sorted(hours.items())))
    # diurnal check of hour convention: temperature max hour at a few US stations (UTC ~20-23 if UTC)
    tmax_hours = []
    for f in sel[:30]:
        a = np.loadtxt(gzip.open(f), dtype=float, ndmin=2)
        a = a[a[:, 4] > -9999]
        if len(a) < 2000:
            continue
        df = pd.DataFrame(a[:, [3, 4]], columns=["h", "t"])
        tmax_hours.append(int(df.groupby("h").t.mean().idxmax()))
    res["mean_temp_peak_hour_by_station"] = tmax_hours
    st = pd.read_csv(RAW / "weather" / "selected_isd_stations.csv")
    res["stations_lon_range"] = [float(st.LON.min()), float(st.LON.max())]
    json.dump(res, open(OUT / "b6_isd.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        globals()[a]()
