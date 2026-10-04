"""Reporting-propensity analyses: do report waves follow media events?

1. Event study around each curated media event: NUFORC reports by POSTED
   date and by EVENT date in [-30, +60] days; share of posted reports whose
   event happened >= 1 year earlier (retrospective) before vs after.
   An increase that is mostly retrospective indicates reporting propensity,
   not increased phenomenon frequency.
2. 2015-2023 daily counts vs Wikipedia 'Unidentified flying object' pageviews
   (log-log Poisson with weekday + year-month FE): attention elasticity.
3. Composition: share of HIGH_QUALITY / UNEXPLAINED events within 30 d after
   media events vs other days (case-crossover handled in discovery_tests).
Uses all NUFORC years because media events are positive-control-like
'reporting-bias' checks; the outputs are descriptive and are not candidate
discoveries. Exposure-time denominators: US population (World Bank) and
annual night hours.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

import covariates as C
import stats as S
from common import PROCESSED, RESULTS, RAW


def main():
    r = pd.read_parquet(PROCESSED / "reports_scored.parquet",
                        columns=["SOURCE", "DATE", "POSTED_DATE", "REPORT_DELAY_DAYS", "COUNTRY", "INFORMATION_QUALITY", "P_EXPLAINED"])
    r = r[(r.SOURCE == "NUFORC") & (r.COUNTRY == "US")]
    r["posted"] = pd.to_datetime(r.POSTED_DATE, errors="coerce")
    r["event"] = pd.to_datetime(r.DATE, errors="coerce")
    rows = []
    for d, label in C.MEDIA_EVENTS:
        d = pd.Timestamp(d)
        if d.year < 1996:
            continue
        pw = r[(r.posted >= d - pd.Timedelta(days=30)) & (r.posted < d + pd.Timedelta(days=60))]
        pre = pw[pw.posted < d]
        post = pw[(pw.posted >= d) & (pw.posted < d + pd.Timedelta(days=30))]
        ew = r[(r.event >= d - pd.Timedelta(days=30)) & (r.event < d + pd.Timedelta(days=30))]
        rows.append(dict(date=d.date(), event=label,
                         posted_per_day_pre30=len(pre) / 30, posted_per_day_post30=len(post) / 30,
                         retro_share_pre=(pre.REPORT_DELAY_DAYS >= 365).mean(), retro_share_post=(post.REPORT_DELAY_DAYS >= 365).mean(),
                         events_per_day_pre30=(ew.event < d).sum() / 30, events_per_day_post30=(ew.event >= d).sum() / 30,
                         hq_share_pre=(ew[ew.event < d].INFORMATION_QUALITY >= 7).mean(),
                         hq_share_post=(ew[ew.event >= d].INFORMATION_QUALITY >= 7).mean()))
    es = pd.DataFrame(rows)
    es["posted_ratio"] = es.posted_per_day_post30 / es.posted_per_day_pre30
    es["event_ratio"] = es.events_per_day_post30 / es.events_per_day_pre30
    print(es.round(3).to_string())
    es.to_csv(RESULTS / "media_event_study.csv", index=False)
    # Wikipedia attention elasticity
    w = C.wiki_pageviews()
    w = w[w.article == "Unidentified_flying_object"].set_index("date").views
    daily = r.dropna(subset=["event"]).groupby("event").size()
    idx = pd.date_range("2015-07-01", "2023-05-31", freq="D")
    df = pd.DataFrame({"n": daily.reindex(idx, fill_value=0), "views": w.reindex(idx)}).dropna()
    df["lv"] = np.log(df.views)
    df["ym"] = df.index.strftime("%Y-%m")
    df["dow"] = df.index.dayofweek.astype(str)
    X = sm.add_constant(pd.concat([df[["lv"]], pd.get_dummies(df[["ym", "dow"]], drop_first=True).astype(float)], axis=1))
    m = sm.GLM(df.n, X, family=sm.families.Poisson()).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    el = dict(elasticity=m.params["lv"], ci_low=m.conf_int().loc["lv", 0], ci_high=m.conf_int().loc["lv", 1],
              p=m.pvalues["lv"], n_days=len(df))
    print("attention elasticity", el)
    S.register(family="MEDIA_attention", phase="descriptive", hypothesis="daily NUFORC event counts vs log Wikipedia UFO pageviews (2015-2023)",
               variable_a="log_wiki_views", model="poisson_HAC_ym_dow_FE", subset="ALL", split="all_years",
               n_cases=int(df.n.sum()), effect_measure="elasticity", effect=el["elasticity"], ci_low=el["ci_low"],
               ci_high=el["ci_high"], p_value=el["p"])
    pd.DataFrame([el]).to_csv(RESULTS / "media_attention_elasticity.csv", index=False)
    # Annual exposure-time rates: reports per million residents per night-hour
    pop = {}
    for it in json.load(open(RAW / "population" / "worldbank_pop_USA.json"))[1]:
        if it["value"]:
            pop[int(it["date"])] = it["value"]
    net = {}
    for it in json.load(open(RAW / "population" / "worldbank_internet_USA.json"))[1]:
        if it["value"]:
            net[int(it["date"])] = it["value"]
    yr = r.dropna(subset=["event"]).groupby(r.event.dt.year).size()
    ann = pd.DataFrame({"reports": yr})
    ann["pop_million"] = ann.index.map(lambda y: pop.get(y, np.nan) / 1e6)
    ann["internet_pct"] = ann.index.map(lambda y: net.get(y, np.nan))
    ann["night_hours_per_year"] = 365.25 * 11.0  # mean CONUS hours with sun < -6 deg (approx. 11 h/day)
    ann["reports_per_million_per_night_hour"] = ann.reports / ann.pop_million / ann.night_hours_per_year
    ann["per_million_internet_users"] = ann.reports / (ann.pop_million * ann.internet_pct / 100)
    ann.loc[1990:2023].to_csv(RESULTS / "exposure_annual_rates.csv")
    print(ann.loc[1995:2023].round(4).to_string())


if __name__ == "__main__":
    main()
