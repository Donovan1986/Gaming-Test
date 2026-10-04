"""Repair CS1/CS2 referents after the DST-arithmetic fix (audit item 1).

Regenerates CS1/CS2 with calendar offsets on naive local time, compares with
the stored points, and recomputes covariates ONLY for referents whose UTC time
changed or which were added; referents no longer valid are removed. CASE and CS4
rows, event splits and all other rows are untouched. Writes a provenance record
to results/referent_dst_fix.json and verifies zero clock/weekday mismatches.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import analysis_data as A
import spatial
import weather
import features as F
from common import PROCESSED, RESULTS

META = ["SOURCE", "SPLIT", "TIMEZONE", "YEAR", "HIGH_QUALITY", "HQ_UNEXPLAINED", "UNEXPLAINED", "MULTI_SENSOR",
        "MULTI_SENSOR_RADAR", "EVENT_QUALITY", "INFORMATION_QUALITY", "P_EXPLAINED", "TIME_UNCERTAINTY_MIN", "COUNTRY",
        "GEO_REGION", "GEIPAN_CLASS", "REPORT_COUNT", "SHAPE"]


def main():
    pts = pd.read_parquet(PROCESSED / "points.parquet")
    ev = A.load_cases()
    new = pd.concat([A.cs1_referents(ev), A.cs2_referents(ev)], ignore_index=True)
    key = ["EVENT_ID", "strategy", "offset"]
    old = pts[pts.strategy.isin(["CS1", "CS2"])][key + ["utc_ts"]].reset_index()
    m = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    removed = m[m._merge == "left_only"]
    added = m[m._merge == "right_only"]
    both = m[m._merge == "both"]
    changed = both[both.utc_ts_old != both.utc_ts_new]
    print(f"referents old={len(old)} new={len(new)} unchanged={len(both) - len(changed)} changed={len(changed)} "
          f"added={len(added)} removed={len(removed)}", flush=True)
    todo = pd.concat([changed, added])[key + ["utc_ts_new"]].rename(columns={"utc_ts_new": "utc_ts"})
    lat = ev.set_index("EVENT_ID")[["LATITUDE", "LONGITUDE"]]
    todo = todo.join(lat, on="EVENT_ID").rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})
    todo = todo.join(ev.set_index("EVENT_ID")[META], on="EVENT_ID")
    todo["is_case"] = 0
    todo["YEAR_PT"] = todo.utc_ts.dt.year
    todo = todo.reset_index(drop=True)
    feats = A.compute_features(todo)
    sp = spatial.spatial_features(todo.lat.to_numpy(), todo.lon.to_numpy(), todo.YEAR_PT.to_numpy())
    wx = weather.weather_features(F.to_sec(todo.utc_ts), todo.lat.to_numpy(float), todo.lon.to_numpy(float))
    sp.index = todo.index
    wx.index = todo.index
    rec = pd.concat([todo, feats, sp, wx], axis=1)
    rec = rec.loc[:, ~rec.columns.duplicated()]
    drop_idx = pd.concat([changed["index"], removed["index"]]).dropna().astype(int).to_numpy()
    keep = pts.drop(index=drop_idx)
    for c in keep.columns:
        if c not in rec:
            rec[c] = np.nan
    rec = rec[keep.columns]
    for c in keep.columns:  # align dtypes where possible
        try:
            rec[c] = rec[c].astype(keep[c].dtype)
        except (TypeError, ValueError):
            pass
    out = pd.concat([keep, rec], ignore_index=True)
    refs = out[out.strategy.isin(["CS1", "CS2"])][["EVENT_ID", "utc_ts"]]
    chk = A.check_referents(ev, refs)
    print("verification:", chk, flush=True)
    assert chk["clock_mismatch"] == 0 and chk["weekday_mismatch"] == 0
    tmp = PROCESSED / "points.tmp.parquet"
    out.to_parquet(tmp, index=False)
    tmp.rename(PROCESSED / "points.parquet")
    prov = dict(referents_old=len(old), referents_new=len(new), unchanged=int(len(both) - len(changed)),
                changed_recomputed=int(len(changed)), added=int(len(added)), removed=int(len(removed)),
                changed_by_strategy=changed.strategy.value_counts().to_dict(),
                verification=chk, points_rows=len(out),
                policy="calendar offsets on naive local time; nonexistent wall time -> referent dropped; "
                       "ambiguous wall time -> first (daylight-time) instance")
    json.dump(prov, open(RESULTS / "referent_dst_fix.json", "w"), indent=1, default=int)
    print(prov)


if __name__ == "__main__":
    main()
