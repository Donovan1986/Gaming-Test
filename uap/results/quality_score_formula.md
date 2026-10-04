
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
