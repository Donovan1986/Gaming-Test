"""Measure the known-object matcher against independent labels:
GEIPAN investigator phenomenon labels (A/B cases). Reports sensitivity and
false-positive rate per category for (i) context-only components (no text;
cannot leak investigator conclusions) and (ii) full matcher (text may include
GEIPAN's own conclusion -> optimistic)."""
import numpy as np
import pandas as pd
from common import PROCESSED, RESULTS

r = pd.read_parquet(PROCESSED / "reports_scored.parquet")
g = r[(r.SOURCE == "GEIPAN") & r.GEIPAN_CLASS.isin(["A", "B"])].copy()
lab = g.GEIPAN_PHENOMENON.fillna("").str.lower()
CATS = {
    "lantern": lab.str.contains("lanterne"),
    "satellite": lab.str.contains(r"\biss\b|satellite|station spatiale|iridium|starlink"),
    "planet": lab.str.contains(r"v[ée]nus|jupiter|mars|saturne|[ée]toile|sirius|plan[èe]te|lune"),
    "meteor": lab.str.contains(r"bolide|m[ée]t[ée]or|[ée]toile filante"),
    "reentry": lab.str.contains(r"rentr[ée]e"),
    "aircraft": lab.str.contains(r"avion|h[ée]licopt|a[ée]ronef|ulm|planeur"),
    "balloon": lab.str.contains(r"ballon"),
    "launch": lab.str.contains(r"fus[ée]e|lancement|missile"),
    "drone": lab.str.contains(r"drone"),
}
rows = []
for cat, m in CATS.items():
    for comp in ("ctx", "txt", "P"):
        col = f"{comp}_{cat}" if comp != "P" else f"P_{cat}"
        if col not in g:
            continue
        s = g[col].fillna(0)
        rows.append(dict(category=cat, component=comp, n_labelled=int(m.sum()), n_other=int((~m).sum()),
                         sensitivity_p03=float((s[m] >= 0.3).mean()) if m.any() else np.nan,
                         false_pos_rate_p03=float((s[~m] >= 0.3).mean()),
                         mean_score_labelled=float(s[m].mean()) if m.any() else np.nan,
                         mean_score_other=float(s[~m].mean())))
out = pd.DataFrame(rows)
# overall: explained (A/B) vs D
allg = r[(r.SOURCE == "GEIPAN")]
summ = allg.groupby("GEIPAN_CLASS")[["P_EXPLAINED", "P_EXPLAINED_CTX_ONLY"]].agg(["mean", lambda s: (s >= 0.3).mean()])
print(out.to_string())
print(summ)
out.to_csv(RESULTS / "explanation_matcher_validation_geipan.csv", index=False)
summ.to_csv(RESULTS / "explanation_matcher_by_geipan_class.csv")
n = r[r.SOURCE == "NUFORC"]
print("NUFORC notes in TT summaries:", n.d_nuforc_note_txt.sum())
print(n.TOP_EXPLANATION.value_counts().head(20))
print(n.KNOWN_OBJECT_STATUS.value_counts(normalize=True))

# ---- English: NUFORC analyst notes as (imperfect) labels; flags recomputed on text with the note removed
import re
import textparse as T
nn = r[(r.SOURCE == "NUFORC") & r.d_nuforc_note_txt].copy()
nn["note"] = nn.ORIGINAL_TEXT.str.extract(r"(?i)(nuforc note.*)$")[0].fillna("")
nn["body"] = nn.ORIGINAL_TEXT.str.replace(r"(?i)\(\(?\s*nuforc note.*$", "", regex=True)
fl = pd.DataFrame([T.desc_flags(s) for s in nn.body], index=nn.index)
labs = {"starlink": nn.note.str.contains(r"(?i)starlink|satellites in (?:a )?(?:line|train)"),
        "satellite": nn.note.str.contains(r"(?i)\bsatellite|\biss\b|space station|iridium"),
        "planet": nn.note.str.contains(r"(?i)venus|jupiter|mars|saturn|planet|sirius|\bstar\b"),
        "meteor": nn.note.str.contains(r"(?i)meteor|fireball|bolide"),
        "aircraft": nn.note.str.contains(r"(?i)aircraft|airliner|\bjet\b|plane|helicopter"),
        "launch": nn.note.str.contains(r"(?i)launch|rocket|missile|vandenberg|falcon|spacex"),
        "lantern": nn.note.str.contains(r"(?i)lantern")}
flagmap = {"starlink": ["d_line_formation_txt", "d_starlink_word_txt"], "satellite": ["d_satellite_word_txt", "d_iss_word_txt"],
           "planet": ["d_planet_word_txt"], "meteor": ["d_meteor_word_txt", "d_fireball_shape_txt"],
           "aircraft": ["d_aircraft_word_txt", "d_red_green_txt"], "launch": ["d_rocket_word_txt"],
           "lantern": ["d_lantern_word_txt", "d_orange_txt"]}
ctxmap = {"starlink": "ctx_starlink", "satellite": "ctx_satellite", "planet": "ctx_planet", "meteor": "ctx_meteor",
          "aircraft": "ctx_aircraft", "launch": "ctx_launch", "lantern": "ctx_lantern"}
rows = []
for k, m in labs.items():
    txt = fl[flagmap[k]].any(axis=1)
    ctx = nn[ctxmap[k]].fillna(0) >= 0.2
    rows.append(dict(category=k, n_note_labelled=int(m.sum()), n_other_notes=int((~m).sum()),
                     txt_sensitivity=txt[m].mean(), txt_fpr=txt[~m].mean(),
                     ctx_sensitivity=ctx[m].mean(), ctx_fpr=ctx[~m].mean()))
o2 = pd.DataFrame(rows)
print(o2.to_string())
o2.to_csv(RESULTS / "explanation_matcher_validation_nuforc_notes.csv", index=False)
