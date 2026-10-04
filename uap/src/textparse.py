"""Transparent regex-based extraction from free-text report fields.

Everything extracted from text carries the suffix _txt in the event tables so
it is never confused with a structured field. Extraction is deliberately
conservative: if a pattern is not matched the value is NaN/False (NOT
OBSERVED), never an imputed number.
"""
from __future__ import annotations

import re

import numpy as np

NUMWORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
            "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45,
            "fifty": 50, "sixty": 60, "ninety": 90, "half": 0.5, "few": 3, "a few": 3,
            "several": 5, "couple": 2, "a couple": 2, "couple of": 2}
UNIT_S = [(r"s(?:ec(?:ond)?s?)?|secs?", 1), (r"m(?:in(?:ute)?s?)?|mins?", 60),
          (r"h(?:(?:ou)?rs?)?|hrs?", 3600), (r"days?", 86400)]
_NUM = r"(\d+(?:\.\d+)?|\d+/\d+|" + "|".join(sorted(map(re.escape, NUMWORDS), key=len, reverse=True)) + r")"
_DUR = re.compile(_NUM + r"(?:\s*(?:-|to|or|–)\s*" + _NUM + r")?\s*\+?\s*(seconds?|secs?|s\b|minutes?|mins?|m\b|hours?|hrs?|h\b|days?)", re.I)


def _num(tok: str) -> float:
    tok = tok.lower().strip()
    if tok in NUMWORDS:
        return float(NUMWORDS[tok])
    if "/" in tok:
        a, b = tok.split("/")
        return float(a) / float(b) if float(b) else np.nan
    return float(tok)


def parse_duration_s(text) -> float:
    """Duration in seconds from free text; NaN when not parseable.
    Ranges ("2-3 min") return the geometric midpoint."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    t = text.lower().replace("1/2", "0.5").replace("½", "0.5")
    if re.search(r"\bhalf (an |a )?hour\b", t):
        return 1800.0
    if re.search(r"\b(an|one) hour\b", t) and not re.search(r"\d", t):
        return 3600.0
    m = _DUR.search(t)
    if not m:
        if re.search(r"\b(instant|split second|flash|blink)\b", t):
            return 1.0
        if re.search(r"\bfew seconds\b|\bseconds\b", t):
            return 5.0
        if re.search(r"\bfew minutes\b|\bminutes\b", t):
            return 180.0
        if re.search(r"\bhours\b", t):
            return 7200.0
        return np.nan
    a = _num(m.group(1))
    b = _num(m.group(2)) if m.group(2) else None
    unit = m.group(3).lower()
    mult = 1
    if unit.startswith("s"):
        mult = 1
    elif unit.startswith("m"):
        mult = 60
    elif unit.startswith("h"):
        mult = 3600
    elif unit.startswith("d"):
        mult = 86400
    v = np.sqrt(a * b) if (b and a > 0 and b > 0) else a
    return float(v * mult)


_WIT_N = re.compile(r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten|several|many|multiple|dozens of|hundreds of)\s+"
                    r"(?:other\s+)?(witnesses|people|persons|observers|of us|friends|family members|neighbors|kids|adults|employees|officers|guys)\b", re.I)
_WIT_COMP = re.compile(r"\b(my|our)\s+(wife|husband|son|daughter|friend|friends|girlfriend|boyfriend|mother|father|"
                       r"mom|dad|brother|sister|kids|children|family|neighbor|neighbors|co-?worker|coworkers|partner|fianc[eé]e?)\b", re.I)
_WE = re.compile(r"\b(we|us|both of us)\b", re.I)


def witness_count_txt(text) -> float:
    """Lower-bound estimate of witnesses from text. NaN if nothing indicates a count."""
    if not isinstance(text, str):
        return np.nan
    m = _WIT_N.search(text)
    if m:
        tok = m.group(1).lower()
        val = {"several": 4, "many": 10, "multiple": 3, "dozens of": 24, "hundreds of": 200}.get(tok)
        if val is None:
            val = _num(tok)
        if m.group(2).lower() in ("of us",):
            return float(val)
        return float(val) + 1.0
    if _WIT_COMP.search(text):
        return 2.0
    if _WE.search(text):
        return 2.0
    if re.search(r"\b(i|me|my)\b", text, re.I):
        return 1.0
    return np.nan


OBSERVER_PATTERNS = {
    "pilot": r"\b(pilot|co-?pilot|first officer|captain of (?:a|the) (?:plane|aircraft|flight)|flight crew|aircrew|airline crew|cockpit)\b",
    "police": r"\b(police|officer|deputy|sheriff|trooper|patrolm[ae]n|law enforcement|state patrol)\b",
    "military": r"\b(military|air force|usaf|navy|naval|army|marine corps|marines|soldier|airman|sergeant|lieutenant|colonel|major\s+\w+|veteran|national guard|nco)\b",
    "astronomer": r"\b(astronomer|astronomy|telescope|observatory|astrophotograph|amateur astronomer)\b",
    "atc": r"\b(air traffic control|controller|tower operator|faa)\b",
    "scientist_engineer": r"\b(engineer|scientist|physicist|meteorologist|professor)\b",
    "radar": r"\b(radar|scope|blip|painted|tracked on)\b",
}
_OBS_RE = {k: re.compile(v, re.I) for k, v in OBSERVER_PATTERNS.items()}


def observer_flags(text) -> dict:
    out = {f"obs_{k}_txt": False for k in OBSERVER_PATTERNS}
    if not isinstance(text, str):
        return out
    for k, r in _OBS_RE.items():
        out[f"obs_{k}_txt"] = bool(r.search(text))
    return out


DESC_PATTERNS = {
    "line_formation": r"\b(line of|in a line|straight line of|single file|in a row|train of|string of|chain of|evenly spaced|one after (?:the )?other|following each other|in formation)\b",
    "starlink_word": r"\b(starlink|elon|spacex satellites)\b",
    "satellite_word": r"\bsatellites?\b",
    "iss_word": r"\b(iss|international space station|space station)\b",
    "meteor_word": r"\b(meteor|meteorite|shooting star|falling star|fireball|bolide|streak(?:ed|ing)?|fell|falling)\b",
    "fireball_shape": r"\bfire\s*ball\b",
    "rocket_word": r"\b(rocket|launch|missile|plume|exhaust|contrail|jellyfish|spiral)\b",
    "planet_word": r"\b(venus|jupiter|mars|saturn|planet|star-like|like a star|bright star)\b",
    "moon_word": r"\bmoon\b",
    "aircraft_word": r"\b(plane|airplane|aircraft|jet|airliner|helicopter|chopper|blimp)\b",
    "blinking": r"\b(blink(?:ing)?|flash(?:ing|ed)?|strobe|strobing|pulsing|pulsat(?:ing|ed))\b",
    "red_green": r"\b(red and green|green and red|red,? white,? (?:and )?green|navigation lights)\b",
    "sound": r"\b(sound|noise|hum(?:ming)?|roar|engine|buzz(?:ing)?|rumble|whir+)\b",
    "silent": r"\b(silent|no sound|no noise|noiseless|without (?:any )?sound|quiet)\b",
    "hover": r"\b(hover(?:ed|ing)?|stationary|motionless|not moving|stood still|hung|suspended)\b",
    "fast": r"\b(fast|rapid(?:ly)?|high speed|incredible speed|shot off|zoom(?:ed|ing)?|darted|took off|accelerat\w+)\b",
    "erratic": r"\b(zig-?zag\w*|erratic(?:ally)?|sharp turns?|right angles?|changed direction|instant(?:ly)? stop\w*|bobbing|jerky|darting)\b",
    "orange": r"\b(orange|amber|reddish-orange|red-orange|orangish)\b",
    "lantern_word": r"\b(lantern|chinese lantern|sky lantern|candle|flicker(?:ing)?|drift(?:ed|ing)?|float(?:ed|ing)?)\b",
    "drone_word": r"\b(drones?|quad-?copter|uav)\b",
    "balloon_word": r"\b(balloons?|mylar)\b",
    "firework_word": r"\b(fireworks?|firecrackers?)\b",
    "lightning_word": r"\b(lightning|thunder|storm|thunderstorm)\b",
    "searchlight_word": r"\b(searchlights?|spot ?lights?|beams? of light|light beams?|laser)\b",
    "flare_word": r"\b(flares?|parachute flare)\b",
    "cloud_word": r"\b(cloud|lenticular|haze|fog)\b",
    "em_effects": r"\b(car (?:died|stalled|stopped)|engine (?:died|stalled|quit)|radio (?:static|interference|went dead)|"
                  r"lights? (?:went out|flickered|dimmed)|power (?:outage|went out|failure)|static on|electrical (?:interference|failure)|"
                  r"tv (?:static|interference)|phone (?:died|went dead|malfunction)|compass)\b",
    "physio_effects": r"\b(nause\w+|headache|burn(?:s|ed)?|tingl\w+|paraly[sz]\w+|sick(?:ness)?|dizz\w+|sunburn|rash|eye (?:pain|irritation)|hair stood)\b",
    "photo_word": r"\b(photo(?:graph)?s?|pictures?|picture|camera|took a pic|filmed|video(?:ed|taped)?|recorded|footage)\b",
    "hoax_note": r"\b(hoax|prank|not a serious report|fabricat\w+|joke)\b",
    "nuforc_note": r"nuforc note",
    "nuforc_note_starlink": r"nuforc note[^)]*?(starlink|satellites? in (?:a )?(?:line|train))",
    "nuforc_note_iss": r"nuforc note[^)]*?(iss|international space station)",
    "nuforc_note_planet": r"nuforc note[^)]*?(venus|jupiter|mars|saturn|planet|sirius|star\b|twinkling star)",
    "nuforc_note_satellite": r"nuforc note[^)]*?(satellite|iridium)",
    "nuforc_note_aircraft": r"nuforc note[^)]*?(aircraft|airliner|jet|plane)",
    "nuforc_note_meteor": r"nuforc note[^)]*?(meteor|fireball|bolide)",
    "nuforc_note_launch": r"nuforc note[^)]*?(launch|rocket|missile|vandenberg|spacex|falcon)",
    "nuforc_note_lantern": r"nuforc note[^)]*?(lantern|balloon)",
    "nuforc_note_hoax": r"nuforc note[^)]*?(hoax|prank|not serious|fabricat|joke|serious\?)",
}
_DESC_RE = {k: re.compile(v, re.I) for k, v in DESC_PATTERNS.items()}


NEGATABLE = {"aircraft_word", "planet_word", "satellite_word", "meteor_word", "balloon_word", "drone_word",
             "lantern_word", "iss_word", "starlink_word", "firework_word", "sound", "rocket_word", "blinking",
             "hover", "searchlight_word", "flare_word", "moon_word"}
_NEG = re.compile(r"\b(not|no|never|nor|unlike|than|neither|without|nothing like|wasn.t|isn.t|didn.t|wasnt|isnt|didnt|"
                  r"definitely not|ruled out|other than)\b[^.!?;]{0,25}$", re.I)


def _affirmed(regex, text) -> bool:
    """True if at least one match is not preceded (same clause, <=25 chars) by a negation/comparison."""
    for m in regex.finditer(text):
        pre = text[max(0, m.start() - 40):m.start()]
        if not _NEG.search(pre):
            return True
    return False


def desc_flags(text) -> dict:
    out = {f"d_{k}_txt": False for k in DESC_PATTERNS}
    if not isinstance(text, str):
        return out
    for k, r in _DESC_RE.items():
        out[f"d_{k}_txt"] = _affirmed(r, text) if k in NEGATABLE else bool(r.search(text))
    return out


COLOR_WORDS = ["white", "red", "orange", "yellow", "green", "blue", "purple", "silver", "gold", "black",
               "gray", "grey", "amber", "pink", "multi-colored", "multicolored"]
_COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_WORDS) + r")\b", re.I)


def colors_txt(text) -> str:
    if not isinstance(text, str):
        return ""
    return ",".join(sorted({c.lower().replace("grey", "gray") for c in _COLOR_RE.findall(text)}))


_DIR_RE = re.compile(r"\b(?:to(?:ward)?s?(?: the)?|heading|moving|travel(?:l?ing|ed)?|going|flew|headed)\s+"
                     r"(north(?:east|west)?|south(?:east|west)?|east|west|n|s|e|w|ne|nw|se|sw)\b", re.I)


def direction_txt(text) -> str:
    if not isinstance(text, str):
        return ""
    m = _DIR_RE.search(text)
    return m.group(1).lower() if m else ""


SHAPE_MAP = {
    "light": "light", "circle": "circle", "triangle": "triangle", "fireball": "fireball",
    "unknown": "unknown", "other": "other", "sphere": "sphere", "disk": "disk", "disc": "disk",
    "oval": "oval", "formation": "formation", "cigar": "cigar", "changing": "changing",
    "flash": "flash", "rectangle": "rectangle", "cylinder": "cylinder", "diamond": "diamond",
    "chevron": "chevron", "egg": "egg", "teardrop": "teardrop", "cone": "cone", "cross": "cross",
    "orb": "sphere", "star": "star", "delta": "triangle", "round": "circle", "crescent": "other",
    "pyramid": "other", "flare": "flash", "hexagon": "other", "dome": "disk", "changed": "changing",
}


def norm_shape(s) -> str:
    if not isinstance(s, str) or not s.strip():
        return "unknown"
    return SHAPE_MAP.get(s.strip().lower(), "other")


# French equivalents (GEIPAN). Same keys as DESC_PATTERNS / OBSERVER_PATTERNS so
# downstream code is language-agnostic.
DESC_PATTERNS_FR = {
    "line_formation": r"(en ligne|align[ée]s?|file indienne|queue leu leu|les uns derri[èe]re les autres|en formation|chapelet|train de)",
    "starlink_word": r"(starlink)",
    "satellite_word": r"(satellite)",
    "iss_word": r"(\biss\b|station spatiale)",
    "meteor_word": r"(m[ée]t[ée]or|bolide|[ée]toile filante|boule de feu|tra[îi]n[ée]e|chute)",
    "fireball_shape": r"(boule de feu)",
    "rocket_word": r"(fus[ée]e|lancement|missile|panache)",
    "planet_word": r"(v[ée]nus|jupiter|mars|saturne|plan[èe]te|[ée]toile brillante|comme une [ée]toile)",
    "moon_word": r"(\blune\b)",
    "aircraft_word": r"(avion|a[ée]ronef|h[ée]licopt[èe]re|jet|ulm|dirigeable)",
    "blinking": r"(clignot|flash|scintill|puls)",
    "red_green": r"(rouge et vert|vert et rouge|feux de navigation)",
    "sound": r"(bruit|son\b|vrombiss|bourdonn|grondement|sifflement)",
    "silent": r"(silencieu|sans bruit|aucun bruit|aucun son)",
    "hover": r"(stationnaire|immobile|fixe dans le ciel|fait du surplace|surplace)",
    "fast": r"(rapide|tr[èe]s vite|grande vitesse|fulgurant|acc[ée]l[ée]r)",
    "erratic": r"(zigzag|erratique|changement(s)? de direction|angle droit|brusque)",
    "orange": r"(orange|orang[ée]|ambr[ée])",
    "lantern_word": r"(lanterne|lampion)",
    "drone_word": r"(drone)",
    "balloon_word": r"(ballon)",
    "firework_word": r"(feu d.artifice|p[ée]tard)",
    "lightning_word": r"([ée]clair|orage|foudre)",
    "searchlight_word": r"(projecteur|faisceau|laser|skytracer|lumi[èe]re de discoth[èe]que)",
    "flare_word": r"(fus[ée]e [ée]clairante|leurre)",
    "cloud_word": r"(nuage|brume|brouillard)",
    "em_effects": r"(panne de (?:voiture|moteur|courant)|interf[ée]rence|parasites|perturbation [ée]lectr|boussole)",
    "physio_effects": r"(naus[ée]e|maux de t[êe]te|br[ûu]lure|picotement|paralys|malaise)",
    "photo_word": r"(photo|vid[ée]o|film[ée]|cam[ée]ra|cliché)",
    "hoax_note": r"(canular|blague)",
}
OBSERVER_PATTERNS_FR = {
    "pilot": r"(pilote|[ée]quipage|commandant de bord|copilote)",
    "police": r"(gendarme|policier|police)",
    "military": r"(militaire|arm[ée]e de l.air|marine nationale|soldat|base a[ée]rienne)",
    "astronomer": r"(astronome|t[ée]lescope|observatoire|astronomie)",
    "atc": r"(contr[ôo]leur a[ée]rien|tour de contr[ôo]le|contr[ôo]le a[ée]rien)",
    "scientist_engineer": r"(ing[ée]nieur|scientifique|physicien|m[ée]t[ée]orologue)",
    "radar": r"(radar)",
}
_DESC_RE_FR = {k: re.compile(v, re.I) for k, v in DESC_PATTERNS_FR.items()}
_OBS_RE_FR = {k: re.compile(v, re.I) for k, v in OBSERVER_PATTERNS_FR.items()}


_NEG_FR = re.compile(r"\b(pas|ni|aucun|aucune|sans|jamais|plus que|que|contrairement|exclu|[ée]limin)\b[^.!?;]{0,30}$", re.I)


def desc_flags_fr(text) -> dict:
    out = {f"d_{k}_txt": False for k in DESC_PATTERNS}
    if isinstance(text, str):
        for k, r in _DESC_RE_FR.items():
            if k in NEGATABLE:
                ok = False
                for m in r.finditer(text):
                    if not _NEG_FR.search(text[max(0, m.start() - 45):m.start()]):
                        ok = True
                        break
                out[f"d_{k}_txt"] = ok
            else:
                out[f"d_{k}_txt"] = bool(r.search(text))
    return out


def observer_flags_fr(text) -> dict:
    out = {f"obs_{k}_txt": False for k in OBSERVER_PATTERNS}
    if isinstance(text, str):
        for k, r in _OBS_RE_FR.items():
            out[f"obs_{k}_txt"] = bool(r.search(text))
    return out


_FR_TIME = re.compile(r"(?:\b[àa]|vers|environ|aux alentours de|aux environs de|entre)\s+(\d{1,2})\s*[hH]\s*(\d{2})?", re.I)


def parse_time_fr(text):
    """(hh, mm, approx) from French narrative, or None. 'vers 22h' -> (22, 0, True)."""
    if not isinstance(text, str):
        return None
    m = _FR_TIME.search(text)
    if not m:
        return None
    hh = int(m.group(1))
    if hh > 23:
        return None
    mm = int(m.group(2)) if m.group(2) else 0
    approx = (m.group(2) is None) or bool(re.match(r"(vers|environ|aux)", m.group(0), re.I))
    return hh, mm, approx


_FR_DUR = re.compile(r"(?:pendant|durant|dur[ée]e? (?:de|d.environ)?|environ)\s+(\d+(?:[.,]\d+)?)\s*(secondes?|s\b|minutes?|min|mn|heures?|h\b)", re.I)


def parse_duration_fr(text) -> float:
    if not isinstance(text, str):
        return np.nan
    m = _FR_DUR.search(text)
    if not m:
        return np.nan
    v = float(m.group(1).replace(",", "."))
    u = m.group(2).lower()
    return v * (1 if u.startswith("s") else 3600 if u.startswith("h") else 60)


_FR_WIT = re.compile(r"\b(deux|trois|quatre|cinq|six|sept|huit|neuf|dix|plusieurs|\d+)\s+t[ée]moins\b", re.I)
_FR_NUM = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "plusieurs": 3}


def witness_count_fr(text) -> float:
    if not isinstance(text, str):
        return np.nan
    m = _FR_WIT.search(text)
    if m:
        t = m.group(1).lower()
        return float(_FR_NUM.get(t, t if t.isdigit() else 2))
    if re.search(r"\b(nous|on a|avec (?:mon|ma|mes)|ma femme|mon mari|mes enfants|des amis)\b", text, re.I):
        return 2.0
    if re.search(r"\b(un t[ée]moin|le t[ée]moin|la t[ée]moin)\b", text, re.I):
        return 1.0
    return np.nan
