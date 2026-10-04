"""Curated historical infrastructure lists (public sources; see DATA_SOURCES.md).

Coordinates for curated entries are obtained by geocoding the named base or
town through geo.geocode() (Census gazetteer / DoD MIRTA) rather than typed by
hand, so every coordinate is traceable. ICBM missile *fields* are modelled as
disks around an approximate field centroid offset from the support base; the
offsets and radii are coarse (+-30 km) and documented here. Analyses using
these layers are therefore only run at radii >= 25 km.

Operating periods: from USAF wing histories / Wikipedia articles listed in
DATA_SOURCES.md. "end=None" = still operating at 2023.
"""
import numpy as np
import pandas as pd

import geo

# name, geocode string, field-centroid offset (km, bearing deg), field radius km, start, end, system
ICBM_FIELDS = [
    ("341 MW Malmstrom", "Malmstrom AFB, MT", (60, 80), 130, 1962, None, "Minuteman"),
    ("91 MW Minot", "Minot AFB, ND", (20, 180), 100, 1962, None, "Minuteman"),
    ("90 MW F.E. Warren", "Francis E. Warren AFB, WY", (90, 70), 110, 1964, None, "Minuteman/Peacekeeper"),
    ("44 MW Ellsworth", "Ellsworth AFB, SD", (90, 45), 110, 1963, 1994, "Minuteman"),
    ("351 MW Whiteman", "Whiteman AFB, MO", (10, 0), 90, 1964, 1997, "Minuteman"),
    ("321 MW Grand Forks", "Grand Forks AFB, ND", (60, 270), 90, 1966, 1998, "Minuteman"),
    ("381 SMW McConnell", "McConnell AFB, KS", (0, 0), 70, 1963, 1986, "Titan II"),
    ("390 SMW Davis-Monthan", "Davis-Monthan AFB, AZ", (0, 0), 60, 1963, 1984, "Titan II"),
    ("308 SMW Little Rock", "Little Rock AFB, AR", (0, 0), 70, 1962, 1987, "Titan II"),
]

# DOE / AEC nuclear-weapons complex sites (production, design, storage, testing)
DOE_SITES = [
    ("Los Alamos National Laboratory", "Los Alamos, NM", 1943, None),
    ("Sandia / Kirtland (Manzano storage)", "Kirtland AFB, NM", 1945, None),
    ("Nevada Test Site", "Mercury, NV", 1951, None),
    ("Hanford Site", "Richland, WA", 1943, None),
    ("Oak Ridge (Y-12)", "Oak Ridge, TN", 1943, None),
    ("Savannah River Site", "Jackson, SC", 1951, None),
    ("Rocky Flats Plant", "Superior, CO", 1952, 1992),
    ("Pantex Plant", "Panhandle, TX", 1951, None),
    ("Idaho National Laboratory", "Atomic City, ID", 1949, None),
    ("Lawrence Livermore National Laboratory", "Livermore, CA", 1952, None),
    ("Kansas City Plant", "Grandview, MO", 1949, None),
    ("Mound Laboratory", "Miamisburg, OH", 1948, 2003),
    ("Pinellas Plant", "Largo, FL", 1957, 1994),
    ("Fernald Feed Materials", "Ross, OH", 1951, 1989),
    ("Paducah Gaseous Diffusion", "Paducah, KY", 1952, 2013),
    ("Portsmouth Gaseous Diffusion", "Piketon, OH", 1954, 2001),
    ("White Sands Missile Range / Trinity", "White Sands Missile Range, NM", 1945, None),
]

NUCLEAR_TEST_SITES = [
    ("Nevada Test Site", "Mercury, NV"), ("Trinity", "San Antonio, NM"),
    ("Project Gnome", "Carlsbad, NM"), ("Project Shoal", "Fallon, NV"),
    ("Project Rulison", "Parachute, CO"), ("Project Rio Blanco", "Rangely, CO"),
    ("Project Salmon/Sterling", "Purvis, MS"), ("Amchitka", "Adak, AK"),
]

# Permanently shut down US commercial power reactors (NRC decommissioning list),
# geocoded by nearest town. name, town, first power year, shutdown year
DECOMMISSIONED_NUCLEAR = [
    ("Rancho Seco", "Herald, CA", 1974, 1989), ("Trojan", "Rainier, OR", 1975, 1992),
    ("Zion", "Zion, IL", 1973, 1998), ("Yankee Rowe", "Rowe, MA", 1960, 1991),
    ("Maine Yankee", "Wiscasset, ME", 1972, 1996), ("Connecticut Yankee", "Haddam, CT", 1967, 1996),
    ("Big Rock Point", "Charlevoix, MI", 1962, 1997), ("San Onofre", "San Clemente, CA", 1968, 2013),
    ("Humboldt Bay", "Eureka, CA", 1963, 1976), ("La Crosse", "Genoa, WI", 1967, 1987),
    ("Fort St. Vrain", "Platteville, CO", 1976, 1989), ("Crystal River", "Crystal River, FL", 1977, 2009),
    ("Kewaunee", "Kewaunee, WI", 1974, 2013), ("Vermont Yankee", "Vernon, VT", 1972, 2014),
    ("Fort Calhoun", "Fort Calhoun, NE", 1973, 2016), ("Oyster Creek", "Forked River, NJ", 1969, 2018),
    ("Pilgrim", "Plymouth, MA", 1972, 2019), ("Three Mile Island", "Middletown, PA", 1974, 2019),
    ("Indian Point", "Buchanan, NY", 1962, 2021), ("Duane Arnold", "Palo, IA", 1975, 2020),
    ("Millstone 1", "Waterford, CT", 1970, 1998), ("Dresden 1", "Morris, IL", 1960, 1978),
    ("Palisades", "Covert, MI", 1971, 2022), ("Shippingport", "Shippingport, PA", 1957, 1982),
]


# Sites the gazetteer cannot resolve correctly; coordinates hand-entered from the
# operator's published location (approximate, +-5 km), flagged note="hand-entered".
HAND = {
    "Nevada Test Site": (37.10, -116.05), "White Sands Missile Range / Trinity": (32.38, -106.48),
    "Three Mile Island": (40.153, -76.725), "Yankee Rowe": (42.728, -72.928),
    "Connecticut Yankee": (41.482, -72.499), "Palisades": (42.322, -86.315),
    "Amchitka": (51.45, 179.10), "Vermont Yankee": (42.779, -72.513),
}


def build():
    rows = []
    for name, loc, (off_km, brg), rad, a, b, sysn in ICBM_FIELDS:
        g = geo.geocode(loc)
        lat, lon = g["lat"], g["lon"]
        if off_km and np.isfinite(lat):
            lat, lon = geo.offset_point(lat, lon, off_km, brg)
        rows.append(dict(layer="icbm_field", name=name, lat=lat, lon=lon, radius_km=rad, start=a, end=b,
                         note=f"{sysn}; base geocode {g['precision']}:{g['matched']}"))
    for name, loc, a, b in DOE_SITES:
        g = geo.geocode(loc)
        rows.append(dict(layer="doe_weapons_site", name=name, lat=g["lat"], lon=g["lon"], radius_km=g["radius_km"],
                         start=a, end=b, note=f"{g['precision']}:{g['matched']}"))
    for name, loc in NUCLEAR_TEST_SITES:
        g = geo.geocode(loc)
        rows.append(dict(layer="nuclear_test_site", name=name, lat=g["lat"], lon=g["lon"], radius_km=g["radius_km"],
                         start=1945, end=None, note=f"{g['precision']}:{g['matched']}"))
    for name, loc, a, b in DECOMMISSIONED_NUCLEAR:
        g = geo.geocode(loc)
        rows.append(dict(layer="nuclear_plant_closed", name=name, lat=g["lat"], lon=g["lon"], radius_km=g["radius_km"],
                         start=a, end=b, note=f"{g['precision']}:{g['matched']}"))
    d = pd.DataFrame(rows)
    for name, (la, lo) in HAND.items():
        m = d.name == name
        d.loc[m, ["lat", "lon"]] = (la, lo)
        d.loc[m, "radius_km"] = 5.0
        d.loc[m, "note"] = "hand-entered (approx. +-5 km)"
    return d


if __name__ == "__main__":
    d = build()
    print(d.to_string())
