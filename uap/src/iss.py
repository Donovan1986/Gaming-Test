"""ISS visibility from historical TLEs (J. McDowell archive, 1998-11 .. 2020-06).

visible(t, lat, lon) = ISS altitude > 10 deg, ISS sunlit (cylindrical Earth
shadow), observer sun altitude < -4 deg. A report 'matches' an ISS pass if
the ISS is visible at any minute within +-W minutes of the reported time
(W = max(10, time uncertainty), capped at 30). Outside the TLE coverage the
field is NaN (DATA_UNAVAILABLE).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

from common import RAW, fetch

TLE = RAW / "space" / "iss_25544_tle_mcdowell.txt"


def acquire():
    fetch("https://planet4589.org/space/elements/25500/S25544", TLE,
          "ISS (25544) historical TLEs, J. McDowell planet4589 archive")


@lru_cache(maxsize=1)
def tles():
    acquire()
    lines = [l.rstrip("\n") for l in open(TLE) if l.startswith(("1 ", "2 "))]
    sats, ep = [], []
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            try:
                s = Satrec.twoline2rv(lines[i][:69].ljust(69), lines[i + 1][:69].ljust(69))
                sats.append(s)
                ep.append(s.jdsatepoch + s.jdsatepochF)
            except Exception:
                pass
            i += 2
        else:
            i += 1
    ep = np.array(ep)
    o = np.argsort(ep)
    return [sats[k] for k in o], ep[o]


def _gmst(jd):
    T = (jd - 2451545.0) / 36525.0
    g = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T ** 2 - T ** 3 / 38710000.0
    return np.radians(g % 360)


def _sun_eci(jd):
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def iss_visible_window(t_sec, lat, lon, half_window_min) -> np.ndarray:
    """1.0 if ISS visible at any minute in [t-W, t+W]; 0.0 if not; NaN if no TLE within 5 days."""
    sats, ep = tles()
    t_sec = np.asarray(t_sec, float)
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    W = np.clip(np.nan_to_num(np.asarray(half_window_min, float), nan=10), 10, 30).astype(int)
    out = np.full(len(t_sec), np.nan)
    jd0 = t_sec / 86400.0 + 2440587.5
    ok = np.isfinite(jd0) & np.isfinite(lat)
    k = np.clip(np.searchsorted(ep, jd0), 1, len(ep) - 1)
    k = np.where(np.abs(ep[k - 1] - jd0) < np.abs(ep[k] - jd0), k - 1, k)
    near = np.abs(ep[k] - jd0) <= 5
    ok &= near
    out[np.isfinite(jd0) & ~near] = np.nan
    offs = np.arange(-30, 31)
    for kk in np.unique(k[ok]):
        idx = np.where(ok & (k == kk))[0]
        s = sats[kk]
        jd = jd0[idx][:, None] + offs[None, :] / 1440.0
        jdf = jd.ravel()
        e, r, _ = s.sgp4_array(np.floor(jdf - 0.5) + 0.5, jdf - (np.floor(jdf - 0.5) + 0.5))
        r = r.reshape(len(idx), len(offs), 3)
        th = _gmst(jd)
        la = np.radians(lat[idx])[:, None]
        lo = np.radians(lon[idx])[:, None] + th
        Re = 6378.137
        obs = np.stack([Re * np.cos(la) * np.cos(lo), Re * np.cos(la) * np.sin(lo), Re * np.sin(la) * np.ones_like(lo)], -1)
        up = obs / np.linalg.norm(obs, axis=-1, keepdims=True)
        rel = r - obs
        alt = np.degrees(np.arcsin(np.clip((rel * up).sum(-1) / np.linalg.norm(rel, axis=-1), -1, 1)))
        sun = _sun_eci(jd)
        # observer sun altitude
        sun_alt = np.degrees(np.arcsin(np.clip((sun * up).sum(-1), -1, 1)))
        # ISS sunlit: not in cylindrical shadow
        proj = (r * sun).sum(-1)
        perp = np.linalg.norm(r - proj[..., None] * sun, axis=-1)
        lit = (proj > 0) | (perp > Re)
        vis = (alt > 10) & lit & (sun_alt < -4) & (e.reshape(len(idx), len(offs)) == 0)
        win = np.abs(offs)[None, :] <= W[idx][:, None]
        out[idx] = (vis & win).any(axis=1).astype(float)
    return out
