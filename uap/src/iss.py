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
    t_sec, lat, lon, half_window_min = np.broadcast_arrays(
        np.atleast_1d(np.asarray(t_sec, float)), np.asarray(lat, float),
        np.asarray(lon, float), np.asarray(half_window_min, float))
    if t_sec.ndim != 1:
        raise ValueError("ISS inputs must be one-dimensional")
    W = np.clip(np.nan_to_num(half_window_min, nan=10), 10, 30).astype(int)
    out = np.full(len(t_sec), np.nan)
    jd0 = t_sec / 86400.0 + 2440587.5
    ok = np.isfinite(jd0) & np.isfinite(lat) & np.isfinite(lon)
    ok &= (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    if not ok.any():
        return out
    sats, ep = tles()
    ep = np.asarray(ep, float)
    if not len(ep):
        return out
    if len(sats) != len(ep) or not np.isfinite(ep).all() or (np.diff(ep) < 0).any():
        raise ValueError("Invalid ISS TLE epoch table")
    following = np.clip(np.searchsorted(ep, jd0), 0, len(ep) - 1)
    preceding = np.clip(following - 1, 0, len(ep) - 1)
    k = np.where(np.abs(ep[preceding] - jd0) < np.abs(ep[following] - jd0), preceding, following)
    near = np.abs(ep[k] - jd0) <= 5
    ok &= near & (jd0 >= ep[0]) & (jd0 <= ep[-1])
    for kk in np.unique(k[ok]):
        idx = np.where(ok & (k == kk))[0]
        width = int(W[idx].max())
        offs = np.arange(-width, width + 1)
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
        propagated = (e.reshape(len(idx), len(offs)) == 0) & np.isfinite(r).all(axis=-1)
        vis = (alt > 10) & lit & (sun_alt < -4) & propagated
        win = np.abs(offs)[None, :] <= W[idx][:, None]
        detected = (vis & win).any(axis=1)
        complete = (propagated | ~win).all(axis=1)
        # A failed propagation cannot establish that no visible pass occurred.
        out[idx] = np.where(detected, 1.0, np.where(complete, 0.0, np.nan))
    return out
