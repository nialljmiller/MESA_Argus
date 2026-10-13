"""When a pointing can be observed: the night's observable steps, the nearest night it is up, the night its
variables change most, and BJD_TDB at the site. Used by observe.py.
"""
from __future__ import annotations

import numpy as np

SITES = {"argus": None,  # ArgusSim's configured site
         "sro": dict(latitude=37.07, longitude=-119.41, altitude=1400.0)}  # Sierra Remote Observatories
MAX_AIRMASS = 2.0  # [night] max_airmass


def night_times(ra, dec, date, site, start, end, max_airmass=None):
    """BJD-like JD (UTC + light time ignored, < 10 min) of the 5-minute steps the field is observable on a night."""
    max_airmass = MAX_AIRMASS if max_airmass is None else max_airmass
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
    from astropy.time import Time

    if site is None:
        from argus_sim import c as cfg
        site = dict(latitude=cfg.observatory.latitude, longitude=cfg.observatory.longitude, altitude=cfg.observatory.altitude)
    loc = EarthLocation(lat=site["latitude"] * u.deg, lon=site["longitude"] * u.deg, height=site["altitude"] * u.m)
    t = Time(f"{date} 23:00:00") + np.arange(0, 14 * 60, 5) * u.min
    frame = AltAz(obstime=t, location=loc)
    alt = SkyCoord(ra * u.deg, dec * u.deg).transform_to(frame).alt.deg
    ok = (get_sun(t).transform_to(frame).alt.deg < -18) & (alt > 0) & (1 / np.sin(np.radians(np.clip(alt, 1e-3, 90))) <= max_airmass)
    clock = np.array([(h - 24 if h >= 20 else h) for h in (t.datetime64.astype("datetime64[m]").astype(int) % 1440) / 60])
    if start:
        ok &= clock >= (lambda h: h - 24 if h >= 20 else h)(int(start[:2]) + int(start[3:5]) / 60)
    if end:
        ok &= clock <= (lambda h: h - 24 if h >= 20 else h)(int(end[:2]) + int(end[3:5]) / 60)
    return t[ok].jd


def variability_score(jd, picks):
    """How visibly each chosen variable varies during a night, from its ephemeris, as a fraction 0-1 summed over the
    stars (so every star counts equally); ties broken by amplitude. Eclipsing binaries: 1 if the night covers a Gaia
    eclipse from -1 to +1 Gaussian sigma, 0.75 if only its centre, 0.5 if it comes within 1.5 sigma. Pulsators: half for the fraction of
    the cycle covered, half if maximum light or the rise to it (phase 0.85-1.05) is covered."""
    if len(jd) == 0:
        return 0.0
    score, amp = 0.0, 0.0
    for r in picks:
        if r["kind"] == "EB":
            v, p = r["veb"], r["period"]
            best = 0.0
            for k in (1, 2):
                ph, sig = v.get(f"PhaseGauss{k}", np.nan), v.get(f"sigPhaseGauss{k}", np.nan)
                if not np.isfinite(ph):
                    continue
                t_ecl = v["TimeRef"] + 2455197.5 + ph * p
                n_c = np.ceil((jd[0] - t_ecl) / p)
                centre = t_ecl + n_c * p
                w = (sig if np.isfinite(sig) else 0.02) * p
                if centre - w >= jd[0] and centre + w <= jd[-1]:  # ingress to egress (+-1 sigma) in the night
                    best = 1.0
                elif centre <= jd[-1]:
                    best = max(best, 0.75)
                elif np.isfinite(sig) and np.any(np.abs(((jd - t_ecl) / p + 0.5) % 1.0 - 0.5) <= 1.5 * sig):
                    best = max(best, 0.5)
            score += best
            amp += r["amp_G"] * best
        else:
            if not np.isfinite(r.get("epoch_bjd", np.nan)):
                continue
            ph = ((jd - r["epoch_bjd"]) / r["period"]) % 1.0
            s_ = 0.5 * min(1.0, (jd[-1] - jd[0]) / r["period"]) + (0.5 if np.any((ph >= 0.85) | (ph <= 0.05)) else 0.0)
            score += s_
            amp += r["amp_G"] * s_
    return score + 1e-3 * amp


def best_night(ra, dec, date, site, start, end, picks, min_minutes=180):
    """The night within half a year of `date` (up >= min_minutes) on which the chosen variables change most."""
    from datetime import date as Date, timedelta

    d0 = Date.fromisoformat(date)
    best = (-1.0, None, None)
    for offset in range(-182, 183):
        d = (d0 + timedelta(days=offset)).isoformat()
        jd = night_times(ra, dec, d, site, start, end)
        if 5 * len(jd) < min_minutes:
            continue
        sc = variability_score(jd, picks)
        if sc > best[0] or (sc == best[0] and abs(offset) < abs((Date.fromisoformat(best[1]) - d0).days)):
            best = (sc, d, jd)
    if best[1] is None:
        raise SystemExit("never observable for 3 h from this site (dark, airmass <= 2)")
    return best[1], best[0]


def nearest_night(ra, dec, date, site, start, end, min_minutes=180):
    """The night nearest `date` (up to half a year away) when the field is up for >= min_minutes."""
    from datetime import date as Date, timedelta

    d0 = Date.fromisoformat(date)
    for offset in sorted(range(-182, 183, 7), key=abs):  # weekly steps (visibility changes slowly)
        d = (d0 + timedelta(days=offset)).isoformat()
        minutes, span = observable(ra, dec, d, site, start, end)
        if minutes >= min_minutes:
            return d, minutes, span
    raise SystemExit("never observable for 3 h from this site (dark, airmass <= 2)")


def observable(ra, dec, date, site, start, end, max_airmass=None):
    """Minutes the field is dark (Sun < -18) at airmass <= max_airmass on the night of `date` (within the window)."""
    max_airmass = MAX_AIRMASS if max_airmass is None else max_airmass
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
    from astropy.time import Time

    if site is None:
        from argus_sim import c as cfg
        site = dict(latitude=cfg.observatory.latitude, longitude=cfg.observatory.longitude, altitude=cfg.observatory.altitude)
    loc = EarthLocation(lat=site["latitude"] * u.deg, lon=site["longitude"] * u.deg, height=site["altitude"] * u.m)
    t = Time(f"{date} 23:00:00") + np.arange(0, 14 * 60, 5) * u.min
    frame = AltAz(obstime=t, location=loc)
    alt = SkyCoord(ra * u.deg, dec * u.deg).transform_to(frame).alt.deg
    ok = (get_sun(t).transform_to(frame).alt.deg < -18) & (alt > 0) & (1 / np.sin(np.radians(np.clip(alt, 1e-3, 90))) <= max_airmass)
    clock = np.array([(h - 24 if h >= 20 else h) for h in (t.datetime64.astype("datetime64[m]").astype(int) % 1440) / 60])
    if start:
        ok &= clock >= (lambda h: h - 24 if h >= 20 else h)(int(start[:2]) + int(start[3:5]) / 60)
    if end:
        ok &= clock <= (lambda h: h - 24 if h >= 20 else h)(int(end[:2]) + int(end[3:5]) / 60)
    return 5 * int(ok.sum()), (t[ok][0].iso[11:16], t[ok][-1].iso[11:16]) if ok.any() else (None, None)


def bjd_tdb(t, ra, dec, site):
    """BJD_TDB of astropy Time(s) t (UTC) for the field seen from the site."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord

    if site is None:
        from argus_sim import c as cfg
        site = dict(latitude=cfg.observatory.latitude, longitude=cfg.observatory.longitude, altitude=cfg.observatory.altitude)
    loc = EarthLocation(lat=site["latitude"] * u.deg, lon=site["longitude"] * u.deg, height=site["altitude"] * u.m)
    return (t.tdb + t.light_travel_time(SkyCoord(ra * u.deg, dec * u.deg), location=loc)).jd


def resolve_time(value, start, ra, dec, site):
    """BJD_TDB of a run-file time: a number (BJD_TDB), "start+D" / "start-D" (days from the night's first observable
    step), or an ISO UTC time."""
    from astropy.time import Time

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    v = str(value).strip()
    if v.startswith("start"):
        return float(bjd_tdb(start, ra, dec, site)) + (float(v[5:]) if v[5:] else 0.0)
    return float(bjd_tdb(Time(v, scale="utc"), ra, dec, site))
