"""When a pointing is observed, from ArgusSim's own survey rules: the night's exposures, the nearest night it is up, the
night its variables change most, and BJD_TDB at the site. Used by observe.py and night_movie.py.

The exposures are ArgusSim's schedule (coverage_exact.Schedule, built as Survey builds it from config.survey): a
base_cadence_s cadence in ratchets of ratchet_len minutes on UTC boundaries, the last reset_min of each a reset, and
the Sun below survey.min_sun_alt. Of those, the field is observed while it is at or above observatory.min_alt (the
lowest the array points); [night] max_airmass, if given, is a further limit of the observation's own. Airmass is
ArgusSim's (Pickering 2002), as its sky model and survey use.
"""
from __future__ import annotations

import numpy as np

SITES = {"argus": None,  # ArgusSim's configured site
         "sro": dict(latitude=37.07, longitude=-119.41, altitude=1400.0)}  # Sierra Remote Observatories
MAX_AIRMASS = None  # [night] max_airmass: a further limit (None / 0: ArgusSim's min_alt only)


def _config(site):
    """ArgusSim's configuration, with the site moved if asked (the instrument stays Argus)."""
    import copy

    from argus_sim import c as ARGUS_CONFIG

    cfg = copy.deepcopy(ARGUS_CONFIG)
    for k, v in (site or {}).items():
        setattr(cfg.observatory, k, v)
    return cfg


def _location(cfg):
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    o = cfg.observatory
    return EarthLocation(lat=o.latitude * u.deg, lon=o.longitude * u.deg, height=o.altitude * u.m)


def exposures(ra, dec, date, site=None, start=None, end=None, max_airmass=None, cfg=None, coarse=False):
    """The exposures of the night starting on local `date` (UTC 23:00 + 14 h), as ArgusSim schedules them (module
    docstring), with the field at or above min_alt (and at or below max_airmass), within the UT window start-end.
    coarse: every 5th minute only, by the same rules (for searches over many nights; ~5x faster).
    Returns (mid-exposure times, airmass, alt, az)."""
    import astropy.units as u
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time
    from argus_sim.coverage_exact import Schedule
    from argus_sim.observatory import pickering_airmass

    cfg = cfg or _config(site)
    max_airmass = MAX_AIRMASS if max_airmass is None else max_airmass
    sv = cfg.survey
    sched = Schedule(exptime_s=sv.base_cadence_s, track_min=sv.ratchet_len - sv.reset_min, reset_min=sv.reset_min,
                     sun_alt_max_deg=sv.min_sun_alt)
    t0 = Time(f"{date} 23:00:00", scale="utc")
    if coarse:  # the schedule's own rules on a 5-minute grid: ratchet tracking minutes, the Sun below the limit
        from astropy.coordinates import get_sun

        t = t0 + (np.arange(0, 14 * 60, 5) + 0.5 * sv.base_cadence_s / 60) * u.min
        in_track = (t.datetime64.astype("datetime64[s]").astype(np.int64) / 60) % sv.ratchet_len < sv.ratchet_len - sv.reset_min
        sun = get_sun(t).transform_to(AltAz(obstime=t, location=_location(cfg))).alt.deg
        t = t[in_track & (sun < sv.min_sun_alt)]
    else:
        t = sched.exposures(t0, t0 + 14 * u.hour, _location(cfg))["time"]
    if len(t) == 0:
        return t, np.array([]), np.array([]), np.array([])
    here = SkyCoord(ra * u.deg, dec * u.deg).transform_to(AltAz(obstime=t, location=_location(cfg)))
    alt, az = here.alt.deg, here.az.deg
    airmass = np.where(alt > 0, pickering_airmass(np.clip(alt, 1e-3, 90)), np.inf)
    ok = alt >= cfg.observatory.min_alt
    if max_airmass:
        ok &= airmass <= max_airmass
    clock = np.array([(h - 24 if h >= 20 else h) for h in (t.datetime64.astype("datetime64[s]").astype(np.int64) % 86400) / 3600])
    if start:
        ok &= clock >= (lambda h: h - 24 if h >= 20 else h)(int(start[:2]) + int(start[3:5]) / 60)
    if end:
        ok &= clock <= (lambda h: h - 24 if h >= 20 else h)(int(end[:2]) + int(end[3:5]) / 60)
    return t[ok], airmass[ok], alt[ok], az[ok]


def night_times(ra, dec, date, site, start, end, max_airmass=None):
    """JD (UTC) of the night's exposures (exposures())."""
    return exposures(ra, dec, date, site, start, end, max_airmass)[0].jd


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
        jd = exposures(ra, dec, d, site, start, end, coarse=True)[0].jd
        if 5 * len(jd) < min_minutes:
            continue
        sc = variability_score(jd, picks)
        if sc > best[0] or (sc == best[0] and abs(offset) < abs((Date.fromisoformat(best[1]) - d0).days)):
            best = (sc, d, jd)
    if best[1] is None:
        raise SystemExit("never observable for 3 h from this site (dark, above the array's minimum altitude)")
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
    raise SystemExit("never observable for 3 h from this site (dark, above the array's minimum altitude)")


def observable(ra, dec, date, site, start, end, max_airmass=None):
    """Minutes the field is observed on the night of `date` (exposures(), 5-minute steps), and the UT of the first
    and last step."""
    t = exposures(ra, dec, date, site, start, end, max_airmass, coarse=True)[0]
    minutes = 5.0 * len(t)
    return minutes, ((t[0].iso[11:16], t[-1].iso[11:16]) if len(t) else (None, None))


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
