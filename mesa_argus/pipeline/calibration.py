"""The frame's photometric zero point, against an external reference catalogue (nothing about the simulation is used).

atlas_refcat2: hdps's zero point (hdps.photometry.zeropoint) against ATLAS-Refcat2 in the frame's band (beta: g,
rho: r): detections matched by position to Refcat2 stars, unsaturated and measured at S/N >= min_snr. hdps's two
tiers: the plane, ref - det = zp + gradients across the frame + c (g - r - 0.6), clipped (fit_zeropoint_plane, at
least 30 stars); with fewer, a clipped median (fit_zeropoint, at least 20). Refcat2 is read
with hdps's own reader (hdps.astrometry.providers.ATLASRefcat2; ~/.rico_cache/atlas_refcat2), proper motions applied
to the frame's epoch. Every detection then gets mag = -2.5 log10(flux) + the plane at its position (its colour term
from the matched Refcat2 star, else at the pivot colour) and mag_err from its flux error.
"""
import numpy as np

SETTINGS = dict(match_arcsec=1.5, min_snr=20.0, refcat_radius_deg=0.12, refcat_max_mag=19.0)
_REFCAT = {}


def reference_stars(ra, dec, radius, max_mag, epoch_year):
    """Refcat2 around (ra, dec): its table with positions moved to the epoch (proper motions in mas/yr from 2015.5)."""
    key = (round(ra, 4), round(dec, 4), radius, max_mag)
    if key not in _REFCAT:
        from hdps.astrometry.providers import ATLASRefcat2

        _REFCAT[key] = ATLASRefcat2().get_reference_stars(ra, dec, radius, min_g=0.0, max_g=max_mag).table
    t = _REFCAT[key]
    dt = epoch_year - 2015.5
    ra_now = np.asarray(t["ra"]) + np.nan_to_num(np.asarray(t["pmra"], float)) * dt / 3.6e6 / np.cos(np.radians(t["dec"]))
    dec_now = np.asarray(t["dec"]) + np.nan_to_num(np.asarray(t["pmdec"], float)) * dt / 3.6e6
    return t, ra_now, dec_now


def atlas_refcat2(catalogue, frame, cfg):
    """Adds mag, mag_err to the catalogue; returns the zero point (zp, gx, gy, c, rms, n) or None if it cannot be fitted."""
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    from hdps.photometry.zeropoint import GR_PIVOT, MINIPIX_WIDTH, atlas_column_for_band, fit_zeropoint, fit_zeropoint_plane

    c = {**SETTINGS, **cfg}
    band = frame.metadata["band"]
    epoch = Time(frame.metadata["date_obs"]).jyear
    ref, ref_ra, ref_dec = reference_stars(frame.ra_center, frame.dec_center, c["refcat_radius_deg"],
                                           c["refcat_max_mag"], epoch)
    flux = np.asarray(catalogue["flux"], float)
    det_mag = -2.5 * np.log10(np.clip(flux, 1e-12, None)) + 23.0  # hdps's instrumental magnitude (DET_MAG_ZP 23)
    det = SkyCoord(np.asarray(catalogue["ra"]) * u.deg, np.asarray(catalogue["dec"]) * u.deg)
    j, sep, _ = det.match_to_catalog_sky(SkyCoord(ref_ra * u.deg, ref_dec * u.deg))
    matched = sep.arcsec <= c["match_arcsec"]
    gr = np.where(matched, np.asarray(ref["g"], float)[j] - np.asarray(ref["r"], float)[j], np.nan)
    good = (matched & ~np.asarray(catalogue["saturated"], bool)
            & (flux >= c["min_snr"] * np.asarray(catalogue["flux_err"], float)))
    ref_mag = np.where(good, np.asarray(ref[atlas_column_for_band(band)], float)[j], np.nan)
    ny, nx = frame.data.shape
    x, y = np.asarray(catalogue["xcentroid"], float), np.asarray(catalogue["ycentroid"], float)
    fit = fit_zeropoint_plane(det_mag, ref_mag, x, y, (nx / 2, ny / 2), gr)
    tier = "plane"
    if fit is None:
        scalar = fit_zeropoint(det_mag, ref_mag)
        if scalar is None:
            catalogue["mag"], catalogue["mag_err"], catalogue["refcat_match"] = np.nan, np.nan, matched
            return None
        fit, tier = (scalar[0], 0.0, 0.0, None, scalar[1], scalar[2]), "scalar"
    zp, gx, gy, colour, rms, n = fit
    plane = zp + gx * (x - nx / 2) / MINIPIX_WIDTH + gy * (y - ny / 2) / MINIPIX_WIDTH
    if colour is not None:
        plane = plane + colour * (np.where(np.isfinite(gr), gr, GR_PIVOT) - GR_PIVOT)
    catalogue["mag"] = det_mag - 23.0 + plane
    catalogue["mag_err"] = 2.5 / np.log(10) * np.asarray(catalogue["flux_err"], float) / flux
    catalogue["refcat_match"] = matched
    return dict(zp=zp, gx=gx, gy=gy, colour=colour, rms=rms, n=n, tier=tier)
