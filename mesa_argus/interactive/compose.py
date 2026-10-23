"""The edited star added into hdps's own frames (the field pack's block cut-outs of hdps's science tile and difference
image), and the airmass offset applied to them.

Stand-ins, each stated on screen:
- the star's PSF: ArgusSim's (ZemaxPSF at the frame's seeing and field angle) resampled onto hdps's tile grid
  (0.879"), without the small rotation between the sensor and the tile;
- its noise: Poisson in the electrons it adds, the same draw in the science and the difference image (the reference
  is deep); the frame's own noise is hdps's;
- in the difference image it adds (now - reference) x PSF: SFFT's match taken as perfect for it;
- airmass offset dX >= 0 (observing it lower): every frame dimmed by the band's extinction, blurred to the seeing at
  the new airmass (X^0.6), and given the extra sky noise (sky ~ X); the field's own alerts and photometry stay as
  observed.
"""
from functools import lru_cache

import numpy as np

SKY_PX = 900.0 / 1024.0  # hdps tile pixel (arcsec)


@lru_cache(maxsize=1)
def _psfs():
    from argus_sim import c as cfg

    from mesa_argus import sensor

    return sensor.PSFs(cfg)


def plate():
    from argus_sim import c as cfg

    return float(cfg.telescope.plate_scale)


@lru_cache(maxsize=4096)
def _stamp(band, seeing, angle_index):
    return np.asarray(_psfs().stamp(band, round(float(seeing), 2), int(angle_index), 0, 0), float)


def psf_on_tile(band, seeing, field_angle, dx, dy, extent, shape, half_arcsec=20.0):
    """Unit-flux PSF of a star at (dx, dy) arcsec on a cut-out with this extent (x0, x1, y0, y1 arcsec) and shape:
    (rows slice, cols slice, values) of the patch around it."""
    from scipy.ndimage import map_coordinates

    angles = _psfs().angles(band)
    stamp = _stamp(band, seeing, int(np.abs(angles - field_angle).argmin()))
    yy, xx = np.indices(stamp.shape)
    cy, cx = float((stamp * yy).sum()), float((stamp * xx).sum())
    ny, nx = shape
    px_x = (extent[1] - extent[0]) / nx
    px_y = (extent[3] - extent[2]) / ny
    i0 = max(int((dx - half_arcsec - extent[0]) / px_x), 0)
    i1 = min(int((dx + half_arcsec - extent[0]) / px_x) + 1, nx)
    j0 = max(int((dy - half_arcsec - extent[2]) / px_y), 0)
    j1 = min(int((dy + half_arcsec - extent[2]) / px_y) + 1, ny)
    if i1 <= i0 or j1 <= j0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0))
    x = extent[0] + (np.arange(i0, i1) + 0.5) * px_x  # tile pixel centres (arcsec)
    y = extent[2] + (np.arange(j0, j1) + 0.5) * px_y
    u = cx + (x - dx) / plate()  # where they fall on the sensor-pixel stamp
    v = cy + (y - dy) / plate()
    vv, uu = np.meshgrid(v, u, indexing="ij")
    values = np.clip(map_coordinates(stamp, [vv, uu], order=1, mode="constant", cval=0.0), 0.0, None)
    total = values.sum()
    return slice(j0, j1), slice(i0, i1), (values / total if total > 0 else values)


def extinction_ratio(band, x0, x1):
    """The fraction of a star's light left at airmass x1 against x0 in a band (ArgusSim's atmosphere, a flat
    spectrum); airmasses to 0.01."""
    return _extinction_ratio(band, round(float(x0), 2), round(float(x1), 2))


@lru_cache(maxsize=4096)
def _extinction_ratio(band, x0, x1):
    import astropy.units as u

    import night_movie as nm
    from mesa_argus.interactive.templates import photometry

    ab = photometry()
    wave = np.linspace(3000.0, 10500.0, 400) * u.AA
    flat = np.full(400, 1e-16) * nm.FLAM
    e0 = ab.electrons_from_sed(nm.BANDS[band], wave, flat, airmass=x0).value
    e1 = ab.electrons_from_sed(nm.BANDS[band], wave, flat, airmass=x1).value
    return float(e1 / e0)


def frame(cut, band, seeing, field_angle, airmass, star_e, ref_e, dx, dy, d_airmass=0.0, seed=0, sky=None):
    """One frame's cut-outs with the edited star added (and the airmass offset): dict(science, difference, extent,
    psf (rows, cols, values), added (star electrons per pixel), seeing (now), keep (the extinction factor applied)). star_e / ref_e: the star's electrons in this exposure
    and in the reference (already at the offset airmass); sky: the science cut-out's sky level (its median if not
    given)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    sci = np.array(cut["science"], dtype=np.float64)
    diff = np.array(cut["difference"], dtype=np.float64)
    extent, shape = cut["extent"], sci.shape
    seeing_now, keep = seeing, 1.0
    if d_airmass > 0:  # observed lower: extinction, broader seeing, more sky
        x1 = airmass + d_airmass
        sky = float(np.nanmedian(sci)) if sky is None else float(sky)
        keep = extinction_ratio(band, airmass, x1)
        seeing_now = seeing * (x1 / airmass) ** 0.6
        sigma = np.sqrt(max(seeing_now ** 2 - seeing ** 2, 0.0)) / 2.3548 / SKY_PX
        stars = np.nan_to_num(sci - sky)
        sci = gaussian_filter(stars, sigma) * keep + sky * (x1 / airmass)
        diff = gaussian_filter(np.nan_to_num(diff), sigma) * keep
        extra = rng.normal(0.0, np.sqrt(max(sky * (x1 / airmass - 1.0), 0.0)), shape)
        sci += extra
        diff += extra
    rows, cols, psf = psf_on_tile(band, seeing_now, field_angle, dx, dy, extent, shape)
    added = star_e * psf
    noise = rng.normal(0.0, 1.0, psf.shape) * np.sqrt(np.clip(added, 0.0, None))
    sci[rows, cols] += added + noise
    diff[rows, cols] += (star_e - ref_e) * psf + noise
    return dict(science=sci, difference=diff, extent=extent, psf=(rows, cols, psf), added=added,
                seeing=seeing_now, keep=keep)
