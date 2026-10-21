"""The PSF of a frame or reference: what difference imaging uses as its matched filter.

hdps_moffat: hdps's per-tile PSF fit, as its TilePSFTask makes it (tasks.py): clean, isolated ATLAS-Refcat2 stars in
the magnitude window (hdps.photometry.psf.select_stars), then a spatially varying pixel-integrated Moffat field fitted
to them (FramePSFBuilder: Legendre degree 2 fields, beta fitted, oversample 8; NumPy backend without a GPU). hdps's
ZOGY uses this analytic model as the filter, not the stacked ePSF (ZOGYSubtractionTask: prefer_measured_epsf off, as
a stacked ePSF "carries sub-pixel phase and pedestal error"); the ePSF's residuals enter as the reference's
PSF-model-error variance instead (reference.py).

psf_min_stars is 8, FramePSFBuilder's own floor, not TilePSFTask's 12: the frames here are cut-outs of a tile (the
movie's block, ~548 px of the tile's 2048), with ~1/14 of a tile's stars; f294's beta cut-out has 10 clean ones.
With that few the builder fits a constant PSF across the cut-out (Legendre degree 0).

psf_cadence "per_ratchet" (hdps TilePSFTask's build_cadence): the model is fitted on a ratchet's first frame (one
telescope at one pointing; frame metadata tel_id, ratchet) and used for the ratchet's other frames. The simulation's
seeing and PSF are set per ratchet, so this is the same PSF; the fit is ~85% of a frame's pipeline time. "per_frame"
fits every frame. Frames without a ratchet (ideal observing) are fitted every frame.
"""
import numpy as np

SETTINGS = dict(psf_mag_bright=12.0, psf_mag_faint=17.0, psf_select_stamp=25, psf_max_stars=200, psf_min_stars=8,
                psf_stamp=41, psf_cadence="per_ratchet")


def with_cadence(fit, cfg):
    """fit (a psf stage function) applied at psf_cadence: per_ratchet reuses a ratchet's model for its other frames."""
    if {**SETTINGS, **cfg}["psf_cadence"] == "per_frame":
        return fit
    if {**SETTINGS, **cfg}["psf_cadence"] != "per_ratchet":
        raise SystemExit(f"psf_cadence {cfg['psf_cadence']!r}: per_ratchet or per_frame")
    models = {}

    def per_ratchet(frame, cfg):
        key = (frame.metadata.get("tel_id"), frame.metadata.get("ratchet"))
        if key[1] is None:
            return fit(frame, cfg)
        if key not in models:
            models[key] = fit(frame, cfg)
        return models[key]

    return per_ratchet


def hdps_moffat(frame, cfg):
    """The frame's MoffatPSFModel, or None when too few stars (hdps then skips the tile)."""
    from astropy.table import Table
    from astropy.time import Time
    from hdps.photometry.psf import BuildError, FramePSFBuilder, select_stars
    from hdps.photometry.zeropoint import atlas_column_for_band

    from .calibration import SETTINGS as CAL, reference_stars

    c = {**SETTINGS, **CAL, **cfg}
    ny, nx = frame.data.shape
    ra0, dec0 = frame.full_wcs.wcs_pix2world([(nx - 1) / 2], [(ny - 1) / 2], 0)
    refcat, ra, dec = reference_stars(float(ra0[0]), float(dec0[0]), c["refcat_radius_deg"], c["refcat_max_mag"],
                                      Time(frame.metadata["date_obs"]).decimalyear)
    stars = Table(dict(rix=np.arange(len(refcat)), ref_ra=ra, ref_dec=dec,
                       ref_mags=np.asarray(refcat[atlas_column_for_band(frame.metadata["band"])], float)))
    selection = select_stars(frame, stars, mag_range=(c["psf_mag_bright"], c["psf_mag_faint"]),
                             stamp_size=c["psf_select_stamp"], max_stars=c["psf_max_stars"])
    if len(selection) < c["psf_min_stars"]:
        return None
    try:
        return FramePSFBuilder(poly_degree=2, beta0=3.0, fit_beta=True, min_stars=c["psf_min_stars"], oversample=8,
                               backend="numpy").build_image(frame, selection)
    except BuildError:
        return None


def fwhm_px(model, frame, fallback=4.5):
    """The model's FWHM (tile pixels) at the frame's centre; fallback (hdps.alerts.features' survey-typical 4.5 px)
    without a model."""
    if model is None:
        return fallback
    ny, nx = frame.data.shape
    ra, dec = frame.full_wcs.wcs_pix2world([(nx - 1) / 2], [(ny - 1) / 2], 0)
    return float(np.atleast_1d(model.fwhm(ra=np.atleast_1d(float(ra[0])), dec=np.atleast_1d(float(dec[0]))))[0])


def stamp(model, frame, cfg):
    """The model at the frame's centre: a unit-sum psf_stamp x psf_stamp image (what ZOGY takes per tile)."""
    size = {**SETTINGS, **cfg}["psf_stamp"]
    ny, nx = frame.data.shape
    ra, dec = frame.full_wcs.wcs_pix2world([(nx - 1) / 2], [(ny - 1) / 2], 0)
    image = np.asarray(model.image(ra=float(ra[0]), dec=float(dec[0]), size=size), float)
    return image / image.sum()
