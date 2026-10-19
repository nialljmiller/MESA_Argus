"""Difference imaging: transient candidates in a frame, against its reference.

hdps_zogy: hdps's ZOGY path, as its ZOGYSubtractionTask and ZOGYDetectionTask run it (tasks.py):
- inputs: the frame and the reference each minus its robust background level, with its robust sigma (median and
  clipped std of every 4th pixel); the PSFs: the frame's fitted by the psf stage, the reference's from its file; flux
  ratio F_n = 10^(0.4 (frame zero point - reference zero point)); the reference's PSF-model-error variance plane.
- subtraction: hdps.diffimg ZOGY (ZOGYReferenceSubtractor on the CPU, the same equations as ZOGYGPUSubtractor), with
  hdps's astrometric noise term (sigma 0.032 px).
- candidates: S_corr divided by its own per-region robust scale (hdps.diffimg.detection.calibrate_scorr, 256 px), then
  positive and negative peaks above `dia_threshold` sigma (extract_candidates, 4 px apart), flux = S / F_S.
Two departures, for want of hdps's real/bogus classifier (AlertVettingTask, which needs a trained model): the
threshold is above hdps's 5 sigma, and no candidate is taken within `dia_saturation_px` of a pixel saturated in the
frame or in any of the reference's frames. No candidate is taken within `dia_edge_px` of the cut-out's edge either:
ZOGY's FFTs treat the image as periodic, so a star near one edge leaves a residual at the opposite edge. A frame
without enough PSF stars is skipped, as hdps skips the tile.
"""
import numpy as np

from .reference import robust_background

SETTINGS = dict(dia_threshold=8.0, dia_region_size=256, dia_min_separation=4, astrometric_sigma=0.032,
                dia_saturation_px=10, dia_edge_px=20)


def hdps_zogy(frame, reference, frame_zp, fit_psf, cfg):
    """The frame's candidates (astropy Table: x, y, scorr, flux, flux_err, npix_valid_3x3, is_negative, ra, dec) and
    the difference image (ZOGY's D; None when skipped)."""
    from astropy.table import Table
    from hdps.diffimg import ZOGYReferenceSubtractor
    from hdps.diffimg.detection import CANDIDATE_DTYPE, calibrate_scorr, extract_candidates
    from scipy.ndimage import binary_dilation

    from . import psf

    c = {**SETTINGS, **cfg}
    model = fit_psf(frame, cfg)
    if model is None:
        candidates = Table(np.zeros(0, CANDIDATE_DTYPE))
        candidates["ra"], candidates["dec"] = np.zeros(0), np.zeros(0)
        return candidates, None
    level_n, sigma_n = robust_background(frame.data)
    level_r, sigma_r = robust_background(reference["sci"])
    f_n = 10 ** (0.4 * (frame_zp - reference["magzp"])) if np.isfinite(frame_zp) and np.isfinite(reference["magzp"]) else 1.0
    zogy = ZOGYReferenceSubtractor(include_astrometric_noise=True, astrometric_sigma=c["astrometric_sigma"])
    result = zogy.subtract(reference["sci"] - level_r, frame.data - level_n, reference["psf"], psf.stamp(model, frame, cfg),
                           sigma_r, sigma_n, F_r=1.0, F_n=f_n, extra_var_r=reference["pmerr"])
    saturated = reference["dq"] | (frame.data >= frame.metadata["satlevel"])
    valid = (reference["npix"] > 0) & ~binary_dilation(saturated, iterations=c["dia_saturation_px"])
    e = c["dia_edge_px"]
    valid[:e], valid[-e:], valid[:, :e], valid[:, -e:] = False, False, False, False
    s_cal, _ = calibrate_scorr(result.S_corr, valid, region_size=c["dia_region_size"])
    rows = extract_candidates(s_cal, result.S / result.F_S, valid, threshold=c["dia_threshold"],
                              min_separation=c["dia_min_separation"])
    candidates = Table(rows)
    ra, dec = frame.full_wcs.wcs_pix2world(candidates["x"], candidates["y"], 0) if len(candidates) else ([], [])
    candidates["ra"], candidates["dec"] = ra, dec
    return candidates, result.D
