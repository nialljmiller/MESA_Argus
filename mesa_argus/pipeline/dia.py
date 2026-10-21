"""Difference imaging: hdps's production chain, as every hdps pipeline config runs it (docker/demo/config*.yaml):
SFFTSubtractionTask -> DiffDetectionTask -> TransientOutputTask (then AlertVettingTask: alerts.py).

hdps_sfft mirrors those tasks (hdps.orchestrate.tasks) on the CPU, with hdps's own NumPy code where it has it:
- SFFTSubtractionTask: hdps.diffimg.SFFTReferenceSubtractor (hdps's NumPy SFFT, the oracle its CUDA SFFT is tested
  against; the GPU's constrained_kernel_sum is its fp32 conditioning device, the NumPy solve is fp64) with the
  deployed config: kernel half-width 6, constant kernel (kernel_poly_order 0), background order 2, one pass of
  normalised-residual variable-source rejection (3 sigma, 2 px grow); non-finite pixels zeroed; the fit masks the
  science's saturated pixels dilated by the kernel half-width (_saturated_fit_mask). Var(D) as
  _store_difference_variance: Var(science) + max(0, science - sky) + Var(reference) (x) K^2 (zero-padded direct
  convolution, variance_conv "direct"), inf where the reference has no coverage. The science sky level is SFFT's
  metric, the science mean.
- DiffDetectionTask: hdps's SourceDetector (its CPU mirror) on D, detect_sigma 5, 256 px regions, 4 px matched filter,
  Var(D) as the per-pixel noise model, and the background from _build_diff_background: zero level, rms measured from
  the matched-filtered D per region (three median/MAD 3-sigma clips, diff_detection.cu), floored at
  sqrt(0.663 sky).
- TransientOutputTask: the detections with flux above 1e-6 e- (the dipole floor), with RA/Dec.

One input differs from production because the simulation has no reprojection inverse-variance plane (W): Var(science)
is the frame's measured sky variance (clipped std of every 4th pixel, as reference.robust_background), where hdps
reads 1/W.
"""
import numpy as np

from .reference import robust_background

SETTINGS = dict(sfft_kernel_half_width=6, sfft_kernel_poly_order=0, sfft_background_order=2, sfft_rejection_iters=1,
                sfft_rejection_sigma=3.0, sfft_rejection_grow=2, diff_detect_sigma=5.0, diff_bg_region_size=256,
                diff_matched_filter_width=4.0)
DIPOLE_FLUX_FLOOR = 1e-6  # tasks._DIPOLE_FLUX_FLOOR


def _dilate_box(mask, radius):
    """SFFTSubtractionTask._dilate_box: a pixel within +-radius (box) of a set pixel is set."""
    from scipy.ndimage import maximum_filter

    return maximum_filter(mask.astype(np.uint8), size=2 * int(radius) + 1, mode="constant") > 0 if radius else mask


def _convolve_reference_variance(var_ref, kernel):
    """Var(ref) (x) kernel^2, zero-padded direct convolution (variance_conv "direct"), clamped at 0."""
    from scipy.ndimage import convolve

    return np.maximum(convolve(var_ref, kernel**2, mode="constant", cval=0.0), 0.0)


def _regional_clipped_rms(filtered, valid, region):
    """diff_detection.cu regional_clipped_rms: per region, three iterations of median / MAD and a 3 x 1.4826 MAD clip,
    then 1.4826 x the final MAD; NaN where a region has no clean pixel."""
    n_y, n_x = filtered.shape[0] // region, filtered.shape[1] // region
    out = np.full((n_y, n_x), np.nan, np.float32)
    f = np.where(valid, filtered, np.nan).astype(np.float32)
    for j in range(n_y):
        for i in range(n_x):
            x = f[j * region:(j + 1) * region, i * region:(i + 1) * region].ravel()
            x = x[np.isfinite(x)]
            for _ in range(3):
                if x.size == 0:
                    break
                med = np.median(x)
                cut = 3.0 * 1.4826 * np.median(np.abs(x - med))
                if np.isfinite(cut):
                    x = x[~(np.abs(x - med) > cut)]
            if x.size:
                out[j, i] = 1.4826 * np.median(np.abs(x - np.median(x)))
    return out


def _diff_background(D, var_d, sky_level, width, region):
    """DiffDetectionTask._build_diff_background: (bg_map zeros, bg_rms_map) per region."""
    n_y, n_x = D.shape[0] // region, D.shape[1] // region
    sky_noise = np.sqrt(0.663 * max(sky_level, 0.0))
    mf_sigma = width / (2.355 * 2.0)
    fw = int(width * 1.5)
    yy, xx = np.mgrid[-fw:fw, -fw:fw]
    k = np.exp(-(yy**2 + xx**2) / (2.0 * mf_sigma**2)).astype(np.float32)
    valid = np.isfinite(D) & np.isfinite(var_d)
    clean = np.where(valid, D, 0.0).astype(np.float32)
    kpad = np.zeros(D.shape, np.float32)
    kpad[:k.shape[0], :k.shape[1]] = k
    kpad = np.roll(kpad, (-(k.shape[0] // 2), -(k.shape[1] // 2)), axis=(0, 1))
    filtered = np.fft.irfft2(np.fft.rfft2(clean) * np.fft.rfft2(kpad), s=D.shape)
    rms = _regional_clipped_rms(filtered, valid, region) / np.sqrt(float((k**2).sum()))
    rms = np.maximum(np.where(np.isfinite(rms), rms, 0.0), sky_noise).astype(np.float32)
    return np.zeros((n_y, n_x), np.float32), rms


def hdps_sfft(frame, reference, cfg):
    """The frame differenced against its reference. Returns dict(D, var, kernel, sky, science, reference,
    transients (astropy Table: TransientOutputTask's columns), solve_failed)."""
    from astropy.table import Table
    from hdps.diffimg import SFFTReferenceSubtractor
    from hdps.photometry import SourceDetector

    from . import frames

    c = {**SETTINGS, **cfg}
    hw = int(c["sfft_kernel_half_width"])
    science = np.nan_to_num(np.asarray(frame.data, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.nan_to_num(np.asarray(reference["sci"], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    saturated = np.asarray(frame.data) >= frame.metadata["satlevel"]  # the science's DQ SATURATED
    fit_mask = ~_dilate_box(saturated, hw) if saturated.any() else None
    var_sci = robust_background(frame.data)[1] ** 2  # in place of 1/W (see the module notes)
    var_ref = np.where(reference["npix"] > 0, reference["bkgvar"] / np.maximum(reference["npix"], 1), 0.0)
    subtractor = SFFTReferenceSubtractor(kernel_half_width=hw, background_order=int(c["sfft_background_order"]),
                                         kernel_poly_order=int(c["sfft_kernel_poly_order"]))
    try:
        result = subtractor.subtract(ref, science, mask=fit_mask)
        for _ in range(int(c["sfft_rejection_iters"])):  # _rejection_masks
            D = np.asarray(result.difference, np.float32)
            sky = float(np.mean(science))
            var = var_sci + np.maximum(science - sky, 0.0) + float(np.sum(np.asarray(result.kernel) ** 2)) * var_ref
            bad = np.abs(D) > c["sfft_rejection_sigma"] * np.sqrt(var)
            if fit_mask is not None:
                bad &= fit_mask
            bad = _dilate_box(bad, int(c["sfft_rejection_grow"]))
            if not bad.any():
                break
            new_mask = ~bad if fit_mask is None else (fit_mask & ~bad)
            try:
                result = subtractor.subtract(ref, science, mask=new_mask)
                fit_mask = new_mask
            except np.linalg.LinAlgError:  # _merge_refit: keep the first pass
                break
    except np.linalg.LinAlgError:
        return dict(solve_failed=True, transients=None)

    D = np.asarray(result.difference, np.float32)
    sky = float(np.mean(science))  # metrics.science_sky_level
    kernel = np.asarray(result.kernel, np.float64)
    var_d = (var_sci + np.maximum(science - sky, 0.0)).astype(np.float32)
    if np.any(var_ref):
        var_d += _convolve_reference_variance(var_ref, kernel).astype(np.float32)
    var_d[reference["npix"] <= 0] = np.inf  # reference gap: NO_REFERENCE

    # DiffDetectionTask + TransientOutputTask
    detector = SourceDetector(detect_sigma=c["diff_detect_sigma"], bg_region_size=int(c["diff_bg_region_size"]),
                              max_sources=100_000, matched_filter_width=c["diff_matched_filter_width"])
    diff_frame = frames.make_frame(D, frame.full_wcs, frame.tile_id, dict(frame.metadata, sfft_science_sky_level=sky))
    background = _diff_background(D, var_d, sky, c["diff_matched_filter_width"], int(c["diff_bg_region_size"]))
    table = detector.detect_sources(diff_frame, precomputed_background=background, pixel_variance=var_d).sources.table
    keep = np.asarray(table["flux"], float) > DIPOLE_FLUX_FLOOR
    t = table[keep]
    x, y = np.asarray(t["xcentroid"], float), np.asarray(t["ycentroid"], float)
    ra, dec = frame.full_wcs.all_pix2world(x, y, 0) if len(t) else (np.zeros(0), np.zeros(0))
    transients = Table(dict(tanseg_id=np.full(len(t), int(frame.tile_id)), x=x, y=y, ra_deg=ra, dec_deg=dec,
                            flux=np.asarray(t["flux"], float), snr=np.asarray(t["snr"], float),
                            radius=np.asarray(t["radius"], float), is_saturated=np.asarray(t["is_saturated"], bool),
                            obs_time=[frame.metadata["date_obs"]] * len(t)))
    return dict(D=D, var=var_d, kernel=kernel, sky=sky, science=science, reference=ref, transients=transients,
                solve_failed=False)
