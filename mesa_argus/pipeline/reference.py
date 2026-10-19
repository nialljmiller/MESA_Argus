"""The reference image a frame is subtracted from, written as hdps's CoaddReferenceProvider reads it.

sacrificial_night: the per-pixel median of every tile of one night observed before the observation (observe.py
observes it first: the same field and stars, without the transients), each tile's robust background level removed
first. Then, on the median, as for any frame: the zero point (calibration stage) and the PSF (psf stage); and hdps's
PSF commissioning products against that PSF (hdps.photometry.psf.commissioning.measure_coadd_psf_products, anchored on
the fitted model): the measured ePSF and the PSF-model-error kernels and catalogue, from which hdps's ZOGY builds
the reference's PSF-model-error variance plane (V_err; it suppresses false candidates on stars).

The file (hdps's contract, tasks.CoaddReferenceProvider): <name>_tile_NNNNNN_<band>.fits; SCI (electrons per
exposure, background 0, with the cut-out's WCS), NPIX (frames combined per pixel), EPSF, PMERRKER, PMERRCAT; primary
header NCOMBINE, BKGVAR (mean per-frame background variance), MAGZP, BAND. Also, ours: PSF (the fitted model at the
tile centre, which hdps's psf_provider supplies instead), DQ (1 where any frame reached its saturation level), and what
an alert reports about its reference: TSTART / TSTOP (Argus epoch, TT seconds since 2025.0, of the first exposure's
start and the last one's end) and DEPTH5 (5 sigma point-source limiting AB magnitude: zero point and the median's
measured noise through its PSF).
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def robust_background(image):
    """(level, sigma) of the sky: median and clipped standard deviation of every 4th pixel (as hdps's ZOGY task)."""
    sub = image[::4, ::4].ravel()
    sub = sub[np.isfinite(sub)]
    med = float(np.median(sub))
    mad = 1.4826 * float(np.median(np.abs(sub - med)))
    core = sub[np.abs(sub - med) < 4 * mad]
    return med, float(np.std(core)) if core.size > 100 else mad


def sacrificial_night(frame_files, out_dir, name, band, stages, cfg):
    """Median of the night's tiles (FITS written by the engine) -> the reference file; returns its path. `stages`:
    the extraction, calibration and psf functions."""
    from hdps.core.timestamp import isot_to_argus_epoch
    from hdps.photometry.psf.commissioning import measure_coadd_psf_products

    from . import frames, psf

    stack, bkg_var, header, epochs, saturated = [], [], None, [], None
    for path in frame_files:
        with fits.open(path) as h:
            image = np.asarray(h[0].data, float)
            header = header or h[0].header
            epochs.append(isot_to_argus_epoch(h[0].header["DATE_OBS"]))
            hit = image >= float(h[0].header["SATLEVEL"])
        saturated = hit if saturated is None else saturated | hit
        level, sigma = robust_background(image)
        stack.append(image - level)
        bkg_var.append(sigma**2)
    median = np.median(stack, axis=0)
    wcs = WCS(header)
    tile_id = int(header["TILE_ID"])
    meta = dict(band=band, date_obs=header["DATE_OBS"], satlevel=float(header["SATLEVEL"]), exptime=float(header["EXPTIME"]))
    frame = frames.make_frame(median, wcs, tile_id, meta)
    zp = stages["calibration"](stages["extraction"](frame, cfg), frame, cfg)
    model = stages["psf"](frame, cfg)
    if model is None:
        raise SystemExit(f"reference {band}: too few PSF stars in the median of {len(stack)} frames")
    psf_r = psf.stamp(model, frame, cfg)
    products = measure_coadd_psf_products(median, ~saturated, psf_r)
    sigma_r = robust_background(median)[1]
    depth = float(zp["zp"]) - 2.5 * np.log10(5 * sigma_r / np.sqrt(np.sum(psf_r**2))) if zp else np.nan
    half_exposure = float(header["EXPTIME"]) / 2

    primary = fits.PrimaryHDU()
    primary.header.update(NCOMBINE=len(stack), BKGVAR=float(np.mean(bkg_var)), BAND=band, TILE_ID=tile_id,
                          MAGZP=float(zp["zp"]) if zp else np.nan, DEPTH5=depth, TSTART=min(epochs) - half_exposure,
                          TSTOP=max(epochs) + half_exposure, ORIGIN="MESA_ARGUS pipeline.reference.sacrificial_night")
    hdus = [primary, fits.ImageHDU(median.astype(np.float32), wcs.to_header(), name="SCI"),
            fits.ImageHDU(np.full(median.shape, len(stack), np.int32), name="NPIX"),
            fits.ImageHDU(saturated.astype(np.uint8), name="DQ"), fits.ImageHDU(psf_r.astype(np.float32), name="PSF")]
    if products is not None:
        kernels = np.asarray(products["kernels"], np.float32)
        hdus += [fits.ImageHDU(np.asarray(products["epsf"], np.float32), name="EPSF"),
                 fits.ImageHDU(kernels.reshape(-1, *kernels.shape[-2:]), name="PMERRKER"),
                 fits.ImageHDU(np.asarray(products["catalog"], np.float32), name="PMERRCAT")]
    out = Path(out_dir) / f"{name}_tile_{tile_id:06d}_{band}.fits"
    out.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(out, overwrite=True)
    return out


def load(reference_dir, tile_id, band):
    """The reference for a tile and band, or None: dict(sci, npix, dq, psf, pmerr (hdps's V_err plane, or None),
    magzp, ncombine, tstart, tstop, depth5, tile_id)."""
    matches = sorted(Path(reference_dir).glob(f"*_tile_{tile_id:06d}_{band}.fits"))
    if not matches:
        return None
    with fits.open(matches[-1]) as h:
        hd = h[0].header
        sci = np.asarray(h["SCI"].data, float)
        pmerr = None
        if "PMERRKER" in h and len(h["PMERRCAT"].data):
            from hdps.photometry.psf.commissioning import assemble_psf_model_error_plane

            kernels = np.asarray(h["PMERRKER"].data, float)
            nb = int(round(np.sqrt(kernels.shape[0])))
            pmerr = assemble_psf_model_error_plane(np.asarray(h["PMERRCAT"].data, float),
                                                   kernels.reshape(nb, nb, *kernels.shape[1:]), sci.shape)
        return dict(sci=sci, npix=np.asarray(h["NPIX"].data, float), dq=np.asarray(h["DQ"].data, bool),
                    psf=np.asarray(h["PSF"].data, float), pmerr=pmerr, magzp=float(hd["MAGZP"]),
                    ncombine=int(hd["NCOMBINE"]), tstart=float(hd["TSTART"]), tstop=float(hd["TSTOP"]),
                    depth5=float(hd["DEPTH5"]), tile_id=int(hd["TILE_ID"]))
