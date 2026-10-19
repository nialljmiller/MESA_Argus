"""Every source in a frame, found and measured blind (nothing about the simulation is used).

hdps: hdps.photometry.SourceDetector (GPUSourceDetector on a CUDA host, its CPU mirror otherwise) with the settings
hdps's SourceDetectionTask runs in production: background in 64 px regions, local maxima above detect_sigma confirmed
by a matched filter (4 px FWHM), cosmic-ray test, iterated centroid, flux in a 4 px radius aperture minus a 15-25 px
annulus background, detections within 6 px merged. RA/Dec come from the frame's tile WCS.

saturated: the brightest pixel within 2 px of a detection is at the frame's saturation level (metadata satlevel, from
the detector's full well). hdps's own is_saturated is set at 0.9999 x 65534, a 16-bit ADC level these electron
images never reach.
"""
import numpy as np

SETTINGS = dict(detect_sigma=3.0, bg_region_size=64, flux_radius=4.0, matched_filter_width=4.0, de_dup_radius=6.0,
                cr_detect_contrast=0.25)


def hdps(frame, cfg):
    """The frame's catalogue: xcentroid, ycentroid, flux, flux_err, snr, ... (hdps's columns) + ra, dec."""
    from hdps.photometry import SourceDetector

    detector = SourceDetector(**{k: cfg.get(k, v) for k, v in SETTINGS.items()})
    table = detector.detect_sources(frame).sources.table
    ra, dec = frame.full_wcs.wcs_pix2world(np.asarray(table["xcentroid"], float), np.asarray(table["ycentroid"], float), 0)
    table["ra"], table["dec"] = ra, dec
    image, (ny, nx) = frame.data, frame.data.shape
    peak = [image[max(int(round(b)) - 2, 0):min(int(round(b)) + 3, ny), max(int(round(a)) - 2, 0):min(int(round(a)) + 3, nx)].max()
            for a, b in zip(table["xcentroid"], table["ycentroid"])]
    table["saturated"] = np.asarray(peak, float) >= frame.metadata["satlevel"]
    return table
