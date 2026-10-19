"""Source extraction and photometry of the simulated exposures (night_movie.py), with hdps's own source detector.

Each exposure, in three steps:

1. extraction: hdps.photometry.SourceDetector (GPUSourceDetector on a CUDA host, its CPU mirror otherwise) with the
   settings hdps's SourceDetectionTask runs in production: background in 64 px regions, local maxima above 3 sigma
   confirmed by a matched filter (4 px FWHM), cosmic-ray test, iterated centroid, flux in a 4 px radius circular
   aperture around it (sub-pixel rim weights) minus a 15-25 px annulus background, detections within 6 px merged
   (the brightest kept). The result is the exposure's full catalogue of detections.
2. association with the simulated stars: the stars in order of expected flux, brightest first, each take the nearest
   detection not yet taken within match_radius pixels. One detection belongs to at most one star; a star with no
   detection (too faint, or merged into a brighter neighbour's detection) is unmeasured in that exposure.
3. zero point: catalogue stars (the least blended of the bright ones, unsaturated, with Gaia colours, measured at
   S/N >= min_cal_snr) give
   mag + 2.5 log10(flux) = a + b (BP-RP), clipped at clip_sigma (MAD) and refitted.

Saturation: a star is flagged when the brightest pixel within 2 px of its detection is at 98% of the full well.
(hdps flags saturation at 0.9999 x 65534, a 16-bit ADC level; these images are in electrons with a smaller full well.)
"""
import time

import numpy as np

# every setting an observation can change ([measure] in observe.py's file); the detector's are hdps's production values
SETTINGS = dict(
    detect_sigma=3.0,          # detection threshold (sigma)
    bg_region_size=64,         # background region (pixels)
    flux_radius=4.0,           # aperture radius (pixels)
    matched_filter_width=4.0,  # matched filter FWHM (pixels)
    de_dup_radius=6.0,         # detections closer than this (pixels) are merged, the brightest kept
    cr_detect_contrast=0.25,   # cosmic-ray test: neighbours below this fraction of the peak
    match_radius=2.0,          # association with the simulated stars (pixels)
    n_calibrators=300,         # calibration stars per exposure (the least blended of the brightest 4x this many)
    min_cal_snr=50.0,          # a calibration star counts only if measured at least this well (flux / error)
    colour_term=True,          # zero point a + b (BP-RP); false: a only
    clip_sigma=3.0,            # zero-point fit: reject calibrators beyond this many (MAD) sigma, refit
)
DETECTOR_KEYS = ("detect_sigma", "bg_region_size", "flux_radius", "matched_filter_width", "de_dup_radius",
                 "cr_detect_contrast")


def settings(given=None):
    """SETTINGS with `given` applied; unknown keys stop the run."""
    given = dict(given or {})
    bad = sorted(set(given) - set(SETTINGS))
    if bad:
        raise SystemExit(f"[measure]: unknown keys {bad}; it takes {sorted(SETTINGS)}")
    return {**SETTINGS, **given}


def calibration_stars(x, y, cat_flux, n, k=300, edge=16, radius=8.0):
    """The k bright stars least contaminated by neighbours, away from the edge. Contamination of a star: catalogue
    flux of its neighbours, Gaussian-weighted by distance (sigma = radius/2 pixels), over its own flux."""
    candidates = np.nonzero((x > edge) & (x < n - edge) & (y > edge) & (y < n - edge) & np.isfinite(cat_flux))[0]
    candidates = candidates[np.argsort(-cat_flux[candidates])][: 4 * k]
    contamination = []
    for i in candidates:
        d = np.hypot(x - x[i], y - y[i])
        near = (d > 0) & (d < 3 * radius)
        contamination.append(np.sum(cat_flux[near] * np.exp(-0.5 * (d[near] / (radius / 2)) ** 2)) / cat_flux[i])
    return candidates[np.argsort(contamination)[:k]]


def extract(image, cfg):
    """hdps's source detector on one exposure (electrons): its catalogue as an astropy Table."""
    from hdps.core.images import FullFrameImage
    from hdps.core.telemetry import Telemetry
    from hdps.photometry import SourceDetector

    detector = SourceDetector(**{k: cfg[k] for k in DETECTOR_KEYS})
    frame = FullFrameImage(data=np.asarray(image, np.float32), metadata={}, timestamp=time.time(),
                           telemetry=Telemetry(output_enabled=False))
    return detector.detect_sources(frame).sources.table


def associate(det_x, det_y, star_x, star_y, star_flux, radius):
    """For each star, the index of its detection, or -1: stars brightest first, each taking the nearest detection
    not yet taken within `radius`."""
    match = np.full(len(star_x), -1)
    taken = np.zeros(len(det_x), bool)
    for i in np.argsort(-np.nan_to_num(star_flux, nan=0.0), kind="stable"):
        d = np.hypot(det_x - star_x[i], det_y - star_y[i])
        d[taken] = np.inf
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= radius:
            match[i] = j
            taken[j] = True
    return match


class Photometry:
    """measure(image) -> the targets' flux, sigma, mag, mag_err (nan where not detected), saturation flags, and the
    exposure's full catalogue (every detection, with the simulated star it was matched to, or -1)."""

    def __init__(self, x, y, cat_mag, cat_flux, n, fullwell, targets, colour, cfg=None):
        self.cfg = settings(cfg)
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.cat_mag, self.cat_flux = np.asarray(cat_mag, float), np.asarray(cat_flux, float)
        self.colour = np.asarray(colour, float)
        self.fullwell, self.targets = fullwell, list(targets)
        picks = calibration_stars(self.x, self.y, self.cat_flux, n, k=self.cfg["n_calibrators"])
        self.cal = np.setdiff1d(picks[np.isfinite(self.colour[picks])], self.targets)

    def measure(self, image):
        table = extract(image, self.cfg)
        dx, dy = np.asarray(table["xcentroid"], float), np.asarray(table["ycentroid"], float)
        flux, ferr = np.asarray(table["flux"], float), np.asarray(table["flux_err"], float)
        star_of = associate(dx, dy, self.x, self.y, self.cat_flux, self.cfg["match_radius"])
        if len(dx) == 0:  # nothing detected
            nan = np.full(len(self.targets), np.nan)
            return dict(flux=nan, sigma=nan, mag=nan, mag_err=nan, saturated=np.zeros(len(self.targets), bool),
                        catalogue=table, zeropoint=(np.nan, np.nan))
        ny, nx = image.shape
        peak = np.array([image[max(int(round(b)) - 2, 0):min(int(round(b)) + 3, ny),
                               max(int(round(a)) - 2, 0):min(int(round(a)) + 3, nx)].max() for a, b in zip(dx, dy)])
        saturated = peak >= 0.98 * self.fullwell

        # zero point with a colour term: mag + 2.5 log10(flux) = a + b (BP-RP), clipped
        j = star_of[self.cal]
        use_cal = (j >= 0)
        use_cal[use_cal] &= (flux[j[use_cal]] >= self.cfg["min_cal_snr"] * ferr[j[use_cal]]) & ~saturated[j[use_cal]]
        cal_y = self.cat_mag[self.cal][use_cal] + 2.5 * np.log10(flux[j[use_cal]])
        cal_c = self.colour[self.cal][use_cal]
        use = np.ones(cal_y.size, bool)
        a, b = (np.median(cal_y) if cal_y.size else np.nan), 0.0  # fewer than 5 calibration stars: no colour term
        for _ in range(10):
            if use.sum() < 5:
                break
            if self.cfg["colour_term"]:
                b, a = np.polyfit(cal_c[use], cal_y[use], 1)
            else:
                a = float(np.median(cal_y[use]))
            r = cal_y - (a + b * cal_c)
            new = np.abs(r) <= self.cfg["clip_sigma"] * 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
            if np.array_equal(new, use):
                break
            use = new
        c_t = self.colour[self.targets]
        zp = a + b * np.where(np.isfinite(c_t), c_t, np.median(cal_c[use]) if use.any() else 0.0)

        jt = star_of[self.targets]
        found = jt >= 0
        f = np.where(found & (flux[jt] > 0), flux[jt], np.nan)
        s = np.where(found, ferr[jt], np.nan)
        star = np.full(len(dx), -1)
        star[star_of[star_of >= 0]] = np.nonzero(star_of >= 0)[0]
        table["star"], table["saturated"] = star, saturated
        table["mag"] = a + b * np.where(star >= 0, self.colour[np.maximum(star, 0)], np.nan) - 2.5 * np.log10(flux)
        return dict(flux=f, sigma=s, mag=zp - 2.5 * np.log10(f), mag_err=2.5 / np.log(10) * s / f,
                    saturated=found & saturated[jt], catalogue=table, zeropoint=(float(a), float(b)))
