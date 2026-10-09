"""SExtractor photometry of the synthetic Argus exposures (m3_night_movie.py).

SEP (Barbary 2016; SExtractor's algorithms, Bertin & Arnouts 1996) on every exposure, with SExtractor's defaults:
- background: mesh 64 x 64 px, 3 x 3 median filter (BACK_SIZE 64, BACK_FILTERSIZE 3);
- detection: 1.5 sigma above background over >= 5 connected pixels after the default 3 x 3 convolution
  (DETECT_THRESH 1.5, DETECT_MINAREA 5), deblending 32 thresholds / 0.005 contrast;
- neighbours: other objects' pixels masked from each Kron radius and aperture via the segmentation map (as
  SExtractor's MASK_TYPE does for MAG_AUTO);
- positions: windowed centroids (XWIN_IMAGE / YWIN_IMAGE);
- photometry: MAG_AUTO, a Kron elliptical aperture of 2.5 Kron radii, at least 3.5 (PHOT_AUTOPARAMS 2.5, 3.5).
Saturated pixels are not masked (SExtractor keeps saturated stars; they are flagged).

Each exposure is calibrated on its own: catalogue stars (least-blended, unsaturated, with Gaia colours) matched to
their detections give the zero point with a linear colour term in BP-RP (3-sigma clipped). Catalogue stars
(variables included) are cross-matched to the nearest detection within 0.5 FWHM; none: no measurement in that
exposure.
"""

from __future__ import annotations

import numpy as np
import sep

MATCH_FWHM = 0.5


def calibration_stars(x, y, cat_flux, n, k=300, edge=16, radius=8.0):
    """The k bright stars least contaminated by neighbours, away from the edge.

    Contamination of a star: catalogue flux of its neighbours, Gaussian-weighted by distance (sigma = radius/2
    pixels), over its own flux.
    """
    candidates = np.nonzero((x > edge) & (x < n - edge) & (y > edge) & (y < n - edge) & np.isfinite(cat_flux))[0]
    candidates = candidates[np.argsort(-cat_flux[candidates])][: 4 * k]  # the brightest
    contamination = []
    for i in candidates:
        d = np.hypot(x - x[i], y - y[i])
        near = (d > 0) & (d < 3 * radius)
        contamination.append(np.sum(cat_flux[near] * np.exp(-0.5 * (d[near] / (radius / 2)) ** 2)) / cat_flux[i])
    return candidates[np.argsort(contamination)[:k]]


def extract(image, fwhm):
    """SExtractor's defaults with MAG_AUTO as SExtractor measures it (see the module docstring).

    Returns x, y (windowed), flux, flux error and peak of every detection."""
    data = np.ascontiguousarray(image, dtype=np.float64)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    sub = data - bkg.back()
    sep.set_extract_pixstack(10_000_000)
    sep.set_sub_object_limit(8192)
    obj, seg = sep.extract(sub, 1.5, err=bkg.globalrms, minarea=5, deblend_nthresh=32, deblend_cont=0.005,
                           segmentation_map=True)
    ids = np.arange(1, len(obj) + 1, dtype=np.int32)
    x, y = np.asarray(obj["x"], float), np.asarray(obj["y"], float)
    xw, yw, fl = sep.winpos(sub, x, y, fwhm / 2.3548)
    good = fl == 0
    a, b, th = np.asarray(obj["a"], float), np.asarray(obj["b"], float), np.asarray(obj["theta"], float)
    th = np.clip(th, -np.pi / 2, np.pi / 2)
    kr, kf = sep.kron_radius(sub, x, y, a, b, th, 6.0, seg_id=ids, segmap=seg)
    # PHOT_AUTOPARAMS 2.5, 3.5: the elliptical aperture is max(2.5 x Kron radius, 3.5) in units of (a, b)
    r = np.maximum(2.5 * np.nan_to_num(kr, nan=0.0), 3.5)
    flux, ferr, ff = sep.sum_ellipse(sub, x, y, a, b, th, r, err=bkg.globalrms, subpix=1, seg_id=ids, segmap=seg)
    return np.where(good, xw, x), np.where(good, yw, y), np.asarray(flux, float), np.asarray(ferr, float), obj["peak"]


class SExtractorPhotometry:
    """measure(image) -> the targets' flux, sigma, mag, mag_err (nan where not detected) and saturation flags."""

    def __init__(self, x, y, cat_mag, cat_flux, n, fullwell, targets, colour, fwhm):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.cat_mag, self.colour = np.asarray(cat_mag, float), np.asarray(colour, float)
        self.fullwell, self.targets, self.fwhm = fullwell, list(targets), fwhm
        picks = calibration_stars(self.x, self.y, np.asarray(cat_flux, float), n)
        self.cal = np.setdiff1d(picks[np.isfinite(self.colour[picks])], self.targets)

    def measure(self, image):
        x, y, flux, ferr, peak = extract(image, self.fwhm)
        sat_level = 0.98 * self.fullwell

        def nearest(i):  # standard cross-match: the nearest detection within MATCH_FWHM x FWHM
            d = np.hypot(x - self.x[i], y - self.y[i])
            j = int(np.argmin(d))
            return j if d[j] <= MATCH_FWHM * self.fwhm else -1

        # zero point with a colour term: mag + 2.5 log10(flux) = a + b (BP-RP), 3-sigma clipped
        cal_y, cal_c = [], []
        for i in self.cal:
            j = nearest(i)
            if j >= 0 and flux[j] > 0 and peak[j] < sat_level:
                cal_y.append(self.cat_mag[i] + 2.5 * np.log10(flux[j]))
                cal_c.append(self.colour[i])
        cal_y, cal_c = np.array(cal_y), np.array(cal_c)
        use = np.ones(cal_y.size, bool)
        a, b = np.median(cal_y), 0.0  # fewer than 5 calibration stars: no colour term
        for _ in range(10):
            if use.sum() < 5:
                break
            b, a = np.polyfit(cal_c[use], cal_y[use], 1)
            r = cal_y - (a + b * cal_c)
            new = np.abs(r) <= 3.0 * 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
            if np.array_equal(new, use):
                break
            use = new
        c_t = self.colour[self.targets]
        zp = a + b * np.where(np.isfinite(c_t), c_t, np.median(cal_c[use]))

        j = np.array([nearest(t) for t in self.targets])
        f = np.where((j >= 0) & (flux[j] > 0), flux[j], np.nan)
        s = np.where(j >= 0, ferr[j], np.nan)
        return dict(flux=f, sigma=s, mag=zp - 2.5 * np.log10(f), mag_err=2.5 / np.log(10) * s / f,
                    saturated=(j >= 0) & (peak[j] >= sat_level))
