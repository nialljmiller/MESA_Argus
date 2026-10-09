"""Swappable photometry for the synthetic Argus exposures (m3_night_movie.py).

Every method measures the same thing: the flux, in electrons, of chosen target stars in one exposure, and
its uncertainty, given only what that method is allowed to know.

  ideal   Forced simultaneous PSF fit at the catalogue (Gaia) positions with the TRUE PSF, the TRUE sky
          and weights from the noise-free model. Nothing is estimated: the best case, kept as a benchmark.
  psf     DAOPHOT-style, single pass: background estimated from the image (photutils Background2D), an
          empirical PSF built once from the brightest, least-contaminated stars (photutils EPSFBuilder),
          then a simultaneous linear fit of every catalogue star at its Gaia position with that PSF.
  psf-iter  As psf, with DAOPHOT's iterative PSF: fit all stars, subtract the PSF stars' fitted
          neighbours, rebuild the PSF; three rounds. An independent method from psf.
  dia     Difference-image photometry, the usual method for variables in crowded fields. Reference brightness
          from the reference image alone: stars DETECTED and fitted DAOPHOT/ALLSTAR-style (photutils
          IterativePSFPhotometry: detection, grouped PSF fits with free positions, local annulus sky, iterating on
          the residuals); a variable takes the flux of the detected source at its position, blended neighbours
          included. Change: forced PSF photometry on the difference image at the variable's catalogue position. Blended light in the
          reference flux dilutes the amplitude in magnitudes, as it would in Argus. PSF: ArgusSim's PSF at each
          image's seeing stands in for the PSF a full-frame pipeline would measure (not simulated here).

Magnitudes: every method's fluxes are converted with a zero point measured in the same exposure from
calibration stars (bright, unsaturated, isolated, constant) whose catalogue magnitudes are known, as a
pipeline calibrates against a catalogue. The `ideal` method can also use the exact zero point.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


# ---------------------------------------------------------------- shared pieces
def linear_fit(stamps, image, background, variance, good, targets):
    """Simultaneous weighted least squares for every star's flux, given unit-flux PSF columns.

    stamps: sparse (pixels x stars). Returns fluxes of all stars and the 1-sigma error of each target.
    """
    w = np.where(good, 1.0 / variance, 0.0)
    Aw = stamps.multiply(w[:, None]).tocsc()
    normal = (stamps.T @ Aw).tocsc() + sparse.identity(stamps.shape[1], format="csc") * 1e-12
    lu = splu(normal)
    flux = lu.solve(Aw.T @ (image - background))
    sigma = np.full(stamps.shape[1], np.nan)
    for k in targets:
        e = np.zeros(stamps.shape[1])
        e[k] = 1.0
        sigma[k] = np.sqrt(max(lu.solve(e)[k], 0.0))
    return flux, sigma


def stamp_matrix(psf_at, x, y, n, half=14):
    """Sparse (n*n pixels x stars) matrix of unit-flux PSFs: psf_at(dx, dy) gives the PSF on integer offsets
    (dx, dy) from the star's sub-pixel position, normalised over the stamp."""
    rows, cols, vals = [], [], []
    oy, ox = np.indices((2 * half + 1, 2 * half + 1)) - half
    for k, (xk, yk) in enumerate(zip(x, y)):
        ix, iy = int(np.round(xk)), int(np.round(yk))
        px, py = ox + ix, oy + iy
        p = psf_at(px - xk, py - yk)
        p = np.clip(p, 0.0, None)
        p = p / p.sum() if p.sum() > 0 else p
        ok = (px >= 0) & (px < n) & (py >= 0) & (py < n) & (p > 0)
        rows.append((py[ok] * n + px[ok]).ravel())
        cols.append(np.full(int(ok.sum()), k))
        vals.append(p[ok].ravel())
    return sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, len(x)))


def calibration_stars(x, y, cat_flux, n, sat_flux, k=40, edge=16, radius=8.0):
    """The k bright, unsaturated stars least contaminated by neighbours, as DAOPHOT picks PSF stars.

    Contamination of a star: catalogue flux of its neighbours, Gaussian-weighted by distance (sigma = radius/2
    pixels), over its own flux. In a crowded core no star is truly isolated, so the least-contaminated are
    used; their neighbours are then removed iteratively (empirical_psf). Saturated stars (expected flux above
    sat_flux) and stars near the edge are excluded.
    """
    candidates = np.nonzero((x > edge) & (x < n - edge) & (y > edge) & (y < n - edge) & (cat_flux < sat_flux))[0]
    candidates = candidates[np.argsort(-cat_flux[candidates])][: 4 * k]  # the brightest usable stars
    contamination = []
    for i in candidates:
        d = np.hypot(x - x[i], y - y[i])
        near = (d > 0) & (d < 3 * radius)
        contamination.append(np.sum(cat_flux[near] * np.exp(-0.5 * (d[near] / (radius / 2)) ** 2)) / cat_flux[i])
    return candidates[np.argsort(contamination)[:k]]


def zero_point(flux, mags, picks):
    """Median ZP from calibration stars: mag = ZP - 2.5 log10(flux)."""
    ok = picks[flux[picks] > 0]
    return float(np.median(mags[ok] + 2.5 * np.log10(flux[ok])))


# ---------------------------------------------------------------- empirical PSF and background (photutils)
def empirical_psf(image, background, x, y, picks, size=25, oversampling=4, maxiters=8):
    """Build an empirical PSF (photutils EPSFBuilder) from the picked stars; return psf_at(dx, dy)."""
    from astropy.nddata import NDData
    from astropy.table import Table
    from photutils.psf import EPSFBuilder, extract_stars

    stars = extract_stars(NDData(image - background), Table({"x": x[picks], "y": y[picks]}), size=(size, size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epsf, _ = EPSFBuilder(oversampling=oversampling, maxiters=maxiters, progress_bar=False)(stars)

    def psf_at(dx, dy):
        return epsf.evaluate(dx, dy, flux=1.0, x_0=0.0, y_0=0.0)

    return psf_at, epsf


def estimate_background(image, saturated, box=32):
    """Sky from the image itself (photutils Background2D, sigma-clipped medians in 32-pixel boxes)."""
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # crowded fields: sigma clipping rejects most star-covered pixels, so a box is kept as long as 5% of
        # its pixels survive (photutils' default, 90%, would reject every box in a cluster core)
        bkg = Background2D(image, box, filter_size=3, sigma_clip=SigmaClip(sigma=3.0), mask=saturated,
                           bkg_estimator=MedianBackground(), exclude_percentile=95.0)
    return bkg.background


# ---------------------------------------------------------------- the methods
class Photometry:
    """Common interface: measure(exposure) -> dict(flux, sigma, mag, mag_err) for the targets."""

    name = "base"

    def __init__(self, x, y, cat_mag, cat_flux_guess, n, readnoise, fullwell, targets):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.cat_mag, self.cat_flux = np.asarray(cat_mag, float), np.asarray(cat_flux_guess, float)
        self.n, self.readnoise, self.fullwell = n, readnoise, fullwell
        self.targets = list(targets)
        self.not_targets = np.setdiff1d(np.arange(len(self.x)), self.targets)

    def magnitudes(self, flux, sigma, zp):
        out = {"flux": flux[self.targets], "sigma": sigma[self.targets]}
        f = out["flux"]
        out["mag"] = np.where(f > 0, zp - 2.5 * np.log10(np.where(f > 0, f, 1.0)), np.nan)
        out["mag_err"] = np.where(f > 0, 2.5 / np.log(10) * out["sigma"] / np.where(f > 0, f, 1.0), np.nan)
        out["zp"] = zp
        return out


class IdealPhotometry(Photometry):
    """Forced fit with the true PSF matrix, true sky and model weights (the benchmark)."""

    name = "ideal"

    def measure(self, exp):
        good = ~exp["saturated"].ravel()
        flux, sigma = linear_fit(exp["true_psf"], exp["image"].ravel(), exp["true_background"],
                                 exp["true_mean"] + self.readnoise**2, good, self.targets)
        picks = calibration_stars(self.x, self.y, self.cat_flux, self.n, 0.5 * self.fullwell / exp["true_peak_fraction"])
        picks = np.setdiff1d(picks, self.targets)
        return self.magnitudes(flux, sigma, zero_point(flux, self.cat_mag, picks))


class PSFPhotometry(Photometry):
    """Empirical PSF and estimated sky from the exposure; forced simultaneous fit at catalogue positions."""

    name = "psf"
    iterations = 1

    def fit(self, image, saturated, iterations=None):
        """DAOPHOT-style: background from the image; empirical PSF built iteratively (fit all stars, subtract the
        PSF stars' fitted neighbours, rebuild); final simultaneous fit of every catalogue star."""
        bkg = estimate_background(image, saturated)
        # PSF stars: bright and least contaminated, unsaturated in THIS image (peak below 70% of full well)
        picks = calibration_stars(self.x, self.y, self.cat_flux, self.n, np.inf, k=80)
        peak = np.array([image[int(round(self.y[i])), int(round(self.x[i]))] for i in picks])
        picks = np.setdiff1d(picks[peak < 0.7 * self.fullwell], self.targets)[:40]
        variance = np.clip(image, 0, None).ravel() + self.readnoise**2  # from the data, as a pipeline would
        good = ~saturated.ravel()
        cleaned = image - bkg
        for _ in range(iterations or self.iterations):
            psf_at, _ = empirical_psf(cleaned + bkg, bkg, self.x, self.y, picks)
            stamps = stamp_matrix(psf_at, self.x, self.y, self.n)
            flux, sigma = linear_fit(stamps, image.ravel(), bkg.ravel(), variance, good, self.targets)
            neighbours = flux.copy()
            neighbours[picks] = 0.0  # keep the PSF stars, remove everything else
            cleaned = image - bkg - (stamps @ neighbours).reshape(self.n, self.n)
        return flux, sigma, picks, psf_at, bkg

    def measure(self, exp):
        flux, sigma, picks, _, _ = self.fit(exp["image"], exp["saturated"])
        return self.magnitudes(flux, sigma, zero_point(flux, self.cat_mag, picks))


class IterativePSFPhotometry(PSFPhotometry):
    """psf with DAOPHOT's iterative PSF building (three rounds of neighbour subtraction)."""

    name = "psf-iter"
    iterations = 3


class DIAPhotometry(Photometry):  # "dia-psf": the earlier variant (photutils iterative PSF extraction)
    """Difference-image photometry with the reference brightness measured from the reference image alone.

    Reference: stars are DETECTED in the reference image and fitted with photutils IterativePSFPhotometry (the
    DAOPHOT/ALLSTAR scheme: DAOFIND-style detection, grouped PSF fits with free positions, local sky from an
    annulus, iterating on the residual image). A variable's reference flux is that of the detected source
    brightest within one PSF FWHM of its catalogue position, blended neighbours included; if none, it is
    not detected. Zero point: detections matched to the least-blended bright catalogue stars over the whole
    simulated area, fitted with a linear colour term in Gaia BP-RP (3-sigma clipped); each target gets the zero
    point at its own Gaia BP-RP (the calibration stars' median colour if it has none).
    Change: forced PSF photometry on the PSF-matched difference image at each variable's catalogue position.
    PSF: ArgusSim's PSF at each image's seeing (stand-in for a full-frame measurement; not simulated here).
    """

    name = "dia-psf"

    def __init__(self, *args, reference, reference_saturated, reference_psf, fwhm, n_stack, colour, match_radius=None,
                 **kw):
        super().__init__(*args, **kw)
        from photutils.background import LocalBackground, MMMBackground
        from photutils.detection import DAOStarFinder
        from photutils.psf import IterativePSFPhotometry, SourceGrouper

        # per-pixel noise from the detector's noise model (photon + read noise; a median of n_stack frames);
        # detection threshold 5 sigma of that noise (not the image's pixel scatter, which crowding dominates)
        error = np.sqrt((np.clip(reference, 0, None) + self.readnoise**2) * (np.pi / 2) / n_stack)
        phot = IterativePSFPhotometry(
            reference_psf, fit_shape=(7, 7), finder=DAOStarFinder(threshold=5.0 * float(np.median(error)), fwhm=fwhm),
            grouper=SourceGrouper(min_separation=2.0 * fwhm), aperture_radius=fwhm, maxiters=3,
            local_bkg_estimator=LocalBackground(2.5 * fwhm, 4.0 * fwhm, bkg_estimator=MMMBackground()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = phot(reference, error=error, mask=reference_saturated)
        ok = np.isfinite(np.asarray(table["flux_fit"], float))
        self.det_x = np.asarray(table["x_fit"], float)[ok]
        self.det_y = np.asarray(table["y_fit"], float)[ok]
        self.det_flux = np.asarray(table["flux_fit"], float)[ok]
        self.det_err = np.asarray(table["flux_err"], float)[ok] if "flux_err" in table.colnames else np.full(ok.sum(), np.nan)
        # each target -> the detected source containing it: the brightest within one PSF FWHM (in crowded fields the
        # detection is a blend whose fitted centre is offset from the variable)
        match_radius = fwhm if match_radius is None else match_radius
        self.match = []
        for k in self.targets:
            # the brightest detection within the radius: the iterative fit also detects faint residuals of bright stars
            # on top of them, and the nearest detection can be such a residual rather than the star
            near = np.nonzero(np.hypot(self.det_x - self.x[k], self.det_y - self.y[k]) <= match_radius)[0]
            self.match.append(int(near[np.argmax(self.det_flux[near])]) if near.size else -1)
        # zero point with a colour term: mag + 2.5 log10(flux) = a + b (BP-RP), over detections matched to the
        # least-blended bright catalogue stars with Gaia colours, unsaturated in the reference
        colour = np.asarray(colour, float)
        peak = np.array([reference[min(max(int(round(yy)), 0), self.n - 1), min(max(int(round(xx)), 0), self.n - 1)]
                         for xx, yy in zip(self.x, self.y)])
        picks = calibration_stars(self.x, self.y, self.cat_flux, self.n, np.inf, k=300)
        picks = np.setdiff1d(picks[(peak[picks] < 0.7 * self.fullwell) & np.isfinite(colour[picks])], self.targets)
        cal_y, cal_c = [], []
        for i in picks:
            near = np.nonzero(np.hypot(self.det_x - self.x[i], self.det_y - self.y[i]) <= match_radius)[0]
            if near.size and self.det_flux[near].max() > 0:
                cal_y.append(self.cat_mag[i] + 2.5 * np.log10(self.det_flux[near].max()))
                cal_c.append(colour[i])
        cal_y, cal_c = np.array(cal_y), np.array(cal_c)
        use = np.ones(cal_y.size, bool)
        a, b = (float(np.median(cal_y)) if cal_y.size else np.nan), 0.0
        for _ in range(10):
            if use.sum() < 5:
                break
            b, a = np.polyfit(cal_c[use], cal_y[use], 1)
            r = cal_y - (a + b * cal_c)
            new = np.abs(r) <= 3.0 * 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
            if np.array_equal(new, use):
                break
            use = new
        self.zp_a, self.zp_b = float(a), float(b)
        c_med = float(np.median(cal_c[use])) if use.any() else 0.0
        c_t = colour[self.targets]
        self.ref_zp = self.zp_a + self.zp_b * np.where(np.isfinite(c_t), c_t, c_med)
        self.zp_scatter = float(np.std(cal_y[use] - (a + b * cal_c[use]))) if use.sum() > 1 else np.nan
        zp = cal_y[use]
        self.n_detected, self.n_zp = int(ok.sum()), len(zp)

    def measure(self, exp):
        """exp: difference (nan at the border), image, saturated, kernel_sum, psf_at (the exposure's PSF)."""
        m = np.array(self.match)
        found = m >= 0
        # the flux change is forced photometry at each variable's own catalogue position on the difference image
        # (the blend's fitted centre can sit a PSF width away from the variable and would miss its signal)
        stamps = stamp_matrix(exp["psf_at"], self.x[self.targets], self.y[self.targets], self.n)
        diff = exp["difference"].ravel()
        good = np.isfinite(diff) & ~exp["saturated"].ravel()
        variance = np.clip(exp["image"], 0, None).ravel() + self.readnoise**2
        dflux, dsig = linear_fit(stamps, np.nan_to_num(diff), np.zeros_like(diff), variance, good, range(len(self.targets)))
        k = exp["kernel_sum"]
        ref = np.where(found, self.det_flux[np.clip(m, 0, None)], np.nan)
        ref_err = np.where(found, self.det_err[np.clip(m, 0, None)], np.nan)
        flux = np.full(len(self.x), np.nan)
        sigma = np.full(len(self.x), np.nan)
        flux[self.targets] = ref + dflux / k  # reference units
        sigma[self.targets] = np.hypot(dsig / k, np.nan_to_num(ref_err))
        out = self.magnitudes(np.nan_to_num(flux, nan=-1.0), sigma, self.ref_zp)
        out["delta_flux"], out["detected"] = dflux, found
        return out


class DIAAperturePhotometry(Photometry):
    """Difference imaging with standard SExtractor extraction and aperture photometry.

    Reference: sources extracted with SEP (Barbary 2016; SExtractor's algorithms, Bertin & Arnouts 1996) using
    SExtractor's defaults: background mesh 64 px, background filter 3x3, detection at 1.5 sigma above the local
    background over >= 5 connected pixels after the default 3x3 convolution, deblending 32 thresholds / 0.005 contrast;
    positions refined with windowed centroids (XWIN/YWIN). Saturated pixels are not masked from detection (a saturated
    star is still a source); stars with saturated pixels in the aperture are flagged.
    Brightness: circular aperture of radius APER_FWHM x FWHM, background-subtracted, on the reference; the change on
    each PSF-matched difference image is the same aperture sum at the same position, divided by the kernel's flux
    scale. Saturated pixels (exposure or reference) are left out of the difference sum and the star is flagged.
    A variable takes the brightest extracted source within one FWHM of its catalogue position (blends included).
    Zero point: the same aperture on the least-blended bright catalogue stars, with a linear colour term in BP-RP.
    """

    name = "dia"
    APER_FWHM = 1.5

    def __init__(self, *args, reference, reference_saturated, fwhm, n_stack, colour, reference_psf=None,
                 match_radius=None, **kw):
        super().__init__(*args, **kw)
        import sep

        data = np.ascontiguousarray(reference, dtype=np.float64)
        self.sat_ref = np.asarray(reference_saturated, bool)
        bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
        sub = data - bkg.back()
        self.background_rms = float(bkg.globalrms)
        sep.set_extract_pixstack(5_000_000)
        sep.set_sub_object_limit(4096)
        obj = sep.extract(sub, 1.5, err=bkg.globalrms, minarea=5, deblend_nthresh=32, deblend_cont=0.005)
        x, y = np.asarray(obj["x"], float), np.asarray(obj["y"], float)
        xw, yw, flag = sep.winpos(sub, x, y, fwhm / 2.3548)
        ok = flag == 0
        x, y = np.where(ok, xw, x), np.where(ok, yw, y)
        self.radius = self.APER_FWHM * fwhm
        error = np.sqrt((np.clip(reference, 0, None) + self.readnoise**2) * (np.pi / 2) / n_stack)
        f, fe, _ = sep.sum_circle(sub, x, y, self.radius, err=error)
        self.det_x, self.det_y, self.det_flux, self.det_err = x, y, np.asarray(f, float), np.asarray(fe, float)
        match_radius = fwhm if match_radius is None else match_radius
        self.match = []
        for k in self.targets:
            near = np.nonzero(np.hypot(self.det_x - self.x[k], self.det_y - self.y[k]) <= match_radius)[0]
            self.match.append(int(near[np.argmax(self.det_flux[near])]) if near.size else -1)
        # zero point (colour term) from the same aperture on the least-blended bright catalogue stars
        colour = np.asarray(colour, float)
        picks = calibration_stars(self.x, self.y, self.cat_flux, self.n, np.inf, k=300)
        picks = np.setdiff1d(picks[np.isfinite(colour[picks])], self.targets)
        cal_y, cal_c = [], []
        for i in picks:
            near = np.nonzero(np.hypot(self.det_x - self.x[i], self.det_y - self.y[i]) <= match_radius)[0]
            if near.size and not self._saturated_in_aperture(self.sat_ref, self.det_x[near[0]], self.det_y[near[0]]):
                j = near[np.argmax(self.det_flux[near])]
                if self.det_flux[j] > 0:
                    cal_y.append(self.cat_mag[i] + 2.5 * np.log10(self.det_flux[j]))
                    cal_c.append(colour[i])
        cal_y, cal_c = np.array(cal_y), np.array(cal_c)
        use = np.ones(cal_y.size, bool)
        a, b = (float(np.median(cal_y)) if cal_y.size else np.nan), 0.0
        for _ in range(10):
            if use.sum() < 5:
                break
            b, a = np.polyfit(cal_c[use], cal_y[use], 1)
            r = cal_y - (a + b * cal_c)
            new = np.abs(r) <= 3.0 * 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
            if np.array_equal(new, use):
                break
            use = new
        self.zp_a, self.zp_b = float(a), float(b)
        c_med = float(np.median(cal_c[use])) if use.any() else 0.0
        c_t = colour[self.targets]
        self.ref_zp = self.zp_a + self.zp_b * np.where(np.isfinite(c_t), c_t, c_med)
        self.zp_scatter = float(np.std(cal_y[use] - (a + b * cal_c[use]))) if use.sum() > 1 else np.nan
        self.n_detected, self.n_zp = len(self.det_x), int(use.sum())
        m = np.array(self.match)
        self.ref_saturated = np.array([mi >= 0 and self._saturated_in_aperture(self.sat_ref, self.det_x[mi], self.det_y[mi])
                                       for mi in m])

    def _saturated_in_aperture(self, sat, x, y):
        yy, xx = np.ogrid[: self.n, : self.n]
        r = self.radius
        win = (slice(max(int(y - r - 1), 0), int(y + r + 2)), slice(max(int(x - r - 1), 0), int(x + r + 2)))
        return bool(np.any(sat[win] & ((yy[win[0]] - y) ** 2 + (xx[:, win[1]] - x) ** 2 <= r * r)))

    def measure(self, exp):
        """exp: difference (nan at the border), image, saturated, kernel_sum."""
        import sep

        m = np.array(self.match)
        found = m >= 0
        xs = np.where(found, self.det_x[np.clip(m, 0, None)], self.x[self.targets])
        ys = np.where(found, self.det_y[np.clip(m, 0, None)], self.y[self.targets])
        diff = np.ascontiguousarray(exp["difference"], dtype=np.float64)
        bad = ~np.isfinite(diff) | exp["saturated"] | self.sat_ref
        variance = np.clip(exp["image"], 0, None) + self.readnoise**2
        df, dfe, _ = sep.sum_circle(np.where(bad, 0.0, diff), xs, ys, self.radius,
                                    err=np.sqrt(variance).astype(np.float64), mask=bad)
        k = exp["kernel_sum"]
        ref = np.where(found, self.det_flux[np.clip(m, 0, None)], np.nan)
        ref_err = np.where(found, self.det_err[np.clip(m, 0, None)], np.nan)
        flux = np.full(len(self.x), np.nan)
        sigma = np.full(len(self.x), np.nan)
        flux[self.targets] = ref + df / k
        sigma[self.targets] = np.hypot(dfe / k, np.nan_to_num(ref_err))
        out = self.magnitudes(np.nan_to_num(flux, nan=-1.0), sigma, self.ref_zp)
        sat_now = np.array([self._saturated_in_aperture(exp["saturated"], x_, y_) for x_, y_ in zip(xs, ys)])
        out["delta_flux"], out["detected"], out["saturated"] = df, found, self.ref_saturated | sat_now
        return out


class SExtractorPhotometry(Photometry):
    """SExtractor on every exposure (SEP: Barbary 2016, SExtractor's algorithms, Bertin & Arnouts 1996), with
    SExtractor's defaults:
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

    name = "sextractor"
    MATCH_FWHM = 0.5

    def __init__(self, *args, colour, fwhm, **kw):
        super().__init__(*args, **kw)
        self.colour = np.asarray(colour, float)
        self.fwhm = fwhm
        picks = calibration_stars(self.x, self.y, self.cat_flux, self.n, np.inf, k=300)
        self.cal = np.setdiff1d(picks[np.isfinite(self.colour[picks])], self.targets)

    @staticmethod
    def extract(image, fwhm):
        """SExtractor defaults (background mesh 64 / filter 3; default 3x3 detection filter; 1.5 sigma of the global
        background RMS over >= 5 pixels; deblending 32 / 0.005; windowed centroids), with MAG_AUTO measured as
        SExtractor does: other objects' pixels masked from the Kron radius and the aperture via the segmentation
        map (MASK_TYPE). Tested on a crowded field: a PSF-matched Gaussian filter, lower deblending contrast or an
        RMS-map threshold all lose stars or accuracy there; neighbour masking is what matters."""
        import sep

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

    def measure(self, exp):
        x, y, flux, ferr, peak = self.extract(exp["image"], self.fwhm)
        sat_level = 0.98 * self.fullwell

        def nearest(i):  # standard cross-match: the nearest detection within MATCH_FWHM x FWHM
            d = np.hypot(x - self.x[i], y - self.y[i])
            j = int(np.argmin(d)) if d.size else -1
            return j if j >= 0 and d[j] <= self.MATCH_FWHM * self.fwhm else -1

        cal_y, cal_c = [], []
        for i in self.cal:
            j = nearest(i)
            if j >= 0 and flux[j] > 0 and peak[j] < sat_level:
                cal_y.append(self.cat_mag[i] + 2.5 * np.log10(flux[j]))
                cal_c.append(self.colour[i])
        cal_y, cal_c = np.array(cal_y), np.array(cal_c)
        use = np.ones(cal_y.size, bool)
        a, b = (float(np.median(cal_y)) if cal_y.size else np.nan), 0.0
        for _ in range(10):
            if use.sum() < 5:
                break
            b, a = np.polyfit(cal_c[use], cal_y[use], 1)
            r = cal_y - (a + b * cal_c)
            new = np.abs(r) <= 3.0 * 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
            if np.array_equal(new, use):
                break
            use = new
        c_med = float(np.median(cal_c[use])) if use.any() else 0.0
        c_t = self.colour[self.targets]
        zp = a + b * np.where(np.isfinite(c_t), c_t, c_med)
        f_all = np.full(len(self.x), np.nan)
        s_all = np.full(len(self.x), np.nan)
        found, sat = [], []
        for t in self.targets:
            j = nearest(t)
            found.append(j >= 0)
            sat.append(j >= 0 and peak[j] >= sat_level)
            if j >= 0:
                f_all[t], s_all[t] = flux[j], ferr[j]
        out = self.magnitudes(np.nan_to_num(f_all, nan=-1.0), s_all, zp)
        out["detected"], out["saturated"], out["n_detected"], out["n_zp"] = np.array(found), np.array(sat), len(x), int(use.sum())
        if getattr(self, "check", None) is not None:  # diagnostic: the same measurement for constant stars
            res = []
            for i in self.check:
                j = nearest(i)
                if j >= 0 and flux[j] > 0:
                    res.append((self.cat_mag[i], a + b * (self.colour[i] if np.isfinite(self.colour[i]) else c_med)
                                - 2.5 * np.log10(flux[j]) - self.cat_mag[i], peak[j] >= sat_level))
            out["check"] = np.array(res)
        return out


METHODS = {"ideal": IdealPhotometry, "psf": PSFPhotometry, "psf-iter": IterativePSFPhotometry, "dia-psf": DIAPhotometry,
           "dia": DIAAperturePhotometry, "sextractor": SExtractorPhotometry}
