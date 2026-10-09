#!/usr/bin/env python3
"""Movie: one night of Argus 60 s exposures on a 3x3 HDPS-minipix block of M3's core.

    python m3_night_movie.py                          # -> outputs/m3/m3_night.mp4
    python m3_night_movie.py --stride 4 -o test.mp4   # every 4th exposure, for a quick look

M3 (NGC 5272) has ~240 RR Lyrae, more than any other globular cluster; this 84" block of its core
holds 42 of them among ~1,550 Gaia DR3 sources.

Variability comes only from MESA. The followed pulsators -- four RRab and two RRc -- are MESA RSP
(radial stellar pulsation) models (mesa_argus.rsp): M3 composition and mass, tuned so each pulsates
in its own mode at its Gaia period; the SED of every RSP step (<= 25 s apart) comes from the MESA
colors module. Each model is phased to its star's Gaia epoch of maximum and period, so the movie is
a prediction for the chosen night; amplitudes, shapes, colours and spectra are the models'. Every
other star -- including the other Gaia-classified variables and a 15 d Type II Cepheid that RSP
could not model stably -- is constant at its Gaia mean, with the MESA SED (MESA_ARGUS runs/*_sed)
whose Gaia BP-RP matches its colour, scaled to its G. In the crowded core Gaia misses faint stars
and blends BP/RP colours: the unresolved glow is underestimated and blended colours fall back to
the outer cluster's colour at the same G.

Observing (from ArgusSim unless noted): McDonald, the night of --date while M3 is above --max-airmass and the
Sun below -18 deg; 60 s exposures in 15-min ratchets on UTC quarter hours, the last minute a reset. Sky per
exposure from ArgusSim's Observatory model (airglow, zodiacal light, starlight, Moon, site light); seeing the
site's seasonal median scaled to each exposure's airmass (X^0.6); real airmass in the source extinction.
Assumed here: field angle 1.44 deg (area median of the field) all night, continuous coverage by one
telescope, transparency 1. Each exposure is a synthetic image (every star's ArgusSim PSF at its sub-pixel
position, sky, Poisson and read noise, full-well clipping); magnitudes are measured from it with SExtractor
(mesa_argus.photometry). The difference panel is display only. Full list of assumptions: outputs/m3/ASSUMPTIONS.md.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path
import warnings

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy.utils import iers
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import AsinhNorm  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from argus_sim import ABPhot, SpectralSky, SystemThroughput, ZemaxPSF  # noqa: E402
from argus_sim import c as ARGUS_CONFIG  # noqa: E402
from argus_sim import skymap_shim as sh  # noqa: E402

from plot_sed_observation import BANDS, COLORS, FLAM, SYMBOLS, apply_style, system_response  # noqa: E402

iers.conf.auto_download = False
# per cluster: Gaia data folder, cluster centre, HDPS tile and 3x3 block centre, default night, site override
CLUSTERS = {
    "m3": dict(name="M3", centre=(205.5484, 28.3773), tile=345967, block=(34, 32), date="2027-05-05", site=None),
    # M15 for a comparison with ADDEM at Sierra Remote Observatories (approximate SRO coordinates; Argus instrument)
    "m15": dict(name="M15", centre=(322.49304, 12.16700), tile=298720, block=(26, 36), date="2026-10-05",
                site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0),
                # Gaia RRc with P = 0.409 d: at the Teff its period needs, RSP's first overtone is stable even at
                # 0.70 x the cluster L (runs/rsp_m15_failed_P0409), so it is not followed
                exclude={1745948328028784640}),
    # an uncrowded field: one RRab (Gaia DR3 308108245007332224, P = 0.382 d, G = 13.9, 1.25 mag) and a few
    # neighbours, at b = -34 deg; the block is the 3x3 minipixes centred on the star's minipix
    "rr308108": dict(name="field RRab", centre=(17.588538, 28.550090), tile=345631, block=(35, 54), date="2026-10-06",
                     site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0)),
    # a richer uncrowded field: RRab Gaia DR3 416445908838640384 (P = 0.404 d, G = 14.4, 1.19 mag) with 21 stars
    # G < 20 in the block, b = -10 deg; up all night from SRO on 2026-10-06, predicted maximum ~05:55 UT
    "rr416445": dict(name="field RRab", centre=(8.841137, 52.998087), tile=417610, block=(12, 34), date="2026-10-06",
                     site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0)),
}
# fields made by argus_field.py: data/<field>/field.json (the stars to model are the user's selection)
_REPO = Path(__file__).resolve().parent  # data/, runs/, outputs/ are resolved from the repo, not the working directory
for _f in sorted((_REPO / "data").glob("*/field.json")):
    _c = json.loads(_f.read_text())
    CLUSTERS[_f.parent.name] = dict(name=_c["name"], centre=tuple(_c["centre"]), tile=_c["tile"], block=tuple(_c["block"]),
                                    date=_c["date"], site=_c.get("site"), followed_only=True, block_n=_c.get("block_n", 3))
CLUSTER = "m3"
DATA = Path("data/m3")
CENTRE = SkyCoord(205.5484 * u.deg, 28.3773 * u.deg)
TILE, BLOCK = 345967, (34, 32)


def use_cluster(cluster):
    """Point the module (and mesa_argus.rsp) at one cluster's data, block and models."""
    global CLUSTER, DATA, CENTRE, TILE, BLOCK
    from mesa_argus import rsp

    c_ = CLUSTERS[cluster]
    CLUSTER, DATA = cluster, _REPO / "data" / cluster
    CENTRE = SkyCoord(c_["centre"][0] * u.deg, c_["centre"][1] * u.deg)
    TILE, BLOCK = c_["tile"], c_["block"]
    rsp.use(cluster)
MINIPIX_PX, SKY_PX = 32, 900.0 / 1024.0
BLOCK_N = 3  # the block shown: BLOCK_N x BLOCK_N HDPS minipixes centred on the block minipix (odd; --block)
HALF = BLOCK_N / 2 * MINIPIX_PX * SKY_PX  # block half-width (arcsec)
MARGIN = 15.0  # arcsec: stars this far outside still spill light in


def set_block(n):
    """Show an n x n block of minipixes (n odd)."""
    global BLOCK_N, HALF
    BLOCK_N, HALF = n, n / 2 * MINIPIX_PX * SKY_PX


def block_label():
    return f"{BLOCK_N} × {BLOCK_N} minipixes ({2 * HALF:.0f}″)"


# half-width (arcsec) of the simulated area. The movie shows the block; the difference-image kernel and the
# zero point use the whole area, as a pipeline would use a large part of the detector. Set by --field.
AREA_HALF = 240.0
AIRMASS_GRID = np.round(np.arange(1.0, 2.61, 0.2), 2)
PSF_PHASES = 8  # sub-pixel phases per axis for the PSF lookup


# ---------------------------------------------------------------- stars
def block_offsets(ra, dec):
    """Offsets (arcsec) from the block centre in the tile's tangent-plane pixel frame."""
    tra, tdec = (float(v[0]) for v in sh.tile_center(np.array([TILE])))
    cd, sd = np.cos(np.radians(dec)), np.sin(np.radians(dec))
    ctd, std = np.cos(np.radians(tdec)), np.sin(np.radians(tdec))
    dra = np.radians(ra - tra)
    den = std * sd + ctd * cd * np.cos(dra)
    xi = np.degrees(cd * np.sin(dra) / den)
    eta = np.degrees((ctd * sd - std * cd * np.cos(dra)) / den)
    crpix = sh.SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    px, py = -xi / sh.PIXEL_SCALE_DEG + crpix - 0.5, eta / sh.PIXEL_SCALE_DEG + crpix - 0.5
    return (px - (BLOCK[0] + 0.5) * MINIPIX_PX) * SKY_PX, (py - (BLOCK[1] + 0.5) * MINIPIX_PX) * SKY_PX


def column(t, name):
    """A table column as floats, masked entries as nan (np.asarray would turn them into 0)."""
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)


def block_centre_label():
    """The block centre's ICRS coordinates, e.g. 13h42m11.5s +28d22m30s, as display text."""
    tra, tdec = (float(v[0]) for v in sh.tile_center(np.array([TILE])))
    crpix = sh.SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    px, py = (BLOCK[0] + 0.5) * MINIPIX_PX, (BLOCK[1] + 0.5) * MINIPIX_PX
    xi, eta = -(px - crpix + 0.5) * sh.PIXEL_SCALE_DEG, (py - crpix + 0.5) * sh.PIXEL_SCALE_DEG
    centre = SkyCoord(tra * u.deg, tdec * u.deg).spherical_offsets_by(xi * u.deg, eta * u.deg)
    h, m_, s_ = centre.ra.hms
    sign, d, dm, ds = centre.dec.signed_dms
    return f"{int(h):02d}ʰ{int(m_):02d}ᵐ{s_:04.1f}ˢ {'+' if sign > 0 else '−'}{int(d):02d}°{int(dm):02d}′{ds:02.0f}″"


def load_scene():
    g = Table.read(DATA / "gaia_dr3_core.ecsv")
    dx, dy = block_offsets(column(g, "RA_ICRS"), column(g, "DE_ICRS"))
    keep = (np.abs(dx) <= AREA_HALF + MARGIN) & (np.abs(dy) <= AREA_HALF + MARGIN) & np.isfinite(column(g, "Gmag"))
    g = g[keep]
    g["dx"], g["dy"] = dx[keep], dy[keep]
    g["in_block"] = (np.abs(g["dx"]) <= HALF) & (np.abs(g["dy"]) <= HALF)
    cls = Table.read(DATA / "gaia_classifier.ecsv")
    klass = dict(zip(np.asarray(cls["Source"]).astype(np.int64), np.asarray(cls["Class"])))
    g["class"] = [str(klass.get(int(s), "")) for s in np.asarray(g["Source"]).astype(np.int64)]
    return g


def load_pulsators():
    """Gaia SOS classifications: {source: dict(kind, period, epoch of maximum (BJD), mean mags, amplitudes)}.

    Used to choose which stars to model and to phase the MESA models on the night; never as a light curve.
    """
    out = {}
    for name, kind in (("gaia_rrlyrae.ecsv", "RR"), ("gaia_cepheid.ecsv", "CEP")):
        if not (DATA / name).exists():
            continue
        t = Table.read(DATA / name)
        for r in t:
            pf = float(r["PF"]) if np.isfinite(float(r["PF"])) and float(r["PF"]) > 0 else np.nan
            p1 = float(r["P1O"]) if "P1O" in t.colnames and np.isfinite(float(r["P1O"])) and float(r["P1O"]) > 0 else np.nan
            period = pf if np.isfinite(pf) else p1
            if not np.isfinite(period) or not np.isfinite(float(r["EpochG"])):
                continue
            sub = "RRab" if kind == "RR" and np.isfinite(pf) else ("RRc" if kind == "RR" else "Type II Cepheid")
            if kind == "CEP" and "Class" in t.colnames:  # Gaia's own classification
                klass, subclass, mode = str(r["Class"]), str(r["SubClass"]), str(r["ModeClass"])
                if klass == "DCEP" and mode in ("FUNDAMENTAL", "FIRST_OVERTONE"):
                    sub = "DCEP F" if mode == "FUNDAMENTAL" else "DCEP 1O"
                    period = pf if mode == "FUNDAMENTAL" else p1
                elif klass == "T2CEP":
                    sub = {"BL_HER": "BL Her", "W_VIR": "W Vir", "RV_TAU": "RV Tau"}.get(subclass, "Type II Cepheid")
                else:
                    sub = f"{klass} {mode}"
            if kind == "RR" and np.isfinite(pf) and np.isfinite(p1):
                sub = "RRd"
            amp = {b: float(r[f"ptp{b}"]) for b in ("G", "BP", "RP")}
            mean = {b: float(r[f"{b}magavg"]) for b in ("G", "BP", "RP")} if "BPmagavg" in t.colnames else {"G": float(r["Gmagavg"])}
            out[int(r["Source"])] = dict(period=period, epoch=float(r["EpochG"]) + 2455197.5, kind=sub, amp=amp, mean=mean)
    # ephemerides from other catalogues (argus_field.py: e.g. HADS from ASAS-SN), keyed by Gaia DR3 source
    if (DATA / "extra_pulsators.json").exists():
        for source, v in json.loads((DATA / "extra_pulsators.json").read_text()).items():
            out[int(source)] = dict(period=v["period"], epoch=v["epoch_bjd"], kind=v["kind"], amp={"G": v["amp"]},
                                    mean={"G": v["G"]})
    return out


def pulsator_title(follow, pulsators, src):
    """'6 RR Lyrae: MESA RSP', or e.g. '1 BL Her: MESA RSP · 1 eclipsing binary: PHOEBE'."""
    kinds = [pulsators[int(src[k])]["kind"] for k in follow]
    n_eb = sum(k == "EB" for k in kinds)
    puls = [k for k in kinds if k != "EB"]
    parts = []
    if puls:
        parts.append(f"{len(puls)} RR Lyrae: MESA RSP" if all(k.startswith("RR") for k in puls) else
                     f"{len(puls)} pulsator{'s' if len(puls) != 1 else ''} ({', '.join(dict.fromkeys(puls))}): MESA RSP")
    if n_eb:
        parts.append(f"{n_eb} eclipsing binar{'ies' if n_eb != 1 else 'y'}: PHOEBE")
    return " · ".join(parts) or "no modelled variables"


def choose_followed(stars, pulsators, n):
    """Row indices of the pulsators to follow (each modelled with MESA RSP): two RRc, then the largest-amplitude RRab.

    The block's 15.3 d Type II Cepheid is not followed: standard MESA RSP models of it run away
    (the W Vir envelope cools below the opacity tables or diverges), so it stays Gaia-driven.
    """
    src = np.asarray(stars["Source"]).astype(np.int64)
    in_block = np.asarray(stars["in_block"])
    if CLUSTERS[CLUSTER].get("followed_only"):  # an argus_field.py field: exactly the stars the user selected
        chosen = {v["source"] for v in json.loads((DATA / "followed.json").read_text()).values()}
        follow = [k for k in range(len(stars)) if int(src[k]) in chosen]
        return sorted(follow, key=lambda k: -pulsators[int(src[k])]["amp"]["G"])
    exclude = CLUSTERS[CLUSTER].get("exclude", set())
    candidates = [k for k in range(len(stars)) if in_block[k] and int(src[k]) in pulsators and int(src[k]) not in exclude]
    by_amp = sorted(candidates, key=lambda k: -pulsators[int(src[k])]["amp"]["G"])
    kind = {k: pulsators[int(src[k])]["kind"] for k in candidates}
    follow = []
    for name, quota in (("RRc", 2),):
        follow += [k for k in by_amp if kind[k] == name][:quota]
    follow += [k for k in by_amp if kind[k] == "RRab"][: max(0, n - len(follow))]
    return sorted(follow, key=lambda k: -pulsators[int(src[k])]["amp"]["G"])


def followed_targets(n=7):
    """The followed pulsators as {label: dict(source, kind, period, epoch_bjd, G)} (for mesa_argus.rsp)."""
    stars, pulsators = load_scene(), load_pulsators()
    src = np.asarray(stars["Source"]).astype(np.int64)
    out = {}
    for i, k in enumerate(choose_followed(stars, pulsators, n)):
        v = pulsators[int(src[k])]
        out[f"V{i + 1}"] = dict(source=int(src[k]), kind=v["kind"], period=v["period"], epoch_bjd=v["epoch"],
                                G=float(column(stars, "Gmag")[k]), amp_G=v["amp"]["G"])
    return out


# ---------------------------------------------------------------- SED templates
class Templates:
    """MESA SEDs (f_lambda at 10 pc) indexed by Gaia BP-RP, metal-poor track preferred."""

    def __init__(self, stride=1):
        waves, fluxes, colours, g, feh = None, [], [], [], []
        for path in sorted(glob.glob(str(_REPO / "runs" / "*_sed" / "seds.h5"))):
            with h5py.File(path) as f:
                w = f["wavelength"][:]
                waves = w
                sel = slice(None, None, stride)
                fluxes.append(f["flux"][sel])
                colours.append(f["mag/Gbp"][sel] - f["mag/Grp"][sel])
                g.append(f["mag/G"][sel])
                feh.append(np.full(len(g[-1]), float(f.attrs["feh"])))
        self.wave = waves * u.AA
        self.flux, self.colour, self.g, self.feh = (np.concatenate(x) for x in (fluxes, colours, g, feh))

    def index(self, bp_rp):
        """The template matching bp_rp."""
        score = np.abs(self.colour - bp_rp) + 0.004 * np.abs(self.feh + 1.5)  # mild preference for metal-poor (M3 [Fe/H] ~ -1.5)
        return int(np.argmin(score))

    def sed(self, bp_rp, gmag):
        """Observer-frame f_lambda (erg/s/cm2/A) of the template matching bp_rp, scaled to G = gmag."""
        i = self.index(bp_rp)
        return self.flux[i] * 10 ** (-0.4 * (gmag - self.g[i]))


class RSPStar:
    """A MESA RSP limit cycle (runs/rsp_<label>/pulsation.h5), phased so phase 0 is maximum light in rho.

    SEDs are f_lambda at 10 pc; the observer sees them at M3's distance with M3's reddening.
    """

    def __init__(self, path, ab, ebv=None):
        from mesa_argus import m67

        with h5py.File(path) as f:
            self.attrs = dict(f.attrs)
            wave, flux10, t = f["wavelength"][:], f["flux"][:], f["time_days"][:]
        self.wave = wave * u.AA
        # the distance and reddening the model was made for (recorded in the file); ebv: another field's reddening
        dm, ebv = float(self.attrs["distance_modulus"]), float(self.attrs["ebv"]) if ebv is None else ebv
        observed = m67.redden(wave, flux10 * (10.0 / 10 ** (dm / 5 + 1)) ** 2, ebv)
        rho = np.array([float(ab.mag_from_sed("rho", self.wave, f * FLAM)) for f in observed])
        # one full cycle ending at the last maximum light: RSP's own measured period (RRab light curves have a
        # secondary bump, so local maxima alone would mislead), maximum = brightest point in the last period
        self.period = float(self.attrs["nonlinear_period"])
        if t[-1] - t[0] < 2 * self.period - 1e-9:
            # a model stored as exactly one periodic cycle (e.g. a PHOEBE orbit, not an RSP run): unroll it to two
            # periods so the cycle ending at maximum light is complete
            one = t < t[0] + self.period - 1e-9
            t = np.r_[t[one], t[one] + self.period]
            observed, rho = np.r_[observed[one], observed[one]], np.r_[rho[one], rho[one]]
        last = t >= t[-1] - self.period
        if self.attrs.get("kind") == "EB":
            # binaries: phase 0 is the primary eclipse (the file starts there; the ephemeris is Gaia's primary
            # eclipse); a binary has two equal maxima, so "the brightest point" would be ambiguous
            i_max = int(np.nonzero(last)[0][0])
            sel = last
        else:
            i_max = int(np.nonzero(last)[0][np.argmin(rho[last])])
            sel = (t > t[i_max] - self.period) & (t <= t[i_max])
        self.phase = ((t[sel] - t[i_max]) / self.period) % 1.0
        order = np.argsort(self.phase)
        self.phase = self.phase[order]
        self.flux = observed[sel][order]
        self.rho_cycle = rho[sel][order]

    def sed(self, phase):
        """Observer-frame f_lambda at this phase: linear between the bracketing RSP steps."""
        phase = phase % 1.0
        n = len(self.phase)
        i = int(np.searchsorted(self.phase, phase))
        if i == 0:  # before the first stored step: wrap to the cycle's last step
            a, b, pa, pb = n - 1, 0, self.phase[-1] - 1.0, self.phase[0]
        elif i == n:  # after the last stored step: wrap to the first
            a, b, pa, pb = n - 1, 0, self.phase[-1], self.phase[0] + 1.0
        else:
            a, b, pa, pb = i - 1, i, self.phase[i - 1], self.phase[i]
        w = 0.0 if pb == pa else (phase - pa) / (pb - pa)
        return (1 - w) * self.flux[a] + w * self.flux[b]


# ---------------------------------------------------------------- the night
def night_schedule(date, max_airmass, cfg):
    """UTC start times of every 60 s exposure on the night of `date` with M3 up and astronomical dark."""
    site = EarthLocation(lat=cfg.observatory.latitude * u.deg, lon=cfg.observatory.longitude * u.deg,
                         height=cfg.observatory.altitude * u.m)
    start = Time(f"{date} 23:00:00")  # local early evening (UTC-5/6) -> scan 14 h
    grid = start + TimeDelta(np.arange(0, 14 * 60, 1) * 60, format="sec")
    frame = AltAz(obstime=grid, location=site)
    sun = get_sun(grid).transform_to(frame).alt.deg
    m3 = CENTRE.transform_to(frame)
    airmass = np.where(m3.alt.deg > 5, 1 / np.cos(np.radians(90 - m3.alt.deg)), np.inf)
    ok = (sun < -18) & (airmass <= max_airmass)
    ratchet, reset = cfg.survey.ratchet_len, cfg.survey.reset_min
    minute_of_ratchet = (grid.datetime64.astype("datetime64[m]").astype(int)) % int(ratchet)
    ok &= minute_of_ratchet < ratchet - reset
    times = grid[ok]
    ltt = times.light_travel_time(CENTRE, location=site)
    bjd = (times.tdb + ltt).jd + 30.0 / 86400  # mid-exposure
    return times, bjd, airmass[ok], m3.alt.deg[ok], m3.az.deg[ok]


def phases_at(var, bjd):
    return ((bjd - var["epoch"]) / var["period"]) % 1.0


# ---------------------------------------------------------------- detector scene
class Scene:
    """Every star's ArgusSim PSF on one pixel grid, for each seeing on a grid (one PSF set per seeing value)."""

    def __init__(self, stars, args, cfg, ab, seeings):
        self.plate = cfg.telescope.plate_scale
        self.n = int(np.ceil(2 * AREA_HALF / self.plate)) + 30
        self.n += self.n % 2
        self.centre = self.n / 2.0
        model = ZemaxPSF.from_config(cfg)
        self.seeings = list(seeings)
        self.lookup = {}
        for si, seeing in enumerate(self.seeings):
            for name, band in BANDS.items():
                for iy in range(PSF_PHASES):
                    for ix in range(PSF_PHASES):
                        p, _ = model.with_seeing(seeing=seeing, band=band, field_angle_deg=args.field_angle,
                                                 pixel_phase=(iy / PSF_PHASES, ix / PSF_PHASES))
                        p = np.maximum(p, 0.0)
                        self.lookup[name, si, iy, ix] = p / p.sum()
        p0 = self.lookup["rho", 0, 0, 0]
        yy, xx = np.indices(p0.shape)
        self.c0 = ((p0 * yy).sum(), (p0 * xx).sum())
        self.x = self.centre + np.asarray(stars["dx"]) / self.plate
        self.y = self.centre + np.asarray(stars["dy"]) / self.plate
        # origin and phase per star (the stamp centroid moves by -phase)
        self.oy = np.ceil(self.y - self.c0[0]).astype(int)
        self.ox = np.ceil(self.x - self.c0[1]).astype(int)
        self.py = np.clip(np.round((self.oy + self.c0[0] - self.y) * PSF_PHASES).astype(int), 0, PSF_PHASES - 1)
        self.px = np.clip(np.round((self.ox + self.c0[1] - self.x) * PSF_PHASES).astype(int), 0, PSF_PHASES - 1)
        self._designs = {}  # (band, seeing index) -> PSF matrix, built on demand; the most recent few kept
        self.seeing_index = None  # per exposure: set_seeing()
        self.sky = None  # electrons/s/pixel per band per exposure: set_sky()
        self.dark, self.readnoise, self.fullwell = cfg.telescope.dark_current, cfg.telescope.readnoise, cfg.telescope.fullwell
        self.exptime = args.exptime
        yy, xx = np.indices((self.n, self.n))
        self.yy, self.xx = yy.ravel(), xx.ravel()

    def _design(self, name, si):
        """Sparse (pixels x stars) matrix of unit-flux PSFs at seeing self.seeings[si]."""
        stack = np.stack([self.lookup[name, si, iy, ix] for iy in range(PSF_PHASES) for ix in range(PSF_PHASES)])
        p = stack[self.py * PSF_PHASES + self.px]  # (stars, stamp, stamp)
        yy, xx = np.indices(p.shape[1:])
        y = yy[None] + self.oy[:, None, None]
        x = xx[None] + self.ox[:, None, None]
        ok = (y >= 0) & (y < self.n) & (x >= 0) & (x < self.n) & (p > 0)
        cols = np.broadcast_to(np.arange(len(self.x))[:, None, None], p.shape)
        return sparse.csc_matrix((p[ok].astype(np.float32), ((y * self.n + x)[ok], cols[ok])),
                                 shape=(self.n * self.n, len(self.x)))

    def design(self, name, si):
        key = (name, int(si))
        if key not in self._designs:
            if len(self._designs) >= 6:
                self._designs.pop(next(iter(self._designs)))
            self._designs[key] = self._design(name, si)
        return self._designs[key]

    def set_sky(self, sky_rate):
        self.sky = sky_rate

    def set_seeing(self, index):
        self.seeing_index = np.asarray(index)

    def design_at(self, name, j):
        """The PSF matrix for exposure j's seeing."""
        return self.design(name, self.seeing_index[j])

    def background(self, name, j):
        """Sky + dark electrons per pixel in exposure j."""
        return (self.sky[name][j] + self.dark) * self.exptime

    def observe(self, name, rates, j, rng):
        """One exposure: Poisson draw of sky + dark + stars per pixel, clipped at full well, plus read noise."""
        A = self.design_at(name, j)
        bg = self.background(name, j)
        mean = bg + A @ (rates * self.exptime)
        charge = np.minimum(rng.poisson(mean), self.fullwell)
        image = charge + rng.normal(0.0, self.readnoise, mean.shape)
        return dict(image=image.reshape(self.n, self.n), full=(charge >= self.fullwell).reshape(self.n, self.n), mean=mean,
                    background=bg)


# ---------------------------------------------------------------- difference image (display only)
def difference_image(image, image_sat, reference, reference_sat, readnoise, half=3, clip=3.0, fit_pixels=None):
    """PSF-matched difference image: image - (reference (x) K + b), Bramich (2008) delta-function basis.

    K is a (2*half+1)^2 pixel kernel, free in every pixel, and b a constant background, fitted by weighted
    least squares (weights 1/variance of the image) over pixels unsaturated in both images. The kernel
    absorbs the change of seeing and the flux scale between the reference and this exposure.
    Returns the difference (nan where saturated or at the border) and the kernel's sum (flux scale).
    """
    n = image.shape[0]
    core = (slice(half, n - half), slice(half, n - half))
    columns = [reference[half + dy:n - half + dy, half + dx:n - half + dx].ravel()
               for dy in range(-half, half + 1) for dx in range(-half, half + 1)]
    A = np.column_stack(columns + [np.ones(columns[0].size)])
    target = image[core].ravel()
    # saturated pixels (exposure or reference) are left out of the kernel FIT only: clipped values do not follow
    # the reference. The difference itself is shown everywhere, saturated stars included (no display masking).
    bad = (image_sat | reference_sat)[core].ravel()
    if fit_pixels is not None:  # fit the kernel only on these pixels (stamps around bright stars)
        bad = bad | ~fit_pixels[core].ravel()
    # iterative fit (Bramich 2008, sec. 2.3): pixels with |residual| > 3 sigma (sigma from the fitted model's photon
    # + read noise) are rejected and the kernel refitted until the rejected set stops changing. This keeps variable
    # stars, which do not follow the reference, from being absorbed into the kernel's flux scale.
    use = ~bad
    sigma = np.sqrt(np.maximum(target, 1.0) + readnoise**2)
    for _ in range(10):
        coef, *_ = np.linalg.lstsq(A[use] / sigma[use, None], target[use] / sigma[use], rcond=None)
        model = A @ coef
        sigma = np.sqrt(np.maximum(model, 1.0) + readnoise**2)
        if clip <= 0:
            break
        new = ~bad & (np.abs(target - model) <= clip * sigma)
        if np.array_equal(new, use):
            break
        use = new
    diff = np.full_like(image, np.nan, dtype=float)
    d = target - A @ coef
    diff[core] = d.reshape(n - 2 * half, n - 2 * half)
    return diff, float(np.sum(coef[:-1]))


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/m3/m3_night.mp4"))
    p.add_argument("--cluster", choices=list(CLUSTERS), default="m3")
    p.add_argument("--date", default=None,
                   help="local date the night starts; 2027-05-05: new Moon (ArgusSim sky V ~22.0), every followed star "
                        "predicted to peak while M3 is up")
    p.add_argument("--max-airmass", type=float, default=2.0)
    p.add_argument("--start", default=None, help="UT (HH:MM) to start; default: when the field is first observable")
    p.add_argument("--end", default=None, help="UT (HH:MM) to stop; default: when the field is last observable")
    p.add_argument("--exptime", type=float, default=60.0)
    p.add_argument("--seeing", type=float, default=None,
                   help="zenith seeing FWHM (arcsec); default: ArgusSim's seasonal site median for the date")
    p.add_argument("--field-angle", type=float, default=1.44,
                   help="deg from the optical axis; default 1.44 = the area-weighted median over the 2.03 deg field")
    p.add_argument("--stride", type=int, default=1, help="render every Nth exposure (photometry uses all)")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--highlight", type=int, default=6, help="pulsators to follow (MESA RSP models) in the light-curve panel")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--kernel-half", type=int, default=3, help="difference-imaging kernel half-width (pixels); 3 = 7 x 7")
    p.add_argument("--kernel-clip", type=float, default=3.0,
                   help="kernel fit: reject pixels beyond this many sigma and refit (Bramich 2008); 0 = no rejection")
    p.add_argument("--kernel-stamps", type=float, default=20.0,
                   help="fit the kernel only on stamps around reference sources this many sigma above the sky "
                        "(HOTPANTS/ISIS style); 0 = all pixels")
    p.add_argument("--field", type=float, default=8.0,
                   help="side (arcmin) of the simulated area used for the kernel and zero point; "
                        "the movie shows the block at its centre")
    p.add_argument("--block", type=int, default=None,
                   help="minipixes across the block shown (odd; default the field's, else 3)")
    p.add_argument("--band", choices=list(BANDS), default="rho", help="band shown in the images, light curves and CMD")
    args = p.parse_args(argv)
    import os

    if not args.output.is_absolute():  # relative output paths are relative to where the command was run
        args.output = Path.cwd() / args.output
    os.chdir(_REPO)
    use_cluster(args.cluster)
    set_block(args.block or CLUSTERS[args.cluster].get("block_n", 3))
    global AREA_HALF
    AREA_HALF = max(30.0 * args.field, HALF + MARGIN)
    args.date = args.date or CLUSTERS[args.cluster]["date"]

    cfg = copy.deepcopy(ARGUS_CONFIG)
    if CLUSTERS[args.cluster]["site"]:  # the instrument stays Argus; only where it sits changes
        for key, value in CLUSTERS[args.cluster]["site"].items():
            setattr(cfg.observatory, key, value)
    tp = SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=cfg.telescope.throughput_loss)
    ab = ABPhot(throughput=tp, collecting_area=cfg.telescope.collecting_area)
    stars = load_scene()
    pulsators = load_pulsators()
    src = np.asarray(stars["Source"]).astype(np.int64)
    in_block = np.asarray(stars["in_block"])
    G = column(stars, "Gmag")
    gaia_var = np.array([int(s) in pulsators for s in src])  # Gaia-classified pulsators: constant unless followed
    follow = choose_followed(stars, pulsators, args.highlight)
    labels = {k: f"V{i + 1}" for i, k in enumerate(follow)}
    print(f"{len(stars)} Gaia sources ({int(in_block.sum())} in the block); following {len(follow)} pulsators with "
          f"MESA RSP models; the other {int((gaia_var & in_block).sum()) - len(follow)} Gaia-classified variables in "
          "the block are held constant")

    times, bjd, airmass, alt, az = night_schedule(args.date, args.max_airmass, cfg)
    if args.start or args.end:  # UT window; evening UT (>= 20h) counts as before midnight
        clock = lambda h: h - 24 if h >= 20 else h  # noqa: E731
        hours = np.array([clock(t.datetime.hour + t.datetime.minute / 60) for t in times])
        lo = clock(int(args.start[:2]) + int(args.start[3:5]) / 60) if args.start else -np.inf
        hi = clock(int(args.end[:2]) + int(args.end[3:5]) / 60) if args.end else np.inf
        keep = (hours >= lo) & (hours <= hi)
        times, bjd, airmass, alt, az = times[keep], bjd[keep], airmass[keep], alt[keep], az[keep]
    print(f"night of {args.date}: {len(times)} exposures, UT {times[0].iso[11:16]}-{times[-1].iso[11:16]}, "
          f"airmass {airmass.min():.2f}-{airmass.max():.2f}")

    # --- every star: a constant template SED at its Gaia mean G and colour
    templates = Templates()
    colour = column(stars, "BP-RP")
    # crowded-core sources often have no (or blended) BP/RP: give them the median colour of M3
    # stars of the same G outside the core (> 2.5'), where Gaia colours are reliable
    outer = Table.read(DATA / "gaia_dr3_core.ecsv")
    r_out = 60 * np.hypot((column(outer, "RA_ICRS") - CENTRE.ra.deg) * np.cos(CENTRE.dec.rad), column(outer, "DE_ICRS") - CENTRE.dec.deg)
    og, oc = column(outer, "Gmag"), column(outer, "BP-RP")
    good = (r_out > 2.5) & np.isfinite(og) & np.isfinite(oc)
    bins = np.arange(11.5, 20.51, 0.5)
    centres = 0.5 * (bins[1:] + bins[:-1])
    fiducial = np.array([np.median(oc[good & (og >= lo) & (og < hi)]) if np.sum(good & (og >= lo) & (og < hi)) >= 5 else np.nan
                         for lo, hi in zip(bins[:-1], bins[1:])])
    ok_f = np.isfinite(fiducial)
    missing = ~(np.isfinite(colour) & (colour > -0.5) & (colour < 3.0))
    colour = np.where(missing, np.interp(G, centres[ok_f], fiducial[ok_f]), colour)
    for k in np.nonzero(gaia_var)[0]:
        # a constant variable sits at its Gaia intensity-mean magnitudes; blended core colours fall back to a typical value
        v = pulsators[int(src[k])]
        mean_c = v["mean"].get("BP", np.nan) - v["mean"].get("RP", np.nan)
        low, high, typical = (0.2, 0.9, 0.55 if v["kind"] != "RRc" else 0.42) if v["kind"].startswith("RR") else (0.3, 1.3, 0.7)
        colour[k] = mean_c if np.isfinite(mean_c) and low <= mean_c <= high else typical
    print(f"{int(missing.sum())} sources without usable BP-RP take the median colour, at their G, of stars > 2.5' from the centre")
    static_rate = {b: np.zeros((len(stars), len(AIRMASS_GRID))) for b in BANDS}
    static_mag = {b: np.zeros(len(stars)) for b in BANDS}
    print("ArgusSim photometry of every star …", flush=True)
    # each star is its template SED scaled to its G: ArgusSim photometry once per template, then scaled per star
    t_index = np.array([templates.index(c_) for c_ in colour])
    scale = 10 ** (-0.4 * (G - templates.g[t_index]))
    for i_t in np.unique(t_index):
        s_ = templates.flux[i_t] * FLAM
        on = t_index == i_t
        for b, band in BANDS.items():
            static_rate[b][on] = np.outer(scale[on], [ab.electrons_from_sed(band, templates.wave, s_, airmass=x).value
                                                      for x in AIRMASS_GRID])
            static_mag[b][on] = float(ab.mag_from_sed(band, templates.wave, s_)) - 2.5 * np.log10(scale[on])

    # --- the followed pulsators: their MESA RSP SED at every exposure (phase from the star's Gaia ephemeris)
    import json
    from mesa_argus import rsp

    rsp_dir = {v["source"]: f"rsp_{rsp.PREFIX}{name}" for name, v in json.loads((DATA / "followed.json").read_text()).items()}
    rsp_stars, rsp_rate, rsp_mag, rsp_sed, rsp_cycle = {}, {}, {}, {}, {}
    for k in follow:
        model = RSPStar(_REPO / "runs" / rsp_dir[int(src[k])] / "pulsation.h5", ab)
        rsp_stars[k] = model
        seds = np.array([model.sed(ph) for ph in phases_at(pulsators[int(src[k])], bjd)])
        rsp_sed[k] = seds
        rsp_rate[k] = {b: np.array([ab.electrons_from_sed(band, model.wave, f * FLAM, airmass=x).value
                                    for f, x in zip(seds, airmass)]) for b, band in BANDS.items()}
        rsp_mag[k] = {b: np.array([float(ab.mag_from_sed(band, model.wave, f * FLAM)) for f in seds]) for b, band in BANDS.items()}
        grid = np.linspace(0, 1, 120, endpoint=False)
        rsp_cycle[k] = {b: np.array([float(ab.mag_from_sed(band, model.wave, model.sed(ph) * FLAM)) for ph in grid])
                        for b, band in BANDS.items()}
        v = pulsators[int(src[k])]
        print(f"  {labels[k]}: {v['kind']} MESA RSP, period {model.period:.5f} d (Gaia {v['period']:.5f}); "
              f"rho {rsp_cycle[k]['rho'].min():.2f}-{rsp_cycle[k]['rho'].max():.2f} (amplitude "
              f"{np.ptp(rsp_cycle[k]['rho']):.2f}); Gaia mean G {G[k]:.2f}")

    # catalogue (and difference-image reference) brightness: each pulsator's flux averaged over its cycle (intensity mean of the
    # MESA SEDs on a uniform phase grid), seen at each exposure's airmass
    ref_rate = {}
    for k in follow:
        mean_sed = np.mean([rsp_stars[k].sed(ph) for ph in np.linspace(0, 1, 200, endpoint=False)], axis=0) * FLAM
        ref_rate[k] = {b: np.array([ab.electrons_from_sed(band, rsp_stars[k].wave, mean_sed, airmass=x).value
                                    for x in airmass]) for b, band in BANDS.items()}

    def rates_at(b, j):
        x = float(np.clip(airmass[j], AIRMASS_GRID[0], AIRMASS_GRID[-1]))
        i1 = int(np.clip(np.searchsorted(AIRMASS_GRID, x), 1, len(AIRMASS_GRID) - 1))
        f = (x - AIRMASS_GRID[i1 - 1]) / (AIRMASS_GRID[i1] - AIRMASS_GRID[i1 - 1])
        out = static_rate[b][:, i1 - 1] * (1 - f) + static_rate[b][:, i1] * f
        for k in follow:
            out[k] = rsp_rate[k][b][j]
        return out

    def true_mag(b, k, j):
        return float(rsp_mag[k][b][j]) if k in rsp_mag else static_mag[b][k]

    palette = plt.get_cmap("tab10")
    colour_of = {k: palette(i % 10) for i, k in enumerate(follow)}

    # seeing per exposure from ArgusSim's site model: the season's median zenith seeing at McDonald (DIMM, Barker
    # et al. 2003) scaled to each exposure's airmass as X^0.6; PSFs precomputed on a 0.05" grid
    from argus_sim.observatory import seasonal_seeing_params, seeing_at_airmass

    zenith_seeing = args.seeing if args.seeing else seasonal_seeing_params(times[0])[0]
    seeing = seeing_at_airmass(zenith_seeing, airmass)
    seeing_grid = np.round(np.arange(np.floor(seeing.min() * 20) / 20, seeing.max() + 0.05, 0.05), 2)
    print(f"seeing: {zenith_seeing:.2f}\" at zenith -> {seeing.min():.2f}-{seeing.max():.2f}\" (ArgusSim site model); "
          f"{len(seeing_grid)} PSF sets", flush=True)
    print("building the detector scene …", flush=True)
    scene = Scene(stars, args, cfg, ab, seeing_grid)
    scene.set_seeing(np.abs(seeing[:, None] - seeing_grid[None, :]).argmin(axis=1))
    # sky at M3 in every exposure from ArgusSim's model: airglow (solar activity, airmass), zodiacal light
    # (ecliptic position), integrated starlight (Galactic latitude), Moon (Krisciunas & Schaefer 1991), site light
    from argus_sim.observatory import Observatory

    observatory = Observatory(cfg)
    per_exposure = [observatory.sky_components_at(float(a_), float(z_), t_) for a_, z_, t_ in zip(alt, az, times)]
    comp = {key: np.array([float(np.ravel(c_[key])[0]) for c_ in per_exposure]) for key in per_exposure[0]}
    sky_rate = {}
    for name, band in BANDS.items():
        sky = SpectralSky(ab.tp, ab, band)
        sky_rate[name] = np.asarray(sky.sky_electrons(comp["airglow_s10"], comp["zodiacal_s10"], comp["starlight_s10"],
                                                      comp["moon_s10"], comp["artif_s10"], transparency=1.0)) * scene.plate**2
    scene.set_sky(sky_rate)
    print(f"sky (ArgusSim): V {np.min(comp['v_mag']):.2f}-{np.max(comp['v_mag']):.2f} mag/arcsec2; Moon "
          f"{np.mean(comp['illum_frac']) * 100:.0f}% illuminated, altitude {np.min(comp['moon_alt']):.0f} to "
          f"{np.max(comp['moon_alt']):.0f} deg; rho sky {sky_rate['rho'].min():.1f}-{sky_rate['rho'].max():.1f} e/s/pixel")
    ut = (times.jd - np.floor(times[0].jd - 0.5) - 0.5) * 24  # UT hours since 0h on the first date

    # fixed display limits
    jbest = int(np.argmin(airmass))
    B = args.band  # the band shown in the images, light curves and CMD magnitude axis
    shown_px = ((np.abs(scene.xx - scene.centre) * scene.plate <= HALF + 2) & (np.abs(scene.yy - scene.centre) * scene.plate <= HALF + 2))
    peak = max(scene.background(B, jbest) + (scene.design_at(B, jbest) @ (rates_at(B, jbest) * args.exptime))[shown_px].max(), 1)
    vmax = float(min(peak, scene.fullwell))
    lw = float(np.sqrt(np.median([scene.background(B, j) for j in range(len(times))]) + scene.readnoise**2))
    psf_peak = {k: scene.design_at(B, jbest)[:, k].max() for k in follow}
    # colour limit: half the largest pulsator excursion at its peak pixel, so the variation reads strongly
    dlim = 0.5 * max(np.max(np.abs(rsp_rate[k][B] - ref_rate[k][B])) * args.exptime * psf_peak[k] for k in follow)
    blockmask = in_block & (G < 20)
    fixed = np.nonzero(blockmask & ~np.isin(np.arange(len(stars)), follow))[0]
    cmd_c = np.r_[[static_mag["beta"][k] - static_mag["rho"][k] for k in fixed],
                  np.concatenate([rsp_cycle[k]["beta"] - rsp_cycle[k]["rho"] for k in follow])]
    cmd_m = np.r_[[static_mag[B][k] for k in fixed], np.concatenate([rsp_cycle[k][B] for k in follow])]
    follow_m = np.concatenate([rsp_cycle[k][B] for k in follow])

    history = {k: {"t": [], "m": [], "e": [], "true": []} for k in follow}
    seeds = np.random.SeedSequence(args.seed).spawn(len(times))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                                     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    apply_style()
    fig = plt.figure(figsize=(16, 10))
    frames = range(0, len(times), args.stride)
    # reference for the difference panel: median of the best-seeing exposures of the night (each one a full noisy
    # simulated exposure, the same as it appears in the movie), as a survey pipeline would stack them
    ref_frames = np.argsort(seeing, kind="stable")[: max(20, len(times) // 5)]
    stack, stack_sat = [], []
    for j in ref_frames:
        rng_ref = np.random.default_rng(seeds[j])  # both bands in the movie's order: the identical exposure
        o = {b: scene.observe(b, rates_at(b, j), j, rng_ref) for b in BANDS}[B]
        stack.append(o["image"])
        stack_sat.append(o["full"])
    ref_image = np.median(stack, axis=0)
    ref_sat = np.mean(stack_sat, axis=0) >= 0.5  # saturated in the median reference itself
    stamp_pixels = None
    if args.kernel_stamps > 0:
        from scipy import ndimage
        sky_ref = float(np.median(ref_image))
        noise_ref = 1.4826 * float(np.median(np.abs(ref_image - sky_ref)))
        bright = (ref_image - sky_ref) > args.kernel_stamps * noise_ref
        stamp_pixels = ndimage.binary_dilation(bright, structure=np.ones((2 * args.kernel_half + 9,) * 2, bool))
        print(f"kernel stamps: {int(ndimage.label(bright)[1])} sources above {args.kernel_stamps:g} sigma, "
              f"{100 * stamp_pixels.mean():.0f}% of pixels", flush=True)
    print(f"difference reference: median of {len(ref_frames)} best-seeing exposures "
          f"(seeing {seeing[ref_frames].min():.2f}-{seeing[ref_frames].max():.2f}\")", flush=True)

    # photometry (mesa_argus.photometry): the catalogue a pipeline would calibrate against is every star's magnitude
    # (pulsators: their intensity-mean); expected catalogue fluxes pick the calibration stars
    from mesa_argus.photometry import SExtractorPhotometry

    cat_mag = static_mag[B].copy()
    cat_flux = static_rate[B][:, 0] * args.exptime
    for k in follow:
        cat_mag[k] = -2.5 * np.log10(np.mean(10 ** (-0.4 * rsp_cycle[k][B])))
        cat_flux[k] = np.mean(ref_rate[k][B]) * args.exptime
    p0 = scene.lookup[B, int(scene.seeing_index[ref_frames[0]]), 0, 0]
    yy_, xx_ = np.indices(p0.shape)
    cy_, cx_ = (p0 * yy_).sum(), (p0 * xx_).sum()
    ref_fwhm = 2.3548 * np.sqrt(0.5 * (p0 * ((yy_ - cy_) ** 2 + (xx_ - cx_) ** 2)).sum())
    phot = SExtractorPhotometry(scene.x, scene.y, cat_mag, cat_flux, scene.n, scene.fullwell, follow,
                                colour=column(stars, "BP-RP"), fwhm=ref_fwhm)
    rows = []
    print(f"observing and rendering {len(frames)} frames ({len(times)} exposures) -> {args.output}", flush=True)
    def exposure(j):  # both bands drawn from the exposure's own seed: identical in both passes
        rng = np.random.default_rng(seeds[j])
        return {b: scene.observe(b, rates_at(b, j), j, rng) for b in BANDS}

    # pass 1: measure every frame (relative light curves need each star's mean over the night)
    measured_frames = []
    for n, j in enumerate(frames):
        r = phot.measure(exposure(j)[B]["image"])
        frame_m = {}
        for i, k in enumerate(follow):
            ok = np.isfinite(r["mag"][i]) and r["flux"][i] / r["sigma"][i] >= 5
            frame_m[k] = (r["mag"][i] if ok else np.nan, r["mag_err"][i] if ok else np.nan)
            rows.append(dict(exposure=j, ut_hours=ut[j], airmass=airmass[j], seeing=seeing[j],
                             seeing_model=scene.seeings[int(scene.seeing_index[j])], star=labels[k],
                             true_mag=true_mag(B, k, j), saturated=bool(r["saturated"][i]),
                             sextractor_mag=r["mag"][i], sextractor_err=r["mag_err"][i]))
        measured_frames.append(frame_m)
        if (n + 1) % max(1, len(frames) // 10) == 0:
            print(f"  measured {n + 1}/{len(frames)}", flush=True)
    # each star's mean over the night: measured (its own measurements) and model (its own curve)
    rel = {}
    for k in follow:
        mm = np.array([f[k][0] for f in measured_frames], float)
        tt = np.array([true_mag(B, k, j) for j in frames], float)
        rel[k] = dict(measured=float(np.nanmean(mm)) if np.isfinite(mm).any() else np.nan, model=float(np.mean(tt)),
                      span=(np.nanmin(np.r_[mm - np.nanmean(mm) if np.isfinite(mm).any() else [], tt - tt.mean()]),
                            np.nanmax(np.r_[mm - np.nanmean(mm) if np.isfinite(mm).any() else [], tt - tt.mean()])))

    # fixed SED axis: every followed star's SED (300-1000 nm) over every exposure of the night
    sed_vals = np.concatenate([rsp_sed[k][:, (rsp_stars[k].wave.to_value(u.nm) >= 300) & (rsp_stars[k].wave.to_value(u.nm) <= 1000)].ravel()
                               for k in follow]) if follow else np.array([1e-16, 1e-14])
    sed_vals = sed_vals[sed_vals > 0]
    sed_ylim = (float(np.percentile(sed_vals, 3)) / 1.5, float(sed_vals.max()) * 1.5)

    # pass 2: draw (the same exposures: the same seeds)
    with writer.saving(fig, str(args.output), dpi=args.dpi):
        for n, j in enumerate(frames):
            obs = exposure(j)
            obs[B]["difference"], _ = difference_image(obs[B]["image"], obs[B]["full"], ref_image, ref_sat, scene.readnoise,
                                                       half=args.kernel_half, clip=args.kernel_clip, fit_pixels=stamp_pixels)
            for k in follow:
                h = history[k]
                h["t"].append(ut[j])
                h["true"].append(true_mag(B, k, j))
                h["m"].append(measured_frames[n][k][0])
                h["e"].append(measured_frames[n][k][1])
            fig.clear()
            draw(fig, args, stars, scene, obs, j, n, times, ut, airmass, follow, labels, colour_of, history,
                 static_mag, pulsators, src, gaia_var, in_block, G, templates, bjd, true_mag,
                 dict(cycle=rsp_cycle, sed=rsp_sed, stars=rsp_stars),
                 dict(vmax=vmax, lw=lw, dlim=dlim, cmd_c=cmd_c, cmd_m=cmd_m, follow_m=follow_m, ut=(ut[0], ut[-1]),
                      seeing=seeing, centre=block_centre_label(), n_ref=len(ref_frames),
                      rel=rel, sed_ylim=sed_ylim), ab)
            writer.grab_frame()
            if (n + 1) % max(1, len(frames) // 20) == 0:
                print(f"  {n + 1}/{len(frames)}", flush=True)
    import csv

    with open(args.output.with_suffix(".photometry.csv"), "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plt.close(fig)
    print(f"saved {args.output}")


def draw(fig, args, stars, scene, obs, j, n, times, ut, airmass, follow, labels, colour_of, history, static_mag,
         pulsators, src, gaia_var, in_block, G, templates, bjd, true_mag, rsp, lim, ab):
    grid = fig.add_gridspec(2, 2, width_ratios=[5.4, 4.6], height_ratios=[6, 4], wspace=0.16, hspace=0.3,
                            left=0.055, right=0.975, bottom=0.06, top=0.88)
    cmd = fig.add_subplot(grid[0, 0])
    pair = grid[0, 1].subgridspec(1, 2, wspace=0.06)
    image_ax = fig.add_subplot(pair[0])
    diff_ax = fig.add_subplot(pair[1], sharex=image_ax, sharey=image_ax)
    sed_ax = fig.add_subplot(grid[1, 1])
    light = grid[1, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    lc = fig.add_subplot(light[0])
    cont_ax = fig.add_subplot(light[1], sharex=lc)

    # --- CMD: every constant star in the block; the followed MESA pulsators with their model cycle
    others = np.nonzero(in_block & (G < 20) & ~np.isin(np.arange(len(G)), follow))[0]
    B = args.band
    cmd.scatter([static_mag["beta"][k] - static_mag["rho"][k] for k in others], [static_mag[B][k] for k in others],
                s=4, color="#b9c4cc", zorder=1, label="Constant stars")
    for k in follow:
        cyc = rsp["cycle"][k]
        cmd.plot(np.r_[cyc["beta"] - cyc["rho"], cyc["beta"][0] - cyc["rho"][0]], np.r_[cyc[B], cyc[B][0]],
                 color=colour_of[k], lw=1.3, alpha=0.9, zorder=3)
        now_c, now_m = true_mag("beta", k, j) - true_mag("rho", k, j), true_mag(B, k, j)
        cmd.scatter(now_c, now_m, s=60, color=colour_of[k], edgecolor="#2b2b2b", linewidth=0.6, zorder=5)
    cmd.set(xlim=(np.percentile(lim["cmd_c"], 0.5) - 0.15, np.percentile(lim["cmd_c"], 99.5) + 0.15),
            ylim=(min(20.2, np.max(lim["cmd_m"]) + 0.3), np.min(lim["cmd_m"]) - 0.5),
            xlabel=r"$\beta-\rho$ (AB mag)", ylabel=rf"$m_{{\{B}}}$ (AB mag)")
    cmd.set_title("CMD", loc="left", weight="bold")
    legend = cmd.legend(loc="lower left", fontsize=9, framealpha=0.85, edgecolor="none")
    legend.set_zorder(2)
    cmd.grid(alpha=0.15)

    # --- light curves of the followed pulsators: measured (deblended PSF fit) rho, with the MESA model behind
    model_t = np.linspace(lim["ut"][0], lim["ut"][1], 800)
    model_b = bjd[0] + (model_t - ut[0]) / 24.0
    for k in follow:
        h = history[k]
        t = np.array(h["t"])
        v = pulsators[int(src[k])]
        cyc = rsp["cycle"][k][B]
        model = np.interp(((model_b - v["epoch"]) / v["period"]) % 1.0, np.linspace(0, 1, len(cyc) + 1), np.r_[cyc, cyc[0]])
        # each star relative to its own mean over the night: the model to the model's, the measurements to theirs
        lc.plot(model_t, model - lim["rel"][k]["model"], color=colour_of[k], lw=1.0, alpha=0.35)
        m, e = np.array(h["m"]), np.array(h["e"])
        ok = np.isfinite(m)
        lc.errorbar(t[ok], m[ok] - lim["rel"][k]["measured"], yerr=e[ok], fmt="o", ms=2.5, lw=0.5, color=colour_of[k],
                    label=f"{labels[k]} {v['kind']} {v['period']:.3f} d")
        cont_ax.plot(t, np.array(h["m"]) - np.array(h["true"]), color=colour_of[k], lw=0.8)
    kinds_ = {pulsators[int(src[k])]["kind"] for k in follow}
    model_name = " / ".join(([("MESA RSP")] if kinds_ - {"EB"} else []) + (["PHOEBE"] if "EB" in kinds_ else []))
    lc.plot([], [], color="0.5", lw=1.0, alpha=0.5, label=f"{model_name} model")
    for ax in (lc, cont_ax):
        ax.axvline(ut[j], color="#455868", lw=0.8, alpha=0.35)
        ax.grid(alpha=0.15)
        ax.set_xlim(lim["ut"][0] - 0.1, lim["ut"][1] + 0.1)
    lo_ = min(lim["rel"][k]["span"][0] for k in follow)
    hi_ = max(lim["rel"][k]["span"][1] for k in follow)
    pad = 0.08 * max(hi_ - lo_, 0.05)
    lc.set(ylim=(hi_ + pad, lo_ - 2.5 * pad), ylabel=rf"$\Delta m_{{\{B}}}$ (mag)")
    lc.legend(loc="upper left", ncol=4, fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none", columnspacing=0.8,
              handletextpad=0.3)
    lc.set_title("Light curves (SExtractor MAG_AUTO)", loc="left", weight="bold")
    lc.set_title(f"UT {times[j].iso[11:16]} · airmass {airmass[j]:.2f} · seeing {lim['seeing'][j]:.2f}″", loc="right",
                 fontsize=11, color="#3c5262")
    plt.setp(lc.get_xticklabels(), visible=False)
    cont_ax.axhline(0, color="0.5", lw=0.6, ls=":")
    cont_ax.set(xlabel=f"UT hours ({args.date} night)", ylabel="measured − true\n(mag)")
    lc.yaxis.set_label_coords(-0.075, 0.5)
    cont_ax.yaxis.set_label_coords(-0.075, 0.5)
    cont_ax.invert_yaxis()

    # --- image
    img = obs[B]["image"]
    extent = (np.array([0, scene.n, 0, scene.n]) - scene.centre) * scene.plate
    norm = AsinhNorm(linear_width=lim["lw"], vmin=0.0, vmax=lim["vmax"])
    shown = image_ax.imshow(img, origin="lower", interpolation="nearest", cmap="magma", norm=norm, extent=extent)
    step = MINIPIX_PX * SKY_PX
    for v in np.arange(-HALF, HALF + 1e-6, step):
        image_ax.plot([v, v], [-HALF, HALF], color="white", lw=0.5, alpha=0.35)
        image_ax.plot([-HALF, HALF], [v, v], color="white", lw=0.5, alpha=0.35)
    image_ax.add_patch(Rectangle((-step / 2, -step / 2), step, step, fill=False, ec="#7fd1ff", lw=1.2))
    for k in follow:
        image_ax.add_patch(Circle((stars["dx"][k], stars["dy"][k]), 3.0, fill=False, ec=colour_of[k], lw=1.2))
        image_ax.text(stars["dx"][k] + 3.2, stars["dy"][k] + 3.2, labels[k], fontsize=8, color=colour_of[k], weight="bold")
    lim_a = HALF + 2
    image_ax.set(xlim=(-lim_a, lim_a), ylim=(-lim_a, lim_a), xlabel="Offset (arcsec)", ylabel="Offset (arcsec)")
    image_ax.set_title(f"Exposure ({SYMBOLS[B]})", loc="left", weight="bold")
    bar = fig.colorbar(shown, ax=image_ax, orientation="horizontal", fraction=0.05, pad=0.13, label="electrons / pixel")
    bar.set_ticks(norm.inverse(np.linspace(0, 1, 5)))
    bar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    bar.ax.tick_params(labelsize=7)
    diverging = plt.get_cmap("RdBu_r").copy()
    diverging.set_bad("#bfbfbf")
    diff = diff_ax.imshow(obs[B]["difference"], origin="lower", interpolation="nearest", cmap=diverging,
                          vmin=-lim["dlim"], vmax=lim["dlim"], extent=extent)
    for k in follow:
        diff_ax.add_patch(Circle((stars["dx"][k], stars["dy"][k]), 3.0, fill=False, ec=colour_of[k], lw=1.0))
        diff_ax.text(stars["dx"][k] + 3.2, stars["dy"][k] + 3.2, labels[k], fontsize=8, color=colour_of[k], weight="bold")
    diff_ax.set_title("Difference (PSF-matched)", loc="left", weight="bold")
    diff_ax.text(0.03, 0.97, f"ref: median of {lim['n_ref']} best-seeing exposures", transform=diff_ax.transAxes, va="top",
                 fontsize=7.5, color="#333", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2})
    diff_ax.set(xlim=(-lim_a, lim_a), ylim=(-lim_a, lim_a), xlabel="Offset (arcsec)")
    plt.setp(diff_ax.get_yticklabels(), visible=False)
    dbar = fig.colorbar(diff, ax=diff_ax, orientation="horizontal", fraction=0.05, pad=0.13,
                        label="difference (electrons / pixel)")
    dbar.ax.tick_params(labelsize=7)
    image_ax.text(0.03, 0.97, f"{args.exptime:g} s · exposure {j + 1}", transform=image_ax.transAxes, va="top",
                  fontsize=8, color="white", bbox={"facecolor": "#111726", "alpha": 0.65, "edgecolor": "none", "pad": 3})

    # --- spectra of the followed pulsators now
    shown = []
    for k in follow:
        w, f = rsp["stars"][k].wave.to_value(u.nm), rsp["sed"][k][j]
        band = (w >= 300) & (w <= 1000)
        sed_ax.plot(w[band], f[band], color=colour_of[k], lw=0.9)
        shown.append(f[band])
        # the SED as each Argus band records it: f_lambda x (filter x optics x QE x atmosphere^airmass)
        tw = ab.tp.wav.to_value(u.nm)
        on_grid = np.interp(tw, w, f, left=0.0, right=0.0)
        for name, bnd in BANDS.items():
            seen = on_grid * system_response(ab.tp, bnd, airmass[j])
            ok = seen > 0
            sed_ax.plot(tw[ok], seen[ok], color=colour_of[k], lw=0.9, ls=":")
    sed_ax.set(yscale="log", ylim=lim["sed_ylim"], xlim=(300, 1000),
               xlabel="Wavelength (nm)", ylabel=r"$f_\lambda$ at Argus (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    sed_ax.set_title("SED", loc="left", weight="bold")
    response_ax = sed_ax.twinx()
    response_ax.spines["right"].set_visible(True)
    for name, band in BANDS.items():
        response_ax.fill_between(ab.tp.wav.to_value(u.nm), system_response(ab.tp, band, airmass[j]), color=COLORS[name], alpha=0.10)
    response_ax.set(ylim=(0, 1), ylabel="Detected response")
    sed_ax.plot([], [], color="0.4", lw=0.9, label="SED")
    sed_ax.plot([], [], color="0.4", lw=0.9, ls=":", label=r"SED × $\beta$, $\rho$ system response")
    sed_ax.legend(loc="upper right", fontsize=8, frameon=False)
    sed_ax.grid(alpha=0.12)

    fig.suptitle(
        f"ARGUS — {lim['centre']} ({CLUSTERS[args.cluster]['name']}) — {block_label()} — {args.date} — 60 s exposures — "
        + pulsator_title(follow, pulsators, src),
        fontsize=13.5,
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
