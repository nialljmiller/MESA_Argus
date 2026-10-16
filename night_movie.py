#!/usr/bin/env python3
"""The engine: one night of Argus 60 s exposures of a scene, simulated and measured; the movie (CMD, colour exposure,
difference image, light curves, SEDs) and the photometry CSV. observe.py writes the scene and runs this, once per band.

    python night_movie.py --scene outputs/f294/scene --band rho -o outputs/f294/test.mp4 --stride 10   # by hand
    python night_movie.py --field m3                                                                    # a hand-made field

The night, step by step (main):
  1. the stars: the catalogue (constant stars) and the followed stars (real variables, placed stars, transients)
  2. the night: ArgusSim's exposure schedule at the field (mesa_argus.night), airmass
  3. constant stars: the MESA SED matched to each star's Gaia BP-RP, scaled to its G -> electrons/s per band (ArgusSim)
  4. followed stars: their SED at every exposure (pulsation / orbit phase, or time since t0), through the observation's
     dust -> electrons/s
  5. conditions per exposure (ArgusSim): seeing (seasonal median x airmass^0.6), Zemax PSF with that seeing, sky
  6. the difference-image reference: the median of the best-seeing exposures
  7. every exposure: PSFs + sky -> Poisson, full well, read noise -> SExtractor photometry (mesa_argus.photometry)
  8. the movie, frame by frame (the same exposures again, from the same seeds)

Assumptions (field angle, coverage, seeing, transparency, ...): docs/ASSUMPTIONS.md.
"""
import argparse
import copy
import csv
import json
import textwrap
import warnings
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.utils import iers
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import AsinhNorm  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage, sparse  # noqa: E402

from argus_sim import ABPhot, SpectralSky, SystemThroughput, ZemaxPSF  # noqa: E402
from argus_sim import c as ARGUS_CONFIG  # noqa: E402
from argus_sim import skymap_shim as sh  # noqa: E402
from argus_sim.observatory import Observatory, seasonal_seeing_params, seeing_at_airmass  # noqa: E402

from mesa_argus import dust  # noqa: E402
from mesa_argus.argus import BANDS, COLORS, FLAM, SYMBOLS, apply_style, system_response  # noqa: E402
from mesa_argus.seds import Templates  # noqa: E402

iers.conf.auto_download = False
_REPO = Path(__file__).resolve().parent  # data/, runs/, outputs/ are in the repo, wherever this is run from

# hand-made fields (before observe.py): Gaia data in data/<field>/, centre, HDPS tile and block centre, default night,
# site override
FIELDS = {
    "m3": dict(name="M3", centre=(205.5484, 28.3773), tile=345967, block=(34, 32), date="2027-05-05", site=None),
    # M15 for a comparison with ADDEM at Sierra Remote Observatories (approximate SRO coordinates; Argus instrument)
    "m15": dict(name="M15", centre=(322.49304, 12.16700), tile=298720, block=(26, 36), date="2026-10-05",
                site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0),
                # an RRc whose first overtone RSP finds stable even at 0.70 x the cluster L: not followed
                exclude={1745948328028784640}),
    # one RRab (Gaia DR3 308108245007332224, P 0.382 d, G 13.9) and a few neighbours, b = -34 deg
    "rr308108": dict(name="field RRab", centre=(17.588538, 28.550090), tile=345631, block=(35, 54), date="2026-10-06",
                     site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0)),
    # RRab Gaia DR3 416445908838640384 (P 0.404 d, G 14.4) with 21 stars G < 20 in the block, b = -10 deg
    "rr416445": dict(name="field RRab", centre=(8.841137, 52.998087), tile=417610, block=(12, 34), date="2026-10-06",
                     site=dict(latitude=37.07, longitude=-119.41, altitude=1400.0)),
}
for _f in sorted((_REPO / "data").glob("*/field.json")):  # any other data/<field>/field.json (older argus_field.py fields)
    _c = json.loads(_f.read_text())
    FIELDS[_f.parent.name] = dict(name=_c["name"], centre=tuple(_c["centre"]), tile=_c["tile"], block=tuple(_c["block"]),
                                  date=_c["date"], site=_c.get("site"), followed_only=True, block_n=_c.get("block_n", 3),
                                  synthetic=_c.get("synthetic", False))

# the field being observed (use_field sets these)
FIELD = "m3"
DATA = Path("data/m3")
CENTRE = SkyCoord(205.5484 * u.deg, 28.3773 * u.deg)
TILE, BLOCK = 345967, (34, 32)

MINIPIX_PX, SKY_PX = 32, 900.0 / 1024.0  # HDPS minipix (pixels) and tessellation pixel (arcsec)
BLOCK_N = 3  # the block shown: BLOCK_N x BLOCK_N minipixes centred on the block minipix (set_block)
HALF = BLOCK_N / 2 * MINIPIX_PX * SKY_PX  # its half-width (arcsec)
MARGIN = 15.0  # arcsec: stars this far outside the simulated area still spill light in
AREA_HALF = 240.0  # half-width (arcsec) of the simulated area (--area): the kernel and zero point use all of it
AIRMASS_GRID = np.round(np.arange(1.0, 2.61, 0.2), 2)  # constant stars' rates are tabulated on it
PSF_PHASES = 8  # sub-pixel phases per axis in the PSF lookup


def use_field(field, scene=None):
    """Point the module (and mesa_argus.rsp) at a scene folder (observe.py) or a hand-made field's data/<field>/."""
    global FIELD, DATA, CENTRE, TILE, BLOCK
    from mesa_argus import rsp

    if scene is not None:
        scene = Path(scene).resolve()
        c = json.loads((scene / "field.json").read_text())
        FIELDS[field] = dict(name=c["name"], centre=tuple(c["centre"]), tile=c["tile"], block=tuple(c["block"]),
                             date=c["date"], site=c.get("site"), followed_only=True, block_n=c.get("block_n", 3),
                             synthetic=c.get("synthetic", False))
    f = FIELDS[field]
    FIELD, DATA = field, (scene if scene is not None else _REPO / "data" / field)
    CENTRE = SkyCoord(f["centre"][0] * u.deg, f["centre"][1] * u.deg)
    TILE, BLOCK = f["tile"], f["block"]
    if scene is None:
        rsp.use(field)


def set_block(n):
    """Show an n x n block of minipixes (n odd)."""
    global BLOCK_N, HALF
    BLOCK_N, HALF = n, n / 2 * MINIPIX_PX * SKY_PX


def block_label():
    return f"{BLOCK_N} × {BLOCK_N} minipixes ({2 * HALF:.0f}″)"


# ---------------------------------------------------------------- the stars
def block_offsets(ra, dec):
    """Offsets (arcsec) from the block centre in the HDPS tile's tangent-plane pixel frame."""
    tile_ra, tile_dec = (float(v[0]) for v in sh.tile_center(np.array([TILE])))
    cd, sd = np.cos(np.radians(dec)), np.sin(np.radians(dec))
    ctd, std = np.cos(np.radians(tile_dec)), np.sin(np.radians(tile_dec))
    dra = np.radians(ra - tile_ra)
    den = std * sd + ctd * cd * np.cos(dra)
    xi = np.degrees(cd * np.sin(dra) / den)
    eta = np.degrees((ctd * sd - std * cd * np.cos(dra)) / den)
    crpix = sh.SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    px, py = -xi / sh.PIXEL_SCALE_DEG + crpix - 0.5, eta / sh.PIXEL_SCALE_DEG + crpix - 0.5
    return (px - (BLOCK[0] + 0.5) * MINIPIX_PX) * SKY_PX, (py - (BLOCK[1] + 0.5) * MINIPIX_PX) * SKY_PX


def column(t, name):
    """A table column as floats, masked entries as nan (np.asarray would make them 0)."""
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)


def block_centre_label():
    """The block centre as display text, e.g. 13ʰ42ᵐ11.5ˢ +28°22′30″."""
    tile_ra, tile_dec = (float(v[0]) for v in sh.tile_center(np.array([TILE])))
    crpix = sh.SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    px, py = (BLOCK[0] + 0.5) * MINIPIX_PX, (BLOCK[1] + 0.5) * MINIPIX_PX
    xi, eta = -(px - crpix + 0.5) * sh.PIXEL_SCALE_DEG, (py - crpix + 0.5) * sh.PIXEL_SCALE_DEG
    centre = SkyCoord(tile_ra * u.deg, tile_dec * u.deg).spherical_offsets_by(xi * u.deg, eta * u.deg)
    h, m, s = centre.ra.hms
    sign, d, dm, ds = centre.dec.signed_dms
    return f"{int(h):02d}ʰ{int(m):02d}ᵐ{s:04.1f}ˢ {'+' if sign > 0 else '−'}{int(d):02d}°{int(dm):02d}′{ds:02.0f}″"


def star_catalogue():
    """The field's star list: Gaia DR3, or a synthetic field's (a real field's stars at random positions)."""
    return DATA / ("synthetic_stars.ecsv" if FIELDS[FIELD].get("synthetic") else "gaia_dr3_core.ecsv")


def load_followed():
    """A hand-made field's followed stars (data/<field>/followed.json); {} for a scene."""
    path = DATA / "followed.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_placed():
    """The stars placed by observe.py (sources.json), in order: real Gaia stars (positive ids) and placed stars and
    transients (-1, -2, ...), each with its SED file, position, timing and dust."""
    path = DATA / "sources.json"
    return json.loads(path.read_text()) if path.exists() else []


def engine_of(v):
    """What computed a followed star's light (a load_pulsators entry)."""
    if "transient" in v:
        return v["transient"].get("engine", v["transient"].get("generator", "sncosmo"))
    if "placed" in v:
        return v["placed"]["engine"]
    return "PHOEBE" if v["kind"] == "EB" else "MESA RSP"


def load_scene():
    """Every star near the block: the catalogue, plus the placed stars (appended at their own offsets). Columns: Source,
    RA, Dec, Gmag, BP-RP, dx, dy (arcsec from the block centre), in_block, class (Gaia's variability class)."""
    stars = Table.read(star_catalogue())
    placed = [v for v in load_placed() if v["source"] < 0]  # real Gaia stars are in the catalogue already
    if placed:
        extra = Table({"Source": [v["source"] for v in placed], "RA_ICRS": [np.nan] * len(placed),
                       "DE_ICRS": [np.nan] * len(placed)})
        if any(v["timing"] == "periodic" for v in placed):  # periodic: the catalogue G and colour of its mean; transient: none
            extra["Gmag"] = [v.get("G", np.nan) for v in placed]
            extra["BP-RP"] = [v.get("bp_rp", np.nan) for v in placed]
        keep = [c for c in ("Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP", "distance_pc", "ebv_own", "ebv") if c in stars.colnames]
        stars = vstack([stars[keep], extra])
    dx, dy = block_offsets(column(stars, "RA_ICRS"), column(stars, "DE_ICRS"))
    if placed:
        dx[-len(placed):] = [v["dx"] for v in placed]
        dy[-len(placed):] = [v["dy"] for v in placed]
    near = (np.abs(dx) <= AREA_HALF + MARGIN) & (np.abs(dy) <= AREA_HALF + MARGIN)
    has_light = np.isfinite(column(stars, "Gmag")) | (np.asarray(stars["Source"]).astype(np.int64) < 0)
    stars = stars[near & has_light]
    stars["dx"], stars["dy"] = dx[near & has_light], dy[near & has_light]
    stars["in_block"] = (np.abs(stars["dx"]) <= HALF) & (np.abs(stars["dy"]) <= HALF)
    if FIELDS[FIELD].get("synthetic"):
        stars["class"] = [""] * len(stars)
        return stars
    classifier = Table.read(DATA / "gaia_classifier.ecsv")
    gaia_class = dict(zip(np.asarray(classifier["Source"]).astype(np.int64), np.asarray(classifier["Class"])))
    stars["class"] = [str(gaia_class.get(int(s), "")) for s in np.asarray(stars["Source"]).astype(np.int64)]
    return stars


def load_pulsators():
    """{source: dict(kind, period, epoch (BJD of maximum), amp, mean mags)}: Gaia's RR Lyrae and Cepheids, the
    ephemerides observe.py adds (HADS, binaries), and the placed stars. Used to phase the models, never as light curves."""
    out = {}
    for name, family in (("gaia_rrlyrae.ecsv", "RR"), ("gaia_cepheid.ecsv", "CEP")):
        if not (DATA / name).exists():
            continue
        t = Table.read(DATA / name)
        for r in t:
            pf = float(r["PF"]) if np.isfinite(float(r["PF"])) and float(r["PF"]) > 0 else np.nan
            p1 = float(r["P1O"]) if "P1O" in t.colnames and np.isfinite(float(r["P1O"])) and float(r["P1O"]) > 0 else np.nan
            period = pf if np.isfinite(pf) else p1
            if not np.isfinite(period) or not np.isfinite(float(r["EpochG"])):
                continue
            kind = "RRab" if family == "RR" and np.isfinite(pf) else ("RRc" if family == "RR" else "Type II Cepheid")
            if family == "CEP" and "Class" in t.colnames:  # Gaia's own classification
                klass, subclass, mode = str(r["Class"]), str(r["SubClass"]), str(r["ModeClass"])
                if klass == "DCEP" and mode in ("FUNDAMENTAL", "FIRST_OVERTONE"):
                    kind = "DCEP F" if mode == "FUNDAMENTAL" else "DCEP 1O"
                    period = pf if mode == "FUNDAMENTAL" else p1
                elif klass == "T2CEP":
                    kind = {"BL_HER": "BL Her", "W_VIR": "W Vir", "RV_TAU": "RV Tau"}.get(subclass, "Type II Cepheid")
                else:
                    kind = f"{klass} {mode}"
            if family == "RR" and np.isfinite(pf) and np.isfinite(p1):
                kind = "RRd"
            amp = {b: float(r[f"ptp{b}"]) for b in ("G", "BP", "RP")}
            if "BPmagavg" in t.colnames:
                mean = {b: float(r[f"{b}magavg"]) for b in ("G", "BP", "RP")}
            else:
                mean = {"G": float(r["Gmagavg"])}
            out[int(r["Source"])] = dict(period=period, epoch=float(r["EpochG"]) + 2455197.5, kind=kind, amp=amp, mean=mean)
    if (DATA / "extra_pulsators.json").exists():  # ephemerides from other catalogues (HADS from ASAS-SN; binaries)
        for source, v in json.loads((DATA / "extra_pulsators.json").read_text()).items():
            out[int(source)] = dict(period=v["period"], epoch=v["epoch_bjd"], kind=v["kind"], amp={"G": v["amp"]},
                                    mean={"G": v["G"]})
    for v in load_placed():  # placed stars: -1, -2, ...; a transient's epoch is its t0
        if v["source"] > 0:  # a real Gaia star: phased from the Gaia tables above
            continue
        if v["timing"] == "transient":
            out[v["source"]] = dict(period=np.nan, epoch=v["t0_bjd"], kind=v["kind"], amp={"G": np.nan}, mean={}, transient=v)
        else:
            out[v["source"]] = dict(period=v["period"], epoch=v["epoch_bjd"], kind=v["kind"], amp={"G": v["amp_G"]},
                                    mean={"G": v["G"]}, placed=v)
    return out


def pulsator_title(follow, pulsators, source):
    """What is followed, for the movie's title: e.g. '1 BL Her: MESA RSP · 1 eclipsing binary: PHOEBE'."""
    entries = [pulsators[int(source[k])] for k in follow]
    catalogued = [v["kind"] for v in entries if "transient" not in v and "placed" not in v]
    n_binaries = sum(kind == "EB" for kind in catalogued)
    pulsating = [kind for kind in catalogued if kind != "EB"]
    parts = []
    if pulsating:
        if all(kind.startswith("RR") for kind in pulsating):
            parts.append(f"{len(pulsating)} RR Lyrae: MESA RSP")
        else:
            parts.append(f"{len(pulsating)} pulsator{'s' if len(pulsating) != 1 else ''} "
                         f"({', '.join(dict.fromkeys(pulsating))}): MESA RSP")
    if n_binaries:
        parts.append(f"{n_binaries} eclipsing binar{'ies' if n_binaries != 1 else 'y'}: PHOEBE")
    placed = [v for v in entries if "placed" in v]
    if placed:
        groups = {}
        for v in placed:
            groups[(v["kind"], engine_of(v))] = groups.get((v["kind"], engine_of(v)), 0) + 1
        parts.append(", ".join(f"{n} {kind}: {engine}" for (kind, engine), n in groups.items()))
    transients = [v for v in entries if "transient" in v]
    if transients:
        parts.append(f"{len(transients)} transient{'s' if len(transients) != 1 else ''}: "
                     + ", ".join(dict.fromkeys(f"{v['kind']} ({engine_of(v)})" for v in transients)))
    return " · ".join(parts) or "no modelled variables"


def choose_followed(stars, pulsators, n):
    """Row indices of the followed stars. A scene: exactly the stars placed (real ones by amplitude, then the placed
    ones in order). A hand-made field: two RRc, then the largest-amplitude RRab, up to n."""
    source = np.asarray(stars["Source"]).astype(np.int64)
    in_block = np.asarray(stars["in_block"])
    if FIELDS[FIELD].get("followed_only"):
        chosen = {v["source"] for v in load_followed().values()} | {v["source"] for v in load_placed() if v["source"] > 0}
        real = [k for k in range(len(stars)) if int(source[k]) in chosen]
        placed = sorted((k for k in range(len(stars)) if int(source[k]) < 0), key=lambda k: -int(source[k]))
        return sorted(real, key=lambda k: -pulsators[int(source[k])]["amp"]["G"]) + placed
    exclude = FIELDS[FIELD].get("exclude", set())
    candidates = [k for k in range(len(stars)) if in_block[k] and int(source[k]) in pulsators and int(source[k]) not in exclude]
    by_amplitude = sorted(candidates, key=lambda k: -pulsators[int(source[k])]["amp"]["G"])
    kind = {k: pulsators[int(source[k])]["kind"] for k in candidates}
    follow = [k for k in by_amplitude if kind[k] == "RRc"][:2]
    follow += [k for k in by_amplitude if kind[k] == "RRab"][: max(0, n - len(follow))]
    return sorted(follow, key=lambda k: -pulsators[int(source[k])]["amp"]["G"])


class RSPStar:
    """One cycle of a periodic model (MESA RSP limit cycle, or a PHOEBE orbit), phased so phase 0 is maximum light in
    rho (binaries: the primary eclipse). The file's SEDs are at 10 pc; they are seen at the file's distance modulus,
    through E(B-V) `ebv` (the observation's; the file's own if not given)."""

    def __init__(self, path, ab, ebv=None):
        with h5py.File(path) as f:
            self.attrs = dict(f.attrs)
            wave, flux_10pc, t = f["wavelength"][:], f["flux"][:], f["time_days"][:]
        self.wave = wave * u.AA
        dm = float(self.attrs["distance_modulus"])
        ebv = float(self.attrs["ebv"]) if ebv is None else ebv
        seen = dust.redden(wave, flux_10pc * (10.0 / 10 ** (dm / 5 + 1)) ** 2, ebv)
        rho = np.array([float(ab.mag_from_sed("rho", self.wave, f * FLAM)) for f in seen])
        # one full cycle ending at the last maximum, with the model's own period (RRab have a bump, so local maxima
        # alone would mislead)
        self.period = float(self.attrs["nonlinear_period"])
        if t[-1] - t[0] < 2 * self.period - 1e-9:  # stored as exactly one cycle (a PHOEBE orbit): unroll it to two
            one = t < t[0] + self.period - 1e-9
            t = np.r_[t[one], t[one] + self.period]
            seen, rho = np.r_[seen[one], seen[one]], np.r_[rho[one], rho[one]]
        last = t >= t[-1] - self.period
        if self.attrs.get("kind") == "EB":  # phase 0 = the primary eclipse = the file's start (two equal maxima)
            i_zero = int(np.nonzero(last)[0][0])
            cycle = last
        else:
            i_zero = int(np.nonzero(last)[0][np.argmin(rho[last])])
            cycle = (t > t[i_zero] - self.period) & (t <= t[i_zero])
        phase = ((t[cycle] - t[i_zero]) / self.period) % 1.0
        order = np.argsort(phase)
        self.phase = phase[order]
        self.flux = seen[cycle][order]
        self.rho_cycle = rho[cycle][order]

    def sed(self, phase):
        """f_lambda at Argus at this phase: linear between the two stored steps around it (wrapping the cycle)."""
        phase = phase % 1.0
        n = len(self.phase)
        i = int(np.searchsorted(self.phase, phase))
        if i == 0:
            a, b, pa, pb = n - 1, 0, self.phase[-1] - 1.0, self.phase[0]
        elif i == n:
            a, b, pa, pb = n - 1, 0, self.phase[-1], self.phase[0] + 1.0
        else:
            a, b, pa, pb = i - 1, i, self.phase[i - 1], self.phase[i]
        w = 0.0 if pb == pa else (phase - pa) / (pb - pa)
        return (1 - w) * self.flux[a] + w * self.flux[b]


class TransientSource:
    """A transient: observer-frame SEDs against days since t0, through E(B-V) `ebv` (the observation's t0 and dust;
    the file's own if not given). No light before the model's first step; after its last step is an error."""

    def __init__(self, path, t0=None, ebv=None):
        with h5py.File(path) as f:
            self.attrs = dict(f.attrs)
            wave, flux, self.t = f["wavelength"][:], f["flux"][:], f["time_days"][:]
        self.wave = wave * u.AA
        self.flux = dust.redden(wave, flux, float(self.attrs["ebv"] if ebv is None else ebv))
        self.t0 = float(self.attrs["t0_bjd"] if t0 is None else t0)

    def sed_at(self, bjd):
        """f_lambda at Argus at BJD bjd: linear between the two stored steps around it."""
        d = bjd - self.t0
        if d > self.t[-1]:
            raise ValueError(f"{self.attrs['model']}: {d:.2f} d after t0 is past the model's last step ({self.t[-1]:.2f} d)")
        if d < self.t[0]:
            return np.zeros(self.flux.shape[1])
        i = int(np.clip(np.searchsorted(self.t, d), 1, len(self.t) - 1))
        w = (d - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        return (1 - w) * self.flux[i - 1] + w * self.flux[i]

    def sed_exposure(self, bjd, exptime, n=61):
        """Mean f_lambda over an exposure centred on bjd (a fast transient changes within one exposure)."""
        return np.mean([self.sed_at(b) for b in bjd + np.linspace(-0.5, 0.5, n) * exptime / 86400], axis=0)


# ---------------------------------------------------------------- the night
def night_schedule(date, max_airmass, cfg, start=None, end=None):
    """Every exposure of the night at the field, as ArgusSim schedules it (mesa_argus.night.exposures), in the UT
    window: mid-exposure UTC times, BJD_TDB, airmass, altitude, azimuth."""
    from mesa_argus import night

    times, airmass, alt, az = night.exposures(CENTRE.ra.deg, CENTRE.dec.deg, date, start=start, end=end,
                                              max_airmass=max_airmass, cfg=cfg)
    return times, night.bjd_tdb(times, CENTRE.ra.deg, CENTRE.dec.deg, cfg=cfg), airmass, alt, az


def survey_schedule(band, cfg):
    """The survey's exposures of the field in one band (scene/exposures.ecsv, written by observe.py from
    mesa_argus.survey): mid-exposure UTC times, BJD_TDB, airmass, altitude, azimuth, and the table's rows (telescope,
    field angle, the ratchet's zenith seeing)."""
    from astropy.time import Time

    from mesa_argus import night

    rows = Table.read(DATA / "exposures.ecsv")
    rows = rows[rows["band"] == band]
    if len(rows) == 0:
        raise SystemExit(f"the survey takes no {band} exposures of this field in the window")
    times = Time(list(rows["time"]), scale="utc")
    airmass, alt, az = night.sky_position(times, CENTRE.ra.deg, CENTRE.dec.deg, night.location(cfg))
    return times, night.bjd_tdb(times, CENTRE.ra.deg, CENTRE.dec.deg, cfg=cfg), airmass, alt, az, rows


def phases_at(var, bjd):
    return ((bjd - var["epoch"]) / var["period"]) % 1.0


# ---------------------------------------------------------------- the detector
class Scene:
    """Every star's ArgusSim PSF on one pixel grid, for each PSF set (a seeing, at a field angle: --field-angle unless
    given per set); sky and dark per exposure; observe() makes an exposure."""

    def __init__(self, stars, args, cfg, ab, seeings, field_angles=None):
        self.plate = cfg.telescope.plate_scale
        self.n = int(np.ceil(2 * AREA_HALF / self.plate)) + 30
        self.n += self.n % 2
        self.centre = self.n / 2.0
        psf_model = ZemaxPSF.from_config(cfg)
        self.seeings = list(seeings)
        self.field_angles = list(field_angles) if field_angles is not None else [args.field_angle] * len(self.seeings)
        self.lookup = {}  # (band, PSF set index, y phase, x phase) -> unit-flux PSF stamp
        for si, (seeing, angle) in enumerate(zip(self.seeings, self.field_angles)):
            for name, band in BANDS.items():
                for iy in range(PSF_PHASES):
                    for ix in range(PSF_PHASES):
                        p, _ = psf_model.with_seeing(seeing=seeing, band=band, field_angle_deg=angle,
                                                     pixel_phase=(iy / PSF_PHASES, ix / PSF_PHASES))
                        p = np.maximum(p, 0.0)
                        self.lookup[name, si, iy, ix] = p / p.sum()
        p0 = self.lookup["rho", 0, 0, 0]
        yy, xx = np.indices(p0.shape)
        self.c0 = ((p0 * yy).sum(), (p0 * xx).sum())  # the stamp's centroid
        self.x = self.centre + np.asarray(stars["dx"]) / self.plate
        self.y = self.centre + np.asarray(stars["dy"]) / self.plate
        # each star's stamp origin and sub-pixel phase (the stamp centroid moves by -phase)
        self.oy = np.ceil(self.y - self.c0[0]).astype(int)
        self.ox = np.ceil(self.x - self.c0[1]).astype(int)
        self.py = np.clip(np.round((self.oy + self.c0[0] - self.y) * PSF_PHASES).astype(int), 0, PSF_PHASES - 1)
        self.px = np.clip(np.round((self.ox + self.c0[1] - self.x) * PSF_PHASES).astype(int), 0, PSF_PHASES - 1)
        self._designs = {}  # (band, seeing index) -> PSF matrix, built on demand, the last few kept
        self.seeing_index = None  # per exposure (set_seeing)
        self.sky = None  # electrons/s/pixel per band per exposure (set_sky)
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
        on = (y >= 0) & (y < self.n) & (x >= 0) & (x < self.n) & (p > 0)
        cols = np.broadcast_to(np.arange(len(self.x))[:, None, None], p.shape)
        return sparse.csc_matrix((p[on].astype(np.float32), ((y * self.n + x)[on], cols[on])),
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
        return dict(image=image.reshape(self.n, self.n), full=(charge >= self.fullwell).reshape(self.n, self.n),
                    mean=mean, background=bg)


def difference_image(image, image_sat, reference, reference_sat, readnoise, half=3, clip=3.0, fit_pixels=None):
    """PSF-matched difference image (display only): image - (reference (x) K + b), Bramich (2008) delta-function
    kernel K of (2 half + 1)^2 free pixels plus a constant b, fitted by weighted least squares on pixels unsaturated
    in both images (and in fit_pixels: stamps around bright stars), rejecting pixels beyond clip sigma and refitting
    (Bramich 2008 sec. 2.3) so variables do not leak into the kernel. Returns the difference (nan at the border) and
    the kernel's sum."""
    n = image.shape[0]
    core = (slice(half, n - half), slice(half, n - half))
    shifted = [reference[half + dy:n - half + dy, half + dx:n - half + dx].ravel()
               for dy in range(-half, half + 1) for dx in range(-half, half + 1)]
    A = np.column_stack(shifted + [np.ones(shifted[0].size)])
    target = image[core].ravel()
    excluded = (image_sat | reference_sat)[core].ravel()  # left out of the fit only; the difference is shown everywhere
    if fit_pixels is not None:
        excluded = excluded | ~fit_pixels[core].ravel()
    use = ~excluded
    sigma = np.sqrt(np.maximum(target, 1.0) + readnoise**2)
    for _ in range(10):
        coef, *_ = np.linalg.lstsq(A[use] / sigma[use, None], target[use] / sigma[use], rcond=None)
        model = A @ coef
        sigma = np.sqrt(np.maximum(model, 1.0) + readnoise**2)
        if clip <= 0:
            break
        new = ~excluded & (np.abs(target - model) <= clip * sigma)
        if np.array_equal(new, use):
            break
        use = new
    diff = np.full_like(image, np.nan, dtype=float)
    diff[core] = (target - A @ coef).reshape(n - 2 * half, n - 2 * half)
    return diff, float(np.sum(coef[:-1]))


# ---------------------------------------------------------------- the steps of main
def constant_star_rates(stars, pulsators, ab):
    """Every star as a constant. Its Gaia G and BP-RP were measured through the dust in front of it (column ebv_own):
    that dust is taken out (mesa_argus.dust.deredden: the MESA template whose reddened colour is the measured one, and
    the dust-free G), and the dust where it is observed (column ebv) put back into the template's SED (CCM89). At its
    real position the two are the same, and the star is as Gaia measured it; scattered (a synthetic field), its G and
    BP-RP become those seen through the new sightline. Returns electrons/s on AIRMASS_GRID and above-atmosphere
    magnitudes, per band; the templates; each star's colour as measured (filled where Gaia has none)."""
    templates = Templates()
    G = column(stars, "Gmag")
    colour = column(stars, "BP-RP")
    # no (or a blended) BP-RP, common in crowded cores: the median colour, at the same G, of stars > 2.5' from the centre
    everyone = Table.read(star_catalogue())
    r_arcmin = 60 * np.hypot((column(everyone, "RA_ICRS") - CENTRE.ra.deg) * np.cos(CENTRE.dec.rad),
                             column(everyone, "DE_ICRS") - CENTRE.dec.deg)
    g_all, c_all = column(everyone, "Gmag"), column(everyone, "BP-RP")
    good = (r_arcmin > 2.5) & np.isfinite(g_all) & np.isfinite(c_all)
    bins = np.arange(11.5, 20.51, 0.5)
    centres = 0.5 * (bins[1:] + bins[:-1])
    in_bin = [good & (g_all >= lo) & (g_all < hi) for lo, hi in zip(bins[:-1], bins[1:])]
    median_colour = np.array([np.median(c_all[b]) if np.sum(b) >= 5 else np.nan for b in in_bin])
    known = np.isfinite(median_colour)
    missing = ~(np.isfinite(colour) & (colour > -0.5) & (colour < 3.0))
    colour = np.where(missing, np.interp(G, centres[known], median_colour[known]), colour)
    source = np.asarray(stars["Source"]).astype(np.int64)
    for k in np.nonzero([int(s) in pulsators for s in source])[0]:
        # a Gaia-classified variable: its intensity-mean colour; a blended one a typical colour for its kind
        v = pulsators[int(source[k])]
        mean_colour = v["mean"].get("BP", np.nan) - v["mean"].get("RP", np.nan)
        if v["kind"].startswith("RR"):
            low, high, typical = 0.2, 0.9, (0.42 if v["kind"] == "RRc" else 0.55)
        else:
            low, high, typical = 0.3, 1.3, 0.7
        colour[k] = mean_colour if np.isfinite(mean_colour) and low <= mean_colour <= high else typical
    print(f"{int(missing.sum())} sources without usable BP-RP take the median colour, at their G, of stars > 2.5' from the centre")

    # the dust: in front of the star where Gaia measured it, and where it is observed (none in hand-made fields)
    ebv_own = np.nan_to_num(column(stars, "ebv_own")) if "ebv_own" in stars.colnames else np.zeros(len(stars))
    ebv = np.nan_to_num(column(stars, "ebv")) if "ebv" in stars.colnames else np.zeros(len(stars))
    intrinsic, G0 = colour.copy(), G.copy()
    dusty = ebv_own > 0
    intrinsic[dusty], G0[dusty] = dust.deredden(G[dusty], colour[dusty], ebv_own[dusty])
    moved = ebv != ebv_own  # a synthetic field: seen through other dust than Gaia saw it through
    if moved.any():
        seen_G, seen_colour = dust.redden_gaia(G0[moved], intrinsic[moved], ebv[moved])
        stars["Gmag"][moved], stars["BP-RP"][moved] = seen_G, seen_colour
    print(f"dust: E(B-V) {np.min(ebv):.3f}-{np.max(ebv):.3f} where observed (median {np.median(ebv):.3f}); "
          f"{int(moved.sum())} stars seen through other dust than Gaia saw them through", flush=True)

    print("ArgusSim photometry of every star …", flush=True)
    rate = {b: np.zeros((len(stars), len(AIRMASS_GRID))) for b in BANDS}
    mag = {b: np.zeros(len(stars)) for b in BANDS}
    template = np.array([templates.index(c) for c in intrinsic])
    scale = 10 ** (-0.4 * (G0 - templates.g[template]))
    step = 0.05  # ArgusSim once per template at E(B-V) on this grid, linear in between (in magnitudes)
    for t in np.unique(template):
        these = template == t
        e = ebv[these]
        nodes = np.arange(np.floor(e.min() / step), np.ceil(e.max() / step) + 0.5) * step  # covers every star's E(B-V)
        for b, band in BANDS.items():
            node_mag = np.array([float(ab.mag_from_sed(band, templates.wave, dust.redden(templates.wave.value, templates.flux[t], x) * FLAM))
                                 for x in nodes])
            node_rate = np.array([[ab.electrons_from_sed(band, templates.wave, dust.redden(templates.wave.value, templates.flux[t], x) * FLAM,
                                                         airmass=a).value for a in AIRMASS_GRID] for x in nodes])
            if len(nodes) == 1:
                rate[b][these] = np.outer(scale[these], node_rate[0])
                mag[b][these] = node_mag[0] - 2.5 * np.log10(scale[these])
                continue
            log_rate = np.array([np.interp(e, nodes, np.log(node_rate[:, i])) for i in range(len(AIRMASS_GRID))]).T
            rate[b][these] = scale[these, None] * np.exp(log_rate)
            mag[b][these] = np.interp(e, nodes, node_mag) - 2.5 * np.log10(scale[these])
    return rate, mag, templates, colour


def followed_star_light(k, v, args, ab, bjd, airmass, model_file, model_ebv):
    """A followed star's model and its SED, electrons/s and magnitude at every exposure, and its cycle (or, for a
    transient, its track over the night) for the CMD."""
    if "transient" in v:
        tr = v["transient"]
        model = TransientSource(_REPO / tr["file"], tr.get("t0_bjd"), tr.get("ebv"))
        seds = np.array([model.sed_exposure(b, args.exptime) for b in bjd])
        mag = {b: np.array([float(ab.mag_from_sed(band, model.wave, f * FLAM)) if f.any() else np.inf for f in seds])
               for b, band in BANDS.items()}
        cycle = {b: np.where(np.isfinite(mag[b]), mag[b], np.nan) for b in BANDS}
    else:
        model = RSPStar(model_file, ab, model_ebv)
        seds = np.array([model.sed(ph) for ph in phases_at(v, bjd)])
        mag = {b: np.array([float(ab.mag_from_sed(band, model.wave, f * FLAM)) for f in seds]) for b, band in BANDS.items()}
        grid = np.linspace(0, 1, 120, endpoint=False)
        cycle = {b: np.array([float(ab.mag_from_sed(band, model.wave, model.sed(ph) * FLAM)) for ph in grid])
                 for b, band in BANDS.items()}
    rate = {b: np.array([ab.electrons_from_sed(band, model.wave, f * FLAM, airmass=x).value for f, x in zip(seds, airmass)])
            for b, band in BANDS.items()}
    return model, seds, rate, mag, cycle


def conditions(times, airmass, alt, az, args, stars, cfg, ab, survey=None):
    """Seeing per exposure, the detector scene (PSFs), and the sky per exposure, all from ArgusSim. survey: the survey's
    exposure rows (survey_schedule), each with its ratchet's zenith seeing and its field angle."""
    if survey is None:
        # seeing: the season's median zenith seeing at the site (DIMMs, Barker et al. 2003) x airmass^0.6; PSFs on a 0.05" grid
        zenith_seeing = args.seeing if args.seeing else seasonal_seeing_params(times[0])[0]
        seeing = seeing_at_airmass(zenith_seeing, airmass)
        seeing_grid = np.round(np.arange(np.floor(seeing.min() * 20) / 20, seeing.max() + 0.05, 0.05), 2)
        print(f"seeing: {zenith_seeing:.2f}\" at zenith -> {seeing.min():.2f}-{seeing.max():.2f}\" (ArgusSim site model); "
              f"{len(seeing_grid)} PSF sets", flush=True)
        print("building the detector scene …", flush=True)
        scene = Scene(stars, args, cfg, ab, seeing_grid)
        scene.set_seeing(np.abs(seeing[:, None] - seeing_grid[None, :]).argmin(axis=1))
    else:
        # seeing: each ratchet's zenith draw x airmass^0.6; PSFs at 0.05" in seeing and 0.01 deg in field angle
        seeing = seeing_at_airmass(np.asarray(survey["seeing_zenith"], float), airmass)
        sets = np.c_[np.round(seeing * 20) / 20, np.round(np.asarray(survey["field_angle"], float), 2)]
        unique, index = np.unique(sets, axis=0, return_inverse=True)
        print(f"seeing (survey: per ratchet): {seeing.min():.2f}-{seeing.max():.2f}\"; field angles "
              f"{np.min(survey['field_angle']):.2f}-{np.max(survey['field_angle']):.2f} deg; {len(unique)} PSF sets", flush=True)
        print("building the detector scene …", flush=True)
        scene = Scene(stars, args, cfg, ab, unique[:, 0], unique[:, 1])
        scene.set_seeing(np.ravel(index))
    # sky: airglow (solar activity, airmass), zodiacal light, integrated starlight, Moon (Krisciunas & Schaefer 1991),
    # site light
    observatory = Observatory(cfg)
    per_exposure = [observatory.sky_components_at(float(a), float(z), t) for a, z, t in zip(alt, az, times)]
    component = {key: np.array([float(np.ravel(c[key])[0]) for c in per_exposure]) for key in per_exposure[0]}
    sky_rate = {}
    for name, band in BANDS.items():
        sky = SpectralSky(ab.tp, ab, band)
        sky_rate[name] = np.asarray(sky.sky_electrons(component["airglow_s10"], component["zodiacal_s10"],
                                                      component["starlight_s10"], component["moon_s10"],
                                                      component["artif_s10"], transparency=1.0)) * scene.plate**2
    scene.set_sky(sky_rate)
    print(f"sky (ArgusSim): V {np.min(component['v_mag']):.2f}-{np.max(component['v_mag']):.2f} mag/arcsec2; Moon "
          f"{np.mean(component['illum_frac']) * 100:.0f}% illuminated, altitude {np.min(component['moon_alt']):.0f} to "
          f"{np.max(component['moon_alt']):.0f} deg; rho sky {sky_rate['rho'].min():.1f}-{sky_rate['rho'].max():.1f} e/s/pixel")
    return seeing, scene


def difference_reference(scene, seeing, rates_at, seeds, band, args, bands):
    """The difference image's reference: the median of the best-seeing exposures (each a full noisy exposure, the
    transients left out), and the stamps around bright stars the kernel is fitted on."""
    ref_frames = np.argsort(seeing, kind="stable")[: max(20, len(seeing) // 5)]
    stack, stack_full = [], []
    for j in ref_frames:
        rng = np.random.default_rng(seeds[j])  # the bands in the movie's order: the identical exposure
        exposure = {b: scene.observe(b, rates_at(b, j, reference=True), j, rng) for b in bands}[band]
        stack.append(exposure["image"])
        stack_full.append(exposure["full"])
    ref_image = np.median(stack, axis=0)
    ref_full = np.mean(stack_full, axis=0) >= 0.5  # saturated in the median itself
    stamps = None
    if args.kernel_stamps > 0:
        sky = float(np.median(ref_image))
        noise = 1.4826 * float(np.median(np.abs(ref_image - sky)))
        bright = (ref_image - sky) > args.kernel_stamps * noise
        stamps = ndimage.binary_dilation(bright, structure=np.ones((2 * args.kernel_half + 9,) * 2, bool))
        print(f"kernel stamps: {int(ndimage.label(bright)[1])} sources above {args.kernel_stamps:g} sigma, "
              f"{100 * stamps.mean():.0f}% of pixels", flush=True)
    print(f"difference reference: median of {len(ref_frames)} best-seeing exposures "
          f"(seeing {seeing[ref_frames].min():.2f}-{seeing[ref_frames].max():.2f}\")", flush=True)
    return ref_frames, ref_image, ref_full, stamps


def write_fits(folder, j, exposures, s, args):
    """Exposure j in each band it is taken in as FITS, with what made it in the header."""
    from astropy.io import fits

    folder.mkdir(parents=True, exist_ok=True)
    for band, exposure in exposures.items():
        header = fits.Header()
        header.update(dict(FIELD=FIELD, BAND=band, EXPOSURE=int(j), EXPTIME=args.exptime, DATE_OBS=s.times[j].isot,
                           BJD_TDB=float(s.bjd[j]), AIRMASS=float(s.airmass[j]), SEEING=float(s.seeing[j]),
                           SKY_E=float(s.scene.background(band, j)), FULLWELL=float(s.scene.fullwell),
                           RDNOISE=float(s.scene.readnoise), PLATESCL=float(s.scene.plate), RA_CEN=CENTRE.ra.deg,
                           DEC_CEN=CENTRE.dec.deg, BUNIT="electron", ORIGIN="MESA_ARGUS night_movie.py"))
        if s.survey is not None:  # which telescope took it, and where in its field the block fell
            header.update(dict(TEL_ID=int(s.survey["tel_id"][j]), RATCHET=int(s.survey["ratchet"][j]),
                               FIELDANG=float(s.survey["field_angle"][j])))
        fits.writeto(folder / f"{FIELD}_{band}_{j:04d}.fits", exposure["image"].astype(np.float32), header, overwrite=True)


# ---------------------------------------------------------------- main
# the options an observation file's [output] may set (observe.py passes them on)
MOVIE_OPTIONS = ("max_airmass", "exptime", "seeing", "field_angle", "stride", "fps", "dpi", "seed", "kernel_half",
                 "kernel_clip", "kernel_stamps", "area")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, help="a scene folder written by observe.py")
    p.add_argument("--field", help="a hand-made field (m3, m15, rr308108, rr416445); with --scene, its name")
    p.add_argument("--measure", default="{}", help="photometry settings as JSON (mesa_argus.photometry.SETTINGS)")
    p.add_argument("--fits", type=Path, help="also write every measured exposure (both bands) as FITS here")
    p.add_argument("--no-render", action="store_true", help="measure (and write FITS) but draw no movie")
    p.add_argument("-o", "--output", type=Path, help="default outputs/<field>/<field>_night_<band>.mp4")
    p.add_argument("--date", help="local date the night starts (default: the field's own)")
    p.add_argument("--max-airmass", type=float, default=None, help="a further airmass limit (the array's minimum altitude always applies)")
    p.add_argument("--start", help="UT HH:MM to start (default: when the field is first observable)")
    p.add_argument("--end", help="UT HH:MM to stop (default: when the field is last observable)")
    p.add_argument("--exptime", type=float, default=60.0)
    p.add_argument("--seeing", type=float, default=None, help="zenith seeing FWHM (arcsec); default: ArgusSim's seasonal median")
    p.add_argument("--field-angle", type=float, default=1.44, help="deg from the optical axis; 1.44 = area-weighted median")
    p.add_argument("--stride", type=int, default=1, help="render and measure every Nth exposure")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--highlight", type=int, default=6, help="hand-made fields: pulsators to follow")
    p.add_argument("--seed", type=int, default=3, help="noise")
    p.add_argument("--kernel-half", type=int, default=3, help="difference-imaging kernel half-width (pixels)")
    p.add_argument("--kernel-clip", type=float, default=3.0, help="kernel fit rejection (sigma); 0 = none")
    p.add_argument("--kernel-stamps", type=float, default=20.0, help="fit the kernel on stamps around sources this many sigma above sky; 0 = all pixels")
    p.add_argument("--area", type=float, default=8.0, help="side (arcmin) of the simulated area (kernel and zero point)")
    p.add_argument("--block", type=int, default=None, help="minipixes across the block shown (default: the field's)")
    p.add_argument("--band", choices=list(BANDS), default="rho", help="band shown in the images, light curves and CMD")
    return p


def main(argv=None):
    import os

    args = parser().parse_args(argv)
    if args.scene is not None:  # relative paths are relative to where the command was run
        args.scene = (Path.cwd() / args.scene) if not args.scene.is_absolute() else args.scene
        args.field = args.field or args.scene.resolve().parent.name
    if args.fits is not None and not args.fits.is_absolute():
        args.fits = Path.cwd() / args.fits
    if args.output is None:
        args.output = _REPO / "outputs" / args.field / f"{args.field}_night_{args.band}.mp4"
    if not args.output.is_absolute():
        args.output = Path.cwd() / args.output
    os.chdir(_REPO)
    use_field(args.field, args.scene)
    set_block(args.block or FIELDS[args.field].get("block_n", 3))
    global AREA_HALF
    AREA_HALF = max(30.0 * args.area, HALF + MARGIN)
    args.date = args.date or FIELDS[args.field]["date"]
    B = args.band  # the band shown in the images, light curves and CMD
    cfg = copy.deepcopy(ARGUS_CONFIG)
    for key, value in (FIELDS[args.field]["site"] or {}).items():  # the instrument stays Argus; only the site moves
        setattr(cfg.observatory, key, value)
    ab = ABPhot(throughput=SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=cfg.telescope.throughput_loss),
                collecting_area=cfg.telescope.collecting_area)

    # 1. the stars
    stars = load_scene()
    pulsators = load_pulsators()
    source = np.asarray(stars["Source"]).astype(np.int64)  # Gaia id, or -1, -2, ... for placed stars
    in_block = np.asarray(stars["in_block"])
    G = column(stars, "Gmag")
    gaia_var = np.array([int(s) in pulsators for s in source])  # Gaia-classified: constant unless followed
    follow = choose_followed(stars, pulsators, args.highlight)  # row indices; k below is always a row index
    placed_label = {v["source"]: v["label"] for v in load_placed()}
    labels = {k: placed_label[int(source[k])] if source[k] < 0 else f"V{i + 1}" for i, k in enumerate(follow)}
    n_placed = int(np.sum(source < 0))
    print(f"{len(stars) - n_placed} {'synthetic stars' if FIELDS[args.field].get('synthetic') else 'Gaia sources'} "
          f"({int(in_block.sum()) - n_placed} in the block)" + (f"; {n_placed} transients" if n_placed else "")
          + f"; following {len(follow) - n_placed} pulsators with MESA RSP models; the other "
          f"{int((gaia_var & in_block).sum()) - len(follow)} Gaia-classified variables in the block are held constant")

    # 2. the night
    if args.exptime > cfg.survey.base_cadence_s:
        raise SystemExit(f"--exptime {args.exptime} s is longer than ArgusSim's cadence ({cfg.survey.base_cadence_s} s)")
    survey = None  # observing "survey": the exposures ArgusSim's array takes of the field, each in one band
    if (DATA / "exposures.ecsv").exists():
        times, bjd, airmass, alt, az, survey = survey_schedule(B, cfg)
        print(f"survey ({B}): {len(times)} exposures through {len(np.unique(survey['tel_id']))} telescopes in "
              f"{len(np.unique(survey['ratchet']))} ratchets, UT {times[0].iso[11:16]}-{times[-1].iso[11:16]}, "
              f"airmass {airmass.min():.2f}-{airmass.max():.2f}")
    else:
        times, bjd, airmass, alt, az = night_schedule(args.date, args.max_airmass, cfg, args.start, args.end)
        print(f"night of {args.date}: {len(times)} exposures, UT {times[0].iso[11:16]}-{times[-1].iso[11:16]}, "
              f"airmass {airmass.min():.2f}-{airmass.max():.2f}")
    bands = [B] if survey is not None else list(BANDS)  # the bands each exposure is taken in
    ut = (times.jd - np.floor(times[0].jd - 0.5) - 0.5) * 24  # UT hours since 0h on the first date

    # 3. constant stars
    static_rate, static_mag, templates, _ = constant_star_rates(stars, pulsators, ab)
    G = column(stars, "Gmag")  # as seen where observed (a synthetic field: through its own sightline)

    # 4. followed stars: real variables and placed stars from their model files, with the observation's dust
    from mesa_argus import rsp

    model_file = {v["source"]: _REPO / "runs" / f"rsp_{rsp.PREFIX}{name}" / "pulsation.h5"
                  for name, v in load_followed().items()}
    model_file.update({v["source"]: _REPO / v["file"] for v in load_placed() if v["timing"] == "periodic"})
    model_ebv = {v["source"]: v["ebv"] for v in load_placed() if v["timing"] == "periodic"}
    models, follow_sed, follow_rate, follow_mag, follow_cycle = {}, {}, {}, {}, {}
    transient = {k for k in follow if "transient" in pulsators[int(source[k])]}
    for k in follow:
        v = pulsators[int(source[k])]
        models[k], follow_sed[k], follow_rate[k], follow_mag[k], follow_cycle[k] = followed_star_light(
            k, v, args, ab, bjd, airmass, model_file.get(int(source[k])), model_ebv.get(int(source[k])))
        cyc = follow_cycle[k]["rho"]
        if k in transient:
            tr = v["transient"]
            print(f"  {labels[k]}: {tr['kind']} {tr.get('engine', tr.get('generator', 'sncosmo'))} {tr['model']}, "
                  f"z {tr['z']:g}, t0 BJD {tr['t0_bjd']:.4f} (night starts {bjd[0] - tr['t0_bjd']:+.3f} d after t0); "
                  + (f"rho {np.nanmin(cyc):.2f}-{np.nanmax(cyc):.2f} over the night" if np.isfinite(cyc).any()
                     else "no light during the night"))
        else:
            print(f"  {labels[k]}: {v['kind']} {engine_of(v)}, period {models[k].period:.5f} d "
                  f"({'given' if 'placed' in v else 'Gaia'} {v['period']:.5f}); rho {cyc.min():.2f}-{cyc.max():.2f} "
                  f"(amplitude {np.ptp(cyc):.2f}); Gaia mean G {G[k]:.2f}")
    # the catalogue / reference brightness of a periodic star: its cycle-mean SED; a transient is not in the reference
    ref_rate = {}
    for k in follow:
        if k in transient:
            ref_rate[k] = {b: np.zeros(len(airmass)) for b in BANDS}
            continue
        mean_sed = np.mean([models[k].sed(ph) for ph in np.linspace(0, 1, 200, endpoint=False)], axis=0) * FLAM
        ref_rate[k] = {b: np.array([ab.electrons_from_sed(band, models[k].wave, mean_sed, airmass=x).value for x in airmass])
                       for b, band in BANDS.items()}

    def rates_at(b, j, reference=False):
        """Electrons/s of every star in exposure j (reference: as the difference reference sees it, no transients)."""
        x = float(np.clip(airmass[j], AIRMASS_GRID[0], AIRMASS_GRID[-1]))
        i1 = int(np.clip(np.searchsorted(AIRMASS_GRID, x), 1, len(AIRMASS_GRID) - 1))
        f = (x - AIRMASS_GRID[i1 - 1]) / (AIRMASS_GRID[i1] - AIRMASS_GRID[i1 - 1])
        out = static_rate[b][:, i1 - 1] * (1 - f) + static_rate[b][:, i1] * f  # linear in airmass
        for k in follow:
            out[k] = 0.0 if (reference and k in transient) else follow_rate[k][b][j]
        return out

    def true_mag(b, k, j):
        return float(follow_mag[k][b][j]) if k in follow_mag else static_mag[b][k]

    # 5. conditions per exposure
    seeing, scene = conditions(times, airmass, alt, az, args, stars, cfg, ab, survey)

    # 6. the difference reference
    seeds = np.random.SeedSequence(args.seed).spawn(len(times))  # one per exposure: the same exposure in every pass
    ref_frames, ref_image, ref_full, stamps = difference_reference(scene, seeing, rates_at, seeds, B, args, bands)

    def exposure(j):  # every band it is taken in, from the exposure's own seed
        rng = np.random.default_rng(seeds[j])
        return {b: scene.observe(b, rates_at(b, j), j, rng) for b in bands}

    # 7. every exposure, measured. The catalogue a pipeline calibrates against: every star's magnitude (periodic
    # stars: their intensity mean; transients: not in it); the expected fluxes pick the calibration stars.
    from mesa_argus.photometry import SExtractorPhotometry

    cat_mag = static_mag[B].copy()
    cat_flux = static_rate[B][:, 0] * args.exptime
    for k in follow:
        if k in transient:
            cat_mag[k], cat_flux[k] = np.nan, 0.0
        else:
            cat_mag[k] = -2.5 * np.log10(np.mean(10 ** (-0.4 * follow_cycle[k][B])))
            cat_flux[k] = np.mean(ref_rate[k][B]) * args.exptime
    psf = scene.lookup[B, int(scene.seeing_index[ref_frames[0]]), 0, 0]
    yy, xx = np.indices(psf.shape)
    cy, cx = (psf * yy).sum(), (psf * xx).sum()
    ref_fwhm = 2.3548 * np.sqrt(0.5 * (psf * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum())
    phot = SExtractorPhotometry(scene.x, scene.y, cat_mag, cat_flux, scene.n, scene.fullwell, follow,
                                colour=column(stars, "BP-RP"), fwhm=ref_fwhm, cfg=json.loads(args.measure))
    frames = range(0, len(times), args.stride)
    s = SimpleNamespace(times=times, bjd=bjd, airmass=airmass, seeing=seeing, scene=scene, survey=survey)  # for write_fits
    print(f"observing and rendering {len(frames)} frames ({len(times)} exposures) -> {args.output}", flush=True)
    rows, measured = [], []
    for n, j in enumerate(frames):
        exposures = exposure(j)
        if args.fits is not None:
            write_fits(args.fits, j, exposures, s, args)
        r = phot.measure(exposures[B]["image"])
        this_frame = {}
        for i, k in enumerate(follow):
            ok = np.isfinite(r["mag"][i]) and r["flux"][i] / r["sigma"][i] >= 5  # S/N >= 5
            this_frame[k] = (r["mag"][i] if ok else np.nan, r["mag_err"][i] if ok else np.nan)
            rows.append(dict(exposure=j, ut_hours=ut[j], airmass=airmass[j], seeing=seeing[j],
                             seeing_model=scene.seeings[int(scene.seeing_index[j])], star=labels[k],
                             true_mag=true_mag(B, k, j), saturated=bool(r["saturated"][i]),
                             sextractor_mag=r["mag"][i], sextractor_err=r["mag_err"][i]))
            if survey is not None:
                rows[-1].update(band=B, tel_id=int(survey["tel_id"][j]), ratchet=int(survey["ratchet"][j]),
                                field_angle=float(survey["field_angle"][j]), time_utc=survey["time"][j])
        measured.append(this_frame)
        if (n + 1) % max(1, len(frames) // 10) == 0:
            print(f"  measured {n + 1}/{len(frames)}", flush=True)
    with open(args.output.with_suffix(".photometry.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # 8. the movie
    if not args.no_render:
        draw_data = movie_data(args, stars, scene, follow, transient, models, follow_sed, follow_rate, follow_cycle, ref_rate,
                               static_mag, rates_at, true_mag, measured, frames, times, bjd, ut, airmass, seeing, G, in_block,
                               ab, ref_frames)
        draw_data.update(labels=labels, pulsators=pulsators, source=source, gaia_var=gaia_var, templates=templates,
                         survey=survey)
        render(args, exposure, draw_data, ref_image, ref_full, stamps, frames)
    print(f"saved {args.output.with_suffix('.photometry.csv') if args.no_render else args.output}")


# ---------------------------------------------------------------- the movie
def model_hours(ut):
    """Where the light-curve panel draws the models: 800 points over a night; every 3 minutes over a longer span."""
    n = 800 if ut[-1] - ut[0] <= 24 else int((ut[-1] - ut[0]) * 20)
    return np.linspace(ut[0], ut[-1], n)


def movie_data(args, stars, scene, follow, transient, models, follow_sed, follow_rate, follow_cycle, ref_rate, static_mag,
               rates_at, true_mag, measured, frames, times, bjd, ut, airmass, seeing, G, in_block, ab, ref_frames):
    """Everything draw() needs that stays fixed through the movie: models, display limits, colours, each star's
    mean over the night."""
    B = args.band
    # transients' model light curves on the light-curve panel's time grid (periodic models are drawn from their cycle)
    model_curve = {}
    for k in transient:
        when = bjd[0] + (model_hours(ut) - ut[0]) / 24.0
        model_curve[k] = np.array([float(ab.mag_from_sed(BANDS[B], models[k].wave, f * FLAM)) if f.any() else np.nan
                                   for f in (models[k].sed_at(b) for b in when)])
    j_best = int(np.argmin(airmass))
    shown_px = ((np.abs(scene.xx - scene.centre) * scene.plate <= HALF + 2) & (np.abs(scene.yy - scene.centre) * scene.plate <= HALF + 2))
    # the colour exposure (display only): sky-subtracted rho -> red, beta -> blue, their mean -> green; beta scaled so
    # the field's median star is grey; brightness = the shown band's electrons above sky on an asinh stretch (Lupton et
    # al. 2004); limits: the best-airmass exposure's peak and the sky noise
    ratio = rates_at("rho", j_best) / np.where(rates_at("beta", j_best) > 0, rates_at("beta", j_best), np.nan)
    balance = np.isfinite(ratio) & (G > 12) & (G < 17) & ~np.isin(np.arange(len(stars)), follow)
    best = np.minimum(scene.background(B, j_best) + scene.design_at(B, j_best) @ (rates_at(B, j_best) * args.exptime),
                      scene.fullwell)
    rgb = dict(w_beta=float(np.median(ratio[balance])),
               vmax=float((best - scene.background(B, j_best))[shown_px].max()),
               lw=float(np.sqrt(np.median([scene.background(B, j) for j in range(len(times))]) + scene.readnoise**2)))
    # the difference colour scale: half the largest followed-star excursion at its peak pixel (at most the full well)
    psf_peak = {k: scene.design_at(B, j_best)[:, k].max() for k in follow}
    dlim = 0.5 * max(min(np.max(np.abs(follow_rate[k][B] - ref_rate[k][B])) * args.exptime * psf_peak[k], scene.fullwell)
                     for k in follow)
    fixed = np.nonzero(in_block & (G < 20) & ~np.isin(np.arange(len(stars)), follow))[0]
    cmd_c = np.r_[[static_mag["beta"][k] - static_mag["rho"][k] for k in fixed],
                  np.concatenate([follow_cycle[k]["beta"] - follow_cycle[k]["rho"] for k in follow])]
    cmd_m = np.r_[[static_mag[B][k] for k in fixed], np.concatenate([follow_cycle[k][B] for k in follow])]
    if transient:  # a transient has no light (nan) before its model starts
        lit = np.isfinite(cmd_c) & np.isfinite(cmd_m)
        cmd_c, cmd_m = cmd_c[lit], cmd_m[lit]
    # each star's mean over the night: the measurements' own, and the model's (a transient: where it is lit); the
    # light-curve axes cover 99.5% of the measurements (a rare bad cross-match must not flatten the rest; it is drawn
    # at the edge)
    rel = {}
    for k in follow:
        meas = np.array([f[k][0] for f in measured], float)
        model = np.array([true_mag(B, k, j) for j in frames], float)
        if k in transient:
            model = np.where(np.isfinite(model), model, np.nan)
        mean = np.nanmean if k in transient else np.mean
        lit = np.isfinite(meas).any()
        meas_rel = np.nanpercentile(meas - np.nanmean(meas), [0.25, 99.75]) if lit else []
        rel[k] = dict(measured=float(np.nanmean(meas)) if lit else np.nan, model=float(mean(model)),
                      span=(np.nanmin(np.r_[meas_rel, model - mean(model)]), np.nanmax(np.r_[meas_rel, model - mean(model)])),
                      resid=float(np.nanpercentile(np.abs(meas - model), 99.75)) if lit else np.nan)
    # the SED axis: every followed star's SED (300-1000 nm) over the night
    sed_values = np.concatenate([follow_sed[k][:, (models[k].wave.to_value(u.nm) >= 300) & (models[k].wave.to_value(u.nm) <= 1000)].ravel()
                                 for k in follow]) if follow else np.array([1e-16, 1e-14])
    sed_values = sed_values[sed_values > 0]
    if sed_values.size == 0:  # nothing lit all night (e.g. a transient after the window)
        sed_values = np.array([1e-16, 1e-14])
    palette = plt.get_cmap("tab10")
    return dict(stars=stars, scene=scene, follow=follow, transient=transient, models=models, follow_sed=follow_sed,
                follow_cycle=follow_cycle, model_curve=model_curve, static_mag=static_mag, true_mag=true_mag,
                times=times, bjd=bjd, ut=ut, airmass=airmass, seeing=seeing, G=G, in_block=in_block, ab=ab,
                colour_of={k: palette(i % 10) for i, k in enumerate(follow)}, rgb=rgb, dlim=dlim, cmd_c=cmd_c, cmd_m=cmd_m,
                rel=rel, sed_ylim=(float(np.percentile(sed_values, 3)) / 1.5, float(sed_values.max()) * 1.5),
                centre=block_centre_label(), n_ref=len(ref_frames), measured=measured,
                history={k: {"t": [], "m": [], "e": [], "true": []} for k in follow})


def render(args, exposure, d, ref_image, ref_full, stamps, frames):
    """The movie: each rendered exposure again (same seed), its difference image, and the panels."""
    B = args.band
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                                     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    apply_style()
    fig = plt.figure(figsize=(16, 10))
    with writer.saving(fig, str(args.output), dpi=args.dpi):
        for n, j in enumerate(frames):
            exposures = exposure(j)
            exposures[B]["difference"], _ = difference_image(exposures[B]["image"], exposures[B]["full"], ref_image, ref_full,
                                                             d["scene"].readnoise, half=args.kernel_half,
                                                             clip=args.kernel_clip, fit_pixels=stamps)
            for k in d["follow"]:
                h = d["history"][k]
                h["t"].append(d["ut"][j])
                h["true"].append(d["true_mag"](B, k, j))
                h["m"].append(d["measured"][n][k][0])
                h["e"].append(d["measured"][n][k][1])
            fig.clear()
            draw(fig, args, exposures, j, d)
            writer.grab_frame()
            if (n + 1) % max(1, len(frames) // 20) == 0:
                print(f"  {n + 1}/{len(frames)}", flush=True)
    plt.close(fig)


def draw(fig, args, exposures, j, d):
    """One movie frame: CMD, light curves, colour exposure, difference image, SEDs."""
    B = args.band
    follow, colour_of, labels, scene, stars = d["follow"], d["colour_of"], d["labels"], d["scene"], d["stars"]
    true_mag, pulsators, source = d["true_mag"], d["pulsators"], d["source"]
    grid = fig.add_gridspec(2, 2, width_ratios=[5.4, 4.6], height_ratios=[6, 4], wspace=0.16, hspace=0.3,
                            left=0.055, right=0.975, bottom=0.06, top=0.88)
    cmd = fig.add_subplot(grid[0, 0])
    pair = grid[0, 1].subgridspec(1, 2, wspace=0.06)
    image_ax = fig.add_subplot(pair[0])
    diff_ax = fig.add_subplot(pair[1], sharex=image_ax, sharey=image_ax)
    sed_ax = fig.add_subplot(grid[1, 1])
    light = grid[1, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    lc = fig.add_subplot(light[0])
    resid_ax = fig.add_subplot(light[1], sharex=lc)

    # CMD: the constant stars in the block, each followed star's cycle (a transient: its track) and where it is now
    others = np.nonzero(d["in_block"] & (d["G"] < 20) & ~np.isin(np.arange(len(d["G"])), follow))[0]
    static_mag = d["static_mag"]
    cmd.scatter([static_mag["beta"][k] - static_mag["rho"][k] for k in others], [static_mag[B][k] for k in others],
                s=4, color="#b9c4cc", zorder=1, label="Constant stars")
    for k in follow:
        cyc = d["follow_cycle"][k]
        if k in d["model_curve"]:
            cmd.plot(cyc["beta"] - cyc["rho"], cyc[B], color=colour_of[k], lw=1.3, alpha=0.9, zorder=3)
        else:  # a closed cycle
            cmd.plot(np.r_[cyc["beta"] - cyc["rho"], cyc["beta"][0] - cyc["rho"][0]], np.r_[cyc[B], cyc[B][0]],
                     color=colour_of[k], lw=1.3, alpha=0.9, zorder=3)
        cmd.scatter(true_mag("beta", k, j) - true_mag("rho", k, j), true_mag(B, k, j), s=60, color=colour_of[k],
                    edgecolor="#2b2b2b", linewidth=0.6, zorder=5)
    cmd.set(xlim=(np.percentile(d["cmd_c"], 0.5) - 0.15, np.percentile(d["cmd_c"], 99.5) + 0.15),
            ylim=(min(20.2, np.max(d["cmd_m"]) + 0.3), np.min(d["cmd_m"]) - 0.5),
            xlabel=r"$\beta-\rho$ (AB mag)", ylabel=rf"$m_{{\{B}}}$ (AB mag)")
    cmd.set_title("CMD", loc="left", weight="bold")
    cmd.legend(loc="lower left", fontsize=9, framealpha=0.85, edgecolor="none").set_zorder(2)
    cmd.grid(alpha=0.15)

    # light curves: measured (MAG_AUTO) points and the model behind, each relative to its own mean over the night
    ut = d["ut"]
    model_t = model_hours(ut)
    model_bjd = d["bjd"][0] + (model_t - ut[0]) / 24.0
    shown_m, shown_r = {}, {}  # what is drawn, to mark points beyond the axes
    for k in follow:
        h = d["history"][k]
        t = np.array(h["t"])
        v = pulsators[int(source[k])]
        if k in d["model_curve"]:
            model = d["model_curve"][k]
        else:
            cyc = d["follow_cycle"][k][B]
            model = np.interp(((model_bjd - v["epoch"]) / v["period"]) % 1.0, np.linspace(0, 1, len(cyc) + 1), np.r_[cyc, cyc[0]])
        lc.plot(model_t, model - d["rel"][k]["model"], color=colour_of[k], lw=1.0, alpha=0.35)
        m, e = np.array(h["m"]), np.array(h["e"])
        ok = np.isfinite(m)
        if k in d["model_curve"]:
            label = f"{labels[k]} {v['kind']}" + (f" z {v['transient']['z']:g}" if np.isfinite(v["transient"]["z"]) else "")
        else:
            label = f"{labels[k]} {v['kind']} {v['period']:.3f} d"
        lc.errorbar(t[ok], m[ok] - d["rel"][k]["measured"], yerr=e[ok], fmt="o", ms=2.5, lw=0.5, color=colour_of[k], label=label)
        shown_m[k] = (t[ok], m[ok] - d["rel"][k]["measured"])
        resid = np.array(h["m"]) - np.array(h["true"])
        if d["survey"] is not None:  # not across the gaps between ratchets
            gap = np.r_[False, np.diff(t) > 0.1]
            t, resid = np.insert(t, np.nonzero(gap)[0], np.nan), np.insert(resid, np.nonzero(gap)[0], np.nan)
        resid_ax.plot(t, resid, color=colour_of[k], lw=0.8)
        shown_r[k] = (t, resid)
    engines = [engine_of(pulsators[int(source[k])]) for k in follow]
    model_name = " / ".join([e for e in ("MESA RSP", "PHOEBE") if e in engines]
                            + list(dict.fromkeys(e for e in engines if e not in ("MESA RSP", "PHOEBE"))))
    lc.plot([], [], color="0.5", lw=1.0, alpha=0.5, label=f"{model_name} model")
    for ax in (lc, resid_ax):
        ax.axvline(ut[j], color="#455868", lw=0.8, alpha=0.35)
        ax.grid(alpha=0.15)
        ax.set_xlim(ut[0] - 0.1, ut[-1] + 0.1)
    lo = np.nanmin([d["rel"][k]["span"][0] for k in follow])
    hi = np.nanmax([d["rel"][k]["span"][1] for k in follow])
    if not np.isfinite(lo):  # nothing lit all night
        lo, hi = -0.5, 0.5
    pad = 0.08 * max(hi - lo, 0.05)
    lc.set(ylim=(hi + pad, lo - 2.5 * pad), ylabel=rf"$\Delta m_{{\{B}}}$ (mag)")
    resid_lim = max(0.05, 1.3 * np.nanmax([d["rel"][k]["resid"] for k in follow] + [0.0]))
    resid_ax.set_ylim(-resid_lim, resid_lim)
    for k in follow:  # points beyond the axes: a triangle at the edge
        for ax, (t, y), lo_edge, hi_edge in ((lc, shown_m[k], lo - 2.5 * pad, hi + pad), (resid_ax, shown_r[k], -resid_lim, resid_lim)):
            for beyond, edge, marker in ((y > hi_edge, hi_edge, "v"), (y < lo_edge, lo_edge, "^")):
                if np.any(beyond):
                    ax.plot(t[beyond], np.full(int(np.sum(beyond)), edge), marker, ms=6, mfc="none", color=colour_of[k], clip_on=False)
    lc.legend(loc="upper left", ncol=4, fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none", columnspacing=0.8,
              handletextpad=0.3)
    lc.set_title("Light curves (SExtractor MAG_AUTO)", loc="left", weight="bold")
    lc.set_title(f"UT {d['times'][j].iso[11:16]} · airmass {d['airmass'][j]:.2f} · seeing {d['seeing'][j]:.2f}″", loc="right",
                 fontsize=11, color="#3c5262")
    plt.setp(lc.get_xticklabels(), visible=False)
    resid_ax.axhline(0, color="0.5", lw=0.6, ls=":")
    span_days = (d["times"][-1].jd - d["times"][0].jd) > 1
    resid_ax.set(xlabel=f"UT hours since {d['times'][0].iso[:10]} 0h" if span_days else f"UT hours ({args.date} night)",
                 ylabel="measured − true\n(mag)")
    lc.yaxis.set_label_coords(-0.075, 0.5)
    resid_ax.yaxis.set_label_coords(-0.075, 0.5)
    resid_ax.invert_yaxis()

    # the colour exposure (see movie_data); a pixel at full well in either band is white. The survey takes one band at
    # a time: that band alone, grey
    extent = (np.array([0, scene.n, 0, scene.n]) - scene.centre) * scene.plate
    rgb = d["rgb"]
    above_sky = {b: exposures[b]["image"] - scene.background(b, j) for b in exposures}
    if len(exposures) == len(BANDS):
        red, blue = np.maximum(above_sky["rho"], 0.0), np.maximum(rgb["w_beta"] * above_sky["beta"], 0.0)
        channels = np.stack([red, 0.5 * (red + blue), blue], axis=-1)
        hue = (channels + rgb["lw"]) / (channels.mean(axis=-1, keepdims=True) + rgb["lw"])  # softened: blank sky is grey
    else:
        hue = np.ones(exposures[B]["image"].shape + (3,))
    hue[np.any([exposures[b]["image"] >= scene.fullwell - 5 * scene.readnoise for b in exposures], axis=0)] = 1.0
    norm = AsinhNorm(linear_width=rgb["lw"], vmin=0.0, vmax=rgb["vmax"])
    colour_img = hue * np.clip(norm(np.maximum(above_sky[B], 0.0)), 0, 1)[..., None]
    top = colour_img.max(axis=-1, keepdims=True)
    colour_img = np.where(top > 1, colour_img / np.where(top > 0, top, 1.0), colour_img)  # saturate to white, keeping hue
    image_ax.imshow(colour_img, origin="lower", interpolation="nearest", extent=extent)
    step = MINIPIX_PX * SKY_PX
    for v in np.arange(-HALF, HALF + 1e-6, step):  # the minipix grid
        image_ax.plot([v, v], [-HALF, HALF], color="white", lw=0.5, alpha=0.35)
        image_ax.plot([-HALF, HALF], [v, v], color="white", lw=0.5, alpha=0.35)
    image_ax.add_patch(Rectangle((-step / 2, -step / 2), step, step, fill=False, ec="#7fd1ff", lw=1.2))  # the central minipix
    for k in follow:
        image_ax.add_patch(Circle((stars["dx"][k], stars["dy"][k]), 3.0, fill=False, ec=colour_of[k], lw=1.2))
        image_ax.text(stars["dx"][k] + 3.2, stars["dy"][k] + 3.2, labels[k], fontsize=8, color=colour_of[k], weight="bold")
    edge = HALF + 2
    image_ax.set(xlim=(-edge, edge), ylim=(-edge, edge), xlabel="Offset (arcsec)", ylabel="Offset (arcsec)")
    if d["survey"] is None:
        image_ax.set_title(f"Exposure ({SYMBOLS['beta']} blue, {SYMBOLS['rho']} red)", loc="left", weight="bold")
    else:
        image_ax.set_title(f"Exposure ({SYMBOLS[B]})", loc="left", weight="bold")
        image_ax.text(0.97, 0.97, f"telescope {int(d['survey']['tel_id'][j])}\n{float(d['survey']['field_angle'][j]):.2f}° off-axis",
                      transform=image_ax.transAxes, va="top", ha="right", fontsize=8, color="white",
                      bbox={"facecolor": "#111726", "alpha": 0.65, "edgecolor": "none", "pad": 3})
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="gray"), ax=image_ax, orientation="horizontal", fraction=0.05,
                       pad=0.13, label=f"{SYMBOLS[B]} electrons / pixel above sky")
    bar.set_ticks([v for v in (0, 10, 100, 1000, 10000) if v <= rgb["vmax"]])
    bar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bar.ax.tick_params(labelsize=7)
    image_ax.text(0.03, 0.97, f"{args.exptime:g} s · exposure {j + 1}", transform=image_ax.transAxes, va="top",
                  fontsize=8, color="white", bbox={"facecolor": "#111726", "alpha": 0.65, "edgecolor": "none", "pad": 3})

    # the difference image (saturated pixels grey)
    diverging = plt.get_cmap("RdBu_r").copy()
    diverging.set_bad("#bfbfbf")
    diff = diff_ax.imshow(exposures[B]["difference"], origin="lower", interpolation="nearest", cmap=diverging,
                          vmin=-d["dlim"], vmax=d["dlim"], extent=extent)
    for k in follow:
        diff_ax.add_patch(Circle((stars["dx"][k], stars["dy"][k]), 3.0, fill=False, ec=colour_of[k], lw=1.0))
        diff_ax.text(stars["dx"][k] + 3.2, stars["dy"][k] + 3.2, labels[k], fontsize=8, color=colour_of[k], weight="bold")
    diff_ax.set_title("Difference (PSF-matched)", loc="left", weight="bold")
    diff_ax.text(0.03, 0.97, f"ref: median of {d['n_ref']} best-seeing exposures"
                 + ("\nwithout the transients (pre-event)" if d["model_curve"] else ""), transform=diff_ax.transAxes,
                 va="top", fontsize=7.5, color="#333", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2})
    diff_ax.set(xlim=(-edge, edge), ylim=(-edge, edge), xlabel="Offset (arcsec)")
    plt.setp(diff_ax.get_yticklabels(), visible=False)
    dbar = fig.colorbar(diff, ax=diff_ax, orientation="horizontal", fraction=0.05, pad=0.13,
                        label="difference (electrons / pixel)")
    dbar.ax.tick_params(labelsize=7)

    # SEDs of the followed stars now, and as each Argus band records them (x filter x optics x QE x atmosphere^airmass)
    ab = d["ab"]
    response_wave = ab.tp.wav.to_value(u.nm)
    for k in follow:
        wave, sed = d["models"][k].wave.to_value(u.nm), d["follow_sed"][k][j]
        shown = (wave >= 300) & (wave <= 1000)
        sed_ax.plot(wave[shown], sed[shown], color=colour_of[k], lw=0.9)
        on_grid = np.interp(response_wave, wave, sed, left=0.0, right=0.0)
        for name, band in BANDS.items():
            recorded = on_grid * system_response(ab.tp, band, d["airmass"][j])
            sed_ax.plot(response_wave[recorded > 0], recorded[recorded > 0], color=colour_of[k], lw=0.9, ls=":")
    sed_ax.set(yscale="log", ylim=d["sed_ylim"], xlim=(300, 1000),
               xlabel="Wavelength (nm)", ylabel=r"$f_\lambda$ at Argus (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    sed_ax.set_title("SED", loc="left", weight="bold")
    response_ax = sed_ax.twinx()
    response_ax.spines["right"].set_visible(True)
    for name, band in BANDS.items():
        response_ax.fill_between(response_wave, system_response(ab.tp, band, d["airmass"][j]), color=COLORS[name], alpha=0.10)
    response_ax.set(ylim=(0, 1), ylabel="Detected response")
    sed_ax.plot([], [], color="0.4", lw=0.9, label="SED")
    sed_ax.plot([], [], color="0.4", lw=0.9, ls=":", label=r"SED × $\beta$, $\rho$ system response")
    sed_ax.legend(loc="upper right", fontsize=8, frameon=False)
    sed_ax.grid(alpha=0.12)

    what = pulsator_title(follow, pulsators, source)  # a long list of stars: smaller, and on further lines
    what = "\n" + "\n".join(textwrap.wrap(what, 150, break_long_words=False)) if len(what) > 70 else " — " + what
    fig.suptitle(f"ARGUS — {d['centre']} ({FIELDS[args.field]['name']}) — {block_label()} — {args.date} — 60 s exposures"
                 + what, fontsize=13.5 if len(what) <= 48 else 11.5)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
