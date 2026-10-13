#!/usr/bin/env python3
"""Movie: a 3x3 HDPS-minipix block of M67's core observed by Argus across 1 Myr of evolution.

    python experiments/m67_minipix_movie.py                       # -> outputs/m67/m67_minipix.mp4
    python experiments/m67_minipix_movie.py --frames 60 -o test.mp4

A finished early experiment, kept as it was; the night movies (night_movie.py) grew out of it.

Needs the scene, fit and per-star runs from mesa_argus.m67 (scene -> fit -> run).

Every star is a real Gaia DR3 source at its catalogued position, frozen for the whole window
(over 1 Myr M67's stars really move by arcminutes). Members are MESA models at the cluster age,
distance and reddening; field stars are main-sequence templates matched to their colour and
scaled to their G. Each frame is ONE 60 s Argus exposure at the frame's stellar age: the frames
are snapshots across stellar time, not a survey time series.

Image: every star's ArgusSim PSF (Zemax optics + seeing) at its sub-pixel position on one
detector grid, summed as expected electrons with one sky, then one Poisson + read-noise draw,
clipped at full well. Photometry: a simultaneous PSF fit of all stars at their known positions
(deblended), and a single-PSF fit of each star that ignores its neighbours (naive). Their
difference is the photometric contamination from blending.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import warnings

import astropy.units as u
from astropy.table import Table
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from matplotlib.colors import AsinhNorm, Normalize  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the repo root: mesa_argus, plot_sed_observation

from argus_sim import ABPhot, SpectralSky, SystemThroughput, ZemaxPSF  # noqa: E402
from argus_sim import c as ARGUS_CONFIG  # noqa: E402

from mesa_argus import config as C  # noqa: E402
from mesa_argus import m67  # noqa: E402
from mesa_argus.history import read as read_history  # noqa: E402
from plot_sed_observation import (  # noqa: E402
    BANDS, COLORS, FLAM, MAG_FACTOR, SYMBOLS, apply_style, figure_axes, find_seds, read_mesa, system_response,
)

STAR_CMAP = plt.get_cmap("turbo_r")  # coloured by Teff: red (cool) -> blue (hot), no pale middle
TEFF_NORM = Normalize(3500, 7000)


# ---------------------------------------------------------------- inputs
def load_stars(ab, fit):
    """Every scene star's SEDs over the window at the observer, plus per-model rates and magnitudes."""
    scene = Table.read(m67.SCENE)
    by_label = {s["label"]: s for s in fit["stars"]}
    tracks = {t["label"]: t for t in m67.star_tracks()}
    distance_pc = 10 ** (fit["distance_modulus"] / 5 + 1)
    stars = []
    for row in scene:
        label = row["label"]
        run = C.RUNS / tracks[label]["name"]
        frames, _ = find_seds(run / "SED")
        hist = read_history(tracks[label]["name"])
        index = {int(m): i for i, m in enumerate(hist["model_number"])}
        ages, fluxes, teff, g_model = [], [], [], []
        wave = None
        for model, path in frames:
            if model not in index:
                continue
            w, f, _ = read_mesa(path)
            wave = w if wave is None else wave
            i = index[model]
            ages.append(hist["star_age"][i])
            fluxes.append(f.to_value(FLAM))
            teff.append(10 ** hist["log_Teff"][i])
            g_model.append(hist["G"][i])
        ages, fluxes = np.array(ages), np.array(fluxes)
        if by_label[label]["member"]:
            # MESA SED at 10 pc -> cluster distance, then interstellar reddening
            fluxes = m67.redden(wave.to_value(u.AA), fluxes * (10.0 / distance_pc) ** 2, fit["ebv"], fit["rv"])
            kind = "member"
        else:
            # colour-matched template scaled to the observed G (distance and reddening unknown)
            fluxes = fluxes * 10 ** (-0.4 * (float(row["Gmag"]) - np.array(g_model)))[:, None]
            kind = "field"
        rates = {name: np.array([ab.electrons_from_sed(band, wave, f * FLAM, airmass=1.2).value for f in fluxes])
                 for name, band in BANDS.items()}
        mags = {name: np.array([float(ab.mag_from_sed(band, wave, f * FLAM)) for f in fluxes])
                for name, band in BANDS.items()}
        stars.append(dict(label=label, kind=kind, G=float(row["Gmag"]), dx=float(row["dx_arcsec"]), dy=float(row["dy_arcsec"]),
                          in_block=bool(row["in_block"]), ages=ages, wave=wave, fluxes=fluxes, rates=rates, mags=mags,
                          teff=np.array(teff), phase=by_label[label]["phase"], mass=by_label[label]["mass"]))
        print(f"  {label}  {kind:6s}  {len(ages):4d} SEDs  rho {mags['rho'].mean():6.2f}  beta-rho {np.mean(mags['beta'] - mags['rho']):5.2f}"
              f"  Teff {np.mean(teff):5.0f} K  {by_label[label]['phase']}", flush=True)
    return stars


def at_time(star, key, t, name=None):
    """Linear interpolation in stellar age (rates and fluxes are linear in flux)."""
    values = star[key] if name is None else star[key][name]
    i = int(np.clip(np.searchsorted(star["ages"], t), 1, len(star["ages"]) - 1))
    f = (t - star["ages"][i - 1]) / (star["ages"][i] - star["ages"][i - 1])
    return values[i - 1] + f * (values[i] - values[i - 1])


def isochrone_argus(fit, stars, ab):
    """M67 isochrone in apparent Argus AB magnitudes on ArgusSim's photometric system.

    MESA's beta/rho curves are the full system throughput; ArgusSim's mag_from_sed integrates
    the filter alone, so the two systems differ by a colour term. Every scene star has both
    (MESA history magnitude and ArgusSim magnitude of the same unreddened 10 pc SED), from 3800
    to 6100 K: a quadratic in MESA colour maps one system onto the other. Extinction in the
    Argus bands is measured on the members' reddened SEDs.
    """
    zp = json.loads((C.DATA / "zp_offsets.json").read_text())["offsets"]
    tracks = {t["label"]: t["name"] for t in m67.star_tracks()}
    distance_pc = 10 ** (fit["distance_modulus"] / 5 + 1)
    mesa, argus, extinction = [], [], []
    for s in stars:
        h = read_history(tracks[s["label"]])
        i = int(np.argmin(np.abs(h["star_age"] - s["ages"][0])))
        frames, _ = find_seds(C.RUNS / tracks[s["label"]] / "SED")
        path = dict(frames)[int(h["model_number"][i])]
        w, f10, _ = read_mesa(path)
        mesa.append([h[b][i] + zp[b] for b in BANDS])
        argus.append([float(ab.mag_from_sed(band, w, f10)) for band in BANDS.values()])
        if s["kind"] == "member":
            reddened = m67.redden(w.to_value(u.AA), f10.value * (10.0 / distance_pc) ** 2, fit["ebv"], fit["rv"]) * FLAM
            extinction.append([float(ab.mag_from_sed(band, w, reddened)) - a - fit["distance_modulus"]
                               for band, a in zip(BANDS.values(), argus[-1])])
    mesa, argus = np.array(mesa), np.array(argus)
    colour = mesa[:, 0] - mesa[:, 1]
    terms = [np.polyfit(colour, argus[:, k] - mesa[:, k], 2) for k in range(len(BANDS))]
    hist = m67.grid_histories()
    iso = np.array([[np.interp(fit["age"], h["star_age"], h[b]) + zp[b] for b in BANDS]
                    for mass, h in sorted(hist.items()) if h["star_age"][-1] >= fit["age"]])
    c = iso[:, 0] - iso[:, 1]
    iso = iso + np.array([np.polyval(t, c) for t in terms]).T
    return iso + fit["distance_modulus"] + np.median(extinction, axis=0)


# ---------------------------------------------------------------- scene rendering
class Scene:
    """Fixed geometry: every star's PSF at its sub-pixel position on one detector canvas, per band."""

    def __init__(self, stars, cfg, ab, args):
        self.plate = cfg.telescope.plate_scale
        half = m67.block_centre_px()  # only for provenance; geometry is relative to the block centre
        del half
        span = 2 * (m67.MINIPIX_PX * 1.5 * m67.SKY_PX_ARCSEC + m67.MARGIN_ARCSEC) / self.plate + 30
        self.n = int(np.ceil(span / 2) * 2)
        self.centre = self.n / 2.0  # detector-pixel coordinate of the block centre
        model = ZemaxPSF.from_config(cfg)
        self.psf = {name: [] for name in BANDS}
        self.xy = []
        for s in stars:
            x, y = self.centre + s["dx"] / self.plate, self.centre + s["dy"] / self.plate
            self.xy.append((x, y))
        for name, band in BANDS.items():
            p0, _ = model.with_seeing(seeing=args.seeing, band=band, field_angle_deg=args.field_angle, pixel_phase=(0.0, 0.0))
            yy, xx = np.indices(p0.shape)
            c0 = ((p0 * yy).sum() / p0.sum(), (p0 * xx).sum() / p0.sum())
            for x, y in self.xy:
                # the stamp's centroid moves by -phase: pick the origin that puts it on (x, y)
                oy, ox = int(np.ceil(y - c0[0])), int(np.ceil(x - c0[1]))
                phase = (oy + c0[0] - y, ox + c0[1] - x)
                p, _ = model.with_seeing(seeing=args.seeing, band=band, field_angle_deg=args.field_angle,
                                         pixel_phase=(phase[0] % 1.0, phase[1] % 1.0))
                p = np.maximum(p, 0.0)
                p /= p.sum()
                canvas = np.zeros((self.n, self.n))
                sy, sx = p.shape
                y0, x0 = max(oy, 0), max(ox, 0)
                y1, x1 = min(oy + sy, self.n), min(ox + sx, self.n)
                if y1 > y0 and x1 > x0:
                    canvas[y0:y1, x0:x1] = p[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
                self.psf[name].append(canvas)
        sky_s10 = [145.0 * args.airmass, 60.0, 10.0, 0.0, 0.0]
        self.background = {}
        for name, band in BANDS.items():
            sky = SpectralSky(ab.tp, ab, band)
            rate = float(sky.sky_electrons(*sky_s10, transparency=1.0)) * self.plate**2
            self.background[name] = (rate + cfg.telescope.dark_current) * args.exptime
        self.readnoise, self.fullwell = cfg.telescope.readnoise, cfg.telescope.fullwell
        self.exptime = args.exptime
        # naive photometry window: each star alone, within this radius of its position
        yy, xx = np.indices((self.n, self.n))
        self.window = [np.hypot(yy - y, xx - x) <= args.naive_radius for x, y in self.xy]

    def expected(self, name, rates):
        return self.background[name] + self.exptime * sum(r * p for r, p in zip(rates, self.psf[name]))

    def observe(self, name, rates, rng):
        """One exposure, the deblended and naive fluxes of every star, and contamination fractions."""
        mean = self.expected(name, rates)
        charge = np.minimum(rng.poisson(mean), self.fullwell)
        image = charge + rng.normal(0.0, self.readnoise, mean.shape)
        saturated = charge >= self.fullwell
        variance = mean + self.readnoise**2
        good = ~saturated
        A = np.array([p[good] for p in self.psf[name]]).T
        w = 1.0 / variance[good]
        data = (image - self.background[name])[good]
        normal = A.T @ (A * w[:, None])
        cov = np.linalg.pinv(normal)
        fit = cov @ (A.T @ (w * data))
        sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
        truth = np.asarray(rates) * self.exptime
        naive, contamination, sat = [], [], []
        clean = mean - self.background[name]
        for i, p in enumerate(self.psf[name]):
            win = self.window[i] & good
            core = self.window[i] & (p > 0.2 * p.max())
            sat.append(bool(np.any(saturated & core)))
            denom = np.sum(p[win] ** 2 / variance[win])
            naive.append(np.sum(p[win] * (image - self.background[name])[win] / variance[win]) / denom if denom > 0 else np.nan)
            # noise-free version: fraction of the naive flux that belongs to other stars
            own = truth[i] * np.sum(p[win] ** 2 / variance[win]) / denom if denom > 0 else np.nan
            total = np.sum(p[win] * clean[win] / variance[win]) / denom if denom > 0 else np.nan
            contamination.append((total - own) / own if own and own > 0 else np.nan)
        return dict(image=image, saturated=np.array(sat), fit=fit, sigma=sigma, naive=np.array(naive),
                    contamination=np.array(contamination), truth=truth)


# ---------------------------------------------------------------- frame
def draw_frame(fig, args, stars, scene, iso, t, times, history, obs, fixed, fit_info):
    cmd, lc, cont_ax, image_ax, sed_ax = figure_axes(fig, margins=(0.065, 0.95, 0.06, 0.88))
    k = len(history) - 1
    colours = [STAR_CMAP(TEFF_NORM(at_time(s, "teff", t))) for s in stars]
    zero = {name: np.array([at_time(s, "rates", t, name) for s in stars]) for name in BANDS}

    # --- CMD
    cmd.plot(iso[:, 0] - iso[:, 1], iso[:, 1], color="#c9d3da", lw=2.5, zorder=1,
             label=f"M67 isochrone ({fit_info['age'] / 1e9:.2f} Gyr, MESA)")
    for i, s in enumerate(stars):
        if not s["in_block"]:
            continue
        true_c, true_m = at_time(s, "mags", t, "beta") - at_time(s, "mags", t, "rho"), at_time(s, "mags", t, "rho")
        track_c = s["mags"]["beta"] - s["mags"]["rho"]
        cmd.plot(track_c, s["mags"]["rho"], color=colours[i], lw=1.0, alpha=0.6, zorder=2)
        marker = "*" if s["kind"] == "member" else "D"
        cmd.scatter(true_c, true_m, marker=marker, s=150 if marker == "*" else 40, color=colours[i], edgecolor="#2b2b2b",
                    linewidth=0.6, zorder=5)
        b, r = obs["beta"], obs["rho"]
        if not (b["saturated"][i] or r["saturated"][i]) and b["fit"][i] > 0 and r["fit"][i] > 0:
            mb = s_mag(at_time(s, "mags", t, "beta"), b["fit"][i], b["truth"][i])
            mr = s_mag(true_m, r["fit"][i], r["truth"][i])
            snr = min(b["fit"][i] / b["sigma"][i], r["fit"][i] / r["sigma"][i])
            if snr >= 5:
                eb, er = MAG_FACTOR * b["sigma"][i] / b["fit"][i], MAG_FACTOR * r["sigma"][i] / r["fit"][i]
                cmd.errorbar(mb - mr, mr, xerr=np.hypot(eb, er), yerr=er, fmt="o", ms=3.5, color="#203f54", lw=0.8, zorder=6)
            nb, nr = b["naive"][i], r["naive"][i]
            if nb > 0 and nr > 0:
                vb = s_mag(at_time(s, "mags", t, "beta"), nb, b["truth"][i])
                vr = s_mag(true_m, nr, r["truth"][i])
                if np.hypot(vb - vr - true_c, vr - true_m) > 0.02:
                    cmd.annotate("", xy=(vb - vr, vr), xytext=(true_c, true_m), zorder=4,
                                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1, alpha=0.85))
        cmd.annotate(s["label"], (true_c, true_m), xytext=(6, 3), textcoords="offset points", fontsize=8, color="#3c5262")
    cmd.plot([], [], marker="*", ls="none", color="#d9a33b", markeredgecolor="#2b2b2b", ms=11, label="Member (MESA model, true)")
    cmd.plot([], [], marker="D", ls="none", color="#9fb3c8", markeredgecolor="#2b2b2b", ms=6, label="Field star (template, true)")
    cmd.errorbar([], [], xerr=[], yerr=[], fmt="o", ms=3.5, color="#203f54", label="Deblended PSF fit (this exposure)")
    cmd.plot([], [], color="#c0392b", lw=1.1, label="Neighbours ignored → contamination")
    cmd.set(xlim=fixed["cmd_x"], ylim=fixed["cmd_y"], xlabel=r"$\beta-\rho$ (AB mag)", ylabel=r"$m_\rho$ (AB mag)")
    cmd.set_title("Colour–magnitude diagram · stars in the block", loc="left", weight="bold")
    legend = cmd.legend(loc="center right", fontsize=9, framealpha=0.85, edgecolor="none")
    legend.set_zorder(4)
    cmd.grid(alpha=0.15)
    cmd.set_title(f"{sum(s['in_block'] for s in stars)} Gaia DR3 sources · 1 Myr window",
                  loc="right", fontsize=11, color="#3c5262")

    # --- light curves (deblended rho) and contamination
    dt = (np.array(times[: k + 1]) - times[0]) / 1e3
    for i, s in enumerate(stars):
        if not s["in_block"]:
            continue
        m = np.array([h["rho"][i] for h in history])
        e = np.array([h["rho_err"][i] for h in history])
        ok = np.isfinite(m)
        lc.plot((s["ages"] - times[0]) / 1e3, s["mags"]["rho"], color=colours[i], lw=0.8, alpha=0.35)
        lc.errorbar(dt[ok], m[ok], yerr=e[ok], fmt="o", ms=2.2, lw=0.5, color=colours[i])
        cont = np.array([h["contamination"][i] for h in history]) * 100
        cont_ax.plot(dt, np.clip(cont, 1e-3, None), color=colours[i], lw=1.2)
        if np.isfinite(cont[-1]) and cont[-1] > 1e-3:
            cont_ax.annotate(s["label"], (dt[-1], cont[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                             color=colours[i], va="center")
    for ax in (lc, cont_ax):
        ax.set_xlim(-0.02 * fixed["span_kyr"], 1.06 * fixed["span_kyr"])
        ax.grid(alpha=0.15)
    lc.set(ylim=fixed["lc_y"], ylabel=r"$m_\rho$ deblended (AB)")
    lc.set_title(r"Light curves · deblended $\rho$ photometry", loc="left", weight="bold")
    lc.set_title(f"t = {fit_info['age'] / 1e9:.4f} Gyr + {(t - times[0]) / 1e3:.0f} kyr", loc="right", fontsize=11,
                 color="#3c5262")
    plt.setp(lc.get_xticklabels(), visible=False)
    cont_ax.set(yscale="log", ylim=(1e-2, 2e3), xlabel="Stellar time since window start (kyr)",
                ylabel="Neighbour\nflux (%)")
    cont_ax.axhline(1, color="0.5", lw=0.6, ls=":")

    # --- image: the minipix block
    img = obs[args.image_band]["image"]
    extent_px = np.array([0, scene.n, 0, scene.n])
    extent = (extent_px - scene.centre) * scene.plate
    norm = AsinhNorm(linear_width=fixed["linear_width"], vmin=0.0, vmax=fixed["vmax"])
    shown = image_ax.imshow(img, origin="lower", interpolation="nearest", cmap="magma", norm=norm, extent=extent)
    half = 1.5 * m67.MINIPIX_PX * m67.SKY_PX_ARCSEC
    step = m67.MINIPIX_PX * m67.SKY_PX_ARCSEC
    for v in np.arange(-half, half + 1e-6, step):
        image_ax.plot([v, v], [-half, half], color="white", lw=0.6, alpha=0.45)
        image_ax.plot([-half, half], [v, v], color="white", lw=0.6, alpha=0.45)
    image_ax.add_patch(Rectangle((-step / 2, -step / 2), step, step, fill=False, ec="#7fd1ff", lw=1.4))
    for i, s in enumerate(stars):
        if s["in_block"]:
            sat = obs[args.image_band]["saturated"][i]
            image_ax.add_patch(Circle((s["dx"], s["dy"]), 3.2, fill=False, ec="#ff6b6b" if sat else colours[i], lw=0.8))
            image_ax.text(s["dx"] + 3.5, s["dy"] + 3.5, s["label"] + (" sat" if sat else ""), fontsize=6.5, color="white")
    lim = half + 3
    image_ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Offset (arcsec)", ylabel="Offset (arcsec)")
    image_ax.set_title(f"Argus {SYMBOLS[args.image_band]} · 3×3 minipixes", loc="left", weight="bold")
    colorbar = fig.colorbar(shown, ax=image_ax, shrink=0.8, pad=0.025, label="electrons / pixel")
    colorbar.set_ticks(norm.inverse(np.linspace(0, 1, 5)))
    colorbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    image_ax.text(0.03, 0.97, f"{args.exptime:g} s · minipix 28.1″ (32 × 0.879″)\nhighlighted: centre minipix",
                  transform=image_ax.transAxes, va="top", fontsize=7.5, color="white",
                  bbox={"facecolor": "#111726", "alpha": 0.65, "edgecolor": "none", "pad": 4})

    # --- spectra
    for i, s in enumerate(stars):
        if s["in_block"]:
            nm = s["wave"].to_value(u.nm)
            keep = (nm >= 300) & (nm <= 1000)
            sed_ax.plot(nm[keep], at_time(s, "fluxes", t)[keep], color=colours[i], lw=0.8, alpha=0.9)
    sed_ax.set(yscale="log", ylim=fixed["sed_y"], xlim=(300, 1000), xlabel="Wavelength (nm)",
               ylabel=r"$f_\lambda$ at Argus (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    sed_ax.set_title("Spectra of the block's stars", loc="left", weight="bold")
    response_ax = sed_ax.twinx()
    response_ax.spines["right"].set_visible(True)
    for name, band in BANDS.items():
        response = system_response(scene_ab.tp, band, args.airmass)
        response_ax.fill_between(scene_ab.tp.wav.to_value(u.nm), response, color=COLORS[name], alpha=0.10)
    response_ax.set(ylim=(0, 1), ylabel="Detected response")
    sed_ax.grid(alpha=0.12)

    fig.suptitle(
        "ARGUS · M67 (NGC 2682) core · 3×3 HDPS minipixes (84″) · 1 Myr of stellar evolution\n"
        f"Gaia DR3 positions (frozen) · MESA SEDs · ArgusSim PSFs · one {args.exptime:g} s exposure per frame · "
        f"airmass {args.airmass:g} · seeing {args.seeing:g}″",
        fontsize=14, weight="medium",
    )


def s_mag(true_mag, flux, true_flux):
    return true_mag - 2.5 * np.log10(flux / true_flux)


scene_ab = None


def main(argv=None):
    global scene_ab
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/m67/m67_minipix.mp4"))
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--exptime", type=float, default=60.0)
    p.add_argument("--airmass", type=float, default=1.2)
    p.add_argument("--seeing", type=float, default=1.2)
    p.add_argument("--field-angle", type=float, default=1.0)
    p.add_argument("--image-band", choices=list(BANDS), default="rho")
    p.add_argument("--naive-radius", type=float, default=4.0, help="pixels: the single-star fit window")
    p.add_argument("--seed", type=int, default=67)
    args = p.parse_args(argv)

    cfg = copy.deepcopy(ARGUS_CONFIG)
    tp = SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=cfg.telescope.throughput_loss)
    ab = ABPhot(throughput=tp, collecting_area=cfg.telescope.collecting_area)
    scene_ab = ab
    fit_info = json.loads(m67.FIT.read_text())
    print("loading stars")
    stars = load_stars(ab, fit_info)
    iso = isochrone_argus(fit_info, stars, ab)
    start = max(s["ages"][0] for s in stars)
    end = min(s["ages"][-1] for s in stars)
    times = list(np.linspace(start, end, args.frames))
    scene = Scene(stars, cfg, ab, args)

    block = [s for s in stars if s["in_block"]]
    colours_all = np.concatenate([s["mags"]["beta"] - s["mags"]["rho"] for s in block])
    mags_all = np.concatenate([s["mags"]["rho"] for s in block])
    peaks = []
    for t in (start, end):
        rates = [at_time(s, "rates", t, args.image_band) for s in stars]
        peaks.append(scene.expected(args.image_band, rates).max())
    seds = np.concatenate([s["fluxes"][:, (s["wave"].to_value(u.nm) > 300) & (s["wave"].to_value(u.nm) < 1000)].ravel()
                           for s in block])
    fixed = dict(
        cmd_x=(np.min(colours_all) - 0.25, np.max(colours_all) + 0.25),
        cmd_y=(np.max(mags_all) + 0.8, np.min(mags_all) - 0.8),
        lc_y=(np.max(mags_all) + 0.5, np.min(mags_all) - 0.5),
        span_kyr=(end - start) / 1e3,
        vmax=float(min(max(peaks), scene.fullwell)),
        linear_width=float(np.sqrt(scene.background[args.image_band] + scene.readnoise**2)),
        sed_y=(np.percentile(seds[seds > 0], 2), seds.max() * 2),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                                         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    apply_style()
    fig = plt.figure(figsize=(16, 10))
    seeds = np.random.SeedSequence(args.seed).spawn(len(times))
    history = []
    print(f"rendering {len(times)} frames -> {args.output}")
    with writer.saving(fig, str(args.output), dpi=args.dpi):
        for k, t in enumerate(times):
            rng = np.random.default_rng(seeds[k])
            obs = {name: scene.observe(name, [at_time(s, "rates", t, name) for s in stars], rng) for name in BANDS}
            r = obs["rho"]
            ok = ~r["saturated"] & (r["fit"] > 0) & (r["fit"] / np.where(r["sigma"] > 0, r["sigma"], np.inf) >= 5)
            true_rho = np.array([at_time(s, "mags", t, "rho") for s in stars])
            history.append(dict(
                rho=np.where(ok, true_rho - 2.5 * np.log10(np.where(ok, r["fit"], 1) / r["truth"]), np.nan),
                rho_err=np.where(ok, MAG_FACTOR * r["sigma"] / np.where(ok, r["fit"], 1), np.nan),
                contamination=r["contamination"],
            ))
            fig.clear()
            draw_frame(fig, args, stars, scene, iso, t, times, history, obs, fixed, fit_info)
            writer.grab_frame()
            if (k + 1) % max(1, len(times) // 10) == 0:
                print(f"  {k + 1}/{len(times)}", flush=True)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
