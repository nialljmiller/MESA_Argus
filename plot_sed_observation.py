#!/usr/bin/env python3
"""Render one stellar SED as a linked Argus CMD, light curve, image and spectrum.

Point it straight at a MESA colors-module SED (any filter's file holds the full SED)::

    python plot_sed_observation.py runs/<track>/SED/beta_SED_00000010.csv
    python plot_sed_observation.py runs/<track>/SED/beta_SED_00000010.csv --distance-pc 2000

MESA SEDs are f_lambda at 10 pc; --distance-pc rescales them by (10 pc / d)^2
(default: left at 10 pc, which saturates most stars in a 60 s exposure).

Point it at a folder to make a movie, one frame per SED::

    python plot_sed_observation.py runs/<track> --distance-pc 2000            # -> <track>_argus.mp4
    python plot_sed_observation.py runs/<track>/SED --model-range 1 400 -o ms.gif

MESA writes the same SED once per filter, so one file per model is used, ordered by
model number (other folders: by the trailing number in each file name). Every axis is
fixed by the whole sequence; the CMD traces the evolutionary track and the light curve
shows one exposure per state against model number. Long runs are evenly subsampled
to --max-frames (default 600; 0 renders every model). --save-data writes the per-frame
photometry (with age, Teff and log g from LOGS/history.data) as a CSV.

Other CSVs: the default columns are wavelength,flux, in Angstrom and
erg/(s cm2 Angstrom). Values are observer-frame flux above the atmosphere.
Use --flux-kind surface --radius-rsun R --distance-pc D for surface flux.
CSV comments may specify wavelength_unit, flux_unit, flux_kind and distance_pc
as "# key: value". Explicit command-line options override these comments.

One CSV supplies ONE stellar state. By default the light-curve panel has
one epoch. --n-exposures repeats this fixed SED under fixed conditions;
this is a noise demonstration, not stellar evolution or survey scheduling.

All photometry goes through ArgusSim's native SED API: magnitudes are
ABPhot.mag_from_sed (synthetic AB through the filter alone, calibrated by the
AB zero point); detected rates are ABPhot.electrons_from_sed (filter x optics x
QE x atmosphere^airmass x transparency, with the SED inside the integral).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import warnings

import astropy.units as u
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import AsinhNorm  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402
import numpy as np  # noqa: E402

from argus_sim import ABPhot, NoiseBudget, SpectralSky, SystemThroughput, ZemaxPSF  # noqa: E402
from argus_sim import __version__ as ARGUS_VERSION, c as ARGUS_CONFIG  # noqa: E402

from mesa_argus.seds import read_sed_csv as read_mesa_sed  # noqa: E402

FLAM = u.erg / (u.s * u.cm**2 * u.AA)
BANDS = {"beta": "g", "rho": "rho"}
COLORS = {"beta": "#2078b4", "rho": "#cc5940", "colour": "#7853a6"}
SYMBOLS = {"beta": r"$\beta$", "rho": r"$\rho$"}
MAG_FACTOR = 2.5 / np.log(10.0)


MESA_HEADER = "wavelengths,fluxes,convolved_flux,filter_wavelengths,filter_trans"
MESA_NAME = re.compile(r"^(?P<filter>.+)_SED_(?P<model>\d+)$")


def read_mesa(path):
    """Read a MESA colors-module SED CSV: f_lambda at 10 pc in the first two columns.

    The remaining columns are MESA's own filter convolution and are ignored;
    ArgusSim integrates the raw SED through the Argus passbands itself.
    """
    samples = read_mesa_sed(path)
    metadata = {
        "format": "MESA colors-module SED",
        "flux_kind": "observer",
        "distance_pc": "10",
        "wavelength_unit": "Angstrom",
        "flux_unit": "erg / (s cm2 Angstrom)",
    }
    name = MESA_NAME.match(path.stem)
    if name:
        track = path.parent.parent.name if path.parent.name == "SED" else ""
        model = int(name["model"])
        metadata.update(model_number=model, title=f"{track} · model {model}" if track else f"model {model}")
    return samples[:, 0] * u.AA, samples[:, 1] * FLAM, metadata


def read_sed(path, wavelength_column=None, flux_column=None, wavelength_unit=None, flux_unit=None):
    """Read a MESA SED or a headered/two-column CSV; return sorted wavelength, flux density, metadata.

    MESA colors-module files are recognised by their header. For other CSVs,
    unsorted rows are sorted; duplicate wavelengths, negative flux, missing
    values, and non-finite samples are rejected. Column selectors may be
    names or zero-based indices. Fortran D exponents are accepted.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        candidates = sorted(p for p in path.iterdir() if p.suffix.lower() == ".csv" and p.is_file())
        if len(candidates) != 1:
            raise ValueError(f"{path} contains {len(candidates)} CSVs; pass a single SED file.")
        path = candidates[0]
    with path.open(encoding="utf-8-sig") as handle:
        first = handle.readline().strip().replace(" ", "")
    if first == MESA_HEADER and wavelength_column is None and flux_column is None:
        if wavelength_unit or flux_unit:
            raise ValueError("MESA SEDs have fixed units; drop --wavelength-unit/--flux-unit.")
        wave, flux, metadata = read_mesa(path)
        metadata["path"] = str(path.resolve())
        return wave, flux, metadata
    metadata, lines = {}, []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            key, sep, value = stripped[1:].partition(":")
            if sep:
                metadata[key.strip().lower()] = value.strip()
        elif stripped:
            lines.append(line)
    rows = list(csv.reader(lines))
    if not rows:
        raise ValueError("The SED CSV is empty.")

    def number(value):
        # Fortran writes D exponents, and drops the E for 3-digit ones (1.180971-100)
        value = value.strip().replace("D", "E").replace("d", "e")
        return float(re.sub(r"(?<=\d)([+-]\d{3})$", r"E\1", value))

    try:
        [number(value) for value in rows[0]]
        header = None
    except ValueError:
        header = [value.strip().lower() for value in rows.pop(0)]

    def column(selector, aliases, fallback):
        if selector is not None:
            if str(selector).isdigit():
                return int(selector)
            if header is not None and str(selector).lower() in header:
                return header.index(str(selector).lower())
            raise ValueError(f"Column {selector!r} not found; available columns: {header}.")
        matches = [i for i, name in enumerate(header or []) if name in aliases]
        if len(matches) == 1:
            return matches[0]
        if header is None and len(rows[0]) == 2:
            return fallback
        raise ValueError(f"Specify --{'wavelength' if fallback == 0 else 'flux'}-column; columns: {header}.")

    if not rows:
        raise ValueError("The CSV contains a header but no samples.")
    iw = column(wavelength_column, ("wavelength", "wave", "wav", "lambda"), 0)
    iff = column(flux_column, ("flux", "flux_density", "flam", "f_lambda", "fnu", "f_nu"), 1)
    if iw == iff:
        raise ValueError("Wavelength and flux must use different columns.")
    try:
        samples = np.asarray([[number(row[iw]), number(row[iff])] for row in rows], dtype=float)
    except (ValueError, IndexError) as exc:
        raise ValueError("Every CSV data row must contain numeric wavelength and flux values.") from exc
    if len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError("Supply at least two finite wavelength/flux samples.")
    if np.any(samples[:, 0] <= 0) or np.any(samples[:, 1] < 0) or not np.any(samples[:, 1] > 0):
        raise ValueError("Wavelengths must be positive and fluxes non-negative, with nonzero total flux.")
    wu = wavelength_unit or metadata.get("wavelength_unit", "Angstrom")
    fu = flux_unit or metadata.get("flux_unit", "erg / (s cm2 Angstrom)")
    wave = (samples[:, 0] * u.Unit(wu)).to(u.AA)
    flux = samples[:, 1] * u.Unit(fu)
    if not flux.unit.is_equivalent(FLAM, equivalencies=u.spectral_density(wave)):
        raise ValueError(f"--flux-unit {fu!r} is not a spectral flux density.")
    order = np.argsort(wave.value)
    wave, flux = wave[order], flux[order]
    if np.any(np.diff(wave.value) <= 0):
        raise ValueError("Duplicate wavelengths are not allowed.")
    metadata.update(path=str(path.resolve()), wavelength_unit=str(wu), flux_unit=str(fu))
    return wave, flux, metadata


def system_response(tp, band, airmass):
    """Filter x optics x QE x atmosphere^airmass, as in ABPhot.electrons_from_sed (for plotting)."""
    response = np.asarray(tp.filt_wav[band] * tp.throughput_wav * tp.qe_wav, dtype=float)
    return response * np.asarray(tp.filt_wav.atmosphere, dtype=float) ** airmass


def prepare_sed(args, ab, path=None):
    """Read the SED and apply surface dilution, distance rescaling, or brightness normalisation."""
    wave, flux, metadata = read_sed(
        path or args.input, args.wavelength_column, args.flux_column, args.wavelength_unit, args.flux_unit
    )
    kind = args.flux_kind or metadata.get("flux_kind", "observer")
    if kind not in ("observer", "surface"):
        raise ValueError("flux_kind must be observer or surface.")
    native_pc = float(metadata["distance_pc"]) if "distance_pc" in metadata else None
    scale = 1.0
    if kind == "surface":
        if args.radius_rsun is None or args.distance_pc is None:
            raise ValueError("Surface flux requires --radius-rsun and --distance-pc.")
        scale = ((args.radius_rsun * u.R_sun) / (args.distance_pc * u.pc)).to_value(u.one) ** 2
    elif args.radius_rsun is not None:
        raise ValueError("--radius-rsun applies only to --flux-kind surface.")
    elif native_pc is not None and args.rho_mag is None:
        # observer flux at a known distance (MESA: 10 pc): move it to --distance-pc
        if args.distance_pc is None:
            args.distance_pc = native_pc
        scale = (native_pc / args.distance_pc) ** 2
    flux = flux * scale
    if args.rho_mag is not None:
        factor = 10 ** (-0.4 * (args.rho_mag - float(ab.mag_from_sed("rho", wave, flux))))
        flux = flux * factor
        scale *= factor
    metadata.update(flux_kind=kind, flux_scale=scale, observer_flux=True)
    if args.absolute and args.distance_pc is None:
        raise ValueError("--absolute requires the distance of the observer-frame SED (--distance-pc).")
    if args.distance_pc is not None:
        metadata["observer_distance_pc"] = args.distance_pc
    return wave, flux, metadata


def draw_exposure(psf, source_e, background_e, read_noise_e, fullwell_e, rng):
    """Draw Poisson charge plus Gaussian read noise, then perform a known-PSF flux fit.

    Saturation clips the charge before readout. The mock fit uses the known
    background and expected per-pixel variance (forecast weights). Its
    variance therefore describes this estimator, not a generic pipeline.
    """
    mean = source_e * psf + background_e
    variance = mean + read_noise_e**2
    charge = rng.poisson(mean)
    mask = charge >= fullwell_e
    image = np.minimum(charge, fullwell_e) + rng.normal(0.0, read_noise_e, psf.shape)
    precision = np.sum(psf**2 / variance)
    flux_e = float(np.sum(psf * (image - background_e) / variance) / precision)
    sigma_e = float(1.0 / np.sqrt(precision))
    saturated = bool(np.any(mask) or mean.max() >= fullwell_e)
    if saturated:
        flux_e = np.nan
    return {
        "image": image,
        "mean": mean,
        "variance": variance,
        "saturation_mask": mask,
        "flux_e": flux_e,
        "sigma_e": sigma_e,
        "saturated": saturated,
    }


def cmd_covariance(sigma_beta, sigma_rho):
    """Propagate independent band errors to (beta-rho, rho), including shared rho."""
    return np.array([[sigma_beta**2 + sigma_rho**2, -(sigma_rho**2)], [-(sigma_rho**2), sigma_rho**2]])


def observing_setup(args, ab, cfg):
    """PSF, sky and noise model for both bands: fixed by the observing conditions, not the SED."""
    psf_model = ZemaxPSF.from_config(cfg)
    noise = NoiseBudget(
        read_noise=cfg.telescope.readnoise * u.electron,
        dark_current=cfg.telescope.dark_current * u.electron / u.s,
    )
    # Illustrative components in V-band S10, not a survey/sky-position forecast.
    sky_s10 = args.sky_s10 or [145.0 * args.airmass, 60.0, 10.0, 0.0, 0.0]
    bands = {}
    for name, band in BANDS.items():
        angles = psf_model.field_angles(band)
        if not angles.min() <= args.field_angle <= angles.max():
            raise ValueError(f"--field-angle must lie within {angles.min():.3f}–{angles.max():.3f} deg for {name}.")
        psf, hfd = psf_model.with_seeing(
            seeing=args.seeing, band=band, field_angle_deg=args.field_angle, pixel_phase=tuple(args.pixel_phase)
        )
        if not np.isfinite(psf).all() or psf.min() < -1e-12 or psf.sum() <= 0:
            raise ValueError(f"Invalid {name} PSF.")
        psf = np.maximum(psf, 0.0)
        psf /= psf.sum()
        sky = SpectralSky(ab.tp, ab, band, throughputs_dir=args.throughputs_dir)
        sky_rate = float(sky.sky_electrons(*sky_s10, transparency=args.transparency)) * cfg.telescope.plate_scale**2
        bands[name] = {
            "psf": psf,
            "hfd": float(hfd),
            "angle": float(angles[np.argmin(abs(angles - args.field_angle))]),
            "sky_rate": sky_rate,
        }
    return {"noise": noise, "sky_s10": sky_s10, "bands": bands}


def simulate(args, wave, flux, ab, cfg, setup=None, seed=None):
    """Predict both bands and draw paired, fixed-condition point-source exposures."""
    setup = setup or observing_setup(args, ab, cfg)
    noise, sky_s10 = setup["noise"], setup["sky_s10"]
    cadence = args.cadence_s if args.cadence_s is not None else args.exptime
    times = (np.arange(args.n_exposures) * cadence + args.exptime / 2.0) / 60.0
    seed = np.random.SeedSequence(args.seed) if seed is None else seed
    rngs = [np.random.default_rng(s) for s in seed.spawn(2)]
    results = {}
    for (name, band), rng in zip(BANDS.items(), rngs):
        psf, sky_rate = setup["bands"][name]["psf"], setup["bands"][name]["sky_rate"]
        rate = ab.electrons_from_sed(band, wave, flux, airmass=args.airmass, transparency=args.transparency)
        if rate.value <= 0:
            raise ValueError(f"The SED must have positive in-band flux in {name}.")
        mag = float(ab.mag_from_sed(band, wave, flux))
        snr, budget = noise.get_snr(rate, psf, sky_rate * u.electron / u.s, args.exptime * u.s)
        source_e = rate.to_value(u.electron / u.s) * args.exptime
        background_e = (sky_rate + cfg.telescope.dark_current) * args.exptime
        measurements, chosen = [], None
        for epoch in range(args.n_exposures):
            exposure = draw_exposure(psf, source_e, background_e, cfg.telescope.readnoise, cfg.telescope.fullwell, rng)
            fitted, sigma = exposure["flux_e"], exposure["sigma_e"]
            detected = np.isfinite(fitted) and fitted / sigma >= 5.0
            measurements.append(
                {
                    "flux_e": fitted,
                    "sigma_flux_e": sigma,
                    "mag": mag - 2.5 * np.log10(fitted / source_e) if detected else np.nan,
                    "sigma_mag": MAG_FACTOR * sigma / fitted if detected else np.nan,
                    "saturated": exposure["saturated"],
                    "detected": bool(detected),
                }
            )
            if epoch == args.epoch_index:
                chosen = exposure
        results[name] = {
            "band": band,
            "mag": mag,
            "rate_e_s": rate.value,
            "source_e": source_e,
            "sky_e_s_pixel": sky_rate,
            "background_e_pixel": background_e,
            "psf": psf,
            "hfd_arcsec": setup["bands"][name]["hfd"],
            "sampled_field_angle_deg": setup["bands"][name]["angle"],
            "noise_budget_snr": float(snr),
            "noise_budget": budget,
            "fit_snr": source_e / chosen["sigma_e"],
            "sigma_mag_forecast": MAG_FACTOR * chosen["sigma_e"] / source_e,
            "selected_saturated": chosen["saturated"],
            "measurements": measurements,
            "exposure": chosen,
        }
    return {"bands": results, "time_min": times, "sky_s10": sky_s10}


def apply_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#86939e",
            "axes.labelcolor": "#253746",
            "xtick.color": "#435766",
            "ytick.color": "#435766",
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
        }
    )


def figure_axes(fig, margins=None):
    """The 70/30 by 60/40 linked layout: CMD, light/colour curves, exposure, spectrum.

    margins (left, right, bottom, top) pins the layout for movie frames, where a
    constrained layout would shift the panels as tick labels change.
    """
    if margins is None:
        grid = fig.add_gridspec(2, 2, width_ratios=[7, 3], height_ratios=[6, 4], wspace=0.07, hspace=0.08)
    else:
        left, right, bottom, top = margins
        grid = fig.add_gridspec(
            2, 2, width_ratios=[7, 3], height_ratios=[6, 4], wspace=0.2, hspace=0.32,
            left=left, right=right, bottom=bottom, top=top,
        )
    cmd, image_ax, sed_ax = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1]))
    light_grid = grid[1, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.02 if margins is None else 0.08)
    lc = fig.add_subplot(light_grid[0])
    colour_ax = fig.add_subplot(light_grid[1], sharex=lc)
    return cmd, lc, colour_ax, image_ax, sed_ax


def draw_image_panel(fig, image_ax, args, shown, cfg):
    """Synthetic detector stamp of the selected exposure in one band."""
    img = shown["exposure"]["image"]
    ny, nx = img.shape
    scale = cfg.telescope.plate_scale
    extent = np.array([-nx / 2, nx / 2, -ny / 2, ny / 2]) * scale
    norm = AsinhNorm(
        linear_width=max(1, np.sqrt(shown["background_e_pixel"] + cfg.telescope.readnoise**2)),
        vmin=float(min(0, img.min())),
        vmax=float(img.max()),
    )
    plotted = image_ax.imshow(img, origin="lower", interpolation="nearest", cmap="magma", norm=norm, extent=extent)
    image_ax.set_title(f"Synthetic exposure · {SYMBOLS[args.image_band]}", loc="left", weight="bold")
    image_ax.set(xlabel="Offset (arcsec)", ylabel="Offset (arcsec)")
    colorbar = fig.colorbar(plotted, ax=image_ax, shrink=0.8, pad=0.025, label="electrons / pixel")
    colorbar.set_ticks(norm.inverse(np.linspace(0, 1, 5)))
    colorbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    state = "SATURATED" if shown["exposure"]["saturated"] else f"Expected S/N {shown['fit_snr']:.1f}"
    image_ax.text(
        0.035,
        0.965,
        f"{args.exptime:g} s  ·  {state}\nHFD {shown['hfd_arcsec']:.2f} arcsec",
        transform=image_ax.transAxes,
        va="top",
        fontsize=9,
        color="white",
        bbox={"facecolor": "#111726", "alpha": 0.65, "edgecolor": "none", "pad": 5},
    )


def draw_spectrum_panel(sed_ax, args, wave, flux, ab, ylim=None):
    """Input SED over the Argus wavelength grid with both detected responses; ylim switches to log flux."""
    nm = wave.to_value(u.nm)
    flam = flux.to_value(FLAM, equivalencies=u.spectral_density(wave))
    active = (nm >= ab.tp.wav[0].to_value(u.nm)) & (nm <= ab.tp.wav[-1].to_value(u.nm))
    sed_ax.plot(nm[active], flam[active], color="#213e52", lw=1.2, label="Input SED")
    sed_ax.set(
        xlabel="Wavelength (nm)",
        ylabel=r"$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)",
        xlim=(ab.tp.wav[0].to_value(u.nm), ab.tp.wav[-1].to_value(u.nm)),
    )
    sed_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    sed_ax.set_title("Spectrum at the CMD point", loc="left", weight="bold")
    response_ax = sed_ax.twinx()
    response_ax.spines["right"].set_visible(True)
    for name, band in BANDS.items():
        response = system_response(ab.tp, band, args.airmass) * args.transparency
        response_ax.fill_between(ab.tp.wav.to_value(u.nm), response, color=COLORS[name], alpha=0.12)
        response_ax.plot(ab.tp.wav.to_value(u.nm), response, color=COLORS[name], alpha=0.8, lw=0.8)
    response_ax.set(ylim=(0, 1), ylabel="Detected response")
    sed_ax.text(0.03, 0.97, "β", transform=sed_ax.transAxes, va="top", color=COLORS["beta"], weight="bold")
    sed_ax.text(0.10, 0.97, "ρ", transform=sed_ax.transAxes, va="top", color=COLORS["rho"], weight="bold")
    sed_ax.grid(alpha=0.12)
    if ylim is not None:
        sed_ax.set(yscale="log", ylim=ylim)


def plot_figure(args, wave, flux, metadata, result, ab, cfg):
    """Build the requested 70/30 by 60/40 linked panel layout."""
    apply_style()
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    cmd, lc, colour_ax, image_ax, sed_ax = figure_axes(fig)
    data = result["bands"]
    b, r = data["beta"], data["rho"]
    truth_colour = b["mag"] - r["mag"]
    modulus = 5 * np.log10(args.distance_pc / 10.0) if args.absolute else 0.0
    truth_y = r["mag"] - modulus
    uncertainty = cmd_covariance(b["sigma_mag_forecast"], r["sigma_mag_forecast"])
    usable = all(d["fit_snr"] >= 5 and not d["exposure"]["saturated"] for d in (b, r))
    if usable:
        values, vectors = np.linalg.eigh(uncertainty)
        major = vectors[:, 1]
        radii = np.sqrt(values * (-2 * np.log(1 - 0.682689492)))
        cmd.add_patch(
            Ellipse(
                (truth_colour, truth_y),
                2 * radii[1],
                2 * radii[0],
                angle=np.degrees(np.arctan2(major[1], major[0])),
                facecolor="#deedf4",
                edgecolor="#74a7be",
                lw=1.4,
                label="68% joint forecast",
            )
        )
    cmd.scatter(
        truth_colour,
        truth_y,
        marker="*",
        s=190,
        color="#d9a33b",
        edgecolor="#755921",
        linewidth=0.8,
        zorder=5,
        label="Input SED",
    )
    selected = args.epoch_index
    bmag = np.array([m["mag"] for m in b["measurements"]])
    rmag = np.array([m["mag"] for m in r["measurements"]])
    berr = np.array([m["sigma_mag"] for m in b["measurements"]])
    rerr = np.array([m["sigma_mag"] for m in r["measurements"]])
    valid = np.isfinite(bmag) & np.isfinite(rmag)
    if args.n_exposures > 1:
        cmd.scatter(
            (bmag - rmag)[valid], (rmag - modulus)[valid], s=15, color="#a8b8c4", alpha=0.6, label="Repeated fixed SED"
        )
    if valid[selected]:
        cmd.scatter(
            bmag[selected] - rmag[selected],
            rmag[selected] - modulus,
            s=58,
            color="#203f54",
            zorder=6,
            label="Shown exposure",
        )
    dx = max(0.08, 5 * np.sqrt(uncertainty[0, 0])) if usable else 0.5
    dy = max(0.16, 5 * np.sqrt(uncertainty[1, 1])) if usable else 1.0
    cmd.set(
        xlim=(truth_colour - dx, truth_colour + dx),
        ylim=(truth_y + dy, truth_y - dy),
        xlabel=r"$\beta-\rho$ (AB mag)",
        ylabel=r"$M_\rho$ (AB mag)" if args.absolute else r"$m_\rho$ (AB mag)",
    )
    cmd.set_title("Colour–magnitude diagram", loc="left", weight="bold")
    cmd.grid(alpha=0.15)
    cmd.legend(loc="upper right", frameon=False, fontsize=10)
    cmd.text(
        0.025,
        0.97,
        f"INPUT SED\nβ = {b['mag']:.4f} AB\nρ = {r['mag']:.4f} AB\nβ − ρ = {truth_colour:.4f} mag",
        transform=cmd.transAxes,
        va="top",
        linespacing=1.5,
        fontsize=11,
        color="#3c5262",
    )
    cmd.text(0.025, 0.035, "One stellar state · zoomed view", transform=cmd.transAxes, color="#6d7d88", fontsize=10)

    times = result["time_min"]
    for name, mags, errs in (("beta", bmag, berr), ("rho", rmag, rerr)):
        ok = np.isfinite(mags)
        lc.errorbar(
            times[ok],
            mags[ok],
            yerr=errs[ok],
            fmt="o",
            ms=4,
            capsize=2,
            color=COLORS[name],
            label=SYMBOLS[name],
            lw=0.9,
        )
        if args.n_exposures > 1:
            lc.axhline(data[name]["mag"], color=COLORS[name], ls="--", lw=0.8, alpha=0.7)
        if ok[selected]:
            lc.plot(times[selected], mags[selected], "o", ms=8, mfc="none", mec=COLORS[name], mew=1.5)
    colour_ax.errorbar(
        times[valid],
        (bmag - rmag)[valid],
        yerr=np.hypot(berr, rerr)[valid],
        fmt="o",
        ms=4,
        color=COLORS["colour"],
        capsize=2,
        lw=0.9,
    )
    if args.n_exposures > 1:
        colour_ax.axhline(truth_colour, ls="--", lw=0.8, color=COLORS["colour"], alpha=0.7)
        for ax in (lc, colour_ax):
            ax.axvline(times[selected], color="#455868", lw=0.8, alpha=0.35)
    lc.set_title("Light and colour curves", loc="left", weight="bold")
    lc.set_ylabel("AB magnitude")
    lc.invert_yaxis()
    lc.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2, frameon=False, borderaxespad=0)
    plt.setp(lc.get_xticklabels(), visible=False)
    colour_ax.set(xlabel="Elapsed observing time (min)", ylabel=r"$\beta-\rho$ (mag)")
    padding = max(0.5, float(np.ptp(times)) * 0.04)
    colour_ax.set_xlim(times[0] - padding, times[-1] + padding)
    for ax in (lc, colour_ax):
        ax.grid(alpha=0.15)
    if not np.any(valid):
        colour_ax.set_ylim(truth_colour - 0.1, truth_colour + 0.1)
        colour_ax.text(
            0.5,
            0.5,
            "No valid colour: saturation or <5σ measurement",
            ha="center",
            va="center",
            transform=colour_ax.transAxes,
            color="#a84634",
            fontsize=10,
        )
    if not np.isfinite(bmag).any() and not np.isfinite(rmag).any():
        lc.set_ylim(max(b["mag"], r["mag"]) + 0.1, min(b["mag"], r["mag"]) - 0.1)
        lc.text(
            0.5,
            0.5,
            "Saturation or <5σ: magnitudes omitted",
            ha="center",
            va="center",
            transform=lc.transAxes,
            color="#a84634",
            fontsize=10,
        )

    draw_image_panel(fig, image_ax, args, data[args.image_band], cfg)

    draw_spectrum_panel(sed_ax, args, wave, flux, ab)
    title = args.title or metadata.get("title") or Path(metadata["path"]).stem
    mode = "one epoch" if args.n_exposures == 1 else f"{args.n_exposures} repeats of one fixed SED"
    fig.suptitle(
        f"ARGUS · {title}\n{mode}  |  {args.exptime:g} s  |  airmass {args.airmass:g}  |  "
        f"seeing {args.seeing:g}″  |  field {args.field_angle:g}°",
        fontsize=15,
        weight="medium",
    )
    return fig


def find_seds(folder):
    """SED CSVs in a folder, one per stellar state, in evolutionary order: [(key, path)], key label.

    A MESA run folder (runs/<track>) is searched in its SED/ subfolder. MESA writes the same
    SED once per filter (<filter>_SED_<model>.csv), so one file per model is kept (beta's,
    when present) and models are ordered by number. Other CSVs are ordered by the trailing
    number in their names, or naturally by name when there is none.
    """
    folder = Path(folder).expanduser()
    if (folder / "SED").is_dir() and not any(folder.glob("*.csv")):
        folder = folder / "SED"
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and not p.name.endswith(".photometry.csv")
    )
    if not files:
        raise ValueError(f"No CSV files in {folder}.")
    names = [MESA_NAME.match(p.stem) for p in files]
    if all(names):
        by_model = {}
        for path, name in zip(files, names):
            model = int(name["model"])
            if model not in by_model or name["filter"] == "beta":
                by_model[model] = path
        return [(model, by_model[model]) for model in sorted(by_model)], "Model number"
    files.sort(key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)])
    numbers = [re.search(r"(\d+)$", p.stem) for p in files]
    if all(numbers) and len({int(n[1]) for n in numbers}) == len(files):
        return [(int(n[1]), p) for n, p in zip(numbers, files)], "File number"
    return list(enumerate(files, start=1)), "Frame"


def mesa_history(sed_dir):
    """Model number -> (star_age yr, Teff K, log g) from the run's LOGS/history.data, if present."""
    history = Path(sed_dir).parent / "LOGS" / "history.data"
    if Path(sed_dir).name != "SED" or not history.is_file():
        return {}
    h = np.genfromtxt(history, skip_header=5, names=True)
    return {
        int(m): (age, 10.0**teff, logg)
        for m, age, teff, logg in zip(h["model_number"], h["star_age"], h["log_Teff"], h["log_g"])
    }


def format_age(years):
    """Star age at a readable scale."""
    for unit, scale in (("Gyr", 1e9), ("Myr", 1e6), ("kyr", 1e3)):
        if years >= scale:
            return f"{years / scale:.4g} {unit}"
    return f"{years:.4g} yr"


def make_movie(args, frames, key_label, ab, cfg):
    """Observe every SED in turn and write one linked four-panel frame per stellar state.

    Pass 1 reads each SED and its true AB magnitudes, which fix every axis for the whole
    movie. Pass 2 draws one exposure per state under the same conditions (fresh noise for
    each) and renders it: the CMD shows the evolutionary track traced so far, the light curve
    the measured magnitudes against model number.
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if args.model_range is not None:
        low, high = args.model_range
        frames = [f for f in frames if low <= f[0] <= high]
    if len(frames) < 2:
        raise ValueError(f"A movie needs at least two SEDs; {len(frames)} selected.")
    if args.max_frames and len(frames) > args.max_frames:
        keep = np.unique(np.round(np.linspace(0, len(frames) - 1, args.max_frames)).astype(int))
        print(f"{len(frames)} SEDs; rendering {len(keep)} evenly spaced in {key_label.lower()} (--max-frames 0 for all)")
        frames = [frames[i] for i in keep]

    history = mesa_history(frames[0][1].parent)
    states, skipped = [], []
    for key, path in frames:
        try:
            wave, flux, metadata = prepare_sed(args, ab, path)
            mags = {name: float(ab.mag_from_sed(band, wave, flux)) for name, band in BANDS.items()}
        except (ValueError, u.UnitConversionError) as exc:
            skipped.append(f"{path.name}: {exc}")
            continue
        states.append({"key": key, "path": path, "wave": wave, "flux": flux, "metadata": metadata, "mag": mags})
    if skipped:
        warnings.warn(f"Skipped {len(skipped)} unusable SEDs, e.g. {skipped[0]}", stacklevel=1)
    if len(states) < 2:
        raise ValueError("Fewer than two usable SEDs.")

    modulus = 5 * np.log10(args.distance_pc / 10.0) if args.absolute else 0.0
    keys = np.array([s["key"] for s in states], dtype=float)
    true = {name: np.array([s["mag"][name] for s in states]) for name in BANDS}
    true_colour, true_y = true["beta"] - true["rho"], true["rho"] - modulus

    def padded(values, minimum):
        low, high = np.nanmin(values), np.nanmax(values)
        pad = max(minimum, 0.06 * (high - low))
        return low - pad, high + pad

    fixed = {"colour": padded(true_colour, 0.05), "y": padded(true_y, 0.1), "lc": padded(np.r_[true["beta"], true["rho"]], 0.1)}
    lo, hi = ab.tp.wav[0].to_value(u.nm), ab.tp.wav[-1].to_value(u.nm)
    peaks, floors = [], []
    for state in states:
        nm = state["wave"].to_value(u.nm)
        flam = state["flux"].to_value(FLAM)[(nm >= lo) & (nm <= hi)]
        peaks.append(flam.max())
        floors.append(np.percentile(flam[flam > 0], 5) if np.any(flam > 0) else flam.max())
    fixed["sed"] = (max(min(floors) / 2, max(peaks) * 1e-6), max(peaks) * 2)

    setup = observing_setup(args, ab, cfg)
    seeds = np.random.SeedSequence(args.seed).spawn(len(states))
    if args.output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
    elif FFMpegWriter.isAvailable():
        writer = FFMpegWriter(
            fps=args.fps,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
    else:
        raise ValueError("ffmpeg not found on PATH; install it or write a .gif instead.")
    apply_style()
    fig = plt.figure(figsize=(16, 10))
    rows = []
    print(f"Rendering {len(states)} frames to {args.output}")
    with writer.saving(fig, str(args.output), dpi=args.dpi):
        for i, state in enumerate(states):
            result = simulate(args, state["wave"], state["flux"], ab, cfg, setup=setup, seed=seeds[i])
            row = {key_label.lower().replace(" ", "_"): state["key"]}
            if state["key"] in history:
                row.update(zip(("star_age_yr", "teff_K", "log_g"), history[state["key"]]))
            for name in BANDS:
                d = result["bands"][name]
                m = d["measurements"][0]
                row.update({
                    f"{name}_true_mag": d["mag"], f"{name}_mag": m["mag"], f"{name}_sigma_mag": m["sigma_mag"],
                    f"{name}_rate_e_s": d["rate_e_s"], f"{name}_saturated": m["saturated"],
                })
            rows.append(row)
            fig.clear()
            draw_movie_frame(fig, args, states, i, rows, result, key_label, history, fixed, modulus, ab, cfg)
            writer.grab_frame()
            if (i + 1) % max(1, len(states) // 10) == 0 or i + 1 == len(states):
                print(f"  {i + 1}/{len(states)}", flush=True)
    plt.close(fig)
    if args.save_data:
        with args.output.with_suffix(".photometry.csv").open("w", newline="") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(rows)
    saturated = sum(r["beta_saturated"] or r["rho_saturated"] for r in rows)
    print(f"Saved {args.output}: {len(rows)} frames at {args.fps:g} fps")
    if saturated:
        warnings.warn(
            f"{saturated}/{len(rows)} frames saturated in at least one band (magnitudes omitted there). "
            "Move the star further away (--distance-pc) or shorten --exptime.",
            stacklevel=1,
        )


def draw_movie_frame(fig, args, states, i, rows, result, key_label, history, fixed, modulus, ab, cfg):
    """One movie frame: evolutionary CMD and light curve so far, this state's exposure and spectrum."""
    cmd, lc, colour_ax, image_ax, sed_ax = figure_axes(fig, margins=(0.05, 0.95, 0.06, 0.88))
    state = states[i]
    key_name = key_label.lower().replace(" ", "_")
    keys = np.array([r[key_name] for r in rows], dtype=float)
    all_keys = np.array([s["key"] for s in states], dtype=float)
    true = {name: np.array([s["mag"][name] for s in states]) for name in BANDS}
    true_colour, true_y = true["beta"] - true["rho"], true["rho"] - modulus
    meas = {name: np.array([r[f"{name}_mag"] for r in rows]) for name in BANDS}
    err = {name: np.array([r[f"{name}_sigma_mag"] for r in rows]) for name in BANDS}
    b, r = result["bands"]["beta"], result["bands"]["rho"]

    cmd.plot(true_colour, true_y, color="#c9d3da", lw=1.0, zorder=1, label="Full track (input SEDs)")
    cmd.plot(true_colour[: i + 1], true_y[: i + 1], color="#5d7d93", lw=1.6, zorder=2, label="Track so far")
    valid = np.isfinite(meas["beta"]) & np.isfinite(meas["rho"])
    cmd.scatter((meas["beta"] - meas["rho"])[valid], (meas["rho"] - modulus)[valid], s=9, color="#a8b8c4",
                alpha=0.7, zorder=3, label="Measured so far")
    if valid[i]:
        cmd.errorbar(meas["beta"][i] - meas["rho"][i], meas["rho"][i] - modulus,
                     xerr=np.hypot(err["beta"][i], err["rho"][i]), yerr=err["rho"][i],
                     fmt="o", ms=7, color="#203f54", capsize=2, zorder=6, label="This exposure")
    cmd.scatter(true_colour[i], true_y[i], marker="*", s=230, color="#d9a33b", edgecolor="#755921",
                linewidth=0.8, zorder=5, label="Input SED")
    cmd.set(xlim=fixed["colour"], ylim=(fixed["y"][1], fixed["y"][0]), xlabel=r"$\beta-\rho$ (AB mag)",
            ylabel=r"$M_\rho$ (AB mag)" if args.absolute else r"$m_\rho$ (AB mag)")
    cmd.set_title("Colour–magnitude diagram", loc="left", weight="bold")
    cmd.grid(alpha=0.15)
    # the track can pass under any corner: the legend sits beneath the current state, and the
    # state itself is read out in the headings rather than in a box over the data
    legend = cmd.legend(loc="lower right", fontsize=10, framealpha=0.85, edgecolor="none")
    legend.set_zorder(4)
    cmd.set_title(f"β {b['mag']:.3f}   ρ {r['mag']:.3f}   β−ρ {b['mag'] - r['mag']:.3f}  (input SED, AB)",
                  loc="right", fontsize=11, color="#3c5262")

    for name in BANDS:
        lc.plot(all_keys, true[name], color=COLORS[name], lw=0.8, alpha=0.3)
        ok = np.isfinite(meas[name])
        lc.errorbar(keys[ok], meas[name][ok], yerr=err[name][ok], fmt="o", ms=3, capsize=0, lw=0.7,
                    color=COLORS[name], label=SYMBOLS[name])
        lc.plot(keys[i], true[name][i], "o", ms=9, mfc="none", mec=COLORS[name], mew=1.5)
    colour_ax.plot(all_keys, true["beta"] - true["rho"], color=COLORS["colour"], lw=0.8, alpha=0.3)
    colour_ax.errorbar(keys[valid], (meas["beta"] - meas["rho"])[valid],
                       yerr=np.hypot(err["beta"], err["rho"])[valid], fmt="o", ms=3, lw=0.7,
                       color=COLORS["colour"])
    for ax in (lc, colour_ax):
        ax.axvline(keys[i], color="#455868", lw=0.8, alpha=0.35)
        ax.grid(alpha=0.15)
    lc.set_title("Light and colour curves", loc="left", weight="bold")
    lc.set(ylabel="AB magnitude", ylim=(fixed["lc"][1], fixed["lc"][0]))
    lc.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2, frameon=False, borderaxespad=0)
    plt.setp(lc.get_xticklabels(), visible=False)
    span = all_keys[-1] - all_keys[0]
    colour_ax.set(xlabel=key_label, ylabel=r"$\beta-\rho$ (mag)", ylim=fixed["colour"],
                  xlim=(all_keys[0] - 0.02 * span, all_keys[-1] + 0.02 * span))

    draw_image_panel(fig, image_ax, args, result["bands"][args.image_band], cfg)
    draw_spectrum_panel(sed_ax, args, state["wave"], state["flux"], ab, ylim=fixed["sed"])
    title = args.title or (state["path"].parent.parent.name if state["path"].parent.name == "SED" else state["path"].parent.name)
    distance = f"  |  {args.distance_pc:g} pc" if args.distance_pc else ""
    physical = ""
    if state["key"] in history:
        age, teff, logg = history[state["key"]]
        physical = f" · {format_age(age)} · {teff:.0f} K · log g {logg:.2f}"
    fig.suptitle(
        f"ARGUS · {title} · {key_label.lower()} {state['key']}{physical}  ({i + 1}/{len(states)})\n"
        f"one {args.exptime:g} s exposure per state{distance}  |  airmass {args.airmass:g}  |  "
        f"seeing {args.seeing:g}″  |  field {args.field_angle:g}°",
        fontsize=15,
        weight="medium",
    )


def save_data(output, args, metadata, result, cfg):
    """Save numerical photometry, both selected detector stamps, and run provenance."""
    rows = []
    for i, time in enumerate(result["time_min"]):
        row = {"epoch": i, "time_min": float(time)}
        for name in BANDS:
            row.update({f"{name}_{key}": value for key, value in result["bands"][name]["measurements"][i].items()})
        row["beta_minus_rho"] = row["beta_mag"] - row["rho_mag"]
        row["sigma_colour"] = float(np.hypot(row["beta_sigma_mag"], row["rho_sigma_mag"]))
        rows.append(row)
    with output.with_suffix(".photometry.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    hdus = [fits.PrimaryHDU()]
    hdus[0].header["EXPTIME"] = args.exptime
    hdus[0].header["AIRMASS"] = args.airmass
    hdus[0].header["SEEING"] = args.seeing
    hdus[0].header["EPOCH"] = args.epoch_index
    hdus[0].header["SEED"] = args.seed
    hdus[0].header["PIXSCALE"] = cfg.telescope.plate_scale
    hdus[0].header["FULLWELL"] = cfg.telescope.fullwell
    hdus[0].header["READNOIS"] = (cfg.telescope.readnoise, "electron RMS per pixel")
    hdus[0].header["DARKRATE"] = (cfg.telescope.dark_current, "electron/s/pixel")
    for name, d in result["bands"].items():
        for key, values, unit in (
            ("IMAGE", d["exposure"]["image"], "electron"),
            ("MEAN", d["exposure"]["mean"], "electron"),
            ("VAR", d["exposure"]["variance"], "electron2"),
            ("PSF", d["psf"], "1"),
            ("SAT", d["exposure"]["saturation_mask"].astype(np.uint8), "1"),
        ):
            hdu = fits.ImageHDU(values, name=f"{name.upper()}_{key}")
            hdu.header["BUNIT"] = unit
            hdu.header["SRCRATE"] = d["rate_e_s"]
            hdu.header["SKYRATE"] = d["sky_e_s_pixel"]
            hdus.append(hdu)
    fits.HDUList(hdus).writeto(output.with_suffix(".fits"), overwrite=True)
    summary = {
        "argus_sim_version": ARGUS_VERSION,
        "input": metadata,
        "input_sha256": hashlib.sha256(Path(metadata["path"]).read_bytes()).hexdigest(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "magnitude_system": "synthetic AB through the filter (ABPhot.mag_from_sed); no atmosphere",
        "detected_rate": "ABPhot.electrons_from_sed: filter x optics x QE x atmosphere^airmass x transparency",
        "sky_s10": result["sky_s10"],
        "photometry": "known-PSF weighted fit; known background; truth-based variance; omit saturated/<5 sigma magnitudes",
        "instrument": {
            "collecting_area_cm2": cfg.telescope.collecting_area.value,
            "plate_scale_arcsec": cfg.telescope.plate_scale,
            "read_noise_e": cfg.telescope.readnoise,
            "dark_current_e_s_pixel": cfg.telescope.dark_current,
            "fullwell_e": cfg.telescope.fullwell,
            "psf_fits": cfg.telescope.psf_fits_path,
            "ee50_profile": cfg.telescope.ee50_profile_path,
        },
        "bands": {
            name: {key: value for key, value in d.items() if key not in ("psf", "exposure", "measurements")}
            for name, d in result["bands"].items()
        },
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def parser():
    """Construct the standalone command-line interface."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "input",
        type=Path,
        help="one SED CSV (e.g. MESA's runs/<track>/SED/<filter>_SED_<model>.csv) for an image, "
        "or a folder of them (runs/<track>, runs/<track>/SED) for a movie",
    )
    p.add_argument(
        "-o", "--output", type=Path,
        help="image (.png/.pdf/.svg; default sed_observation.png) or, for a folder, movie (.mp4/.gif; default <name>_argus.mp4)",
    )
    p.add_argument("--wavelength-column", help="CSV header name or zero-based index")
    p.add_argument("--flux-column", help="CSV header name or zero-based index")
    p.add_argument("--wavelength-unit", help="Astropy unit; default Angstrom or CSV metadata")
    p.add_argument("--flux-unit", help="Astropy spectral flux-density unit; default cgs F_lambda or CSV metadata")
    p.add_argument("--flux-kind", choices=["observer", "surface"], help="default observer or CSV metadata")
    p.add_argument("--radius-rsun", type=float, help="radius for surface-flux dilution")
    p.add_argument("--distance-pc", type=float, help="observer distance: rescales MESA (10 pc) SEDs, dilutes surface flux, sets M_rho for --absolute")
    p.add_argument("--absolute", action="store_true", help="show M_rho (needs a distance: --distance-pc, or MESA's 10 pc)")
    p.add_argument("--rho-mag", type=float, help="explicitly renormalize the SED to this apparent rho AB magnitude")
    p.add_argument("--exptime", type=float, default=60.0, help="seconds per exposure (default 60)")
    p.add_argument("--airmass", type=float, default=1.2)
    p.add_argument("--seeing", type=float, default=1.2, help="line-of-sight FWHM in arcsec; no further airmass scaling")
    p.add_argument("--transparency", type=float, default=1.0, help="grey transparency in (0,1]")
    p.add_argument("--field-angle", type=float, default=1.0, help="degrees from optical axis")
    p.add_argument("--pixel-phase", type=float, nargs=2, default=[0.0, 0.0], metavar=("Y", "X"))
    p.add_argument(
        "--sky-s10",
        type=float,
        nargs=5,
        metavar=("AIRGLOW", "ZODIACAL", "STARLIGHT", "MOON", "ARTIFICIAL"),
        help="fixed sky components; defaults: 145*airmass,60,10,0,0 (illustrative)",
    )
    p.add_argument("--n-exposures", type=int, default=1, help="repeat the SAME spectrum (default 1)")
    p.add_argument("--cadence-s", type=float, help="start-to-start spacing; defaults to exposure time")
    p.add_argument("--epoch-index", type=int, default=0, help="zero-based exposure shown in image and CMD (default 0)")
    p.add_argument("--image-band", choices=list(BANDS), default="rho")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--psf-fits", type=Path, help="override config's Zemax FITS path")
    p.add_argument("--ee50-profile", type=Path, help="override config's delivered optics profile")
    p.add_argument("--throughputs-dir", help="override ArgusSim's packaged throughput/template directory")
    p.add_argument("--title")
    p.add_argument("--dpi", type=int, help="default 180 for an image, 100 for a movie")
    p.add_argument("--fps", type=float, default=24, help="movie frames per second (default 24)")
    p.add_argument("--max-frames", type=int, default=600,
                   help="movie: evenly subsample to at most this many SEDs (default 600; 0 = all)")
    p.add_argument("--model-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                   help="movie: only models (or file numbers) in this inclusive range")
    p.add_argument("--save-data", action="store_true", help="also write FITS stamps, photometry CSV and metadata JSON")
    return p


def validate_args(args):
    """Reject invalid observing conditions and ambiguous physical scaling."""
    for name in ("exptime", "seeing", "transparency", "radius_rsun", "distance_pc", "cadence_s"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    if not np.isfinite(args.airmass) or args.airmass < 1 or args.transparency > 1:
        raise ValueError("Use airmass >= 1 and transparency in (0,1].")
    if not np.isfinite(args.field_angle) or args.field_angle < 0:
        raise ValueError("--field-angle must be non-negative and finite.")
    if args.sky_s10 is not None and (not np.isfinite(args.sky_s10).all() or min(args.sky_s10) < 0):
        raise ValueError("Sky components must be finite and non-negative.")
    if not all(np.isfinite(v) and 0 <= v < 1 for v in args.pixel_phase):
        raise ValueError("Pixel phases must lie in [0,1).")
    if args.n_exposures < 1 or not 0 <= args.epoch_index < args.n_exposures or args.seed < 0:
        raise ValueError("Invalid exposure count, selected epoch or seed.")
    if (args.dpi is not None and args.dpi < 1) or not np.isfinite(args.fps) or args.fps <= 0 or args.max_frames < 0:
        raise ValueError("Invalid DPI, --fps or --max-frames.")
    if args.cadence_s is not None and args.cadence_s < args.exptime:
        raise ValueError("--cadence-s must be at least --exptime.")
    if args.rho_mag is not None and (not np.isfinite(args.rho_mag) or args.distance_pc is not None):
        raise ValueError("--rho-mag must be finite and cannot be combined with a physical --distance-pc.")


def main(argv=None):
    """Run a single SED simulation and write the four linked panels."""
    p = parser()
    args = p.parse_args(argv)
    try:
        validate_args(args)
        cfg = copy.deepcopy(ARGUS_CONFIG)
        for option, field in ((args.psf_fits, "psf_fits_path"), (args.ee50_profile, "ee50_profile_path")):
            if option is not None:
                setattr(cfg.telescope, field, str(option.expanduser().resolve()))
        for field in ("psf_fits_path", "ee50_profile_path"):
            value = getattr(cfg.telescope, field)
            if value and not Path(value).is_file():
                raise ValueError(
                    f"Missing {field}: {value}. Supply --{'psf-fits' if field == 'psf_fits_path' else 'ee50-profile'}."
                )
        tp = SystemThroughput(
            throughputs_dir=args.throughputs_dir,
            filters={"g": True, "rho": True},
            throughput_loss=cfg.telescope.throughput_loss,
        )
        ab = ABPhot(throughput=tp, collecting_area=cfg.telescope.collecting_area)
        if args.input.expanduser().is_dir():
            frames, key_label = find_seds(args.input)
            if len(frames) > 1:
                if args.n_exposures != 1 or args.rho_mag is not None:
                    raise ValueError("A movie draws one exposure per state at a physical distance: "
                                     "drop --n-exposures and --rho-mag.")
                folder = frames[0][1].parent
                name = folder.parent.name if folder.name == "SED" else folder.name
                args.output = (args.output or Path(f"{name}_argus.mp4")).expanduser()
                if args.output.suffix.lower() not in (".mp4", ".gif"):
                    raise ValueError("A movie must be written as .mp4 or .gif.")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.dpi = args.dpi or 100
                make_movie(args, frames, key_label, ab, cfg)
                return 0
            args.input = frames[0][1]
        args.output = (args.output or Path("sed_observation.png")).expanduser()
        if args.output.suffix.lower() not in (".png", ".pdf", ".svg"):
            raise ValueError("Output extension must be .png, .pdf or .svg.")
        args.dpi = args.dpi or 180
        wave, flux, metadata = prepare_sed(args, ab)
        result = simulate(args, wave, flux, ab, cfg)
        fig = plot_figure(args, wave, flux, metadata, result, ab, cfg)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        plt.close(fig)
        if args.save_data:
            save_data(args.output, args, metadata, result, cfg)
        print(f"Saved {args.output}")
        for name, d in result["bands"].items():
            print(
                f"{name:4s}: {d['mag']:.6f} AB, {d['rate_e_s']:.3f} e-/s, "
                f"NoiseBudget S/N={d['noise_budget_snr']:.2f}, fit S/N={d['fit_snr']:.2f}"
            )
            if d["exposure"]["saturated"]:
                warnings.warn(
                    f"{name}: selected exposure saturated; fitted magnitude omitted. "
                    "Move the star further away (--distance-pc) or shorten --exptime.",
                    stacklevel=1,
                )
            elif d["fit_snr"] < 5:
                warnings.warn(f"{name}: expected S/N <5; inspect fluxes before interpreting magnitudes.", stacklevel=1)
    except (ValueError, OSError, u.UnitConversionError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
