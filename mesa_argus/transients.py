"""Transients for the night movies: SEDs against time, in the same spirit as the MESA RSP / PHOEBE pulsation.h5.

    python -m mesa_argus.transients --list                       # the sncosmo models, by type
    python -m mesa_argus.transients salt3 out.h5 --z 0.03 --t0-bjd 2461326.7 --peak-absmag -19.3 --ebv 0.05

sncosmo (Barbary et al.) models are spectral time series built from observed spectra of real supernovae (SALT2/3,
Hsiao, Nugent, Vincenzi et al. 2019, SNANA templates, ...): no physics is run. A model is placed at redshift z and
scaled so its rest-frame peak absolute magnitude in `peak_band` (sncosmo bandpass, `magsys`) is `peak_absmag`, with
sncosmo's luminosity distance (cosmology below); sncosmo applies the redshift to wavelength and time. Its SEDs are
then sampled on a fixed observer-frame grid and written as

    transient.h5: wavelength (A, observer frame), time_days (observer-frame days since t0), flux (f_lambda at the
    observer, erg/s/cm2/A, before Milky Way dust; per time step); attrs: kind, model, z, t0_bjd, ebv, ...

t0 is the model's phase 0 (sncosmo's convention): B-band maximum for SALT2/3 and Hsiao, the explosion for the Nugent
templates; each file records the observer-frame time of rest-frame B maximum (`peak_days`). Milky Way dust is the
field's E(B-V) (SFD), applied when the movie reads the file (mesa_argus.m67.redden, CCM89, as for every other star).
Before the model's first step the transient adds no light; a night after its last step is refused (no extrapolation).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WAVE_STEP = 5.0  # A, observer frame (the templates are tabulated at ~10 A)
TIME_STEP = 0.25  # observer-frame days between stored SEDs (the templates are tabulated at ~1 d; the movie interpolates linearly)
WAVE_RANGE = (3000.0, 10500.0)  # A: the SED panel's 300-1000 nm, within the model's own range
BAND_RANGE = (3600.0, 8300.0)  # A: where the Argus beta and rho system responses are non-zero (361-827 nm)


def catalogue():
    """[(name, version, type, reference)] of every sncosmo source."""
    from sncosmo.models import _SOURCES

    return [(m["name"], m["version"], m.get("type", ""), m.get("reference", ("", ""))[1]) for m in _SOURCES.get_loaders_metadata()]


def model_info(name, version=None):
    rows = [r for r in catalogue() if r[0] == name and (version is None or r[1] == version)]
    if not rows:
        raise SystemExit(f"no sncosmo model {name!r}" + (f" version {version}" if version else ""))
    return rows[-1]  # sncosmo's own default is the latest version


def make_sncosmo(out, name, z, t0_bjd, peak_absmag, ebv, peak_band="bessellb", magsys="ab", params=None, host_ebv=0.0,
                 version=None):
    """Write one sncosmo transient to `out` (transient.h5); returns its attrs."""
    import h5py
    import sncosmo

    name_, version_, kind, reference = model_info(name, version)
    effects, names, frames = ([sncosmo.CCM89Dust()], ["host"], ["rest"]) if host_ebv > 0 else ([], [], [])
    model = sncosmo.Model(source=sncosmo.get_source(name_, version=version_), effects=effects, effect_names=names,
                          effect_frames=frames)
    model.set(z=z, t0=0.0, **(params or {}))
    if host_ebv > 0:
        model.set(hostebv=host_ebv, hostr_v=3.1)
    # sncosmo's default cosmology for the distance (as argus_survey_model's tutorial)
    model.set_source_peakabsmag(peak_absmag, peak_band, magsys)
    lo, hi = max(model.minwave(), WAVE_RANGE[0]), min(model.maxwave(), WAVE_RANGE[1])
    if lo > BAND_RANGE[0] or hi < BAND_RANGE[1]:
        raise SystemExit(f"{name_} at z = {z} covers {model.minwave():.0f}-{model.maxwave():.0f} A (observer frame); the Argus "
                         f"bands need {BAND_RANGE[0]:.0f}-{BAND_RANGE[1]:.0f} A: not used")
    wave = np.arange(np.ceil(lo / WAVE_STEP) * WAVE_STEP, hi + 1e-6, WAVE_STEP)
    time = np.arange(model.mintime(), model.maxtime() + 1e-9, TIME_STEP)
    flux = np.asarray(model.flux(time, wave), float)
    peak_days = float(model.source.peakphase("bessellb")) * (1 + z)
    attrs = dict(kind=kind, model=name_, model_version=str(version_), reference=reference, z=float(z), t0_bjd=float(t0_bjd),
                 peak_absmag=float(peak_absmag), peak_band=peak_band, magsys=magsys, cosmology="sncosmo default (WMAP9)",
                 host_ebv=float(host_ebv), ebv=float(ebv), peak_days=peak_days,
                 params=", ".join(f"{k} = {model.get(k):.6g}" for k in model.param_names if k not in ("z", "t0")),
                 flux_units="erg s^-1 cm^-2 A^-1 (f_lambda at the observer, before Milky Way dust)",
                 time_note="observer-frame days since t0 (the model's phase 0; BJD_TDB)", source="sncosmo " + sncosmo.__version__)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("wavelength", data=wave)
        f.create_dataset("time_days", data=time)
        f.create_dataset("flux", data=flux, compression="gzip", compression_opts=4)
        f.attrs.update(attrs)
    return attrs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?")
    p.add_argument("out", nargs="?", type=Path)
    p.add_argument("--list", action="store_true")
    p.add_argument("--z", type=float)
    p.add_argument("--t0-bjd", type=float)
    p.add_argument("--peak-absmag", type=float)
    p.add_argument("--peak-band", default="bessellb")
    p.add_argument("--magsys", default="ab")
    p.add_argument("--ebv", type=float, default=0.0, help="Milky Way E(B-V) at the field (applied by the movie)")
    p.add_argument("--host-ebv", type=float, default=0.0)
    p.add_argument("--param", action="append", default=[], help="model parameter, e.g. x1=0.5 (repeatable)")
    args = p.parse_args(argv)
    if args.list:
        for name, version, kind, ref in sorted(catalogue(), key=lambda r: (r[2], r[0])):
            print(f"{kind:10s} {name:22s} {version:8s} {ref}")
        return
    params = {k: float(v) for k, v in (s.split("=") for s in args.param)}
    a = make_sncosmo(args.out, args.model, args.z, args.t0_bjd, args.peak_absmag, args.ebv, args.peak_band, args.magsys,
                     params, args.host_ebv)
    print(a)


if __name__ == "__main__":
    main()
