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
field's E(B-V) (SFD), applied when the movie reads the file (mesa_argus.dust.redden, CCM89, as for every other star).
Before the model's first step the transient adds no light; a night after its last step is refused (no extrapolation).

GRB optical afterglow (make_afterglow): the forward-shock synchrotron afterglow as a power law in time and frequency,
f_nu ∝ t^-alpha nu^-beta (Sari, Piran & Narayan 1998, ApJ 497, L17), with its onset (the shock's deceleration) as a
rise ∝ t^2 (Sari & Piran 1999, ApJ 520, 641), joined smoothly at t_peak:
    f_nu(t) = f_peak (nu / nu_ref)^-beta [((t/t_peak)^(-2 s) + (t/t_peak)^(alpha s)) / 2]^(-1/s),  s = 2,
normalised to the observed AB magnitude `peak_mag` at nu_ref (6400 A) at t_peak. t0 is the burst; the SEDs are
observer frame (no redshift is applied: the power law is the observed one), on a log-spaced time grid.

Microlensing (make_microlensing): a point lens magnifying a point source or a uniform disc (Paczynski 1986;
magnification), the source a MESA template SED; achromatic. Dwarf nova (make_dwarf_nova): a steady optically thick
accretion disc (Shakura & Sunyaev 1973) whose accretion rate rises and decays, plus the white dwarf as a blackbody.
The afterglow, microlensing and dwarf-nova models are parametric models written here, not sncosmo models.
"""

import argparse
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np

from argus_sim import ABPhot, SystemThroughput
from argus_sim import c as ARGUS_CONFIG
from mesa_argus import dust
from mesa_argus.seds import Templates

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
                 time_note="observer-frame days since t0 (the model's phase 0; BJD_TDB)", source="sncosmo " + sncosmo.__version__,
                 generator="sncosmo")
    return write(out, wave, time, flux, attrs)


AFTERGLOW_REF_A = 6400.0  # A: where peak_mag is given (R-band-like; within Argus rho)
AFTERGLOW_DAYS = 3.0  # the stored model's extent (observer-frame days after the burst)
AFTERGLOW_STEPS = 600  # log-spaced steps from 1 s to AFTERGLOW_DAYS (adjacent steps 2% apart in time)


def make_afterglow(out, t0_bjd, peak_mag, ebv, t_peak_s=60.0, alpha=1.0, beta=0.66, smooth=2.0):
    """Write a GRB optical afterglow (module docstring) to `out` (transient.h5); returns its attrs."""
    wave = np.arange(WAVE_RANGE[0], WAVE_RANGE[1] + 1e-6, WAVE_STEP)
    time = np.r_[0.0, np.geomspace(1.0 / 86400, AFTERGLOW_DAYS, AFTERGLOW_STEPS)]  # days; no light at the burst itself
    x = time[1:] * 86400 / t_peak_s
    shape = np.r_[0.0, (0.5 * (x ** (-2 * smooth) + x ** (alpha * smooth))) ** (-1 / smooth)]
    f_nu_peak = 10 ** (-0.4 * (peak_mag + 48.6))  # erg/s/cm2/Hz at AFTERGLOW_REF_A, t_peak
    f_nu = f_nu_peak * (wave / AFTERGLOW_REF_A) ** beta  # nu^-beta = (lambda / lambda_ref)^beta
    flux = shape[:, None] * (f_nu * 2.99792458e18 / wave**2)[None, :]  # f_lambda
    attrs = dict(kind="GRB afterglow", model="afterglow", model_version="power law", generator="power law",
                 reference="Sari, Piran & Narayan 1998; Sari & Piran 1999", z=float("nan"), t0_bjd=float(t0_bjd),
                 peak_mag=float(peak_mag), peak_ref_angstrom=AFTERGLOW_REF_A, t_peak_s=float(t_peak_s), alpha=float(alpha),
                 beta=float(beta), smooth=float(smooth), ebv=float(ebv), peak_days=float(t_peak_s / 86400),
                 params=f"peak {peak_mag:g} AB at {AFTERGLOW_REF_A:.0f} A at t_peak {t_peak_s:g} s, alpha {alpha:g}, "
                        f"beta {beta:g}",
                 flux_units="erg s^-1 cm^-2 A^-1 (f_lambda at the observer, before Milky Way dust)",
                 time_note="observer-frame days since the burst (BJD_TDB)")
    return write(out, wave, time, flux, attrs)


def write(out, wave, time, flux, attrs):
    """transient.h5 (module docstring)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("wavelength", data=wave)
        f.create_dataset("time_days", data=time)
        f.create_dataset("flux", data=flux, compression="gzip", compression_opts=4)
        f.attrs.update(attrs)
    return attrs


def rho_mag(wave, flam, ebv):
    """Argus rho AB magnitude (ArgusSim ABPhot, no airmass) of f_lambda (erg/s/cm2/A) after Milky Way dust (CCM89)."""
    cfg = ARGUS_CONFIG
    ab = ABPhot(throughput=SystemThroughput(filters={"g": True, "rho": True}, throughput_loss=cfg.telescope.throughput_loss),
                collecting_area=cfg.telescope.collecting_area)
    seen = dust.redden(wave, flam, ebv) * u.erg / (u.s * u.cm**2 * u.AA)
    return float(ab.mag_from_sed("rho", wave * u.AA, seen))


# ---------------------------------------------------------------- microlensing
def magnification(separation, rho=0.0):
    """Point-lens magnification (Paczynski 1986) at lens-source separation(s) in Einstein radii:
        point source:                 A(u) = (u^2 + 2) / (u sqrt(u^2 + 4))
        uniform disc of radius rho:   A(u) averaged over the disc,
            A = [pi a sqrt(a^2 + 4) + int 2 arccos((r^2 + u^2 - rho^2) / (2 r u)) r A(r) dr] / (pi rho^2)
    summed in rings of radius r about the lens: rings wholly inside the disc (r < a = rho - u) exactly, as
    int 2 pi r A(r) dr = pi a sqrt(a^2 + 4); rings the disc's edge crosses (|u - rho| < r < u + rho) by Gauss-Legendre,
    each weighted by the arc of it inside the disc. Exact at the singularity (lens inside the disc), where a plain
    average over the disc converges slowly. At u = 0: sqrt(rho^2 + 4) / rho."""
    separation = np.atleast_1d(np.asarray(separation, float))
    if rho <= 0:
        return (separation**2 + 2) / (separation * np.sqrt(separation**2 + 4))
    x, w = np.polynomial.legendre.leggauss(400)
    phi, w_phi = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w  # r = centre - half cos(phi) spreads the nodes to the ends
    out = np.empty_like(separation)
    for i, u_i in enumerate(separation):
        a = max(rho - u_i, 0.0)
        inside = np.pi * a * np.sqrt(a**2 + 4)  # the rings wholly inside the disc
        if u_i <= 0:
            out[i] = inside / (np.pi * rho**2)
            continue
        r_lo = abs(u_i - rho)  # the rings the disc's edge crosses: r_lo to u + rho
        centre, half = 0.5 * (r_lo + u_i + rho), 0.5 * (u_i + rho - r_lo)
        r = centre - half * np.cos(phi)
        arc = 2 * np.arccos(np.clip((r**2 + u_i**2 - rho**2) / (2 * r * u_i), -1, 1))  # angle of the ring in the disc
        crossed = np.sum(w_phi * half * np.sin(phi) * arc * (r**2 + 2) / np.sqrt(r**2 + 4))  # r A(r) = (r^2+2)/sqrt(r^2+4)
        out[i] = (inside + crossed) / (np.pi * rho**2)
    return out


MICROLENS_STEPS = 2001


def make_microlensing(out, t0_bjd, t_e, u0, base_mag, bp_rp, rho=0.0, fs=1.0):
    """Write a microlensing event (point lens; point source, or a uniform disc of radius rho Einstein radii) to `out`.

    The baseline is the MESA template SED of intrinsic BP-RP `bp_rp` (as the field stars: runs/*_sed), scaled so it is
    rho = base_mag (AB) unmagnified, before dust (the observation adds the dust). fs: the source's fraction of the
    baseline light (the rest is unlensed blended light; 1 = no blend). u(t) = sqrt(u0^2 + ((t - t0)/t_E)^2); the
    lensing is achromatic (no limb darkening). The file holds all the light at the position, (fs A + 1 - fs) x the
    baseline SED: the source (and its blend) is not otherwise in the field, so like every transient it is left out of
    the catalogue and the difference reference.
    """
    tpl = Templates()
    i = tpl.index(bp_rp)
    wave = np.arange(WAVE_RANGE[0], WAVE_RANGE[1] + 1e-6, WAVE_STEP)
    sed = np.interp(wave, tpl.wave.value, tpl.flux[i])
    sed *= 10 ** (-0.4 * (base_mag - rho_mag(wave, sed, 0.0)))
    # time grid: steps of ~width near t0 (where A changes fastest), growing to ~span away from it
    width = t_e * max(u0, rho, 1e-3)
    span = max(10 * t_e, 3.0)  # days either side of t0
    s = np.arcsinh(span / width)
    time = width * np.sinh(np.linspace(-s, s, MICROLENS_STEPS))
    A = magnification(np.sqrt(u0**2 + (time / t_e) ** 2), rho)
    flux = (fs * A + 1 - fs)[:, None] * sed[None, :]
    peak_mag = rho_mag(wave, sed * (fs * A.max() + 1 - fs), 0.0)
    attrs = dict(kind="microlensing", model="microlensing", model_version="point lens" + (", finite source" if rho else ""),
                 generator="point lens", reference="Paczynski 1986" + ("; uniform finite source" if rho else ""),
                 z=float("nan"), t0_bjd=float(t0_bjd), t_e_days=float(t_e), u0=float(u0), rho=float(rho), fs=float(fs),
                 base_mag=float(base_mag), bp_rp=float(bp_rp), template_bp_rp=float(tpl.colour[i]),
                 template_feh=float(tpl.feh[i]), ebv=0.0, peak_days=0.0, peak_magnification=float(A.max()),
                 peak_rho=peak_mag,
                 params=f"t_E {t_e:g} d, u0 {u0:g}, rho {rho:g}, fs {fs:g}; source rho {base_mag:g} (AB, before dust), "
                        f"BP-RP {bp_rp:g} (template {tpl.colour[i]:.2f}); peak A {A.max():.1f} (rho {peak_mag:.2f})",
                 flux_units="erg s^-1 cm^-2 A^-1 ((fs A + 1 - fs) x baseline; at the observer, before Milky Way dust)",
                 time_note="days since closest approach (BJD_TDB)")
    return write(out, wave, time, flux, attrs)


# ---------------------------------------------------------------- dwarf nova
G_CGS, SIGMA_SB, H_CGS, C_CGS, K_B = 6.674e-8, 5.6704e-5, 6.62607e-27, 2.99792458e10, 1.380649e-16
MSUN, RSUN, PC, YEAR = 1.98847e33, 6.957e10, 3.0857e18, 3.15576e7
DISC_RINGS = 300
DN_STEPS = 1200


def wd_radius(m_wd):
    """White-dwarf radius (cm) for mass m_wd (Msun): Nauenberg 1972, R = 0.0112 Rsun [(M/Mch)^-2/3 - (M/Mch)^2/3]^1/2,
    Mch = 1.44."""
    x = m_wd / 1.44
    return 0.0112 * RSUN * np.sqrt(x ** (-2 / 3) - x ** (2 / 3))


def planck(wave_a, t):
    """B_lambda (erg/s/cm2/A/sr) at wavelengths wave_a (A), temperatures t (K, broadcast)."""
    lam = wave_a * 1e-8
    with np.errstate(over="ignore"):
        return 2 * H_CGS * C_CGS**2 / lam**5 / np.expm1(H_CGS * C_CGS / (lam * K_B * t)) * 1e-8


def disc_flux(wave_a, mdot, m_wd, r_in, r_out, incl_deg, d_pc):
    """f_lambda at distance d of a steady, optically thick accretion disc (Shakura & Sunyaev 1973; Frank, King & Raine
    2002, eq. 5.43): blackbody rings with T(R)^4 = 3 G M Mdot / (8 pi sigma R^3) [1 - (r_in/R)^1/2] from r_in to r_out,
    seen at inclination i: F = cos i / d^2 int B(T) 2 pi R dR. Also returns the disc's luminosity (both faces)."""
    edges = np.geomspace(r_in, r_out, DISC_RINGS + 1)
    r, dr = np.sqrt(edges[1:] * edges[:-1]), np.diff(edges)
    t4 = 3 * G_CGS * m_wd * MSUN * mdot / (8 * np.pi * SIGMA_SB * r**3) * (1 - np.sqrt(r_in / r))
    t = np.maximum(t4, 0) ** 0.25
    area = 2 * np.pi * r * dr
    f = np.cos(np.radians(incl_deg)) / (d_pc * PC) ** 2 * np.sum(planck(wave_a[None, :], t[:, None]) * area[:, None], axis=0)
    return f, float(2 * np.sum(SIGMA_SB * t**4 * area))


def make_dwarf_nova(out, t0_bjd, d_pc, mdot_q, mdot_peak, t_rise, plateau, tau_decay, r_out, t_wd, ebv, m_wd=0.8,
                    incl=60.0, before=3.0, after=None):
    """Write a dwarf-nova outburst (disc instability) as a steady disc whose accretion rate changes with time, plus the
    white dwarf; to `out`.

    Mdot (Msun/yr) is mdot_q before t0; from t0 it rises exponentially to mdot_peak in t_rise days
    (Mdot = mdot_q (mdot_peak/mdot_q)^(t/t_rise)), stays there for `plateau` days, then decays as
    exp(-(t - t_rise - plateau)/tau_decay) down to mdot_q. The disc (disc_flux) runs from the white dwarf's surface
    (wd_radius) to r_out (Rsun) at inclination incl (deg) and distance d_pc. The white dwarf is a blackbody at t_wd, of
    radius wd_radius(m_wd), seen at d_pc. Not included: the donor star, the bright spot, the boundary layer (its light
    is in the far UV / X-rays), emission lines, superhumps. In quiescence a real disc is not steady-state (it is
    optically thin, with emission lines), so the quiescent light here is only an approximation.
    The file runs from `before` days before t0 to `after` (default: until Mdot is back within 1% of mdot_q).
    """
    r_in = wd_radius(m_wd)
    if r_out * RSUN <= r_in:
        raise SystemExit(f"r_out {r_out} Rsun is inside the white dwarf ({r_in / RSUN:.4f} Rsun)")
    wave = np.arange(WAVE_RANGE[0], WAVE_RANGE[1] + 1e-6, WAVE_STEP)
    end_peak = t_rise + plateau
    after = end_peak + tau_decay * np.log(100 * mdot_peak / mdot_q) if after is None else after
    time = np.unique(np.r_[np.linspace(-before, 0, 31), np.linspace(0, after, DN_STEPS)])

    def mdot_at(t):
        if t <= 0:
            return mdot_q
        if t <= t_rise:
            return mdot_q * (mdot_peak / mdot_q) ** (t / t_rise)
        if t <= end_peak:
            return mdot_peak
        return max(mdot_peak * np.exp(-(t - end_peak) / tau_decay), mdot_q)

    mdot = np.array([mdot_at(t) for t in time])
    wd = np.pi * planck(wave, t_wd) * (r_in / (d_pc * PC)) ** 2
    disc, lum = zip(*(disc_flux(wave, m * MSUN / YEAR, m_wd, r_in, r_out * RSUN, incl, d_pc) for m in mdot))
    flux = np.array(disc) + wd[None, :]
    rho_q, rho_p = (rho_mag(wave, flux[k], ebv) for k in (0, int(np.argmax(mdot))))
    attrs = dict(kind="dwarf nova", model="dwarf_nova", model_version="steady disc + white dwarf", generator="steady disc",
                 reference="Shakura & Sunyaev 1973; Frank, King & Raine 2002; Nauenberg 1972", z=float("nan"),
                 t0_bjd=float(t0_bjd), d_pc=float(d_pc), mdot_q=float(mdot_q), mdot_peak=float(mdot_peak),
                 t_rise=float(t_rise), plateau=float(plateau), tau_decay=float(tau_decay), r_out_rsun=float(r_out),
                 t_wd=float(t_wd), m_wd=float(m_wd), r_wd_rsun=float(r_in / RSUN), incl=float(incl), ebv=float(ebv),
                 peak_days=float(t_rise), rho_quiescent=rho_q, rho_peak=rho_p,
                 disc_luminosity_peak=float(max(lum)), mdot=mdot,
                 params=f"Mdot {mdot_q:g} -> {mdot_peak:g} Msun/yr (rise {t_rise:g} d, plateau {plateau:g} d, decay e-fold "
                        f"{tau_decay:g} d); M_wd {m_wd:g}, T_wd {t_wd:g} K, r_out {r_out:g} Rsun, i {incl:g} deg, "
                        f"d {d_pc:g} pc; rho {rho_q:.2f} quiescent, {rho_p:.2f} peak",
                 flux_units="erg s^-1 cm^-2 A^-1 (f_lambda at the observer, before Milky Way dust)",
                 time_note="days since the outburst starts (BJD_TDB)")
    return write(out, wave, time, flux, attrs)


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
