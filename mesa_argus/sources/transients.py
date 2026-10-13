"""Transient generators: SEDs against time since t0 (mesa_argus.transients does the physics)."""

from .. import transients as T
from . import REQUIRED, Generator, Param


def _supernova(out, p, field):
    return T.make_sncosmo(out / "transient.h5", p["model"], p["z"], p["t0_bjd"], p["M"], field.ebv, p["band"], p["magsys"],
                          {k: float(v) for k, v in p["params"].items()}, p["host_ebv"], p["version"] or None)


def _grb(out, p, field):
    return T.make_afterglow(out / "transient.h5", p["t0_bjd"], p["peak"], field.ebv, p["t_peak"], p["alpha"], p["beta"])


def _microlensing(out, p, field):
    return T.make_microlensing(out / "transient.h5", p["t0_bjd"], p["tE"], p["u0"], p["base"], p["bp_rp"], field.ebv,
                               p["rho"], p["fs"])


def _dwarf_nova(out, p, field):
    return T.make_dwarf_nova(out / "transient.h5", p["t0_bjd"], p["d"], p["mdot_q"], p["mdot_peak"], p["t_rise"],
                             p["plateau"], p["tau_decay"], p["r_out"], p["t_wd"], field.ebv, p["m_wd"], p["incl"])


GENERATORS = [
    Generator(
        "supernova", "transient", "sncosmo",
        "any sncosmo model (python -m mesa_argus.transients --list): spectral templates built from observed spectra, "
        "placed at redshift z and scaled to a rest-frame peak absolute magnitude (sncosmo's default cosmology, WMAP9)",
        dict(model=Param(str, REQUIRED, "sncosmo source name, e.g. salt3, hsiao, nugent-sn2p"),
             z=Param(float, REQUIRED, "redshift"),
             M=Param(float, REQUIRED, "rest-frame peak absolute magnitude in `band`"),
             band=Param(str, "bessellb", "sncosmo bandpass of M"),
             magsys=Param(str, "ab", "magnitude system of M"),
             host_ebv=Param(float, 0.0, "host-galaxy E(B-V) (CCM89, R_V 3.1, rest frame)"),
             version=Param(str, "", "sncosmo model version (default: sncosmo's)"),
             params=Param(dict, {}, "the model's own parameters, e.g. {x1 = 0, c = 0} for SALT2/3")),
        _supernova, "Barbary et al. (sncosmo); the model's own reference", "t0 is the model's phase 0"),
    Generator(
        "GRB", "transient", "power-law afterglow",
        "GRB optical afterglow: f_nu ∝ t^-alpha nu^-beta with a t^2 onset peaking at t_peak; observed frame, "
        "no redshift (written in this repo, mesa_argus/transients.py)",
        dict(peak=Param(float, REQUIRED, "observed AB magnitude at 6400 A at t_peak"),
             t_peak=Param(float, 60.0, "seconds after the burst"),
             alpha=Param(float, 1.0, "temporal decay index"),
             beta=Param(float, 0.66, "spectral index")),
        _grb, "Sari, Piran & Narayan 1998; Sari & Piran 1999", "t0 is the burst"),
    Generator(
        "microlensing", "transient", "point lens",
        "point-lens microlensing of a point or uniform-disc source, achromatic; the source a MESA template SED "
        "(written in this repo, mesa_argus/transients.py)",
        dict(tE=Param(float, REQUIRED, "Einstein time (days)"),
             u0=Param(float, REQUIRED, "impact parameter (Einstein radii)"),
             base=Param(float, REQUIRED, "the unmagnified source's rho (AB, after dust)"),
             bp_rp=Param(float, REQUIRED, "the source's intrinsic BP-RP (picks the MESA template)"),
             rho=Param(float, 0.0, "source radius (Einstein radii); 0: a point source"),
             fs=Param(float, 1.0, "the source's fraction of the baseline light (1: no blend)")),
        _microlensing, "Paczynski 1986", "t0 is the closest approach"),
    Generator(
        "dwarf nova", "transient", "steady disc",
        "dwarf-nova outburst: a steady optically thick accretion disc whose accretion rate rises, plateaus and decays, "
        "plus the white dwarf as a blackbody; no donor, bright spot or lines (written in this repo, "
        "mesa_argus/transients.py)",
        dict(d=Param(float, REQUIRED, "distance (pc)"),
             mdot_q=Param(float, REQUIRED, "quiescent accretion rate (Msun/yr)"),
             mdot_peak=Param(float, REQUIRED, "outburst accretion rate (Msun/yr)"),
             t_rise=Param(float, REQUIRED, "rise time (days, exponential in Mdot)"),
             plateau=Param(float, REQUIRED, "days at the peak rate"),
             tau_decay=Param(float, REQUIRED, "e-folding time of the decline (days)"),
             r_out=Param(float, REQUIRED, "disc outer radius (Rsun)"),
             t_wd=Param(float, REQUIRED, "white-dwarf temperature (K)"),
             m_wd=Param(float, 0.8, "white-dwarf mass (Msun)"),
             incl=Param(float, 60.0, "inclination (deg)")),
        _dwarf_nova, "Shakura & Sunyaev 1973; Frank, King & Raine 2002; Nauenberg 1972", "t0 is the start of the rise"),
]
