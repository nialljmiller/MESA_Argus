"""Transients: observer-frame SEDs before any dust, against days since t0 (the physics is in mesa_argus/transients.py).
When t0 is, and the dust, are the observation's."""
from .. import transients as T
from . import REQUIRED, Generator


def supernova(folder, k, name):
    params = {key: float(value) for key, value in k["params"].items()}
    return T.make_sncosmo(folder / "sed.h5", k["model"], k["z"], 0.0, k["M"], 0.0, k["band"], k["magsys"], params,
                          k["host_ebv"], k["version"] or None)


def grb(folder, k, name):
    return T.make_afterglow(folder / "sed.h5", 0.0, k["peak"], 0.0, k["t_peak"], k["alpha"], k["beta"])


def microlensing(folder, k, name):
    return T.make_microlensing(folder / "sed.h5", 0.0, k["tE"], k["u0"], k["base"], k["bp_rp"], k["rho"], k["fs"])


def dwarf_nova(folder, k, name):
    return T.make_dwarf_nova(folder / "sed.h5", 0.0, k["d"], k["mdot_q"], k["mdot_peak"], k["t_rise"], k["plateau"],
                             k["tau_decay"], k["r_out"], k["t_wd"], 0.0, k["m_wd"], k["incl"])


GENERATORS = [
    Generator("supernova", "transient", "sncosmo",
              "any sncosmo model (python -m mesa_argus.transients --list) at redshift z, scaled to a rest-frame peak "
              "absolute magnitude (sncosmo's default cosmology, WMAP9); t0 is the model's phase 0",
              dict(model=(REQUIRED, "sncosmo source name, e.g. salt3, hsiao, nugent-sn2p"),
                   z=(REQUIRED, "redshift"),
                   M=(REQUIRED, "rest-frame peak absolute magnitude in `band`"),
                   band=("bessellb", "sncosmo bandpass of M"),
                   magsys=("ab", "magnitude system of M"),
                   host_ebv=(0.0, "host-galaxy E(B-V), CCM89 R_V 3.1, rest frame"),
                   version=("", "sncosmo model version (default: sncosmo's)"),
                   params=({}, "the model's own parameters, e.g. {x1 = 0, c = 0} for SALT2/3")),
              supernova, "Barbary et al. (sncosmo); the model's own reference"),
    Generator("GRB", "transient", "power law",
              "GRB optical afterglow, f_nu ~ t^-alpha nu^-beta with a t^2 onset peaking at t_peak; observed frame, no "
              "redshift (written in this repo); t0 is the burst",
              dict(peak=(REQUIRED, "observed AB magnitude at 6400 A at t_peak, before dust"),
                   t_peak=(60.0, "seconds after the burst"),
                   alpha=(1.0, "temporal decay index"),
                   beta=(0.66, "spectral index")),
              grb, "Sari, Piran & Narayan 1998; Sari & Piran 1999"),
    Generator("microlensing", "transient", "point lens",
              "point-lens microlensing of a point or uniform-disc source, achromatic; the source a MESA template SED "
              "(written in this repo); t0 is the closest approach",
              dict(tE=(REQUIRED, "Einstein time (days)"),
                   u0=(REQUIRED, "impact parameter (Einstein radii)"),
                   base=(REQUIRED, "the unmagnified source's apparent rho (AB), before dust"),
                   bp_rp=(REQUIRED, "the source's intrinsic BP-RP (picks the MESA template)"),
                   rho=(0.0, "source radius (Einstein radii); 0 = a point source"),
                   fs=(1.0, "the source's fraction of the baseline light; 1 = no blend")),
              microlensing, "Paczynski 1986"),
    Generator("dwarf nova", "transient", "steady disc",
              "dwarf-nova outburst: a steady optically thick disc whose accretion rate rises, plateaus and decays, plus "
              "the white dwarf as a blackbody; no donor, bright spot or lines (written in this repo); t0 is the start of "
              "the rise",
              dict(d=(REQUIRED, "distance (pc)"),
                   mdot_q=(REQUIRED, "quiescent accretion rate (Msun/yr)"),
                   mdot_peak=(REQUIRED, "outburst accretion rate (Msun/yr)"),
                   t_rise=(REQUIRED, "rise time (days, exponential in Mdot)"),
                   plateau=(REQUIRED, "days at the peak rate"),
                   tau_decay=(REQUIRED, "e-folding time of the decline (days)"),
                   r_out=(REQUIRED, "disc outer radius (Rsun)"),
                   t_wd=(REQUIRED, "white-dwarf temperature (K)"),
                   m_wd=(0.8, "white-dwarf mass (Msun)"),
                   incl=(60.0, "inclination (deg)")),
              dwarf_nova, "Shakura & Sunyaev 1973; Frank, King & Raine 2002; Nauenberg 1972"),
]
