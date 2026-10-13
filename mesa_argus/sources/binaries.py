"""Eclipsing binaries from PHOEBE 2 (mesa_argus/binary.py), from parameters alone (no Gaia).

Two stars with MESA-atmosphere SEDs (Kurucz 2003 surface flux at each Teff and log g), their visible flux through
the orbit from PHOEBE (Roche geometry, blackbody intensities, PHOEBE's default limb darkening, no irradiation), one
orbit of system SEDs. Masses from the main-sequence Teff-mass relation (Pecaut & Mamajek 2013) unless given; the orbit
size from Kepler's third law. Brightness: either `distance` (the physical brightness) or `G` (the out-of-eclipse Gaia G;
the distance is then the one that gives it). epoch is a time of primary eclipse.
"""

import numpy as np

from . import REQUIRED, Generator, Param


def _write(out, p_, contact, p, field, kind_name):
    from .. import binary as B
    from ..dust import redden

    if (p["distance"] > 0) == (p["G"] > 0):
        raise SystemExit(f"[[source]] {kind_name}: give exactly one of distance = (pc) or G = (out-of-eclipse Gaia G)")
    w, f10, r1, r2, m1, m2 = B.component_seds(p_, contact, p["mh"])
    if p["distance"] > 0:
        dm = float(5 * np.log10(p["distance"]) - 5)
    else:  # the distance modulus at which the unobscured system (spheres) has this G through the field's dust
        dm = float(p["G"] - B.vega_mag(w, redden(w, f10[0] + f10[1], field.ebv), "G"))
    phases, seds, t_max = B.orbit_seds(p_, contact, w, f10)
    B.write_orbit(out, w, seds, phases * p_["period"], dict(
        label=out.name, kind="EB", source_kind=kind_name, nonlinear_period=p_["period"], epoch_eclipse_bjd=p["epoch_bjd"],
        distance_modulus=dm, ebv=field.ebv, mass=m1, mass2=m2, teff1=p_["teff1"], teff2=p_["teff2"], R1=r1, R2=r2,
        incl=p_["incl"], q=p_["q"], sma=p_["sma"], ecc=p_.get("ecc", 0.0), contact=bool(contact),
        route="PHOEBE from given parameters"))
    return dict(kind=kind_name, reference="PHOEBE 2 (Prsa et al. 2016); Pecaut & Mamajek 2013",
                params=f"P {p_['period']:g} d, i {p_['incl']:g} deg, q {p_['q']:.3f}, Teff {p_['teff1']:.0f} / "
                       f"{p_['teff2']:.0f} K, R {r1:.3f} / {r2:.3f} Rsun, M {m1:.3f} / {m2:.3f} Msun, a {p_['sma']:.3f} Rsun; "
                       f"distance modulus {dm:.2f}")


def _wuma(out, p, field):
    from .. import binary as B

    m1 = B.ms_mass(p["teff1"])
    sma = B.sma_rsun(p["period"], m1 * (1 + p["q"]))
    p_ = dict(period=p["period"], incl=p["incl"], q=p["q"], sma=sma, teff1=p["teff1"], teff2=p["teff2"],
              requiv1=B.contact_requiv(p["period"], p["q"], sma, p["fillout"]))
    return _write(out, p_, True, p, field, "W UMa")


def _detached(out, p, field):
    from .. import binary as B

    m1 = p["m1"] or B.ms_mass(p["teff1"])
    m2 = p["m2"] or B.ms_mass(p["teff2"])
    p_ = dict(period=p["period"], incl=p["incl"], q=m2 / m1, sma=B.sma_rsun(p["period"], m1 + m2), teff1=p["teff1"],
              teff2=p["teff2"], requiv1=p["r1"], requiv2=p["r2"], m1=m1, m2=m2, ecc=p["ecc"], per0=p["per0"])
    return _write(out, p_, False, p, field, "detached EB")


BRIGHTNESS = dict(distance=Param(float, 0.0, "distance (pc); give this or G"),
                  G=Param(float, 0.0, "out-of-eclipse apparent Gaia G; give this or distance"),
                  mh=Param(float, 0.0, "[M/H] of the atmospheres"))
GENERATORS = [
    Generator("W UMa", "periodic", "PHOEBE", "contact binary (W UMa): two stars sharing a common envelope",
              dict(period=Param(float, REQUIRED, "orbital period (days)"),
                   q=Param(float, REQUIRED, "mass ratio M2/M1"),
                   incl=Param(float, REQUIRED, "inclination (deg)"),
                   teff1=Param(float, REQUIRED, "primary Teff (K); its mass from the main-sequence relation"),
                   teff2=Param(float, REQUIRED, "secondary Teff (K)"),
                   fillout=Param(float, REQUIRED, "PHOEBE fillout factor: 0 inner, 1 outer critical surface"),
                   **BRIGHTNESS),
              _wuma, "Prsa et al. 2016", "epoch is a time of primary eclipse", parallel=False),
    Generator("detached EB", "periodic", "PHOEBE", "detached eclipsing binary, circular or eccentric",
              dict(period=Param(float, REQUIRED, "orbital period (days)"),
                   incl=Param(float, REQUIRED, "inclination (deg)"),
                   teff1=Param(float, REQUIRED, "primary Teff (K)"),
                   teff2=Param(float, REQUIRED, "secondary Teff (K)"),
                   r1=Param(float, REQUIRED, "primary equivalent radius (Rsun)"),
                   r2=Param(float, REQUIRED, "secondary equivalent radius (Rsun)"),
                   m1=Param(float, 0.0, "primary mass (Msun); 0: main-sequence relation"),
                   m2=Param(float, 0.0, "secondary mass (Msun); 0: main-sequence relation"),
                   ecc=Param(float, 0.0, "eccentricity"),
                   per0=Param(float, 90.0, "argument of periastron (deg)"),
                   **BRIGHTNESS),
              _detached, "Prsa et al. 2016", "epoch is a time of primary eclipse", parallel=False),
]
