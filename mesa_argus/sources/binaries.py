"""Eclipsing binaries from PHOEBE 2 (mesa_argus/binary.py), from parameters alone (no Gaia, no dust).

Two stars with MESA-atmosphere SEDs (Kurucz 2003 surface flux at each Teff and log g), their visible flux through
the orbit from PHOEBE (Roche geometry, blackbody intensities, PHOEBE's default limb darkening, no irradiation), one
orbit of system SEDs at 10 pc; the brightness follows from the radii and Teffs, seen at `distance`. Masses from the
main-sequence Teff-mass relation (Pecaut & Mamajek 2013) unless given; the orbit size from Kepler's third law. Phase 0
of the stored orbit is the primary eclipse.
"""

import numpy as np

from . import REQUIRED, Generator, Param


def _write(work, p_, contact, p, ctx, kind_name):
    from .. import binary as B

    w, f10, r1, r2, m1, m2 = B.component_seds(p_, contact, p["mh"])
    dm = float(5 * np.log10(p["distance"]) - 5)
    phases, seds, t_max = B.orbit_seds(p_, contact, w, f10)
    B.write_orbit(work, w, seds, phases * p_["period"], dict(
        label=ctx.name, kind="EB", source_kind=kind_name, nonlinear_period=p_["period"], distance_modulus=dm, ebv=0.0,
        mass=m1, mass2=m2, teff1=p_["teff1"], teff2=p_["teff2"], R1=r1, R2=r2, incl=p_["incl"], q=p_["q"], sma=p_["sma"],
        ecc=p_.get("ecc", 0.0), contact=bool(contact), route="PHOEBE from given parameters"))
    return dict(kind=kind_name, reference="PHOEBE 2 (Prsa et al. 2016); Pecaut & Mamajek 2013", product=work / "pulsation.h5",
                period=p_["period"],
                params=f"P {p_['period']:g} d, i {p_['incl']:g} deg, q {p_['q']:.3f}, Teff {p_['teff1']:.0f} / "
                       f"{p_['teff2']:.0f} K, R {r1:.3f} / {r2:.3f} Rsun, M {m1:.3f} / {m2:.3f} Msun, a {p_['sma']:.3f} Rsun; "
                       f"at {p['distance']:g} pc")


def _wuma(work, p, ctx):
    from .. import binary as B

    m1 = B.ms_mass(p["teff1"])
    sma = B.sma_rsun(p["period"], m1 * (1 + p["q"]))
    p_ = dict(period=p["period"], incl=p["incl"], q=p["q"], sma=sma, teff1=p["teff1"], teff2=p["teff2"],
              requiv1=B.contact_requiv(p["period"], p["q"], sma, p["fillout"]))
    return _write(work, p_, True, p, ctx, "W UMa")


def _detached(work, p, ctx):
    from .. import binary as B

    m1 = p["m1"] or B.ms_mass(p["teff1"])
    m2 = p["m2"] or B.ms_mass(p["teff2"])
    p_ = dict(period=p["period"], incl=p["incl"], q=m2 / m1, sma=B.sma_rsun(p["period"], m1 + m2), teff1=p["teff1"],
              teff2=p["teff2"], requiv1=p["r1"], requiv2=p["r2"], m1=m1, m2=m2, ecc=p["ecc"], per0=p["per0"])
    return _write(work, p_, False, p, ctx, "detached EB")


BRIGHTNESS = dict(distance=Param(float, REQUIRED, "distance (pc)"),
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
              _wuma, "Prsa et al. 2016", parallel=False),
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
              _detached, "Prsa et al. 2016", parallel=False),
]
