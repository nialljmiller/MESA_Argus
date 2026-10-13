"""Radial pulsators from MESA RSP (mesa_argus/rsp.py), from parameters alone (no Gaia, no dust).

The same machinery as for a Gaia star: Teff tuned with RSP's linear analysis until the period of the star's mode equals
`period`, at luminosity L (given, or from the dust-free absolute magnitude M_G with bolometric correction ~0), stepped
down if needed so the mode is the only unstable one; the limit cycle is run and one cycle of SEDs (every <= 25 s, at
10 pc) written. A star with no acceptable model is not built (runs/rsp_star_<name>/FAILED, rejected.json say why).
One model takes ~0.5-2 h.
"""

import numpy as np

from . import REQUIRED, Generator, Param


def _rsp(kinds):
    def make(work, p, ctx):
        from .. import rsp

        if (p["L"] > 0) == (p["M_G"] != 99.0):
            raise SystemExit(f"[[star]] {ctx.name}: give exactly one of L = (Lsun) or M_G = (dust-free absolute G)")
        rsp.build_sed_instrument()
        rsp.PREFIX = "star_"
        kind = kinds[p["mode"]] if isinstance(kinds, dict) else kinds
        dm = float(5 * np.log10(p["distance"]) - 5)
        target = dict(source=0, kind=kind, period=p["period"], epoch_bjd=float("nan"), dm=dm, ebv=0.0, z=p["z"],
                      mass=p["mass"], kap=p["kap"], x=p["hydrogen"])
        if p["L"] > 0:
            target["L"] = p["L"]
        else:
            target.update(G=p["M_G"] + dm, G_mean=p["M_G"] + dm)
        rsp.model_star(ctx.name, target)
        run = ctx.runs / f"rsp_star_{ctx.name}"
        if not (run / "pulsation.h5").exists():
            return None
        import json

        q = json.loads((run / "params.json").read_text())
        return dict(kind=kind, reference="MESA RSP (Paxton et al. 2019)", product=run / "pulsation.h5",
                    period=p["period"],
                    params=f"P {p['period']:g} d (model {q.get('nonlinear_period', float('nan')):.5f}); L {q['L']:.2f} Lsun "
                           f"(x {q.get('L_factor', 1.0)} the L asked for), Teff {q['teff']:.0f} K; mass {p['mass']:g} Msun, "
                           f"X {p['hydrogen']:g}, Z {p['z']:g}, {p['kap']} opacities; at {p['distance']:g} pc")
    return make


def _keys(mode=None, hydrogen=0.75):
    k = {}
    if mode:
        k["mode"] = Param(str, mode[0], "pulsation mode / subtype", tuple(mode))
    k.update(period=Param(float, REQUIRED, "period (days); the model is tuned to it"),
             L=Param(float, 0.0, "luminosity (Lsun); give this or M_G"),
             M_G=Param(float, 99.0, "dust-free absolute Gaia G (bolometric correction ~0); give this or L"),
             distance=Param(float, REQUIRED, "distance (pc)"),
             z=Param(float, REQUIRED, "metal mass fraction Z"),
             mass=Param(float, REQUIRED, "mass (Msun)"),
             hydrogen=Param(float, hydrogen, "hydrogen mass fraction X"),
             kap=Param(str, "a09", "opacity mixture: a09 (alpha-enhanced) or gs98 (scaled solar)", ("a09", "gs98")))
    return k


NOTE = "the X default is the RSP setup's (as M3)"
GENERATORS = [
    Generator("RR Lyrae", "periodic", "MESA RSP", "RR Lyrae star: RRab (fundamental) or RRc (first overtone)",
              _keys(("RRab", "RRc")), _rsp({"RRab": "RRab", "RRc": "RRc"}), "Paxton et al. 2019", NOTE),
    Generator("Cepheid", "periodic", "MESA RSP", "classical Cepheid: F (fundamental) or 1O (first overtone)",
              _keys(("F", "1O")), _rsp({"F": "DCEP F", "1O": "DCEP 1O"}), "Paxton et al. 2019", NOTE),
    Generator("Type II Cepheid", "periodic", "MESA RSP", "Type II Cepheid: BL Her or W Vir",
              _keys(("BL Her", "W Vir")), _rsp({"BL Her": "BL Her", "W Vir": "W Vir"}), "Paxton et al. 2019", NOTE),
    Generator("HADS", "periodic", "MESA RSP", "high-amplitude delta Scuti star, radiative (convection off)",
              _keys(), _rsp("HADS"), "Paxton et al. 2019", NOTE),
]
