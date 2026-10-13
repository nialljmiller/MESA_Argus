"""Radial pulsators from MESA RSP (mesa_argus/rsp.py), from parameters alone (no Gaia).

The same machinery as for a Gaia star: Teff tuned with RSP's linear analysis until the period of the star's mode equals
`period`, L from `G` at `distance` (bolometric correction ~0), stepped down if needed so the mode is the only unstable
one; the limit cycle is run and one cycle of SEDs (every <= 25 s) written. A star with no acceptable model is dropped
(rsp's FAILED / rejected.json say why). One model takes ~0.5-2 h; models are cached per field and label.
"""

import numpy as np

from . import REQUIRED, Generator, Param


def _rsp(kinds):
    def make(out, p, field):
        from .. import rsp

        rsp.build_sed_instrument()
        rsp.PREFIX = f"{field.name}_"
        label = out.name[len(f"rsp_{field.name}_"):]
        kind = kinds[p["mode"]] if isinstance(kinds, dict) else kinds
        target = dict(source=field.extra["source"], kind=kind, period=p["period"], epoch_bjd=p["epoch_bjd"], G=p["G"],
                      G_mean=p["G"], dm=float(5 * np.log10(p["distance"]) - 5), ebv=field.ebv, z=p["z"], mass=p["mass"],
                      kap=p["kap"], x=p["hydrogen"])
        rsp.model_star(label, target)
        if not (out / "pulsation.h5").exists():
            return None
        import json

        q = json.loads((out / "params.json").read_text())
        return dict(kind=kind, reference="MESA RSP (Paxton et al. 2019)",
                    params=f"P {p['period']:g} d (model {q.get('nonlinear_period', float('nan')):.5f}); L {q['L']:.2f} Lsun "
                           f"(x {q.get('L_factor', 1.0)} the L from G {p['G']:g} at {p['distance']:g} pc), Teff "
                           f"{q['teff']:.0f} K; mass {p['mass']:g} Msun, X {p['hydrogen']:g}, Z {p['z']:g}, {p['kap']} opacities")
    return make


def _keys(mode=None, hydrogen=0.75):
    k = {}
    if mode:
        k["mode"] = Param(str, mode[0], "pulsation mode / subtype", tuple(mode))
    k.update(period=Param(float, REQUIRED, "period (days); the model is tuned to it"),
             G=Param(float, REQUIRED, "intensity-mean apparent Gaia G (sets L with the distance)"),
             distance=Param(float, REQUIRED, "distance (pc)"),
             z=Param(float, REQUIRED, "metal mass fraction Z"),
             mass=Param(float, REQUIRED, "mass (Msun)"),
             hydrogen=Param(float, hydrogen, "hydrogen mass fraction X"),
             kap=Param(str, "a09", "opacity mixture: a09 (alpha-enhanced) or gs98 (scaled solar)", ("a09", "gs98")))
    return k


NOTE = "epoch is a time of maximum light; X default is the RSP setup's (as M3)"
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
