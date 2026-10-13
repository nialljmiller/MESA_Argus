"""Real stars from Gaia DR3: one catalogued variable modelled as itself, or every star around a pointing as a set.

Gaia variable: the star's period and ephemeris (Gaia SOS tables, ASAS-SN for HADS), its distance (parallax, or a
period-luminosity relation), the E(B-V) along its sightline (SFD, used only to infer its intrinsic luminosity from its
measured G), composition and mass (mesa_argus/gaia.py physical()), then MESA RSP (pulsators) or PHOEBE fitted to Gaia's
eclipse model (binaries). The SEDs are dust-free at 10 pc, as every star's; the star's own sky position, ephemeris and
sightline E(B-V) are kept with it (`real`), so an observation can put it where and when it really is.

Gaia field: every Gaia DR3 source within 0.12 deg of a pointing (positions, G, BP, RP, parallaxes), with Gaia's
variability tables, as catalogue files. Their brightnesses are as Gaia measured them, i.e. already through their own
dust: an observation uses them as they are (each becomes the colour-matched MESA SED scaled to its G).

The Gaia queries are cached in data/gaia/<pointing>/ (committed), so a star or set can be rebuilt without the network.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from . import REQUIRED, Generator, Param

CACHE = Path(__file__).resolve().parents[2] / "data" / "gaia"


def _cache_with(source):
    """A cached pointing whose Gaia table holds this source, or None."""
    from astropy.table import Table

    for core in sorted(CACHE.glob("*/gaia_dr3_core.ecsv")):
        t = Table.read(core)
        hit = np.nonzero(np.asarray(t["Source"]).astype(np.int64) == int(source))[0]
        if len(hit):
            return core.parent, float(t["RA_ICRS"][hit[0]]), float(t["DE_ICRS"][hit[0]])
    return None


def _prepare_variable(p, ctx):
    from .. import gaia as Gd

    found = _cache_with(p["source"])
    if found is None:
        from astroquery.vizier import Vizier

        r = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS"]).query_constraints(catalog="I/355/gaiadr3", Source=str(p["source"]))
        if not r or not len(r[0]):
            raise SystemExit(f"[[star]] {ctx.name}: Gaia DR3 {p['source']} not found")
        ra, dec = float(r[0]["RA_ICRS"][0]), float(r[0]["DE_ICRS"][0])
        cache = CACHE / Gd.field_name(ra, dec)
        Gd.fetch(ra, dec, cache)
    else:
        cache, ra, dec = found
    Gd.TRY_RRD = p["try_rrd"]
    rows, stars, _, _ = Gd.inventory(cache, ra, dec, 9)
    row = next((r for r in rows if r["source"] == p["source"]), None)
    if row is None:
        raise SystemExit(f"[[star]] {ctx.name}: Gaia DR3 {p['source']} is not a catalogued variable")
    if row["model"] is None:
        raise SystemExit(f"[[star]] {ctx.name}: Gaia DR3 {p['source']} ({row['kind']}) cannot be modelled: {row['why']}")
    ph, why = Gd.physical(row, stars, ra, dec)
    if ph is None:
        raise SystemExit(f"[[star]] {ctx.name}: Gaia DR3 {p['source']} ({row['kind']}) cannot be modelled: {why}")
    target = dict(source=row["source"], kind=row["kind"], period=row["period"], epoch_bjd=row["epoch_bjd"], G=row["G"],
                  **({"period2": row["period2"]} if "period2" in row else {}), G_mean=row["G_mean"], amp_G=row["amp_G"],
                  **{k: v for k, v in ph.items() if k != "notes"}, notes=ph["notes"])
    if row["kind"] == "EB":
        target.update(veb=row["veb"], nss=row.get("nss"))
    ctx.extra.update(target=target, row=row, ra=ra, dec=dec, cache=str(cache))


def _make_variable(work, p, ctx):
    import json

    from .. import binary as B
    from .. import rsp

    t, label = ctx.extra["target"], f"G{p['source']}"
    run = ctx.runs / f"rsp_gaia_{label}"
    epoch = t["epoch_bjd"]
    if t["kind"] == "EB":
        if (run / "pulsation.h5").exists() and (run / "binary.json").exists():
            ok, info = True, json.loads((run / "binary.json").read_text())
        else:
            ok, info = B.model_binary(dict(source=t["source"], veb=t["veb"], nss=t.get("nss"), teff_gsp=t["teff_gsp"],
                                           logg_gsp=t["logg_gsp"], G=t["veb"]["magModRef"]), run, t["dm"], t["ebv"])
        if not ok:
            raise SystemExit(f"[[star]] {ctx.name}: PHOEBE could not model Gaia DR3 {p['source']}: {info['reason']}")
        veb = t["veb"]
        epoch = veb["TimeRef"] + 2455197.5 + veb["PhaseE1"] * t["period"]  # a primary eclipse (the model's phase 0)
        engine, how = "PHOEBE", "; ".join(info[k] for k in ("route", "light_curve_check", "distance_check") if k in info)
    else:
        rsp.build_sed_instrument()
        rsp.PREFIX = "gaia_"
        rsp.model_star(label, t)
        if not (run / "pulsation.h5").exists():
            return None
        q = json.loads((run / "params.json").read_text())
        engine = "MESA RSP"
        how = (f"L {q['L']:.2f} Lsun (x {q.get('L_factor', 1.0)} the L from the distance), Teff {q['teff']:.0f} K, "
               f"non-linear period {q.get('nonlinear_period', float('nan')):.5f} d (Gaia {t['period']:.5f})")
    notes = {k: str(v) for k, v in t["notes"].items()}
    return dict(kind=t["kind"], engine=engine, reference="Gaia DR3; " + ("PHOEBE 2" if engine == "PHOEBE" else "MESA RSP"),
                product=run / "pulsation.h5", period=t["period"],
                params=f"Gaia DR3 {t['source']}, {t['kind']}, P {t['period']:.5f} d; distance modulus {t['dm']}, "
                       f"sightline E(B-V) {t['ebv']}; {how}",
                notes=notes,
                real=dict(source=int(t["source"]), ra=ctx.extra["ra"], dec=ctx.extra["dec"], kind=t["kind"], period=t["period"],
                          epoch_bjd=float(epoch), amp_G=float(t["amp_G"]), G=float(t["G"]), sightline_ebv=float(t["ebv"]),
                          veb=t.get("veb"), gaia_cache=ctx.extra["cache"]))


def _make_field(work, p, ctx):
    from astropy.table import Table

    from .. import gaia as Gd

    cache = CACHE / Gd.field_name(p["ra"], p["dec"])
    Gd.fetch(p["ra"], p["dec"], cache)
    out = work / "catalogue"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(cache.glob("*.ecsv")):
        shutil.copy(f, out / f.name)
    n = len(Table.read(out / "gaia_dr3_core.ecsv"))
    return dict(kind="Gaia field", reference="Gaia DR3 (VizieR I/355, I/357, I/358); ASAS-SN (II/366)", product=out,
                params=f"{n} Gaia DR3 sources within {Gd.RADIUS_DEG} deg of RA {p['ra']:.5f}, Dec {p['dec']:+.5f}",
                centre=[p["ra"], p["dec"]], radius_deg=Gd.RADIUS_DEG, n_stars=n)


GENERATORS = [
    Generator("Gaia variable", "periodic", "MESA RSP / PHOEBE",
              "a real Gaia DR3 variable modelled as itself (python make_stars.py --gaia-list RA DEC lists those that can be)",
              dict(source=Param(int, REQUIRED, "Gaia DR3 source_id"),
                   try_rrd=Param(bool, False, "attempt a double-mode RR Lyrae (no stable solution found so far)")),
              _make_variable, "Gaia DR3; mesa_argus/gaia.py", "keeps its own sky position, ephemeris and sightline E(B-V)",
              parallel=lambda ctx: ctx.extra["target"]["kind"] != "EB", prepare=_prepare_variable),
    Generator("Gaia field", "set", "Gaia DR3",
              "every Gaia DR3 source around a pointing, as constant stars (and Gaia's variability tables)",
              dict(ra=Param(float, REQUIRED, "deg (ICRS)"), dec=Param(float, REQUIRED, "deg (ICRS)")),
              _make_field, "Gaia DR3", parallel=False),
]
