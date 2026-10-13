"""Star generators: every kind of star, variable, transient or star set that make_stars.py can build.

    python -m mesa_argus.sources              # every generator, its keys, their defaults
    python -m mesa_argus.sources "W UMa"      # one generator

A star file's entries name a generator and give its keys; nothing about observing (no dust, no position in an image,
no night, no instrument):

    [[star]]
    name = "wuma_035"              # the folder it is built into: stars/<name>/
    generator = "W UMa"            # case, spaces and punctuation are ignored ("wuma" = "W UMa")
    period = 0.35                  # ... the generator's own keys
    ...

    [[set]]                        # many constant stars at once (generator TIMING "set")
    name = "f294_field"
    generator = "Gaia field"
    ra = 294.7844
    dec = -11.6235

Adding a generator: a module in this package with a list GENERATORS of Generator(...). It is found automatically:
its keys are validated, listed by python -m mesa_argus.sources and usable in star files. `make(work, p, ctx)` builds
the model in the working folder `work` (and may use ctx.runs for long MESA / PHOEBE runs) and returns its metadata:

    dict(kind=..., params="one line describing the model", reference=..., product=<path to the SED file it wrote>)
      transient: observer-frame f_lambda before any dust, against days since t0 (mesa_argus.transients format)
      periodic:  one cycle (or more) of f_lambda at 10 pc against time, with attrs distance_modulus and
                 nonlinear_period (the MESA RSP / PHOEBE format)
      set:       product = a folder of catalogue files (Gaia DR3 tables)

and, for a star with a fixed place and ephemeris (a real star), `real=dict(...)` (mesa_argus.stars documents it).
`prepare(p, ctx)`, if given, runs first, one star at a time (network queries, shared state); `make` may run alongside
other stars when `parallel` is true.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import importlib
import pkgutil
import re
from pathlib import Path
from typing import Any, Callable

REQUIRED = object()


@dataclass
class Param:
    type: type
    default: Any = REQUIRED
    help: str = ""
    choices: tuple = ()


@dataclass
class Generator:
    name: str
    timing: str  # "transient", "periodic" or "set"
    engine: str  # what computes it, e.g. "MESA RSP", "PHOEBE", "sncosmo"
    summary: str
    params: dict
    make: Callable
    reference: str = ""
    notes: str = ""
    parallel: bool = True  # make() may run in a thread alongside other stars (False: one at a time)
    prepare: Callable | None = None


@dataclass
class Context:
    name: str  # the star's name (its folder)
    runs: Path  # where long runs may keep their working files (runs/)
    extra: dict = dc_field(default_factory=dict)


COMMON = {"name": Param(str, REQUIRED, "the star's name: its folder, stars/<name>/"),
          "generator": Param(str, REQUIRED, "the generator's name")}


def key(name):
    """Generator names compare without case, spaces or punctuation."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_REGISTRY: dict | None = None


def registry():
    """{key(name): Generator}, from every module in this package."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {}
        for m in pkgutil.iter_modules(__path__):
            if m.name.startswith("_"):
                continue
            for g in getattr(importlib.import_module(f"{__name__}.{m.name}"), "GENERATORS", []):
                if key(g.name) in _REGISTRY:
                    raise RuntimeError(f"two generators named {g.name!r}")
                clash = set(g.params) & set(COMMON)
                if clash:
                    raise RuntimeError(f"generator {g.name!r}: keys {sorted(clash)} are every star's own; rename them")
                _REGISTRY[key(g.name)] = g
    return _REGISTRY


def get(name):
    g = registry().get(key(name))
    if g is None:
        raise SystemExit(f"no star generator {name!r}; there are: " + ", ".join(sorted(v.name for v in registry().values())))
    return g


def validate(entry, table="star"):
    """(Generator, p) for one [[star]] / [[set]] entry: unknown keys, missing keys, types and choices checked;
    defaults filled."""
    if "generator" not in entry:
        raise SystemExit(f"[[{table}]] {entry}: generator = is required")
    g = get(entry["generator"])
    if (g.timing == "set") != (table == "set"):
        raise SystemExit(f"[[{table}]] {entry.get('name')}: {g.name} is a {'set' if g.timing == 'set' else 'star'}; "
                         f"give it as [[{'set' if g.timing == 'set' else 'star'}]]")
    spec = {**COMMON, **g.params}
    where = f"[[{table}]] {entry.get('name', '?')} ({g.name})"
    unknown = sorted(set(entry) - set(spec))
    if unknown:
        raise SystemExit(f"{where}: unknown keys {unknown}; it takes {sorted(spec)}")
    p = {}
    for k, s in spec.items():
        if k not in entry:
            if s.default is REQUIRED:
                raise SystemExit(f"{where}: {k} = is required ({s.help})")
            p[k] = s.default
            continue
        v = entry[k]
        if s.type is float and isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
        elif s.type is int and isinstance(v, int) and not isinstance(v, bool):
            pass
        elif not isinstance(v, s.type):
            raise SystemExit(f"{where}: {k} = {v!r} should be a {s.type.__name__} ({s.help})")
        if s.choices and v not in s.choices:
            raise SystemExit(f"{where}: {k} = {v!r} is not one of {list(s.choices)}")
        p[k] = v
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", p["name"]):
        raise SystemExit(f"{where}: name = {p['name']!r}: letters, digits and _ . + - only (it is a folder name)")
    return g, p


def describe(g):
    lines = [f"{g.name}  ({g.timing}; {g.engine})", f"    {g.summary}"]
    if g.reference:
        lines.append(f"    reference: {g.reference}")
    if g.notes:
        lines.append(f"    note: {g.notes}")
    for k, s in g.params.items():
        d = "required" if s.default is REQUIRED else f"default {s.default!r}"
        c = f"; one of {list(s.choices)}" if s.choices else ""
        lines.append(f"    {k:12s} {s.help} ({d}{c})")
    return "\n".join(lines)
