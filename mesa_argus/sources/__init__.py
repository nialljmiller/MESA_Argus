"""Source generators: every kind of variable star or transient a run can place in a field.

    python -m mesa_argus.sources              # every generator, its keys, their defaults
    python -m mesa_argus.sources "W UMa"      # one generator

A run file's [[source]] entries name a generator and give its keys:

    [[source]]
    generator = "microlensing"     # any generator's name; case, spaces and punctuation are ignored ("wuma" = "W UMa")
    x = 30                         # position, percent of the image from its lower left (default 50, 50; -1: random)
    y = 70
    t0 = "start+0.15"              # transients: the model's t0; periodic sources: `epoch` instead
    tE = 0.2                       # ... the generator's own keys

Adding a generator: a module in this package with a list GENERATORS of Generator(...). It is found automatically;
nothing else changes. `make(out_dir, p, field)` builds the model and returns its metadata; it writes either

    transient.h5   (TIMING "transient")  observer-frame SEDs against days since t0 (mesa_argus.transients format)
    pulsation.h5   (TIMING "periodic")   one cycle of SEDs at 10 pc against time, with the distance modulus and
                                         E(B-V) it is seen at (the MESA RSP / PHOEBE format)

in out_dir. `p` holds the entry's keys (validated, defaults filled) plus t0_bjd (transients) or epoch_bjd (periodic
sources); `field` gives the field's name, E(B-V) and label of the source (Field below). The metadata must hold
`kind` (what the source is, e.g. "RRab", "SN Ia"), `params` (one line describing the model) and `reference`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import importlib
import pkgutil
import re
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
    timing: str  # "transient" or "periodic"
    engine: str  # what computes it, e.g. "MESA RSP", "PHOEBE", "sncosmo"
    summary: str
    params: dict
    make: Callable
    reference: str = ""
    notes: str = ""
    parallel: bool = True  # make() may run in a thread alongside other sources (False: one at a time)


@dataclass
class Field:
    name: str
    ebv: float
    label: str = ""
    extra: dict = dc_field(default_factory=dict)


# keys every source takes (handled here, not by the generator)
COMMON = {
    "generator": Param(str, REQUIRED, "the generator's name"),
    "x": Param(float, 50.0, "position: percent of the image width from the left; -1 random"),
    "y": Param(float, 50.0, "position: percent of the image height from the bottom; -1 random"),
}
TIME_KEY = {"transient": ("t0", "the model's t0: ISO UTC time, start+D / start-D days from the night's start, or BJD_TDB"),
            "periodic": ("epoch", "a time of maximum light (pulsators) or primary eclipse (binaries): ISO UTC, "
                                  "start+D days, or BJD_TDB")}


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
                clash = set(g.params) & (set(COMMON) | {k for k, _ in TIME_KEY.values()})
                if clash:
                    raise RuntimeError(f"generator {g.name!r}: keys {sorted(clash)} are every source's own; rename them")
                _REGISTRY[key(g.name)] = g
    return _REGISTRY


def get(name):
    g = registry().get(key(name))
    if g is None:
        raise SystemExit(f"no source generator {name!r}; there are: " + ", ".join(sorted(v.name for v in registry().values())))
    return g


def validate(entry):
    """(Generator, p) for one [[source]] entry: unknown keys, missing keys, types and choices checked; defaults filled.
    The time key (t0 / epoch) is left as given (argus_field.py resolves it against the night)."""
    if "generator" not in entry:
        raise SystemExit(f"[[source]] {entry}: generator = is required")
    g = get(entry["generator"])
    t_key, t_help = TIME_KEY[g.timing]
    spec = {**COMMON, t_key: Param(object, REQUIRED, t_help), **g.params}
    where = f"[[source]] {g.name}"
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
        elif s.type is object:
            pass
        elif not isinstance(v, s.type):
            raise SystemExit(f"{where}: {k} = {v!r} should be a {s.type.__name__} ({s.help})")
        if s.choices and v not in s.choices:
            raise SystemExit(f"{where}: {k} = {v!r} is not one of {list(s.choices)}")
        p[k] = v
    for k in ("x", "y"):
        if not (p[k] == -1 or 0 <= p[k] <= 100):
            raise SystemExit(f"{where}: {k} = {p[k]} is a percentage of the image (0-100), or -1 for random")
    return g, p


def describe(g):
    t_key, t_help = TIME_KEY[g.timing]
    lines = [f"{g.name}  ({g.timing}; {g.engine})", f"    {g.summary}"]
    if g.reference:
        lines.append(f"    reference: {g.reference}")
    if g.notes:
        lines.append(f"    note: {g.notes}")
    spec = {t_key: Param(object, REQUIRED, t_help), **g.params, "x": COMMON["x"], "y": COMMON["y"]}
    for k, s in spec.items():
        d = "required" if s.default is REQUIRED else f"default {s.default!r}"
        c = f"; one of {list(s.choices)}" if s.choices else ""
        lines.append(f"    {k:12s} {s.help} ({d}{c})")
    return "\n".join(lines)
