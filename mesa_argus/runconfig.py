"""Run files: everything one argus_field.py run needs, in one TOML file (examples/run_template.toml lists every key).

    [field]      where, which mode, how big
    [night]      which night, which hours, which site
    [[source]]   the variables and transients placed in the field (python -m mesa_argus.sources lists the generators)
    [movie]      rendering and simulation options (night_movie.py)

Values come from, in order of precedence: the command line, the run file, the defaults below. The resolved run (every
key, defaults filled, the night as actually used) is written to outputs/<field>/run.toml, so any dataset can be made
again from it. Machine settings (MESA paths, cores) stay in config.toml: a run file means the same on every machine.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

from . import sources as S

REQUIRED = S.REQUIRED


def _movie_defaults():
    import night_movie as M

    acts = {a.dest: a for a in M.parser()._actions}
    return {k: (acts[k].default, acts[k].type or type(acts[k].default), acts[k].help or "") for k in M.MOVIE_OPTIONS
            if k != "max_airmass"}


# section: {key: (default, type, help, choices)}
SCHEMA = {
    "field": {
        "ra": (REQUIRED, float, "deg (ICRS)", ()),
        "dec": (REQUIRED, float, "deg (ICRS)", ()),
        "block": (3, int, "minipixes across the block shown (odd, up to 9; 9 = 253\")", ()),
        "mode": ("gaia", str, "gaia: the real field and its catalogued variables; synthetic: its stars' magnitudes "
                              "and colours at random positions, every star constant", ("gaia", "synthetic")),
        "seed": (1, int, "synthetic: star positions; random source positions (x or y = -1)", ()),
        "name": ("", str, "field name (default from the coordinates, + _syn<seed> for synthetic)", ()),
        "select": ("", (str, list), "gaia: catalogued variables to model: \"all\", \"none\", or [1, 3] (numbers "
                                    "from --list); unset: asked (interactive runs only)", ()),
        "try_rrd": (False, bool, "gaia: attempt double-mode RR Lyrae (no stable solution found so far)", ()),
    },
    "night": {
        "date": ("", str, "local date the night starts (YYYY-MM-DD; default today; the nearest night the field is up "
                          "if not); gaia mode: \"auto\" = the night the modelled variables change most", ()),
        "start": ("", str, "UT HH:MM to start (default: dusk / when the field rises)", ()),
        "end": ("", str, "UT HH:MM to stop (default: dawn / when the field sets)", ()),
        "site": ("argus", str, "argus = ArgusSim's site; sro = Sierra Remote Observatories", ("argus", "sro")),
        "max_airmass": (2.0, float, "observe while the field is at or below this airmass", ()),
    },
    "movie": {
        "bands": (["beta", "rho"], list, "one movie per band", ()),
        "models_only": (False, bool, "build the models; do not render", ()),
        # + night_movie.py's own options (MOVIE_OPTIONS), filled in by schema()
    },
}


def schema():
    s = copy.deepcopy(SCHEMA)
    for k, (default, typ, help_) in _movie_defaults().items():
        s["movie"][k] = (default, typ, help_, ())
    return s


def load(path):
    """The run file as a dict (relative `base` paths are not used here; tools/stage_batch.py expands batches)."""
    return tomllib.loads(Path(path).read_text())


def merge(run, overrides):
    """run with overrides applied: {"section.key": value}, or {"source": [...]} to replace the sources."""
    out = copy.deepcopy(run)
    for dotted, value in overrides.items():
        if dotted == "source":
            out["source"] = copy.deepcopy(value)
            continue
        section, _, k = dotted.partition(".")
        if not k:
            raise SystemExit(f"override {dotted!r}: give section.key (e.g. field.ra, movie.stride)")
        out.setdefault(section, {})[k] = value
    return out


def resolve(run):
    """Validated run with every default filled: {"field": {...}, "night": {...}, "movie": {...}, "source": [...]}.
    Raises SystemExit naming the bad key."""
    sch = schema()
    unknown = sorted(set(run) - set(sch) - {"source"})
    if unknown:
        raise SystemExit(f"run file: unknown sections {unknown}; there are {sorted(sch) + ['source']}")
    out = {}
    for section, keys in sch.items():
        given = run.get(section, {})
        if not isinstance(given, dict):
            raise SystemExit(f"run file: [{section}] should be a table")
        bad = sorted(set(given) - set(keys))
        if bad:
            raise SystemExit(f"run file [{section}]: unknown keys {bad}; it takes {sorted(keys)}")
        out[section] = {}
        for k, (default, typ, help_, choices) in keys.items():
            if k not in given:
                if default is REQUIRED:
                    raise SystemExit(f"run file [{section}]: {k} = is required ({help_})")
                out[section][k] = copy.deepcopy(default)
                continue
            v = given[k]
            types = typ if isinstance(typ, tuple) else (typ,)
            if float in types and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            if not isinstance(v, types) or (isinstance(v, bool) and bool not in types):
                names = "/".join(t.__name__ for t in types)
                raise SystemExit(f"run file [{section}]: {k} = {v!r} should be a {names} ({help_})")
            if choices and v not in choices:
                raise SystemExit(f"run file [{section}]: {k} = {v!r} is not one of {list(choices)}")
            out[section][k] = v
    f = out["field"]
    if f["block"] % 2 == 0 or not 1 <= f["block"] <= 9:
        raise SystemExit(f"run file [field]: block = {f['block']} should be odd, 1-9")
    for b in out["movie"]["bands"]:
        if b not in ("beta", "rho"):
            raise SystemExit(f"run file [movie]: bands = {out['movie']['bands']}: each one beta or rho")
    entries = run.get("source", [])
    if isinstance(entries, dict):
        entries = [entries]
    out["source"] = []
    for e in entries:
        S.validate(e)  # checks; the entry is kept as given (resolved again when placed)
        out["source"].append(dict(e))
    sel = f["select"]
    if not (isinstance(sel, list) and all(isinstance(i, int) for i in sel)) and not (
            isinstance(sel, str) and (sel in ("all", "none", "") or all(t.strip().isdigit() for t in sel.split(",")))):
        raise SystemExit(f"run file [field]: select = {f['select']!r}: \"all\", \"none\" or a list of numbers")
    return out


# ---------------------------------------------------------------- writing
def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_key(k)} = {_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"cannot write {v!r} to TOML")


def _key(k):
    return k if k.replace("_", "").replace("-", "").isalnum() else _value(k)


def dumps(run, header=""):
    lines = [f"# {line}" for line in header.splitlines()] + ([""] if header else [])
    for section in ("field", "night", "movie"):
        lines.append(f"[{section}]")
        lines += [f"{_key(k)} = {_value(v)}" for k, v in run.get(section, {}).items() if v is not None]
        lines.append("")
    for e in run.get("source", []):
        lines.append("[[source]]")
        lines += [f"{_key(k)} = {_value(v)}" for k, v in e.items() if v is not None]
        lines.append("")
    return "\n".join(lines)


def parse_value(text):
    """A command-line value as TOML (numbers, booleans, "strings", [lists], {tables}); bare words are strings."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text
