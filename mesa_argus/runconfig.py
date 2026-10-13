"""Observation files: everything observe.py needs to observe stars that make_stars.py has built (examples/observations/
template.toml lists every key).

    [field]      where Argus points, how big a block is shown
    [night]      which night, which hours, which site
    [stars]      the constant-star background (a built star set) and its layout
    [[place]]    the built stars put in the field: where and when
    [measure]    how the exposures are measured (mesa_argus.photometry.SETTINGS)
    [output]     what to make (movie, FITS exposures, the interactive page) and the simulation / rendering options

Values come from, in order of precedence: the command line, the file, the defaults below. The resolved observation
(every key, the night as used) is written to outputs/<name>/observation.toml, so any dataset can be made again from it.
Machine settings (MESA paths, cores) stay in config.toml. What a star is lives in its star file (make_stars.py).
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

REQUIRED = object()


def _movie_options():
    import night_movie as M

    acts = {a.dest: a for a in M.parser()._actions}
    return {k: (acts[k].default, acts[k].type or type(acts[k].default), acts[k].help or "", ()) for k in M.MOVIE_OPTIONS
            if k != "max_airmass"}


def schema():
    """{section: {key: (default, type, help, choices)}}."""
    from mesa_argus.photometry import SETTINGS

    s = {
        "field": {
            "ra": (REQUIRED, float, "deg (ICRS)", ()),
            "dec": (REQUIRED, float, "deg (ICRS)", ()),
            "block": (3, int, "minipixes across the block shown (odd, up to 9; 9 = 253\")", ()),
            "name": ("", str, "the observation's name: outputs/<name>/ (default: this file's name)", ()),
        },
        "night": {
            "date": ("", str, "local date the night starts (YYYY-MM-DD; default today; the nearest night the field is "
                              "up if not); \"auto\": the night the placed real stars change most", ()),
            "start": ("", str, "UT HH:MM to start (default: dusk / when the field rises)", ()),
            "end": ("", str, "UT HH:MM to stop (default: dawn / when the field sets)", ()),
            "site": ("argus", str, "argus = ArgusSim's site; sro = Sierra Remote Observatories", ("argus", "sro")),
            "max_airmass": (2.0, float, "observe while the field is at or below this airmass", ()),
        },
        "stars": {
            "background": (REQUIRED, str, "a built star set (make_stars.py [[set]]): the field's constant stars", ()),
            "layout": ("real", str, "real: the set's own positions; random: its stars at random positions around "
                                    "the pointing (a synthetic field)", ("real", "random")),
            "seed": (1, int, "random layout: star positions; any placed star with x or y = -1: its position", ()),
            "missing": ("stop", str, "a placed star that is not built: stop, or skip it (e.g. a Gaia variable whose model "
                                     "failed: it is then a constant star of the background)", ("stop", "skip")),
        },
        "measure": {k: (v, type(v), "mesa_argus.photometry.SETTINGS", ()) for k, v in SETTINGS.items()},
        "output": {
            "movie": (True, bool, "render the movie (else only the photometry and any FITS)", ()),
            "bands": (["beta", "rho"], list, "one movie / photometry CSV per band", ()),
            "fits": (False, bool, "write every measured exposure, both bands, as FITS (outputs/<name>/fits/)", ()),
            "page": (False, bool, "build the interactive page (demo/; real Gaia stars only, for now)", ()),
            **_movie_options(),
        },
    }
    for k in ("n_calibrators", "back_size", "back_filtersize", "detect_minarea", "deblend_nthresh"):
        s["measure"][k] = (s["measure"][k][0], int, s["measure"][k][2], ())
    for k in ("detect_thresh", "deblend_cont", "kron_fact", "min_radius", "match_fwhm", "clip_sigma"):
        s["measure"][k] = (s["measure"][k][0], float, s["measure"][k][2], ())
    return s


PLACE = {
    "star": (REQUIRED, str, "a built star's name (stars/<name>/)", ()),
    "x": (50.0, float, "percent of the image width from the left (-1: random); not for real stars", ()),
    "y": (50.0, float, "percent of the image height from the bottom (-1: random); not for real stars", ()),
    "t0": (None, (str, float), "transients: when t0 is (ISO UTC, start+D days, or BJD_TDB)", ()),
    "epoch": (None, (str, float), "periodic stars: a time of maximum light / primary eclipse (ISO UTC, start+D, BJD_TDB); "
                                  "real stars keep their own", ()),
}


def load(path):
    return tomllib.loads(Path(path).read_text())


def merge(run, overrides):
    """run with overrides applied: {"section.key": value}, or {"place": [...]} to replace the placed stars."""
    out = copy.deepcopy(run)
    for dotted, value in overrides.items():
        if dotted == "place":
            out["place"] = copy.deepcopy(value)
            continue
        section, _, k = dotted.partition(".")
        if not k:
            raise SystemExit(f"override {dotted!r}: give section.key (e.g. field.ra, output.stride)")
        out.setdefault(section, {})[k] = value
    return out


def _check(where, keys, given):
    bad = sorted(set(given) - set(keys))
    if bad:
        raise SystemExit(f"{where}: unknown keys {bad}; it takes {sorted(keys)}")
    out = {}
    for k, (default, typ, help_, choices) in keys.items():
        if k not in given:
            if default is REQUIRED:
                raise SystemExit(f"{where}: {k} = is required ({help_})")
            out[k] = copy.deepcopy(default)
            continue
        v = given[k]
        types = typ if isinstance(typ, tuple) else (typ,)
        if float in types and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if not isinstance(v, types) or (isinstance(v, bool) and bool not in types):
            names = "/".join(t.__name__ for t in types)
            raise SystemExit(f"{where}: {k} = {v!r} should be a {names} ({help_})")
        if choices and v not in choices:
            raise SystemExit(f"{where}: {k} = {v!r} is not one of {list(choices)}")
        out[k] = v
    return out


def resolve(run):
    """Validated observation with every default filled. Raises SystemExit naming the bad key."""
    sch = schema()
    unknown = sorted(set(run) - set(sch) - {"place"})
    if unknown:
        raise SystemExit(f"observation file: unknown sections {unknown}; there are {sorted(sch) + ['place']}")
    out = {}
    for section, keys in sch.items():
        given = run.get(section, {})
        if not isinstance(given, dict):
            raise SystemExit(f"observation file: [{section}] should be a table")
        out[section] = _check(f"observation file [{section}]", keys, given)
    f = out["field"]
    if f["block"] % 2 == 0 or not 1 <= f["block"] <= 9:
        raise SystemExit(f"observation file [field]: block = {f['block']} should be odd, 1-9")
    for b in out["output"]["bands"]:
        if b not in ("beta", "rho"):
            raise SystemExit(f"observation file [output]: bands = {out['output']['bands']}: each one beta or rho")
    places = run.get("place", [])
    out["place"] = []
    for e in [places] if isinstance(places, dict) else places:
        q = _check(f"[[place]] {e.get('star', '?')}", PLACE, e)
        for k in ("x", "y"):
            if not (q[k] == -1 or 0 <= q[k] <= 100):
                raise SystemExit(f"[[place]] {q['star']}: {k} = {q[k]} is a percentage of the image (0-100), or -1")
        out["place"].append({k: v for k, v in q.items() if v is not None and (k not in ("x", "y") or k in e)})
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
    for section in ("field", "night", "stars", "measure", "output"):
        lines.append(f"[{section}]")
        lines += [f"{_key(k)} = {_value(v)}" for k, v in run.get(section, {}).items() if v is not None]
        lines.append("")
    for e in run.get("place", []):
        lines.append("[[place]]")
        lines += [f"{_key(k)} = {_value(v)}" for k, v in e.items() if v is not None]
        lines.append("")
    return "\n".join(lines)


def parse_value(text):
    """A command-line value as TOML (numbers, booleans, "strings", [lists], {tables}); bare words are strings."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text
