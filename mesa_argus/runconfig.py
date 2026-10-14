"""Observation files: their sections, keys and defaults (examples/observations/template.toml explains every key).

    [field]  [night]  [stars]  [[place]] (any number)  [measure]  [output]

An observation is the file's values over these defaults, with command-line overrides over both. observe.py writes
the resolved observation to outputs/<name>/observation.toml, so the same data can be made again.
"""
import copy
import tomllib
from pathlib import Path

REQUIRED = "required"


def defaults():
    import night_movie
    from mesa_argus.photometry import SETTINGS

    output = dict(movie=True, bands=["beta", "rho"], fits=False, page=False)
    engine = {a.dest: a.default for a in night_movie.parser()._actions}  # the engine's own defaults
    output.update({k: engine[k] for k in night_movie.MOVIE_OPTIONS if k != "max_airmass"})
    return dict(
        field=dict(ra=REQUIRED, dec=REQUIRED, block=3, name=""),
        night=dict(date="", start="", end="", site="argus", max_airmass=0.0),
        stars=dict(background=REQUIRED, layout="real", seed=1, missing="stop"),
        measure=dict(SETTINGS),
        output=output,
    )


PLACE = dict(star=REQUIRED, x=50.0, y=50.0, t0=None, epoch=None)


def fill(where, given, keys):
    """given over the defaults; an unknown or missing key stops the run."""
    unknown = set(given) - set(keys)
    missing = [k for k, v in keys.items() if v == REQUIRED and k not in given]
    if unknown or missing:
        raise SystemExit(f"{where}: unknown keys {sorted(unknown)}, missing keys {missing}")
    return {**copy.deepcopy(keys), **given}


def resolve(observation):
    """The observation with every default filled."""
    out = {section: fill(f"[{section}]", observation.get(section, {}), keys) for section, keys in defaults().items()}
    out["place"] = [fill(f"[[place]] {p.get('star')}", p, PLACE) for p in observation.get("place", [])]
    unknown = set(observation) - set(out)
    if unknown:
        raise SystemExit(f"unknown sections {sorted(unknown)}")
    return out


def load(path):
    return tomllib.loads(Path(path).read_text())


def merge(observation, overrides):
    """overrides {"section.key": value} (or {"place": [...]}, replacing the placed stars) applied to the file's values."""
    out = copy.deepcopy(observation)
    for dotted, v in overrides.items():
        if dotted == "place":
            out["place"] = v
        else:
            section, key = dotted.split(".")
            out.setdefault(section, {})[key] = v
    return out


def parse_value(text):
    """A command-line value as TOML (numbers, true/false, "strings", [lists], {tables}); a bare word is a string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def dumps(observation, header):
    from mesa_argus.tomlfile import table

    lines = [f"# {header}", ""]
    for section in ("field", "night", "stars", "measure", "output"):
        lines += table(f"[{section}]", observation[section])
    for p in observation["place"]:
        lines += table("[[place]]", p)
    return "\n".join(lines)
