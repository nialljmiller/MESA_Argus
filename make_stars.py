#!/usr/bin/env python3
"""Simulate stars: each [[star]] / [[set]] of a star file is built into its own folder, stars/<name>/.

    python make_stars.py examples/stars/variables.toml          # build every star in the file (skips those built)
    python make_stars.py examples/stars/f294_field.toml examples/stars/transients.toml   # several files
    python make_stars.py examples/stars/variables.toml --only wuma_035
    python make_stars.py examples/stars/variables.toml --rebuild wuma_035   # rebuild one whose definition changed
    python make_stars.py --gaia-list 294.7844 -11.6235 --write my_field.toml --write-observation my_obs.toml
    python make_stars.py --gaia-list 294.78 -11.62 --gaia-list 273.04 -20.71 --write s.toml --write-batch b.toml
    python make_stars.py --list                                  # every built star

A star knows nothing about observing: no dust, no position in an image, no night, no instrument (observe.py adds
those). Generators: python -m mesa_argus.sources (keys, defaults). Each star folder holds

    star.toml    the definition it was built from, every key (defaults filled)
    meta.json    what was built: kind, engine, model parameters, reference; real stars also their sky position,
                 ephemeris and sightline E(B-V)
    sed.h5       the model: SEDs against time (transients, before dust), or one cycle at 10 pc (periodic)
    catalogue/   (sets) the catalogue files
    summary.md   the same, for reading

A star is built once. Running the file again skips stars whose definition is unchanged; a changed definition under
an existing name is refused unless named with --rebuild.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
STARS = REPO / "stars"


def _toml_table(name, d):
    from mesa_argus.runconfig import _key, _value

    return "\n".join([f"[{name}]"] + [f"{_key(k)} = {_value(v)}" for k, v in d.items()]) + "\n"


def load(*paths):
    """[(table, Generator, definition)] of one or more star files, validated. The same name may appear in several
    files only with the same definition (e.g. a field's star set in both its own file and its variables' file)."""
    from mesa_argus import sources as S

    out, seen = [], {}
    for path in paths:
        doc = tomllib.loads(Path(path).read_text())
        bad = sorted(set(doc) - {"star", "set"})
        if bad:
            raise SystemExit(f"{path}: unknown tables {bad}; a star file has [[star]] and [[set]] entries only")
        n = 0
        for table in ("set", "star"):
            for e in doc.get(table, []):
                g, p = S.validate(e, table)
                n += 1
                if p["name"] in seen:
                    if seen[p["name"]] != (table, p):
                        raise SystemExit(f"{path}: {p['name']!r} is defined differently elsewhere")
                    continue
                seen[p["name"]] = (table, p)
                out.append((table, g, p))
        if not n:
            raise SystemExit(f"{path}: no [[star]] or [[set]] entries")
    return out


def state(table, g, p, rebuild):
    """'built' (unchanged, present), 'build' or raises if the definition changed and is not to be rebuilt."""
    folder = STARS / p["name"]
    if not (folder / "star.toml").exists():
        return "build"
    old = tomllib.loads((folder / "star.toml").read_text()).get(table, {})
    ok = (folder / ("catalogue" if g.timing == "set" else "sed.h5")).exists()
    if old == p and ok:
        return "built"
    if p["name"] in rebuild:
        return "build"
    raise SystemExit(f"stars/{p['name']} was built from another definition: name this one differently, or rebuild it "
                     f"with --rebuild {p['name']}")


def save(table, g, p, meta, folder):
    product = Path(meta.pop("product"))
    tmp = folder.with_name(folder.name + ".new")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if product.is_dir():
        shutil.copytree(product, tmp / "catalogue")
    else:
        shutil.copy(product, tmp / "sed.h5")
    engine = meta.pop("engine", None) or meta.pop("generator", None) or g.engine  # transients call theirs "generator"
    meta.pop("generator", None)
    meta = dict(name=p["name"], generator=g.name, engine=engine, timing=g.timing,
                built=time.strftime("%Y-%m-%d %H:%M:%S"), **meta)

    def plain(o):
        return o.tolist() if hasattr(o, "tolist") else str(o)

    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, default=plain))
    (tmp / "star.toml").write_text(f"# made by make_stars.py; stars/{p['name']}/\n" + _toml_table(table, p))
    lines = [f"# {p['name']}", "", f"**{meta['kind']}** — generator `{g.name}` ({meta['engine']}; {meta.get('reference', '')})",
             "", f"- {meta['params']}", f"- timing: {g.timing}" + (f", period {meta['period']:g} d" if meta.get("period") else "")]
    if meta.get("real"):
        r = meta["real"]
        lines.append(f"- a real star: Gaia DR3 {r['source']} at RA {r['ra']:.5f}, Dec {r['dec']:+.5f}; ephemeris epoch BJD "
                     f"{r['epoch_bjd']:.5f}; sightline E(B-V) {r['sightline_ebv']}")
    for k, v in (meta.get("notes") or {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "Definition: star.toml. Dust, position, timing and instrument are the observation's (observe.py)."]
    (tmp / "summary.md").write_text("\n".join(lines) + "\n")
    if folder.exists():
        shutil.rmtree(folder)
    tmp.rename(folder)


def build(paths, only=(), rebuild=()):
    from mesa_argus import config as C
    from mesa_argus import sources as S

    entries = [e for e in load(*paths) if not only or e[2]["name"] in only]
    todo = []
    for table, g, p in entries:
        st = state(table, g, p, set(rebuild))
        print(f"  {p['name']}: {g.name} — {'built already' if st == 'built' else 'to build'}", flush=True)
        if st == "build":
            todo.append((table, g, p, S.Context(p["name"], REPO / "runs")))
    for table, g, p, ctx in todo:  # one at a time: network queries, shared module state
        if g.prepare:
            g.prepare(p, ctx)
    failed = []

    def one(item):
        table, g, p, ctx = item
        work = REPO / "runs" / "stars" / p["name"]
        work.mkdir(parents=True, exist_ok=True)
        print(f"  {p['name']}: building ({g.engine}) …", flush=True)
        meta = g.make(work, p, ctx)
        if meta is None:
            failed.append(p["name"])
            print(f"  {p['name']}: NOT built (the engine found no acceptable model; see runs/)", flush=True)
            return
        save(table, g, p, meta, STARS / p["name"])
        print(f"  {p['name']}: built -> stars/{p['name']}/ ({meta['kind']}; {meta['params']})", flush=True)

    serial = [t for t in todo if not (t[1].parallel if not callable(t[1].parallel) else t[1].parallel(t[3]))
              or t[1].timing == "transient"]
    for item in serial:
        one(item)
    rest = [t for t in todo if t not in serial]
    if rest:
        with ThreadPoolExecutor(int(C.config()["run"]["workers"])) as ex:
            list(ex.map(one, rest))
    if failed:
        raise SystemExit(f"not built: {failed}")


def gaia_list(pointings, block, write, write_batch=None, write_observation=None):
    """The catalogued variables around each pointing, and a star file for them: each pointing's Gaia field set and
    every variable that can be modelled. write_batch: also an observation batch observing each pointing (its set at
    the real positions, its variables, the night they change most), for tools/stage_batch.py."""
    from mesa_argus import gaia as Gd
    from mesa_argus.runconfig import _value
    from mesa_argus.sources.gaia import CACHE

    text = ["# Gaia fields: each one's stars as a set, and each catalogued variable that can be modelled",
            "# (Gaia variables: MESA RSP ~0.5-2 h each, PHOEBE minutes).",
            f"#   python make_stars.py {write or 'this_file.toml'}", ""]
    fields = []
    for ra, dec in pointings:
        cache = CACHE / Gd.field_name(ra, dec)
        Gd.fetch(ra, dec, cache)
        rows, stars, _, _ = Gd.inventory(cache, ra, dec, block)
        print(f"\nRA {ra}, Dec {dec}:")
        Gd.show(rows, block)
        name = Gd.field_name(ra, dec)
        text += ["[[set]]", f'name = "{name}_field"', 'generator = "Gaia field"', f"ra = {ra}", f"dec = {dec}", ""]
        variables = []
        for r in rows:
            if r["model"]:
                text += [f"[[star]]   # {r['kind']}, G {r['G']:.2f}, P {r['period']:.5f} d ({r['model']})",
                         f'name = "gaia_{r["source"]}"', 'generator = "Gaia variable"', f"source = {r['source']}", ""]
                variables.append(f"gaia_{r['source']}")
        fields.append((ra, dec, f"{name}_field", variables))
    if write:
        Path(write).write_text("\n".join(text))
        print(f"\nwrote {write}")
    else:
        print("\n" + "\n".join(text))
    if write_observation:  # one observation file per field: its set at the real positions, every variable placed
        wo = Path(write_observation)
        for i, (ra, dec, bg, vs) in enumerate(fields):
            path = wo if len(fields) == 1 else wo.with_name(f"{wo.stem}_{bg[:-6]}{wo.suffix}")
            path.write_text("\n".join(
                [f"# Observe the Gaia field at RA {ra}, Dec {dec} on the night its variables change most.",
                 f"#   python make_stars.py {write or '<the star file>'}   # once",
                 f"#   python observe.py {path}", "", "[field]", f"ra = {ra}", f"dec = {dec}", f"block = {block}", "",
                 "[night]", 'date = "auto"', "", "[stars]", f"background = {_value(bg)}", 'layout = "real"',
                 'missing = "skip"     # a variable whose model failed stays a constant star', ""]
                + sum((["[[place]]", f"star = {_value(v)}", ""] for v in vs), [])))
            print(f"wrote {path}")
    if write_batch:
        wb = Path(write_batch)
        base = wb.with_name(wb.stem + "_base.toml")
        base.write_text("\n".join([
            "# The observation each pointing of the batch gets (the batch sets the pointing, background and stars).",
            "[field]", f"ra = {fields[0][0]}", f"dec = {fields[0][1]}", f"block = {block}", "",
            "[night]", 'date = "auto"', "", "[stars]", f'background = "{fields[0][2]}"', 'layout = "real"',
            'missing = "skip"     # a variable whose model failed stays a constant star', ""]) + "\n")
        lines = [f"# Observe each Gaia field of {write or 'the star file'} on the night its variables change most.",
                 f"#   python make_stars.py {write or '<star file>'} && python tools/stage_batch.py {wb}", "",
                 f'base = "{base.name}"', 'combine = "zip"', "", "[vary]",
                 '"field.ra,field.dec,stars.background,field.name" = ['] + [
                 f"  [{ra}, {dec}, {_value(bg)}, {_value(bg[:-6])}]," for ra, dec, bg, _ in fields] + [
                 "]", "place = ["] + [f"  [{', '.join('{star = ' + _value(v) + '}' for v in vs)}]," for *_, vs in fields] + ["]", ""]
        wb.write_text("\n".join(lines))
        print(f"wrote {wb} and {base}")


def list_built():
    for meta in sorted(STARS.glob("*/meta.json")):
        m = json.loads(meta.read_text())
        print(f"  {m['name']:32s} {m['timing']:9s} {m['kind']:16s} {m['generator']} ({m['engine']})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stars", nargs="*", type=Path, help="star file(s) (TOML)")
    p.add_argument("--only", nargs="+", default=[], help="build only these names")
    p.add_argument("--rebuild", nargs="+", default=[], help="rebuild these names even though they are built")
    p.add_argument("--gaia-list", nargs=2, type=float, action="append", metavar=("RA", "DEC"),
                   help="list a Gaia field's variables (repeatable: several fields)")
    p.add_argument("--block", type=int, default=9, help="--gaia-list: minipixes across the block searched (odd, <= 9)")
    p.add_argument("--write", help="--gaia-list: write the star file here")
    p.add_argument("--write-observation", help="--gaia-list: also write an observation file (one per field) here")
    p.add_argument("--write-batch", help="--gaia-list: also write an observation batch for the fields here")
    p.add_argument("--list", action="store_true", help="list the built stars")
    a = p.parse_args(argv)
    if a.list:
        return list_built()
    if a.gaia_list:
        return gaia_list(a.gaia_list, a.block, a.write, a.write_batch, a.write_observation)
    if not a.stars:
        p.error("give a star file (or --gaia-list / --list)")
    build(a.stars, a.only, a.rebuild)


if __name__ == "__main__":
    main()
