#!/usr/bin/env python3
"""Observe built stars with Argus: one night of 60 s exposures of a pointing, measured, as a movie, a photometry CSV,
FITS exposures and/or the interactive page.

    python observe.py examples/observations/f294.toml                 # the real f294 field and its two variables
    python observe.py examples/observations/synthetic_transients.toml
    python observe.py examples/observations/f294.toml --set output.stride=10 --date 2026-07-19   # override any key

The stars must already be built (make_stars.py); an observation never simulates a star. An observation file
(mesa_argus/runconfig.py; examples/observations/template.toml lists every key) says:

    [field]      where Argus points and how big a block is shown
    [night]      which night ("auto": the night the placed real stars change most), hours, site, airmass limit
    [stars]      the constant-star background: a built star set, at its real positions or scattered at random
    [[place]]    each built star: where (x, y percent of the image; real Gaia stars sit where they really are) and when
                 (t0 for transients, epoch for periodic stars; real stars keep their own ephemeris)
    [measure]    how the exposures are measured (mesa_argus.photometry.SETTINGS: SExtractor's defaults)
    [output]     movie, bands, FITS exposures, the interactive page; noise seed, stride, PSF and kernel options

Everything about observing is added here: the Milky Way dust (SFD E(B-V) at the pointing; a real star keeps the
E(B-V) of its own sightline), positions, timing, the night's sky, seeing and airmass (ArgusSim), the instrument and
noise, and the measurement. Outputs, in outputs/<name>/: <name>_night_<band>.mp4 and .photometry.csv,
observation.toml (the resolved file: running it makes the same data again), summary.md, scene/ (what the engine
read) and fits/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
STARS = REPO / "stars"


def arguments(argv=None):
    from mesa_argus import runconfig as R

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observation", type=Path, help="observation file (TOML)")
    p.add_argument("--ra", type=float, help="[field] ra")
    p.add_argument("--dec", type=float, help="[field] dec")
    p.add_argument("--name", help="[field] name")
    p.add_argument("--date", help="[night] date: YYYY-MM-DD or auto")
    p.add_argument("--start", help="[night] start: UT HH:MM")
    p.add_argument("--end", help="[night] end: UT HH:MM")
    p.add_argument("--bands", nargs="+", choices=["beta", "rho"], help="[output] bands")
    p.add_argument("--place", action="append", default=[], metavar="'{star = \"wuma_035\", x = 30, y = 70, epoch = \"start\"}'",
                   help="a [[place]] as a TOML inline table (repeatable; replaces the file's)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="any key of the file, e.g. --set output.stride=10 (repeatable)")
    p.add_argument("--scene-only", action="store_true", help="write outputs/<name>/scene/ (and the resolved file); "
                                                              "observe nothing")
    a = p.parse_args(argv)
    run = R.load(a.observation)
    over = {}
    for flag, dotted in (("ra", "field.ra"), ("dec", "field.dec"), ("name", "field.name"), ("date", "night.date"),
                         ("start", "night.start"), ("end", "night.end"), ("bands", "output.bands")):
        if getattr(a, flag) is not None:
            over[dotted] = getattr(a, flag)
    for item in a.set:
        k, eq, v = item.partition("=")
        if not eq:
            raise SystemExit(f"--set {item}: give SECTION.KEY=VALUE")
        over[k.strip()] = R.parse_value(v.strip())
    if a.place:
        over["place"] = [R.parse_value(t if t.strip().startswith("{") else "{" + t + "}") for t in a.place]
    obs = R.resolve(R.merge(run, over))
    obs["field"]["name"] = obs["field"]["name"] or a.observation.stem
    return obs, a.scene_only


def built(name, kind="star"):
    """A built star's folder and metadata, or a clear stop."""
    folder = STARS / name
    if not (folder / "meta.json").exists():
        raise SystemExit(f"no built {kind} {name!r} in stars/: build it first (python make_stars.py <star file naming it>)")
    return folder, json.loads((folder / "meta.json").read_text())


def scene(obs, out):
    """Write the scene folder the engine reads (outputs/<name>/scene/); return (summary lines, placed rows, night)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from astropy.time import Time
    from argus_sim import skymap_shim as sh

    import night_movie as movie
    from mesa_argus import gaia as Gd
    from mesa_argus import night as N

    f, n, st = obs["field"], obs["night"], obs["stars"]
    ra, dec, name = f["ra"], f["dec"], f["name"]
    N.MAX_AIRMASS = n["max_airmass"] or None
    site = N.SITES[n["site"]]
    start_, end_ = n["start"] or None, n["end"] or None
    bg_folder, bg = built(st["background"], "star set")
    if bg["timing"] != "set":
        raise SystemExit(f"[stars] background = {st['background']!r} is a single star, not a star set")
    places = []
    for p in obs["place"]:
        if st["missing"] == "skip" and not (STARS / p["star"] / "meta.json").exists():
            print(f"[[place]] {p['star']}: not built; skipped ([stars] missing = \"skip\")", flush=True)
            continue
        places.append((p, *built(p["star"])))
    for p, _, m in places:
        if m["timing"] == "set":
            raise SystemExit(f"[[place]] star = {p['star']!r} is a star set: give it as [stars] background")
    real = [(p, fo, m) for p, fo, m in places if m.get("real")]
    # the night
    from datetime import date as Date

    if n["date"] == "auto":
        if not real:
            raise SystemExit("[night] date = \"auto\" chooses the night for placed real stars; there are none")
        picks = [dict(kind=m["real"]["kind"], period=m["real"]["period"], epoch_bjd=m["real"]["epoch_bjd"],
                      amp_G=m["real"]["amp_G"], veb=m["real"].get("veb")) for _, _, m in real]
        date, score = N.best_night(ra, dec, Date.today().isoformat(), site, start_, end_, picks)
        print(f"night chosen for the placed real stars (date auto): {date} (visibility score {score:.2f} of {len(real)})")
    else:
        asked = n["date"] or Date.today().isoformat()
        # the nearest night it is up for 3 h (or, with a UT window, at all within it)
        date, minutes, _ = N.nearest_night(ra, dec, asked, site, start_, end_, 5 if (start_ or end_) else 180)
        if date != asked:
            print(f"not up on the night of {asked}: using the nearest night it is, {date}")
    obs["night"]["date"] = date
    night = N.night_times(ra, dec, date, site, start_, end_)
    start = Time(night[0], format="jd", scale="utc")
    start_bjd = float(N.bjd_tdb(start, ra, dec, site))
    end_bjd = float(N.bjd_tdb(Time(night[-1], format="jd", scale="utc"), ra, dec, site)) + 10 / 1440

    sc = out / "scene"
    if sc.exists():
        shutil.rmtree(sc)
    sc.mkdir(parents=True)
    tid, loc = sh.radec_to_composite_id(np.array([ra]), np.array([dec]))
    tile, block = int(tid[0]), (int(loc[0]) % 64, int(loc[0]) // 64)
    movie.set_block(f["block"])
    movie.TILE, movie.BLOCK = tile, block
    ebv = Gd.reddening(ra, dec)
    rng = np.random.default_rng(st["seed"])
    cat = bg_folder / "catalogue"
    lines = [f"# {name}", "", f"RA {ra:.5f}, Dec {dec:+.5f}; night of {date}; site {n['site']}; UT window "
             f"{start_ or 'dusk'}-{end_ or 'dawn'}; Milky Way E(B-V) {ebv:.3f} (SFD at the pointing)", ""]
    if st["layout"] == "real":
        sep_ = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(*bg["centre"], unit="deg")).deg
        if sep_ > 0.02:
            raise SystemExit(f"[stars] background {st['background']} is centred {sep_:.3f} deg from the pointing; with "
                             f"layout = \"real\" it must cover it (build a Gaia field set at this pointing)")
        for c_ in cat.glob("*.ecsv"):
            shutil.copy(c_, sc / c_.name)
        lines += [f"Background: `{st['background']}` ({bg['params']}), at the stars' real positions; each constant at its "
                  "Gaia G and colour (colour-matched MESA SED).", ""]
    else:
        g = Table.read(cat / "gaia_dr3_core.ecsv")
        m = len(g)
        r, th = Gd.RADIUS_DEG * np.sqrt(rng.random(m)), 2 * np.pi * rng.random(m)
        pos = SkyCoord(ra * u.deg, dec * u.deg).spherical_offsets_by(r * np.cos(th) * u.deg, r * np.sin(th) * u.deg)
        syn = Table({"Source": np.arange(1, m + 1), "RA_ICRS": pos.ra.deg, "DE_ICRS": pos.dec.deg})
        for c_ in ("Gmag", "BPmag", "RPmag", "BP-RP"):
            syn[c_] = g[c_]
        syn.meta.update(dict(synthetic=True, parent=f"star set {st['background']}", seed=st["seed"],
                             note="Gaia G, BP, RP of every source; positions uniform at random around the pointing"))
        syn.write(sc / "synthetic_stars.ecsv", overwrite=True)
        lines += [f"Background: `{st['background']}` ({bg['params']}): its {m} stars' G and colours at positions drawn "
                  f"uniformly at random within {Gd.RADIUS_DEG} deg of the pointing (seed {st['seed']}); every star constant.", ""]
    real_ids = set()
    if st["layout"] == "real":
        real_ids = set(np.asarray(Table.read(sc / "gaia_dr3_core.ecsv")["Source"]).astype(np.int64).tolist())

    # the placed stars
    rows, extra, count = [], {}, {"transient": 0, "periodic": 0}
    for i, (p, folder, m) in enumerate(places, 1):
        file = str((folder / "sed.h5").relative_to(REPO))
        common = dict(star=p["star"], generator=m["generator"], engine=m["engine"], timing=m["timing"], kind=m["kind"],
                      model=m.get("model", m["generator"]), params=m["params"], reference=m.get("reference", ""), file=file)
        if m.get("real"):
            rl = m["real"]
            if st["layout"] != "real" or rl["source"] not in real_ids:
                raise SystemExit(f"[[place]] {p['star']} is the real Gaia DR3 {rl['source']}: it can only be observed in "
                                 f"its own field (a Gaia field set containing it, layout = \"real\")")
            if any(k in p for k in ("x", "y", "epoch", "t0")):
                raise SystemExit(f"[[place]] {p['star']} is a real star: it keeps its own position and ephemeris "
                                 f"(remove x, y, epoch)")
            rows.append(dict(common, label="", source=rl["source"], real=True, ebv=rl["sightline_ebv"],
                             period=rl["period"], epoch_bjd=rl["epoch_bjd"]))
            if rl["kind"] in ("HADS", "EB"):  # ephemerides the Gaia SOS tables do not hold (as argus_field.py did)
                extra[str(rl["source"])] = dict(period=rl["period"], epoch_bjd=rl["epoch_bjd"], kind=rl["kind"],
                                                amp=rl["amp_G"], G=rl["G"])
            continue
        t_key = "t0" if m["timing"] == "transient" else "epoch"
        if t_key not in p:
            raise SystemExit(f"[[place]] {p['star']} ({m['kind']}): {t_key} = is required (when, on this night)")
        when = N.resolve_time(p[t_key], start, ra, dec, site)
        x_pc, y_pc = (rng.uniform(0, 100) if v_ == -1 else v_ for v_ in (p.get("x", 50.0), p.get("y", 50.0)))
        count[m["timing"]] += 1
        label = f"{'T' if m['timing'] == 'transient' else 'P'}{count[m['timing']]}"
        dx, dy = (x_pc / 100 - 0.5) * 2 * movie.HALF, (y_pc / 100 - 0.5) * 2 * movie.HALF
        row = dict(common, label=label, source=-i, x_percent=x_pc, y_percent=y_pc, dx=float(dx), dy=float(dy), ebv=ebv)
        if m["timing"] == "transient":
            import h5py

            with h5py.File(folder / "sed.h5") as h:
                first, last = float(h["time_days"][0]), float(h["time_days"][-1])
            if end_bjd - when > last:
                raise SystemExit(f"[[place]] {p['star']}: the night ends {end_bjd - when:.1f} d after t0, past the model's "
                                 f"last step ({last:.1f} d)")
            if first < 0 and start_bjd - when < first:
                raise SystemExit(f"[[place]] {p['star']}: the night starts {when - start_bjd:.2f} d before t0, before the "
                                 f"model's first step ({first:.2f} d)")
            row.update(t0_bjd=when, z=float(m.get("z", np.nan)), peak_days=float(m.get("peak_days", np.nan)))
        else:
            G_, c_, amp = catalogue_entry(folder / "sed.h5", ebv)
            row.update(period=m["period"], epoch_bjd=when, G=G_, bp_rp=c_, amp_G=amp)
        rows.append(row)
    (sc / "sources.json").write_text(json.dumps(rows, indent=1, default=float))
    (sc / "extra_pulsators.json").write_text(json.dumps(extra, indent=1))
    (sc / "field.json").write_text(json.dumps(dict(
        name="field" if st["layout"] == "real" else f"synthetic, seed {st['seed']}", centre=[ra, dec], tile=tile,
        block=list(block), block_n=f["block"], date=date, site=site, synthetic=st["layout"] == "random", ebv=ebv), indent=1))
    lines += ["## Stars placed", ""]
    for r_ in rows:
        where = ("at its real position, with its own ephemeris and sightline E(B-V) "
                 f"{r_['ebv']}" if r_.get("real") else
                 f"at {r_['x_percent']:.1f}%, {r_['y_percent']:.1f}% of the image; "
                 + (f"t0 BJD {r_['t0_bjd']:.5f}" if r_["timing"] == "transient" else f"epoch BJD {r_['epoch_bjd']:.5f}"))
        lines += [f"- **{r_['label'] or 'V'}** `{r_['star']}`: {r_['kind']} ({r_['generator']}; {r_['engine']}), {where}"]
    lines += ["", "What each star is: stars/<name>/summary.md. Instrument, PSF, sky, noise and photometry: docs/ASSUMPTIONS.md.", ""]
    return lines, rows


def catalogue_entry(path, ebv):
    """Intensity-mean Gaia G, BP-RP and G amplitude of a periodic model as seen through the field's dust: what a
    catalogue would list for it (the movie calibrates against these, as against a Gaia star's)."""
    import h5py

    from mesa_argus import binary as B
    from mesa_argus.dust import redden

    with h5py.File(path) as h:
        w, flux10, t, dm = h["wavelength"][:], h["flux"][:], h["time_days"][:], float(h.attrs["distance_modulus"])
    obs = redden(w, flux10 * 10 ** (-0.4 * dm), ebv)
    wt = np.gradient(t)  # time weights: the samples need not be evenly spaced
    mags = {b: np.array([B.vega_mag(w, f_, b) for f_ in obs]) for b in ("G", "Gbp", "Grp")}
    mean = {b: float(-2.5 * np.log10(np.sum(wt * 10 ** (-0.4 * m)) / np.sum(wt))) for b, m in mags.items()}
    return mean["G"], mean["Gbp"] - mean["Grp"], float(np.ptp(mags["G"]))


def main(argv=None):
    import os

    from mesa_argus import runconfig as R

    obs, scene_only = arguments(argv)
    os.chdir(REPO)
    if not list(Path("runs").glob("*_sed/seds.h5")):  # the constant stars' SEDs (mesa_argus.seds.Templates)
        raise SystemExit("no runs/*_sed/seds.h5: run the SED tracks first (python -m mesa_argus run argus_sim_seds)")
    name = obs["field"]["name"]
    out = REPO / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    lines, rows = scene(obs, out)
    (out / "observation.toml").write_text(R.dumps(obs, f"observe.py: {name}, resolved (python observe.py "
                                                       f"{(out / 'observation.toml').relative_to(REPO)} makes it again)"))
    o, n = obs["output"], obs["night"]
    lines += ["## Outputs", ""] + [f"- {b}: `{out.relative_to(REPO) / f'{name}_night_{b}.mp4'}`" for b in o["bands"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    if scene_only:
        print(f"scene: {(out / 'scene').relative_to(REPO)}/")
        return
    import night_movie as M

    opts = []
    for k in M.MOVIE_OPTIONS:
        v = (n["max_airmass"] or None) if k == "max_airmass" else o.get(k)
        if v is not None:
            opts += [f"--{k.replace('_', '-')}", str(v)]
    if not o["movie"]:
        opts += ["--no-render"]
    for band in o["bands"]:
        mp4 = out / f"{name}_night_{band}.mp4"
        cmd = [sys.executable, "night_movie.py", "--scene", str(out / "scene"), "--field", name, "--band", band,
               "-o", str(mp4), "--measure", json.dumps(obs["measure"]), "--highlight", str(len(rows)), *opts] \
            + (["--start", n["start"]] if n["start"] else []) + (["--end", n["end"]] if n["end"] else []) \
            + (["--fits", str(out / "fits")] if o["fits"] else [])
        print(f"observing {band} -> {mp4.relative_to(REPO)} (log {mp4.with_suffix('.log').relative_to(REPO)})", flush=True)
        with open(mp4.with_suffix(".log"), "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    if o["page"]:
        print("interactive page: python demo/export_demo.py --scene " + str((out / "scene").relative_to(REPO))
              + " && python demo/build_html.py", flush=True)
        subprocess.run([sys.executable, "demo/export_demo.py", "--scene", str(out / "scene")], check=True)
        subprocess.run([sys.executable, "demo/build_html.py", name], check=True)
    print(f"done: {out.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
