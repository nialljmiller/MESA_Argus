#!/usr/bin/env python3
"""Observe built stars with Argus: one night of 60 s exposures of a pointing, measured, as a movie, a photometry CSV,
FITS exposures and/or the interactive page.

    python observe.py examples/observations/f294.toml
    python observe.py examples/observations/synthetic_transients.toml --set output.stride=10 --date 2026-07-19

Stars must be built first (make_stars.py); an observation only names them. The observation file
(examples/observations/template.toml lists every key):

    [field]    where Argus points and how big a block is shown
    [night]    which night ("auto": the night the placed real stars change most), UT window, site
    [stars]    the background: a built star set, at its real positions or scattered at random (a synthetic field)
    [[place]]  each star: where (x, y in percent of the image; real Gaia stars sit where they are) and when (t0 for
               transients, epoch for periodic stars; real stars keep their own ephemeris)
    [measure]  how the exposures are measured (SExtractor settings)
    [output]   movie, bands, FITS, page; noise seed, stride, PSF and kernel options

Everything about observing is added here: the pointing, the night (ArgusSim's schedule), the Milky Way dust (the 3D map
at each star's distance along its sightline, mesa_argus/dust.py), positions and timing. night_movie.py then simulates and
measures the exposures from outputs/<name>/scene/.

Outputs in outputs/<name>/: <name>_night_<band>.mp4 and .photometry.csv, fits/, observation.toml (the resolved file:
running it makes the same data), summary.md, scene/.
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import date as Date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
STARS = REPO / "stars"

from mesa_argus import dust, runconfig  # noqa: E402


def read_observation():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observation", type=Path)
    p.add_argument("--ra", type=float)
    p.add_argument("--dec", type=float)
    p.add_argument("--name")
    p.add_argument("--date", help="YYYY-MM-DD or auto")
    p.add_argument("--start", help="UT HH:MM")
    p.add_argument("--end", help="UT HH:MM")
    p.add_argument("--bands", nargs="+")
    p.add_argument("--place", action="append", default=[], help="a [[place]] as a TOML inline table (replaces the file's)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="any key, e.g. output.stride=10")
    p.add_argument("--scene-only", action="store_true", help="write the scene and stop")
    a = p.parse_args()
    flags = (("ra", "field.ra"), ("dec", "field.dec"), ("name", "field.name"), ("date", "night.date"),
             ("start", "night.start"), ("end", "night.end"), ("bands", "output.bands"))
    overrides = {dotted: getattr(a, flag) for flag, dotted in flags if getattr(a, flag) is not None}
    for item in a.set:
        dotted, _, v = item.partition("=")
        overrides[dotted.strip()] = runconfig.parse_value(v.strip())
    if a.place:
        overrides["place"] = [runconfig.parse_value(t if t.strip().startswith("{") else "{" + t + "}") for t in a.place]
    obs = runconfig.resolve(runconfig.merge(runconfig.load(a.observation), overrides))
    obs["field"]["name"] = obs["field"]["name"] or a.observation.stem
    return obs, a.scene_only


def built_star(name):
    folder = STARS / name
    if not (folder / "meta.json").exists():
        raise SystemExit(f"no built star {name!r}: build it first (python make_stars.py <a star file naming it>)")
    return folder, json.loads((folder / "meta.json").read_text())


def choose_night(f, n, window, real_stars, site, max_airmass):
    """The night asked for, or the nearest one the field is up >= 3 h (with a UT window: at all), or "auto"."""
    from mesa_argus import night

    if n["date"] == "auto":  # the night the placed real stars change most
        picks = [dict(kind=r["kind"], period=r["period"], epoch_bjd=r["epoch_bjd"], amp_G=r["amp_G"], veb=r.get("veb"))
                 for r in real_stars]
        date, score = night.best_night(f["ra"], f["dec"], Date.today().isoformat(), site, *window, max_airmass, picks)
        print(f"night chosen for the placed real stars (date auto): {date} (visibility score {score:.2f} of {len(picks)})")
        return date
    asked = n["date"] or Date.today().isoformat()
    shortest = 5 if any(window) else 180  # minutes
    date, minutes = night.nearest_night(f["ra"], f["dec"], asked, site, *window, max_airmass, shortest)
    if date != asked:
        print(f"not up on the night of {asked}: using the nearest night it is, {date}")
    return date


def write_background(scene_dir, s, star_set, ra, dec, rng):
    """The background stars into the scene: the set's catalogue at the stars' real positions, or at random positions
    within 0.12 deg of the pointing (synthetic). Each star gets its distance and two E(B-V): in front of it where Gaia
    measured it (ebv_own), and where it is observed (ebv); the engine takes the first out and puts the second in.
    Returns a line for the summary."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    from mesa_argus.gaia import RADIUS_DEG, col, distances

    catalogue = STARS / s["background"] / "catalogue"
    gaia_stars = Table.read(catalogue / "gaia_dr3_core.ecsv")
    n = len(gaia_stars)
    distance, how = distances(catalogue, gaia_stars["Source"])
    real_ra, real_dec = col(gaia_stars, "RA_ICRS"), col(gaia_stars, "DE_ICRS")
    ebv_own, which = dust.ebv(real_ra, real_dec, distance)
    counts = (f"distances: {int(np.sum(how == 'Bailer-Jones'))} Bailer-Jones, {int(np.sum(how == 'GSP-Phot'))} GSP-Phot, "
              f"{int(np.sum(how == 'none'))} none (the whole column); dust from {'Bayestar19' if (which == 'bayestar19').all() else 'Bayestar19 and SFD'}")
    if s["layout"] == "real":
        offset = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(*star_set["centre"], unit="deg")).deg
        if offset > 0.02:  # real positions only make sense at the set's own pointing
            raise SystemExit(f"background {s['background']} is centred {offset:.3f} deg from the pointing")
        for f in catalogue.glob("*.ecsv"):
            shutil.copy(f, scene_dir / f.name)
        gaia_stars["distance_pc"], gaia_stars["ebv_own"], gaia_stars["ebv"] = distance, ebv_own, ebv_own
        gaia_stars.write(scene_dir / "gaia_dr3_core.ecsv", overwrite=True)
        return (f"Background: `{s['background']}` ({star_set['params']}), at the stars' real positions; each constant at "
                f"its Gaia G and colour (colour-matched MESA SED through the dust in front of it: {counts}).")
    radius, angle = RADIUS_DEG * np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)  # uniform over the circle
    pos = SkyCoord(ra * u.deg, dec * u.deg).spherical_offsets_by(radius * np.cos(angle) * u.deg,
                                                                 radius * np.sin(angle) * u.deg)
    synthetic = Table({"Source": np.arange(1, n + 1), "RA_ICRS": pos.ra.deg, "DE_ICRS": pos.dec.deg})
    for column in ("Gmag", "BPmag", "RPmag", "BP-RP"):
        synthetic[column] = gaia_stars[column]
    synthetic["distance_pc"], synthetic["ebv_own"] = distance, ebv_own
    synthetic["ebv"] = dust.ebv(pos.ra.deg, pos.dec.deg, distance)[0]
    synthetic.meta.update(dict(synthetic=True, parent=f"star set {s['background']}", seed=s["seed"],
                               note="Gaia G, BP, RP of every source (as Gaia measured them, through ebv_own); positions "
                                    "uniform at random around the pointing, seen through ebv (the engine re-reddens)"))
    synthetic.write(scene_dir / "synthetic_stars.ecsv", overwrite=True)
    return (f"Background: `{s['background']}` ({star_set['params']}): its {n} stars at positions drawn uniformly at "
            f"random within {RADIUS_DEG} deg of the pointing (seed {s['seed']}), at their own distances, each seen through "
            f"the dust in front of it there ({counts}); every star constant.")


def catalogue_entry(sed_file, ebv):
    """Intensity-mean Gaia G, BP-RP and G amplitude of a periodic model through its dust: what a catalogue would
    list for it (the movie calibrates against these, as for a Gaia star)."""
    import h5py

    from mesa_argus.binary import vega_mag

    with h5py.File(sed_file) as h:
        wave, flux_10pc, t, dm = h["wavelength"][:], h["flux"][:], h["time_days"][:], float(h.attrs["distance_modulus"])
    seen = dust.redden(wave, flux_10pc * 10 ** (-0.4 * dm), ebv)
    weight = np.gradient(t)  # the samples need not be evenly spaced in time
    mags = {band: np.array([vega_mag(wave, f, band) for f in seen]) for band in ("G", "Gbp", "Grp")}
    mean = {band: -2.5 * np.log10(np.sum(weight * 10 ** (-0.4 * m)) / np.sum(weight)) for band, m in mags.items()}
    return float(mean["G"]), float(mean["Gbp"] - mean["Grp"]), float(np.ptp(mags["G"]))


def placed_distance(folder, meta):
    """A placed star's distance (pc): a periodic model's, from its file; a transient's if it has one (a dwarf nova);
    none for a supernova or a microlensing source (they sit behind all the dust)."""
    import h5py

    if meta["timing"] == "periodic":
        with h5py.File(folder / "sed.h5") as h:
            return 10 ** (float(h.attrs["distance_modulus"]) / 5 + 1)
    return meta.get("distance_pc", meta.get("d_pc"))


def place_stars(placements, scene_dir, night_info, half_width, rng):
    """One row per placed star for the engine (sources.json): its SED file, position, timing and dust (the 3D map at
    its distance on the pointing's sightline; a real star, at its own position and distance); and the
    ephemerides of real HADS / binaries, which the Gaia SOS tables do not hold (extra_pulsators.json)."""
    import h5py
    from astropy.table import Table

    from mesa_argus import night

    start, night_start_bjd, night_end_bjd, ra, dec, site = night_info
    catalogue = scene_dir / "gaia_dr3_core.ecsv"
    in_catalogue = set(np.asarray(Table.read(catalogue)["Source"]).astype(np.int64).tolist()) if catalogue.exists() else set()
    rows, extra, count = [], {}, {"transient": 0, "periodic": 0}
    for i, (p, folder, meta) in enumerate(placements, 1):
        row = dict(star=p["star"], generator=meta["generator"], engine=meta["engine"], timing=meta["timing"],
                   kind=meta["kind"], model=meta.get("model", meta["generator"]), params=meta["params"],
                   reference=meta.get("reference", ""), file=str((folder / "sed.h5").relative_to(REPO)))

        if meta.get("real"):  # a real star: its own position, ephemeris and sightline dust; already in the catalogue
            real = meta["real"]
            if real["source"] not in in_catalogue:
                raise SystemExit(f"{p['star']} is Gaia DR3 {real['source']}: observe it in its own field (layout real)")
            rows.append(dict(row, label="", source=real["source"], real=True, ebv=real["sightline_ebv"],
                             period=real["period"], epoch_bjd=real["epoch_bjd"]))
            if real["kind"] in ("HADS", "EB"):
                extra[str(real["source"])] = dict(period=real["period"], epoch_bjd=real["epoch_bjd"], kind=real["kind"],
                                                  amp=real["amp_G"], G=real["G"])
            continue

        timing = meta["timing"]
        when = night.resolve_time(p["t0"] if timing == "transient" else p["epoch"], start, ra, dec, site)
        x, y = (rng.uniform(0, 100) if v == -1 else float(v) for v in (p["x"], p["y"]))  # -1: random
        count[timing] += 1
        label = ("T" if timing == "transient" else "P") + str(count[timing])
        dx, dy = (x / 100 - 0.5) * 2 * half_width, (y / 100 - 0.5) * 2 * half_width  # arcsec from the block centre
        distance = placed_distance(folder, meta)
        ebv = float(dust.ebv(ra, dec, distance)[0][0])
        row.update(label=label, source=-i, x_percent=x, y_percent=y, dx=float(dx), dy=float(dy), ebv=ebv,
                   distance_pc=float(distance) if distance else None)
        if timing == "transient":
            with h5py.File(folder / "sed.h5") as h:
                first, last = float(h["time_days"][0]), float(h["time_days"][-1])
            if night_end_bjd - when > last or (first < 0 and night_start_bjd - when < first):  # no extrapolation
                raise SystemExit(f"{p['star']}: the night runs outside its model ({first:.2f} to {last:.2f} d from t0)")
            row.update(t0_bjd=when, z=float(meta.get("z", np.nan)), peak_days=float(meta.get("peak_days", np.nan)))
        else:
            G, bp_rp, amplitude = catalogue_entry(folder / "sed.h5", ebv)
            row.update(period=meta["period"], epoch_bjd=when, G=G, bp_rp=bp_rp, amp_G=amplitude)
        rows.append(row)
    return rows, extra


def run_engine(obs, out, window, n_placed):
    import night_movie

    name, n, o = obs["field"]["name"], obs["night"], obs["output"]
    options = []
    for k in night_movie.MOVIE_OPTIONS:
        v = (n["max_airmass"] or None) if k == "max_airmass" else o.get(k)
        if v is not None:
            options += [f"--{k.replace('_', '-')}", str(v)]
    if not o["movie"]:
        options += ["--no-render"]
    if window[0]:
        options += ["--start", window[0]]
    if window[1]:
        options += ["--end", window[1]]
    if o["fits"]:
        options += ["--fits", str(out / "fits")]
    for band in o["bands"]:
        mp4 = out / f"{name}_night_{band}.mp4"
        print(f"observing {band} -> {mp4.relative_to(REPO)} (log {mp4.with_suffix('.log').relative_to(REPO)})", flush=True)
        with open(mp4.with_suffix(".log"), "w") as log:
            subprocess.run([sys.executable, "night_movie.py", "--scene", str(out / "scene"), "--field", name, "--band", band,
                            "-o", str(mp4), "--measure", json.dumps(obs["measure"]), "--highlight", str(n_placed), *options],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    if o["page"]:
        subprocess.run([sys.executable, "demo/export_demo.py", "--scene", str(out / "scene")], check=True)
        subprocess.run([sys.executable, "demo/build_html.py", name], check=True)


def main():
    import os

    from argus_sim import skymap_shim
    from astropy.time import Time

    import night_movie
    from mesa_argus import night
    obs, scene_only = read_observation()
    os.chdir(REPO)
    f, n, s = obs["field"], obs["night"], obs["stars"]
    name, ra, dec = f["name"], f["ra"], f["dec"]
    window = (n["start"] or None, n["end"] or None)  # UT HH:MM, or the whole dark night
    out = REPO / "outputs" / name
    scene_dir = out / "scene"
    shutil.rmtree(scene_dir, ignore_errors=True)
    scene_dir.mkdir(parents=True)

    # the stars
    star_set = built_star(s["background"])[1]
    placements = []
    for p in obs["place"]:
        if s["missing"] == "skip" and not (STARS / p["star"] / "meta.json").exists():
            print(f"{p['star']}: not built; skipped ([stars] missing = \"skip\")")
            continue
        placements.append((p, *built_star(p["star"])))

    # the night
    site = night.SITES[n["site"]]
    max_airmass = n["max_airmass"] or None
    n["date"] = choose_night(f, n, window, [m["real"] for _, _, m in placements if m.get("real")], site, max_airmass)
    exposures = night.exposures(ra, dec, n["date"], site, *window, max_airmass)[0].jd
    start = Time(exposures[0], format="jd", scale="utc")
    night_start_bjd = float(night.bjd_tdb(start, ra, dec, site))
    night_end_bjd = float(night.bjd_tdb(Time(exposures[-1], format="jd", scale="utc"), ra, dec, site)) + 10 / 1440

    # the field: HDPS tile and block, dust, background, placed stars
    tile_id, minipix = skymap_shim.radec_to_composite_id(np.array([ra]), np.array([dec]))
    tile, block = int(tile_id[0]), (int(minipix[0]) % 64, int(minipix[0]) // 64)
    night_movie.set_block(f["block"])
    ebv = float(dust.ebv(ra, dec)[0][0])  # the whole column at the pointing (for the summary)
    rng = np.random.default_rng(s["seed"])  # background positions first, then any random placements
    background_note = write_background(scene_dir, s, star_set, ra, dec, rng)
    night_info = (start, night_start_bjd, night_end_bjd, ra, dec, site)
    rows, extra = place_stars(placements, scene_dir, night_info, night_movie.HALF, rng)
    (scene_dir / "sources.json").write_text(json.dumps(rows, indent=1, default=float))
    (scene_dir / "extra_pulsators.json").write_text(json.dumps(extra, indent=1))
    field_json = dict(name="field" if s["layout"] == "real" else f"synthetic, seed {s['seed']}", centre=[ra, dec],
                      tile=tile, block=list(block), block_n=f["block"], date=n["date"], site=site,
                      synthetic=s["layout"] == "random", ebv=ebv)
    (scene_dir / "field.json").write_text(json.dumps(field_json, indent=1))

    # what was done, for reading and for making it again
    resolved = out / "observation.toml"
    resolved.write_text(runconfig.dumps(obs, f"observe.py: {name}, resolved (python observe.py "
                                             f"{resolved.relative_to(REPO)} makes it again)"))
    summary = [f"# {name}", "", f"RA {ra:.5f}, Dec {dec:+.5f}; night of {n['date']}; site {n['site']}; UT window "
               f"{window[0] or 'dusk'}-{window[1] or 'dawn'}; Milky Way E(B-V) {ebv:.3f} (the whole column at the pointing; each "
               "star sees the 3D map's dust in front of it, mesa_argus/dust.py)", "",
               background_note, "", "## Stars placed", ""]
    for r in rows:
        if r.get("real"):
            where = f"at its real position, own ephemeris and E(B-V) {r['ebv']:.3f} in front of it"
        else:
            when = f"t0 BJD {r['t0_bjd']:.5f}" if r["timing"] == "transient" else f"epoch BJD {r['epoch_bjd']:.5f}"
            far = f"{r['distance_pc']:.0f} pc" if r.get("distance_pc") else "behind all the dust"
            where = f"at {r['x_percent']:.1f}%, {r['y_percent']:.1f}% of the image; {when}; E(B-V) {r['ebv']:.3f} ({far})"
        summary.append(f"- **{r['label'] or 'V'}** `{r['star']}`: {r['kind']} ({r['generator']}; {r['engine']}), {where}")
    summary += ["", "What each star is: stars/<name>/summary.md. Instrument, PSF, sky, noise, photometry: docs/ASSUMPTIONS.md."]
    (out / "summary.md").write_text("\n".join(summary) + "\n")
    if scene_only:
        print(f"scene: {scene_dir.relative_to(REPO)}/")
        return
    run_engine(obs, out, window, len(rows))
    print(f"done: {out.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
