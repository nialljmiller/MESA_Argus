# MESA_ARGUS

Stellar models from MESA, seen through the **Argus Array**. The repo does two things:

1. **Night movies of Argus fields.** One simulated night of Argus 60 s exposures on any pointing: every star an SED
   (MESA pulsation models, PHOEBE eclipsing binaries, supernovae and other transients), turned into images with
   ArgusSim's instrument, sky and noise, and measured with SExtractor. The output is a movie and a photometry CSV.
2. **How much stellar physics Argus photometry can recover**, compared with Gaia and SDSS: MESA tracks with synthetic
   photometry in all three systems, an interactive 3D CMD, and a Fisher-information analysis.

Both use the same MESA install and the same Argus throughputs, from ArgusSim.

## Setup

Requirements: a working MESA install with `$MESA_DIR` set and its colors data, ArgusSim installed in the same
Python (≥ 3.11), and `pip install -r requirements.txt`. The movies also need `ffmpeg`, and the dust maps
`bash tools/fetch_dust.sh` (once per machine).

```bash
$EDITOR config.toml               # machine settings: colors-data path, parallelism (usually only the path)
make -C mesa_work                 # build MESA's star executable
python -m mesa_argus setup        # assemble the filter set (Gaia + Argus from your colors data, SDSS from data/)
python -m mesa_argus run argus_sim_seds   # the stellar SEDs every movie uses for its constant stars (once)
```

On macOS, wrap long runs in `caffeinate -dims` so sleep does not suspend MESA.

## 1. Night movies: stars, then observations

Two scripts, two kinds of file you write:

| Step | Command | Reads | Writes |
|---|---|---|---|
| simulate stars | `python make_stars.py <star file> [...]` | a **star file**: `[[star]]` and `[[set]]` entries | `stars/<name>/`, one folder per star |
| observe them | `python observe.py <observation file>` | an **observation file**: pointing, night, stars placed, measurement, output | `outputs/pseudo_real/<name>/` or `outputs/fully_synthetic/<name>/`: movie, photometry, FITS |

A star knows nothing about observing: no dust, no position in an image, no night, no instrument. An observation never
simulates a star: it names stars that are already built and adds everything about observing.

### Quick start

```bash
python make_stars.py examples/stars/f294_field.toml examples/stars/transients.toml   # build stars (seconds)
python observe.py examples/observations/synthetic_transients.toml --set output.stride=30   # quick look: every 30th exposure
python observe.py examples/observations/synthetic_transients.toml                          # the whole night (~30 min per band)
```

The first command builds a background (every Gaia star around RA 294.78, Dec −11.62, as constant stars) and four
transients. The second observes a synthetic field (those stars scattered at random) with a microlensing event and a
dwarf nova on the night of 2026-07-19, and writes `outputs/synthetic_transients/synthetic_transients_night_rho.mp4`
and its photometry.

The examples:

| Star file (`examples/stars/`) | Builds | Time |
|---|---|---|
| `f294_field.toml` | the background set for every example observation | seconds |
| `transients.toml` | a microlensing event, a dwarf nova, a SN Ia | seconds |
| `variables.toml` | a W UMa and a detached eclipsing binary (PHOEBE), an RRab (MESA RSP) | minutes; ~1 h for the RRab |
| `f294.toml` | the real f294 field's set and its two Gaia variables (MESA RSP) | ~1–4 h |

| Observation file (`examples/observations/`) | Needs | What it shows |
|---|---|---|
| `synthetic_transients.toml` | f294_field, transients | a short microlensing event and a dwarf nova rising |
| `synthetic_variables.toml` | f294_field, variables | the three hand-made periodic variables |
| `f294.toml` | f294 | the real field with its RRab and HADS on their real ephemerides |
| `template.toml` | — | every key, with its default and meaning |

### Star files

```toml
[[star]]
name = "wuma_035"            # its folder: stars/wuma_035/
generator = "W UMa"          # python -m mesa_argus.sources lists every generator and its keys
period = 0.35
q = 0.4
incl = 80
teff1 = 5800
teff2 = 5600
fillout = 0.2
distance = 620

[[set]]
name = "f294_784m11_623_field"
generator = "Gaia field"     # every Gaia star around a pointing, as constant stars
ra = 294.7844
dec = -11.6235
```

Each star is built once, into:

```
stars/<name>/
  star.toml     the definition it was built from (every key)
  meta.json     what the build produced: kind, engine, model parameters; real stars also position, ephemeris,
                sightline E(B-V)
  sed.h5        the model: SEDs against time (transients, before any dust) or one cycle at 10 pc (periodic)
  catalogue/    (sets) the catalogue files
  summary.md    the same, to read
```

Running a star file again skips stars already built. A changed definition under an existing name is refused unless
named with `--rebuild NAME`; `--only NAME` builds just those; `python make_stars.py --list` lists every built star.
`stars/` is not in git: each machine builds its own from the star files.

### Observation files

```toml
[field]
ra = 294.7844
dec = -11.6235
block = 9                    # minipixes across the block shown (9 = 253")

[night]
date = "2026-07-19"          # or "auto": the night the placed real stars change most

[stars]
background = "f294_784m11_623_field"   # a built star set
layout = "random"            # "real": its own positions; "random": scattered around the pointing (a synthetic field)

[[place]]                    # any number
star = "wuma_035"
x = 30                       # percent of the image from its lower left (default 50, 50; -1: random)
y = 30
epoch = "start+0.05"         # periodic: a primary eclipse / maximum; transients: t0 = ...
                             # ISO UTC, "start+D" days from the start of the night (or the UT window), or BJD_TDB

[measure]                    # how the exposures are measured (defaults: SExtractor's)
[output]                     # movie, bands, fits, page; stride, noise seed, exposure time, PSF and kernel options
```

[`examples/observations/template.toml`](examples/observations/template.toml) lists every key with its default and
meaning. Anything can be overridden on the command line: `--set section.key=value`, `--place '{star = "...", ...}'`
(replaces the file's placements), `--date`, `--start`, `--end`, `--ra`, `--dec`, `--name`, `--bands`. Everything is
checked before any work starts. Outputs, in `outputs/pseudo_real/<name>/` (real Gaia stars where they are, `layout = "real"`)
or `outputs/fully_synthetic/<name>/` (stars at random positions, `layout = "random"`); `<name>`: `[field] name`, else the
file's name:

- `<name>_night_<band>.mp4`: the movie (CMD, β/ρ colour exposure, difference image, light curves, SEDs).
- `<name>_night_<band>.photometry.csv`: every measurement, with the true model magnitude beside it.
- `fits/` (`output.fits = true`): every measured exposure in both bands, with what made it in the header.
- `observation.toml`: the observation as made (every key, the night as used); running it makes the same data.
- `summary.md`; `scene/`: exactly what the engine (`night_movie.py`) read.

The dust is added here: for every star, the E(B−V) in front of it at its distance along its sightline (the Bayestar19
3D map, capped at SFD's whole column; `mesa_argus/dust.py`, docs/ASSUMPTIONS.md). Gaia's constant stars have their
own measured dust taken out first with the same law. Network: Gaia queries the first time a pointing is used (then
cached in `data/gaia/`). The dust maps are local: `bash tools/fetch_dust.sh` once per machine (~860 MB, into
`~/dustmaps_data`).

### Real fields

```bash
python make_stars.py --gaia-list 294.7844 -11.6235 --write my_field.toml --write-observation my_obs.toml
python make_stars.py my_field.toml        # the field's stars as a set + each variable that can be modelled
python observe.py my_obs.toml             # layout "real"; each real variable placed by name only; date "auto"
```

A real star (generator `Gaia variable`) is modelled from Gaia DR3: its period and ephemeris, distance, the E(B−V) in
front of it at that distance (used only to infer its luminosity), composition and mass, then MESA RSP or PHOEBE
fitted to Gaia's eclipses. Its SEDs are dust-free like every star's; an observation puts it at its real position and
ephemeris, seen through the same dust. A Gaia field set's constant stars carry their distances (Bailer-Jones et al.
2021; GSP-Phot where they have none); an observation takes each one's measured dust out and puts the dust where it is
observed back in. Gaia queries are cached in `data/gaia/<pointing>/`.

### Generators

| Generator | Timing | Engine | What it is |
|---|---|---|---|
| `RR Lyrae` (`mode` RRab, RRc) | periodic | MESA RSP | radial pulsator tuned to your period, at `L` or absolute `M_G`, at `distance` |
| `Cepheid` (F, 1O), `Type II Cepheid` (BL Her, W Vir), `HADS` | periodic | MESA RSP | as RR Lyrae |
| `W UMa` | periodic | PHOEBE | contact binary from period, q, inclination, Teffs, fillout factor, distance |
| `detached EB` | periodic | PHOEBE | detached binary from period, inclination, Teffs, radii (masses optional), distance |
| `Gaia variable` | periodic | MESA RSP / PHOEBE | a real Gaia DR3 variable, modelled as itself |
| `supernova` | transient | sncosmo | any sncosmo model at a redshift and peak absolute magnitude |
| `microlensing` | transient | point lens | point or uniform-disc source, a MESA template SED |
| `dwarf nova` | transient | steady disc | accretion disc with a rising and decaying accretion rate, plus the white dwarf |
| `Gaia field` | set | Gaia DR3 | every Gaia star around a pointing, as constant stars |

Microlensing and dwarf nova are parametric models written in this repo (equations and references in
`mesa_argus/transients.py`). MESA RSP models take ~0.5–2 h each and PHOEBE a few minutes.

**Adding a generator:** a module in `mesa_argus/sources/` with a list `GENERATORS` of `Generator(name, timing,
engine, about, keys, make)`, added to `MODULES` in `all_generators()` (`mesa_argus/sources/__init__.py`). Its keys
are then checked in star files and listed by `python -m mesa_argus.sources`, and observations place it like every
other star. A star file names its generator exactly as listed (e.g. `"W UMa"`); an unknown generator or key, or a
missing required key, stops the run with the names.

### How it fits together

```
make_stars.py star file          → stars/<name>/   (generators in mesa_argus/sources/: MESA RSP, PHOEBE, sncosmo,
                                                     parametric transients, Gaia)
observe.py observation file      → outputs/<category>/<name>/scene/  (field, background stars, placed stars with their file,
                                                            position, timing and dust)
  └─ night_movie.py --scene      → the night's exposures: ArgusSim PSF, sky, airmass and noise → images
                                   → SExtractor (SEP) photometry, calibrated per exposure → movie, CSV, FITS
```

Every assumption (what comes from ArgusSim, what is written here, what is assumed) is in
[docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).

### Batches

`tools/stage_batch.py` takes a base file and lists to iterate over, as one TOML file:

- **Observation batches** (base: an observation file): one observation per combination of the `[vary]` lists (`combine
  = "product"` or `"zip"`; values inline or from a CSV), each written out as its own observation file and run with
  `observe.py`, several at once.
- **Star batches** (base: a star file): `[vary]` keys `star.<key>` make a grid of every star in the base, built by one
  `make_stars.py` run.

```toml
base = "../examples/observations/synthetic_transients.toml"
[vary]
"field.ra,field.dec" = [[294.7844, -11.6235], [273.0414, -20.7099]]
place = [
  [{star = "dwarf_nova_400pc", t0 = "start+0.02"}],
  [{star = "microlensing_te02", t0 = "start+0.15", x = 30, y = 70}],
]
```

```bash
python tools/stage_batch.py batches/example_sweep.toml --check       # expand and validate: 4 observations
python tools/stage_batch.py batches/example_sweep.toml               # run them; --status for progress
python tools/stage_batch.py batches/example_star_grid.toml --check   # 9 W UMa stars (without --check: builds them)
```

An observation batch only names stars, so build them first (`example_sweep.toml` needs `f294_field.toml` and
`transients.toml`). Each observation is written to `outputs/batch_<batch>/runs/<name>.toml` before it runs, and can be
rerun on its own with `observe.py`; a rerun of the batch skips observations already done.

`make_stars.py --gaia-list RA DEC [--gaia-list RA DEC ...] --write stars.toml --write-batch batch.toml` writes the
star file and the observation batch for any number of real fields: `batches/overnight.toml` is the 16-field newton
batch made this way. `tools/find_rich_field.py` searches Gaia for the windows with the most kinds of modellable
variable.

### Interactive page

`demo/` builds a single offline HTML page of one observation (real Gaia stars, for now) with a control panel,
simulating and measuring the exposures in the browser (`output.page = true`, or by hand):

```bash
python observe.py examples/observations/f294.toml --scene-only
python demo/export_demo.py --scene outputs/pseudo_real/f294/scene && python demo/build_html.py f294   # -> demo/argus_demo.html
```

`demo/validate/` checks the page's simulator and SExtractor port against the Python ones.

## 2. Argus vs Gaia vs SDSS: the 3D CMD and Fisher analysis

MESA tracks with the colors module give synthetic photometry in all three systems at every timestep.

```bash
python -m mesa_argus run population fisher_stencil      # resumable: re-running skips finished and running tracks
python -m mesa_argus status population fisher_stencil
python -m mesa_argus cmd3d population                   # -> outputs/population_cmd3d.html (one self-contained file)
python -m mesa_argus fisher                             # -> outputs/fisher/*.png, results.csv
```

Add `--live` to `cmd3d` or `fisher` to rebuild as runs finish.

**The analysis.** Tracks are resampled onto equivalent evolutionary points (EEPs), so index *e* is the same stage on
every track. At each EEP, central differences across the stencil tracks give ∂m_band/∂θ for θ = [M/H], ln M and
α_MLT; age (along the track) and the distance modulus (with the prior in `analysis.toml`) are nuisance directions.
The Cramér–Rao bound √(F⁻¹)_θθ, with F = Jᵀ Σ⁻¹ J, is the best precision any unbiased fit to that star's photometry
can reach. Gaia and SDSS use fixed per-band errors (`analysis.toml`); Argus is evaluated at a range of trial errors
(1 mmag to 0.5 mag). Two bands constrain at most two quantities per star (plus the distance prior), so separating
[M/H] from α_MLT needs ≥ 3 bands.

**Physics.** Kurucz2003 (ATLAS9) atmospheres; magnitudes absolute (Vega for Gaia and Argus, AB for SDSS, with AB − Vega
offsets measured from the module). Scaled-solar GS98 composition, no diffusion; Y from the solar calibration
(`data/solar_calibration.json`, α_MLT⊙ = 1.980). MIST-like overshoot (f = 0.016), Reimers/Blöcker winds. Stops at
core He depletion or 13.5 Gyr (25 Gyr for the stencils).

**Caveats.** Scaled-solar models (for α-enhanced halo stars read [M/H] ≈ [Fe/H] + 0.3); no extinction in the Fisher
analysis; atmospheres outside the grid (Teff < 3500 K giants, hot low-g corners) are clamped by MESA and excluded.

### Looking at one MESA model through Argus

`plot_sed_observation.py` shows one SED (or a whole track as a movie) as an Argus CMD point, light curve, synthetic
exposure and spectrum:

```bash
python plot_sed_observation.py runs/<track>/SED/beta_SED_00000010.csv --distance-pc 2000
python plot_sed_observation.py runs/<track> --distance-pc 2000      # movie of the track
```

`grids/argus_sim_seds.py` tracks write the SED at every model; `python -m mesa_argus seds GRID` packs them into
`runs/<track>/seds.h5` (wavelength, f_λ at 10 pc per model, model properties, magnitudes).

## Layout

```
make_stars.py           simulate stars (star files) -> stars/<name>/
observe.py              observe built stars (observation files) -> outputs/<category>/<name>/
night_movie.py          the engine observe.py runs: simulate and measure the exposures, draw the movie
plot_sed_observation.py one SED or one MESA track through Argus
mesa_argus/             the Python package (python -m mesa_argus ...)
  argus.py                Argus bands, units and plot style shared by the movies
  dust.py                 dust: the 3D map (Bayestar19, SFD), CCM89, dereddening Gaia stars
  photometry.py           SExtractor (SEP) photometry and per-exposure calibration
  rsp.py                  MESA RSP pulsator models
  binary.py               PHOEBE eclipsing-binary models
  sources/                the star generators (one module per family; python -m mesa_argus.sources)
  transients.py           the transient physics: sncosmo, microlensing, dwarf nova
  gaia.py                 Gaia DR3 queries, a field's variables, a variable's physical parameters
  night.py                when a pointing is observable; BJD at the site
  runconfig.py            observation files: schema, defaults, validation, overrides
  seds.py                 MESA SED packing; the constant-star SED templates
  runner.py, grids.py     resumable parallel MESA runs; grid definitions → track specs
  history.py, eep.py      history reading, phases, equivalent evolutionary points
  cmd3d.py, fisher.py     the 3D CMD page; Fisher matrices and figures
  calibrate.py, zeropoints.py, filters.py, composition.py, config.py
  m67.py                  the M67 experiment (below)
grids/                  MESA grid definitions, one file per grid
inlists/                the single MESA physics template
mesa_work/              MESA work directory (make -C mesa_work)
data/                   committed inputs: Gaia query caches (data/gaia/<pointing>/), calibrations, filter curves
batches/                batch files for tools/stage_batch.py
examples/               star files (examples/stars/) and observation files (examples/observations/, template.toml)
tools/                  batch runner, field finder, one-off migrations
studies/                characterisation studies (the photometric offset study)
demo/                   the interactive HTML page and its validation
docs/ASSUMPTIONS.md     what the night movies model and assume
experiments/            finished early experiments (the M67 1 Myr minipix movie)
config.toml             machine settings          analysis.toml   Fisher-analysis inputs
templates/, vendor/     the 3D CMD page's template and its bundled plotly
stars/                  built stars (not in git; rebuilt from the star files)
runs/, outputs/         generated (not in git): MESA / PHOEBE working runs; observations, figures, CSVs
```

The hand-made fields `m3`, `m15`, `rr308108` and `rr416445` are the first movies, from before the star /
observation split (run with `night_movie.py --field m3`). `data/sparse/` is the search that found the two RR Lyrae fields;
`data/rich_fields*.ecsv` is the output of `tools/find_rich_field.py`.
