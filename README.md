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
Python (≥ 3.11), and `pip install -r requirements.txt`. The movies also need `ffmpeg`.

```bash
$EDITOR config.toml               # machine settings: colors-data path, parallelism (usually only the path)
make -C mesa_work                 # build MESA's star executable
python -m mesa_argus setup        # assemble the filter set (Gaia + Argus from your colors data, SDSS from data/)
python -m mesa_argus run argus_sim_seds   # the stellar SEDs every movie uses for its constant stars (once)
```

On macOS, wrap long runs in `caffeinate -dims` so sleep does not suspend MESA.

## 1. Night movies

`argus_field.py` is the entry point. Give it a position and a night; it fetches the field, builds the models and
renders one movie per band into `outputs/<field>/`.

```bash
# a real field: list the catalogued variables in the block, then model the ones you choose
python argus_field.py --ra 294.7844 --dec -11.6235 --block 9 --list
python argus_field.py --ra 294.7844 --dec -11.6235 --block 9 --select all --date auto

# a synthetic field: the real field's stars at random positions, all constant, plus transients you place
python argus_field.py --ra 294.7844 --dec -11.6235 --block 9 --date 2026-07-19 --synthetic --seed 1 \
  --transient salt3:z=0.02,M=-19.3,x1=0,c=0,t0=start+0.2 \
  --transient microlensing:tE=0.2,u0=0.02,base=19.5,bp_rp=1.0,t0=start+0.15,x=30,y=70
```

Outputs, per field:

- `outputs/<field>/<field>_night_<band>.mp4`: the movie. Its panels are the CMD, the exposure (β/ρ colour
  composite), the difference image, the light curves (measured and model) and the SEDs.
- `outputs/<field>/<field>_night_<band>.photometry.csv`: every measurement, with the true model magnitude beside it.
- `outputs/<field>/field_summary.md`: what was modelled in this field and how.

To re-render a field that is already set up (for example a quick look at every 10th exposure):

```bash
python night_movie.py --field f294_784m11_623 --band rho --stride 10 -o outputs/f294_784m11_623/test.mp4
```

### How a movie is made

```
argus_field.py
  ├─ the field      Gaia DR3 (and ASAS-SN) around the position → data/<field>/
  │                 (--synthetic: the same stars' magnitudes and colours at random positions)
  ├─ the sources    an SED for every star, in runs/:
  │                   constant stars   MESA SED matched to Gaia BP−RP, scaled to G      (runs/*_sed)
  │                   pulsators        MESA RSP limit cycles              mesa_argus/rsp.py
  │                   eclipsing bins.  PHOEBE + MESA atmospheres          mesa_argus/binary.py
  │                   transients       sncosmo, or our parametric models  mesa_argus/transients.py
  └─ night_movie.py the night's exposures: ArgusSim PSF, sky, airmass and noise → images
                    → SExtractor (SEP) photometry, calibrated per exposure → movie + CSV
```

Every assumption (what comes from ArgusSim, what is written here, what is assumed) is in
[docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).

### Transients

`--transient MODEL:key=value,...`, repeatable. Every transient takes `t0` (ISO UTC time, or `start+D` days from
the night's start) and a position `x`, `y` in percent of the image from its lower left (default 50, 50, the centre;
`-1` picks at random). `python argus_field.py --help` lists every key.

| Model | What it is | Main keys |
|---|---|---|
| any sncosmo source (`python -m mesa_argus.transients --list`) | supernova spectral templates from observed spectra | `z`, `M` (peak absolute mag), model parameters |
| `afterglow` | GRB optical afterglow: power law in time and frequency, t² onset | `peak` (AB at 6400 Å), `t_peak`, `alpha`, `beta` |
| `microlensing` | point lens, point or uniform-disc source | `tE`, `u0`, `base` (source ρ), `bp_rp`, `rho`, `fs` |
| `dwarf_nova` | steady accretion disc with a rising and decaying accretion rate, plus the white dwarf | `d`, `mdot_q`, `mdot_peak`, `t_rise`, `plateau`, `tau_decay`, `r_out`, `t_wd` |

The last three are parametric models written in this repo, not sncosmo models; each file header gives the
equations and references.

### Many fields at once

`tools/stage_batch.py batches/overnight.txt` runs a list of fields (one `ra dec [options]` per line), several at
once, resumably. `tools/find_rich_field.py` searches Gaia for the windows with the most kinds of modellable variable.

### Interactive page

`demo/` builds a single offline HTML page of one field with a control panel, simulating and measuring the exposures
in the browser:

```bash
python demo/export_demo.py && python demo/build_html.py      # -> demo/argus_demo.html
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
argus_field.py          night movies: set up a field (Gaia or synthetic), build its models, render
night_movie.py          night movies: simulate and measure the exposures of one field, draw the movie
plot_sed_observation.py one SED or one MESA track through Argus
mesa_argus/             the Python package (python -m mesa_argus ...)
  argus.py                Argus bands, units and plot style shared by the movies
  dust.py                 CCM89 extinction
  photometry.py           SExtractor (SEP) photometry and per-exposure calibration
  rsp.py                  MESA RSP pulsator models
  binary.py               PHOEBE eclipsing-binary models
  transients.py           sncosmo and parametric transients
  seds.py                 MESA SED packing; the constant-star SED templates
  runner.py, grids.py     resumable parallel MESA runs; grid definitions → track specs
  history.py, eep.py      history reading, phases, equivalent evolutionary points
  cmd3d.py, fisher.py     the 3D CMD page; Fisher matrices and figures
  calibrate.py, zeropoints.py, filters.py, composition.py, config.py
  m67.py                  the M67 experiment (below)
grids/                  MESA grid definitions, one file per grid
inlists/                the single MESA physics template
mesa_work/              MESA work directory (make -C mesa_work)
data/                   committed inputs: one folder per field (data/<field>/), calibrations, filter curves
batches/                field lists for tools/stage_batch.py
tools/                  batch runner, field finder
demo/                   the interactive HTML page and its validation
docs/ASSUMPTIONS.md     what the night movies model and assume
experiments/            finished early experiments (the M67 1 Myr minipix movie)
config.toml             machine settings          analysis.toml   Fisher-analysis inputs
templates/, vendor/     the 3D CMD page's template and its bundled plotly
runs/, outputs/         generated (not in git): MESA runs and models; movies, figures, CSVs
```

The hand-made fields `m3`, `m15`, `rr308108` and `rr416445` (`night_movie.py --field m3`) are the first movies,
set up before `argus_field.py` existed. `data/sparse/` is the search that found the two RR Lyrae fields;
`data/rich_fields*.ecsv` is the output of `tools/find_rich_field.py`.
