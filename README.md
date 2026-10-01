# MESA_ARGUS

How much stellar physics can **Argus Array** photometry recover, compared with **Gaia** and **SDSS**?

This repo runs MESA stellar evolution tracks with the MESA `colors` module. Every track gets
synthetic photometry in all three systems at every timestep, computed from the same model. The
repo then provides two outputs:

1. **An interactive 3D CMD** (`outputs/<grid>_cmd3d.html`): a single self-contained HTML file,
   so you can send it as is.
2. **A Fisher-information analysis** (`outputs/fisher/`): the best achievable precision on
   [M/H], mass and α_MLT for each system, as a function of evolutionary stage and the
   (unknown) Argus photometric error.

## Quick start

Requirements: a working MESA install with `$MESA_DIR` set, the MESA colors data, and Python ≥ 3.11
with `numpy` and `matplotlib` (`pip install -r requirements.txt`).

```bash
# 1. machine config: usually only needed if your colors data is not at $MESA_DIR/data/colors_data
$EDITOR config.toml

# 2. build MESA and assemble the filter set (Gaia + Argus from your colors data, SDSS from data/)
make -C mesa_work
python -m mesa_argus setup

# 3. run grids (resumable: re-running skips finished tracks and ones already running)
python -m mesa_argus run population fisher_stencil
python -m mesa_argus status population fisher_stencil

# 4. outputs
python -m mesa_argus cmd3d population          # -> outputs/population_cmd3d.html
python -m mesa_argus cmd3d population --live   #    ...rebuilt every 5 min while runs progress
python -m mesa_argus fisher                    # -> outputs/fisher/*.png, results.csv
python -m mesa_argus fisher --live             #    ...re-run as each anchor stencil completes
```

On macOS, wrap long runs in `caffeinate -dims` so sleep does not suspend MESA.
A closed lid on battery still sleeps the machine.

`data/solar_calibration.json` and `data/zp_offsets.json` are committed, so a new machine does not
need to redo them. They are reproducible with `python -m mesa_argus calibrate` and `python -m mesa_argus zeropoints`.

## Layout

```
config.toml             machine settings: MESA / colors-data paths, parallelism
analysis.toml           Fisher-analysis inputs: Gaia/SDSS errors, Argus trial errors, distance prior
grids/                  grid definitions, one file per grid; every track says why it is there
  population.py           realistic (M, [M/H], α_MLT) stars for display (35 tracks)
  fisher_stencil.py       ±steps around 5 anchor stars for numerical derivatives (35 tracks)
inlists/
  physics.inlist          the single physics template (MIST-like), rendered per run
  history_columns.list
mesa_work/              MESA work directory (stock src/), `make -C mesa_work` -> build/bin/star
mesa_argus/             Python package (python -m mesa_argus ...)
  runner.py               run dirs, resumable parallel runs, machine-wide process cap
  grids.py                loads grids/*.py into track specs (composition from [M/H])
  history.py              history reading, pre-MS trim, phases, atmosphere-grid coverage
  cmd3d.py                interactive 3D CMD (templates/cmd3d.html + vendor/plotly)
  eep.py                  equivalent evolutionary points (matched stages across tracks)
  fisher.py               derivatives, Fisher matrices, Cramér–Rao bounds, figures
  calibrate.py            solar calibration of (α_MLT, Y)
  zeropoints.py           AB − Vega offsets measured from the colors module
data/                   committed inputs: calibration, zero points, SDSS curves
runs/                   (generated) one directory per track, with its full rendered `inlist`
outputs/                (generated) HTML, figures, CSV, logs
```

To add a grid, drop a new `grids/<name>.py` defining `tracks(make)` (see the existing two).
To add a photometric system, add its curves to `mesa_argus/filters.py` and its bands to
`config.py` / `fisher.py`.

## Physics

- **Atmospheres:** Kurucz2003 (ATLAS9). Magnitudes are absolute (10 pc): Vega for Gaia and
  Argus, AB for SDSS. The AB − Vega offsets are measured from the module itself.
- **Composition:** scaled-solar GS98, with no diffusion. Y = Y_p + (ΔY/ΔZ) Z, where
  Y_p = 0.2485 and ΔY/ΔZ comes from the solar calibration.
- **Mixing length:** α_MLT⊙ = 1.980 from the solar calibration.
- **MIST-like physics:** exponential core overshoot f = 0.016; Reimers η = 0.1 / Blöcker η = 0.2
  winds; Type-2 opacities during He burning.
- **Stopping condition:** core He depletion, or 13.5 Gyr (25 Gyr for the stencils, where age is a
  nuisance parameter rather than a realism cut).

## The discernibility analysis

The question is how well each system resolves changes in [M/H], mass and α_MLT once
evolutionary state and distance are unknown.

1. **Matched stages.** Tracks are resampled onto EEPs, so index *e* is the same evolutionary
   stage on every track:
   - **MS:** matched by central X.
   - **Subgiant / RGB:** matched by fractional HRD path length.
   - **Core He burning:** matched by central Y.
2. **Derivatives.** At each EEP, central differences across the stencil tracks give
   ∂m_band/∂θ for θ = [M/H], ln M and α_MLT.
3. **Nuisances.** Two nuisance directions are added:
   - **Along the track:** age.
   - **Distance modulus:** shifts every band equally, with the prior σ(μ) from `analysis.toml`.
4. **Bounds.** F = Jᵀ Σ⁻¹ J. The Cramér–Rao bound √(F⁻¹)_θθ is the best precision any unbiased
   fit to that star's photometry can reach.

Gaia and SDSS use the fixed per-band errors in `analysis.toml`. Argus is evaluated at each trial
error (1, 3, 5 mmag and 0.1, 0.3, 0.5 mag), giving one figure each. The equal-noise figure gives
every band the same error, which isolates the information carried by the bandpasses themselves.
`summary_vs_argus_sigma.png` shows where Argus crosses Gaia and SDSS.

Two bands constrain at most two quantities per star (plus the distance prior). So for a single
star, Argus can measure one physics parameter together with its evolutionary state. Separating
[M/H] from α_MLT jointly needs ≥ 3 bands, and the `feh|alpha` row shows that explicitly.

## Caveats

- **Scaled-solar models:** atmospheres and opacities are scaled-solar. For α-enhanced halo stars,
  read [M/H] ≈ [Fe/H] + 0.3 (Salaris et al. 1993).
- **Extinction:** no extinction yet. A_V would be another nuisance direction, and a real
  degeneracy with [M/H].
- **Clamped atmospheres:** points outside the atmosphere grid (Teff < 3500 K at giant-branch
  tips, and hot low-g corners) are clamped by the colors module. They are hidden in the HTML and
  excluded from the analysis.
- **Argus throughputs:** Argus curves are read from your MESA colors data
  (`filters/Argus/System`), not committed here.
