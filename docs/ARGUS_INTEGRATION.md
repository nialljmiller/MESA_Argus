# Merging MESA_Argus with the argus-hdps work

Goal: a synthetic Argus catalogue and data stream that runs through hdps, so the pipeline can be tested and tuned on
data whose truth is known. MESA_Argus moves into the argus-hdps organisation, under a better name, with the work it
duplicates removed.

**Main goal (2026-10-16):** a repository of simulated Argus data lasting at least a month, as close to real as possible:
the night sky tiled as Argus sees it (minipixes, with the survey's gaps) for a whole month, from a leave-N-out
simulation of the minipixes; then alerts from it. Everything below serves that. First: proper source extraction on
the direct images, then real DIA (reference, subtraction, detection and photometry on the difference image).

Clones of the other repos: `~/argus_repos/` (synthref, hdps, argus_alert_sim). ArgusSim: `~/ArgusSim`.

## Checklist

### argus_alert_sim (Hank's simulated alert stream; uses MESA_Argus through `plugins/mesa_argus`)
- [x] Update the plugin to current MESA_Argus: PR https://github.com/argus-hdps/argus_alert_sim/pull/1 (open).
      It pinned MESA_Argus at 12031c7 and called `sources.get`, `sources.validate`, `sources.Field` and the `GRB`
      generator, none of which exist now.
- [x] `GRB_AFTERGLOW` dropped from the plugin; the forecast's ORPHAN_AFTERGLOW has no model for now.
- [ ] Re-add the GRB afterglow to MESA_Argus's transient models, then register it in the plugin again.
- [ ] Follow-ups: microlensing and supernova classes; one image/stamp implementation instead of two
      (`argus_alert_sim/stamps.py` is a port of `night_movie.Scene` and `difference_image`); CV_OUTBURSTS uses
      E(B-V) 0 (MESA_Argus files now record `distance_pc`, so sightline dust could be applied); `PeriodicCurve`
      reddens each template by its source star's own sightline E(B-V), which slightly changes the g/rho amplitude
      ratio when the shape is reused on another sightline.
- [ ] Build the star library (newton) as files `plugins/mesa_argus/library.py` reads; improve its matching (it gives
      each Gaia variable a random template of its class) rather than writing a second matcher.

### synthref (Hank's synthetic Gaia-XP subtraction references)
Each of these changes measurements: decide one at a time.
- [ ] System response: measured, beam-averaged rho filter (CT692-276), DR280 vendor optics, beta red edge. We use
      ArgusSim's idealised 560-825 nm rho box and a PlaneWave CDK optics curve. Route: synthref's
      `export_filter_curves.py` -> ArgusSim -> MESA_Argus. Ask: argus_survey_model ships `argus_rho_ct699.csv`
      (CT699-259), a different part from synthref's CT692-276.
- [ ] PSF: fitted misaligned-wavefront DR280 PSF and the static halo (3.6-4% of the flux within 60 px).
- [ ] Constant stars at their real positions from their Gaia XP spectra (dust already in them), with synthref's
      blend handling (Riello et al. 2021 excess). Our template + 3D dust stays for stars without XP and for
      synthetic or moved stars.
- [ ] Gaia positions propagated to the observation epoch with proper motions (we use the 2016 positions).
- Not to pull: catsfft / svsub (subtraction is hdps's), LRN / M31 work, `design_local.py` (Zemax numbers) and the
  IMX461 data (restricted).

### hdps
- [ ] Write our exposures as L2 `.arg` segments with hdps's own `hdps.archive.data_store.encode_frames` (same
      0.879"/px tile grid as ours), and a reference in `CoaddReferenceProvider`'s layout (synthref `render.py`).
- [ ] SFFT fixes (kernel mirror, DC offset, spatially varying background) are on branches `origin/408/16..20`,
      not on main (checked 2026-10-15): needed before tuning hdps on our data.

### ArgusSim
- [ ] MESA_Argus imports the private `~/ArgusSim` (has `ZemaxPSF`); the public argus_sim in argus_survey_model
      (what argus_alert_sim installs) has no `ZemaxPSF`. Settle on one.

### The move and the rename
- [ ] Move to argus-hdps; new package name; argus_alert_sim's submodule URL and `import mesa_argus` follow.
- [ ] Remove what is duplicated (stamps/image code, throughput, PSF) once the above are settled.

## Log
- 2026-10-15: cloned synthref, hdps, argus_alert_sim to `~/argus_repos/`; reviewed synthref against MESA_Argus
  (findings above). argus_alert_sim's plugin is broken against MESA_Argus main (API change, GRB removed).
- 2026-10-15: argus_alert_sim branch `mesa-argus-current` (local, not pushed): plugin follows MESA_Argus dfd20ad;
  GRB_AFTERGLOW dropped from the plugin; 15 tests pass, 7 skip (need Hank's Gaia catalogue); `aas models` lists
  CV_OUTBURSTS; `aas check` not run (needs Hank's visit store). Library loads 49 templates from `stars/`, 31 from
  `runs/`. Waiting on: port the GRB afterglow into argus_alert_sim, or leave ORPHAN_AFTERGLOW without a class.
  Also fixed: the README's separate plugin install fails in uv; installing both packages in one command works.
- 2026-10-15: PR argus-hdps/argus_alert_sim#1 opened (branch `mesa-argus-current`). User's call: drop
  GRB_AFTERGLOW and leave the gap; the afterglow comes back later in MESA_Argus's transients.

- 2026-10-16: difference panel was display-only and drawn on a linear scale set by the brightest variable (looked
  blank); now asinh (cfac795). Source extraction bug found: `week_f325`'s V2 (G 13.9) measured ~18 mag in 16% of
  exposures. Cause: SEP windowed centroids (XWIN/YWIN, window from the reference FWHM 4.34 px) of a G 18.5 neighbour
  8.5 px away migrate onto V2's centre; the nearest-neighbour cross-match then takes the faint detection. Detection
  itself is fine. Also: `--start`/`--end` are ignored in survey mode (night_movie.py).
- 2026-10-16: plan: use hdps for extraction and DIA, with CPU mirrors of its GPU files (same classes and outputs)
  where no NVIDIA GPU exists (the Mac's Apple M5 GPU runs Metal, not CUDA). hdps is edited locally only, branch
  `cpu-mirror` in ~/argus_repos/hdps, no PRs. The simulator will write calibrated sensor frames (hdps
  `FullFrameImage`) so hdps runs detection, WCS solving, reprojection to tiles (`TANSegmentList`), subtraction,
  difference detection and alert vetting. Step 1 done: every hdps package imports without CuPy (4 files edited);
  hdps tests on CPU: astrometry 157, diffimg 140, skymap 171 passed (GPU-marked tests skipped).
  Next: photometry/background_cpu.py + source_detection_cpu.py mirroring background.py, source_detection.py, .cu.
- 2026-10-16: hdps branch `cpu-mirror` commit ecd73320: `photometry/source_detection_cpu.py` (CPUSourceDetector:
  the 4 CUDA kernels of source_detection.cu as numba loops, host steps copied). `hdps.photometry.SourceDetector`
  resolves to GPUSourceDetector on a CUDA host, else CPUSourceDetector (`HDPS_FORCE_CPU=1` forces CPU);
  SourceDetectionTask and DiffDetectionTask construct SourceDetector. CPU copies of hdps's detector tests: 96 pass
  (aperture sum vs hdps's CPU reference, sub-pixel phase, centroid iteration, error model, determinism,
  per-pixel-variance detection, de-duplication). On week_f325 exposure 18 (the SEP failure) it finds V2 at 0.21 px
  with 43519 e and keeps the G 18.5 neighbour separate (0.37 px, 846 e). Not yet compared with the GPU code on a
  GPU (none available).
- 2026-10-19: hdps packages installed into the system Python (pip --user -e; no existing package versions changed).
  MESA_Argus extraction is now hdps's own detector (78b0d94): production settings of SourceDetectionTask, one-to-one
  association with the simulated stars, calibrators need S/N >= 50, per-exposure catalogues (parquet). f325 survey
  night (210 rho exposures): V1 median -0.013, robust scatter 0.020 mag, 4 off by >0.5 mag; V2 +0.023, 0.016, 1 off
  (SEP had V2 off in 16% of exposures). The SEP code is kept as sep_photometry.py for the demo and the offset study
  (the offset study's capture of the engine's photometry object no longer applies; it needs rewriting or retiring).
- 2026-10-19: hdps detector defect found (same on GPU and CPU; the mirror reproduces the kernels): noise peaks ~6 px
  from a bright star (S/N ~3) pass detection; the 3-pass centroid iteration (window up to the 6 px matched-filter
  half-width) moves them onto the bright star; their re-centred apertures take part of its light; the de-dup grid pass
  (keep rule: a later source replaces the kept one if brighter than the previously visited one) can then keep a fake
  and drop the star. f325 exposure 40: V1 real 55258 e (S/N 184) dropped, fake 30999 e (S/N 3.4) kept. Also: hdps
  saturation is 0.9999 x 65534 (16-bit ADC units); simulated images are electrons with a 21657 e full well.
  Survey fields Dec < -22 (f244, f276, f279) never reach 3 h above the array's 32.84 deg minimum altitude.

