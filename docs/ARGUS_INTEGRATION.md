# Merging MESA_Argus with the argus-hdps work

Goal: a synthetic Argus catalogue and data stream that runs through hdps, so the pipeline can be tested and tuned on
data whose truth is known. MESA_Argus moves into the argus-hdps organisation, under a better name, with the work it
duplicates removed.

Clones of the other repos: `~/argus_repos/` (synthref, hdps, argus_alert_sim). ArgusSim: `~/ArgusSim`.

## Checklist

### argus_alert_sim (Hank's simulated alert stream; uses MESA_Argus through `plugins/mesa_argus`)
- [ ] Update the plugin to current MESA_Argus (PR; see the log). It pinned MESA_Argus at 12031c7 and called
      `sources.get`, `sources.validate`, `sources.Field` and the `GRB` generator, none of which exist now.
- [ ] Decide what happens to `GRB_AFTERGLOW` (the GRB generator was removed from MESA_Argus).
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
