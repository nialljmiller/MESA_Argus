"""Every parameter of an observation: the one file that defines a run (examples/observations/template.toml is written
from this module: python -m mesa_argus.schema > examples/observations/template.toml).

Each parameter has a default, a unit, its allowed values (a range, a set of choices, or a text form) and the
justification for them, with the source. runconfig.py checks every value against these before any work starts.

Sources of the defaults:
- [argussim.*]: ArgusSim, read from it (argus_sim.config, argus_sim.coverage_exact.Schedule) when the file is
  resolved, so a run records exactly what it used. ArgusSim defines everything about taking the image.
- [hdps] and [[hdps.tasks]]: hdps's production pipeline, docker/demo/config.yaml (the Task chain and its settings),
  and each Task's constructor defaults (hdps.orchestrate.tasks). hdps processes the raw frames.
- [camera]: the toy camera (mesa_argus/camera.py): what ArgusSim does not define (electrons to ADU).
- the rest: this repository (the scene, the outputs), with the reason for each.

TOML has no null: "none" stands for an unset optional value.
"""
from dataclasses import dataclass, field
from typing import Any, Callable

NONE = "none"
REQUIRED = "required"
POSITIVE = "positive"  # allowed: > 0


@dataclass
class P:
    """One parameter. allowed: (lo, hi) for numbers (None: open), a set of choices, a callable(value, resolved) that
    returns an error text or None, or None (any value of the default's type). why: the justification and source."""
    key: str
    default: Any
    unit: str
    allowed: Any
    why: str
    optional: bool = False  # "none" allowed
    allowed_text: str = ""  # how a callable `allowed` reads in the template


def argus(path):
    """The live ArgusSim default at dotted path (e.g. "telescope.readnoise")."""
    from argus_sim import c

    obj = c
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def schedule(name):
    """ArgusSim's ratchet schedule default (coverage_exact.Schedule)."""
    from argus_sim.coverage_exact import Schedule

    return getattr(Schedule(), name)


def sensor_px():
    from argus_sim import c

    return c.telescope.n_pixels_x, c.telescope.n_pixels_y


def minipix_arcsec():
    from argus_sim import skymap_shim as sh

    return sh.MINIPIX_PIXEL_SIZE * sh.PIXEL_SCALE_DEG * 3600


# ---------------------------------------------------------------- this repository: the field, nights, scene, outputs
FIELD = [
    P("ra", REQUIRED, "deg", (0.0, 360.0), "ICRS right ascension of the field (the block's centre)."),
    P("dec", REQUIRED, "deg", (-90.0, 90.0), "ICRS declination; whether Argus can see it at all is ArgusSim's coverage "
      "(observe.py stops when no exposure covers it)."),
    P("name", "", "", None, "Output folder name; \"\": the observation file's name."),
    P("block", 9, "minipix", (1, 64), "Side of the block shown in the movie and followed in the summary, in hdps "
      "minipixes (32 tile px, 28.1\"). At most 64: one hdps tile (2048 px) is 64 x 64 minipixes."),
    P("window", 56, "minipix", lambda v, r: _window_check(v),
      "Side of the simulated readout window, in minipixes, centred on the field's position on the sensor (shifted to "
      "stay on it); 0: the whole sensor (ArgusSim telescope.n_pixels_x x n_pixels_y). A real camera can read out a "
      "window (XORIGIN, YORIGIN, XSIZE, YSIZE in its header). At least 28 (782 px): hdps's source detector needs 3 of "
      "its 256 px background regions across. Default 56 (1564 px, 26.3'): of the windows tried, the one hdps handles "
      "best end to end (16 exposures of f294, 2026-10-21: WCS solved 13/16, every touched tile reprojected 11/16; 28: "
      "14/16 and 0/16, 40: 14/16 and 7/16, 80: 11/16 and 0/16; production processes whole frames). The field's Gaia "
      "stars must reach the window's far corner: radius ~0.5 deg for 56.",
      allowed_text="0, or 28 .. the sensor's short side (313 for ArgusSim's default sensor)"),
]


def _window_check(v):
    if v == 0:
        return None
    nx, ny = sensor_px()
    from argus_sim import c

    most = int(ny * c.telescope.plate_scale / minipix_arcsec())
    return None if 28 <= v <= most else f"0 (whole sensor) or 28..{most}"


NIGHT = [
    P("date", "auto", "", lambda v, r: None if v == "auto" or _is_date(v) else "YYYY-MM-DD or \"auto\"",
      "Local date of the first night (the night starts on that evening); \"auto\": the night the placed real stars "
      "change most.", allowed_text="YYYY-MM-DD | \"auto\""),
    P("nights", 7, "nights", (1, 366), "Consecutive nights observed. 7 is the standard observation (a week); up to a "
      "year."),
    P("seed", 0, "", (0, 2**31 - 1), "Seed of ArgusSim's per-ratchet draws (seasonal zenith seeing); the same seed "
      "makes the same nights."),
    P("reference_nights", 1, "nights", (1, 30), "Nights observed before the first night to make hdps's references "
      "(its coadd bootstrap: pass 1 coadds these nights' frames with CoaddTapTask, pass 2 subtracts against those "
      "coadds; docker/demo/config.yaml header). Same stars, transients left out. 1: the night before."),
]


def _is_date(v):
    from datetime import date

    try:
        date.fromisoformat(str(v))
        return True
    except ValueError:
        return False


STARS = [
    P("background", REQUIRED, "", None, "The star set (stars/<name>/) every field star comes from: Gaia stars of the "
      "field built by make_stars.py."),
    P("layout", "real", "", {"real", "random"}, "real: each star at its Gaia position; random: scattered uniformly "
      "over the window (a synthetic field)."),
    P("seed", 1, "", (0, 2**31 - 1), "Seed of the random layout and random placements."),
    P("missing", "stop", "", {"stop", "skip"}, "A placed star not built yet: stop the run, or skip it."),
]

PLACE = [
    P("star", REQUIRED, "", None, "A built star (stars/<name>/)."),
    P("x", 50.0, "%", lambda v, r: None if v == -1 or 0 <= v <= 100 else "0..100, or -1 (random)",
      "Position across the block, percent from its left edge; -1: random. Ignored for a real Gaia star (its own "
      "position).", allowed_text="0 .. 100 | -1"),
    P("y", 50.0, "%", lambda v, r: None if v == -1 or 0 <= v <= 100 else "0..100, or -1 (random)",
      "Position up the block, percent from its bottom edge; -1: random.", allowed_text="0 .. 100 | -1"),
    P("t0", NONE, "", None, "Transients: the model's t0 (ISO UTC, \"start+D\" days after the first exposure, or BJD).",
      optional=True),
    P("epoch", NONE, "", None, "Periodic stars: a primary eclipse / maximum (as t0).", optional=True),
    P("distance", NONE, "pc", (0.0, 1e7), "Distance for the dust in front of it (the 3D map); none: behind all the "
      "dust in its sightline, or the star's own.", optional=True),
]

CAMERA = [
    P("gain", None, "e-/ADU", (0.01, 100.0), "Toy: ArgusSim defines the detector in electrons only. Default: "
      "ArgusSim's full well (telescope.fullwell) at hdps's raw saturation level (saturation_adu), so a full pixel is "
      "where hdps flags saturation; 0.373 for 21657 e- (hdps's prototype QHY600M: 0.33)."),
    P("offset", 0, "ADU", (0, 65535), "Toy: bias level (none in ArgusSim)."),
    P("saturation_adu", 58000.0, "ADU", (1.0, 65535.0), "hdps.calibration.ImageReducer's saturation_value: raw "
      "pixels at or above it are flagged saturated."),
]

LIGHTCURVES = [
    P("link_arcsec", 1.5, "arcsec", (0.1, 5.3), "A detection joins the nearest object within this. Above ~1 px "
      "(1.0\") so centroid noise links; below hdps's de-duplication radius (6 px, 6.0\"), inside which two "
      "detections in one frame cannot both exist."),
]

TRUTH = [
    P("match_radius", 2.0, "px", (0.5, 6.0), "Scoring only: a detection or candidate is a simulated star within "
      "this (tile px). Below hdps's de-dup radius (6 px)."),
    P("alert_radius", 2.0, "arcsec", (0.5, 6.0), "Scoring only: an alert is the simulated star within this of its "
      "position (hdps's tile WCS against the true position; the median WCS error is ~0.1\"). About two pixels; "
      "below the 6 px de-dup radius so a neighbour is not taken."),
]

OUTPUT = [
    P("bands", ["beta", "rho"], "", lambda v, r: None if v and set(v) <= {"beta", "rho"} else "beta and/or rho",
      "Argus bands kept (ArgusSim's filter_strategy has b, named g there, and rho).",
      allowed_text="a non-empty list of \"beta\", \"rho\""),
    P("movie", True, "", {True, False}, "Render the movie (the third output)."),
    P("catalogues", True, "", {True, False}, "Keep hdps's per-frame source catalogues."),
    P("alerts", True, "", {True, False}, "Run difference imaging and keep hdps's alert products."),
    P("lightcurves", True, "", {True, False}, "Link detections into catalogue.csv and lightcurves/ (needs "
      "catalogues)."),
    P("pipeline_movie", True, "", {True, False}, "Also render the pipeline movie (each raw frame on its sensor, hdps's "
      "detections and WCS); drawn from the same run, about 15-25% more time."),
    P("raw", False, "", {True, False}, "Keep the raw camera frames (raw/); they are always written for hdps, and "
      "deleted after unless kept."),
    P("stride", 1, "", (1, 1000), "Movie: every Nth frame."),
    P("fps", 24, "frames/s", (1, 120), "Movie frame rate."),
    P("dpi", 100, "", (50, 300), "Movie resolution."),
    P("noise_seed", 3, "", (0, 2**31 - 1), "Seed of the photon and read noise."),
]

# ---------------------------------------------------------------- ArgusSim: everything about taking the image
# Ranges: ArgusSim validates none of these, so each range is what the quantity can physically be; an open bound (no
# upper limit) where nothing limits it.
_A = "ArgusSim argus_sim/config.py"
ARGUSSIM = {
    "observatory": [
        P("latitude", argus("observatory.latitude"), "deg", (-90.0, 90.0), f"Site latitude, Mt Fowlkes ({_A}). A "
          "latitude."),
        P("longitude", argus("observatory.longitude"), "deg", (-180.0, 180.0), f"Site longitude, east positive "
          f"({_A}). A longitude."),
        P("altitude", argus("observatory.altitude"), "m", (-500.0, 9000.0), f"Site height ({_A}). Between the "
          "lowest and highest land on Earth."),
        P("min_alt", argus("observatory.min_alt"), "deg", (0.0, 90.0), f"Lowest altitude the array points at "
          f"({_A}). Above the horizon."),
        P("seasonal_seeing", argus("observatory.seasonal_seeing"), "", {True, False}, f"Zenith seeing per ratchet "
          f"from the site's DIMM seasons (Barker et al. 2003; {_A}, PI 2026-09-19)."),
        P("seeing_mean", argus("observatory.seeing_mean"), "arcsec", POSITIVE, f"Lognormal seeing, used only "
          f"when seasonal_seeing is false ({_A}). Positive."),
        P("seeing_std", argus("observatory.seeing_std"), "arcsec", (0.0, None), f"As seeing_mean ({_A}). Not "
          "negative."),
        P("transparency", argus("observatory.transparency"), "", (0.0, 1.0), f"Grey transparency, 1 = photometric "
          f"({_A}). A transmitted fraction."),
        P("artificial_light_cd_m2", argus("observatory.artificial_light_cd_m2"), "ucd/m2", (0.0, None),
          f"Artificial sky glow ({_A}). Not negative."),
        P("solar_activity", argus("observatory.solar_activity"), "", (0.8, 2.0), f"Airglow scaling when "
          f"survey.solar_activity_prior is \"fixed\"; ArgusSim's stated range, 0.8 (solar minimum) to 2.0 (maximum) "
          f"({_A})."),
    ],
    "telescope": [
        P("aperture_diameter", argus("telescope.aperture_diameter"), "mm", POSITIVE, f"({_A}). Positive."),
        P("central_obstruction_diameter", argus("telescope.central_obstruction_diameter"), "mm", (0.0, None),
          f"({_A}). Not negative; smaller than the aperture."),
        P("focal_length", argus("telescope.focal_length"), "mm", POSITIVE, f"({_A}); with pixel_size, the plate "
          "scale. Positive."),
        P("pixel_size", argus("telescope.pixel_size"), "um", POSITIVE, f"({_A}). Positive."),
        P("n_pixels_x", argus("telescope.n_pixels_x"), "px", (768, None), f"Sensor long axis ({_A}). At least 768: "
          "hdps's source detector needs 3 of its 256 px background regions across."),
        P("n_pixels_y", argus("telescope.n_pixels_y"), "px", (768, None), f"Sensor short axis ({_A}). As "
          "n_pixels_x."),
        P("readnoise", argus("telescope.readnoise"), "e-", (0.0, None), f"({_A}). A noise: not negative."),
        P("dark_current", argus("telescope.dark_current"), "e-/s/px", (0.0, None), f"({_A}). A rate: not "
          "negative."),
        P("fullwell", argus("telescope.fullwell"), "e-", (1, 65535 * 100), f"({_A}). Positive; with [camera] gain "
          "it must fit the 16-bit raw frame (fullwell / gain <= 65535), checked with [camera]."),
        P("throughput_loss", argus("telescope.throughput_loss"), "", (0.0, 1.0), f"Multiplicative throughput "
          f"degradation ({_A}). A fraction."),
        P("psf_model", argus("telescope.psf_model"), "", {"zemax", "gaussian"}, f"Delivered optics (zemax, the "
          f"default since 2026-09-19) or a Gaussian spot; ArgusSim's two models ({_A})."),
        P("spot_size", argus("telescope.spot_size"), "um", POSITIVE, f"RMS spot radius of the Gaussian PSF "
          f"({_A}). Positive."),
        P("ee50_max_px", argus("telescope.ee50_max_px"), "px", POSITIVE, f"Cap on the delivered EE50 diameter "
          f"({_A}); none: uncapped. Positive.", optional=True),
        P("randomize_pixel_phase", argus("telescope.randomize_pixel_phase"), "", {True, False}, f"({_A})"),
    ],
    "survey": [
        P("min_sun_alt", argus("survey.min_sun_alt"), "deg", (-90.0, 0.0), f"Observing only with the Sun below this "
          f"({_A}). Below the horizon."),
        P("min_moon_sep", argus("survey.min_moon_sep"), "deg", (0.0, 180.0), f"({_A}). An angle on the sky."),
        P("solar_activity_prior", argus("survey.solar_activity_prior"), "", {"epoch", "fixed"}, f"Per-night solar "
          f"activity: ArgusSim's 2028-2033 prior, or observatory.solar_activity ({_A})."),
        P("apply_weather", argus("survey.apply_weather"), "", {True, False}, f"Nights gated by the site's clear "
          f"fractions ({_A}, PI 2026-09-19)."),
    ],
    "schedule": [
        P("exptime_s", schedule("exptime_s"), "s", POSITIVE, "Exposure time (ArgusSim coverage_exact.Schedule). "
          "Positive; at most track_min, checked with it."),
        P("track_min", schedule("track_min"), "min", POSITIVE, "Tracking per ratchet (Schedule). Positive."),
        P("reset_min", schedule("reset_min"), "min", (0.0, None), "Reset per ratchet, no exposures (Schedule). Not "
          "negative."),
        P("theta_start_deg", schedule("theta_start_deg"), "deg", (-180.0, 180.0), "Tracking angle at a ratchet's "
          "start (Schedule). An angle."),
        P("sun_alt_max_deg", schedule("sun_alt_max_deg"), "deg", (-90.0, 0.0), "Exposures with the Sun below this "
          "(Schedule). Below the horizon."),
    ],
}

# ---------------------------------------------------------------- hdps: the pipeline
_H = "hdps docker/demo/config.yaml"
HDPS = [
    P("refcat", "~/.rico_cache/atlas_refcat2", "", None, f"ATLAS-Refcat2 for ReferenceRetrievalTask ({_H}: "
      "RICO_CACHE_DIR/atlas_refcat2)."),
]
HDPS_COADD = [  # the config's top-level coadd section (the worker's accumulator)
    P("strategy", "inverse_variance", "", {"inverse_variance", "mean"}, f"Frame weighting in the coadd ({_H})."),
    P("clip_kappa", 3.0, "sigma", POSITIVE, f"Outlier clip on the sky-subtracted residual ({_H}). Positive."),
    P("clip_min_frames", 3, "frames", (1, None), f"Fewest frames before clipping ({_H}). At least 1."),
    P("accumulation_dtype", "float32", "", {"float32", "float64"}, f"({_H}: float32, sub-1e-6 relative difference "
      "from float64)."),
]


def _task(type_, values):
    return dict(type=type_, config=values)


# The production chain (docker/demo/config.yaml), every constructor parameter written out. Changes from it, each a
# toy for something the simulation does not have: ImageReductionTask (the toy camera's gain and serial number, no
# masters: provider "file", no paths); ReferenceRetrievalTask pixscale (ArgusSim's plate scale, filled at resolve);
# AlertVettingTask enabled with no model bundle (real/bogus not run: every score 1). Two hdps output Tasks added for
# the outputs: LightweightFITSOutputTask (the solved frame's catalogues, WCS and zero point; as the AEGIS
# astrometric-reduction config, after TPVWCSSolverTask) and TileImageOutputTask (each differenced tile's science and
# difference planes, for the movies).
TASKS = [
    _task("DataCleanupTask", dict(target_dtype="float32", nan_replacement=0.0)),
    _task("ImageReductionTask", dict(camera_specs={}, bias_path=NONE, dark_path=NONE, flat_path=NONE,
                                     overscan_trimming=False, bias_subtract=False, gain=NONE, gpu_reduce=True,
                                     provider="file")),
    _task("SourceDetectionTask", dict(detect_sigma=3.0, bg_region_size=256, max_sources=10000000, flux_radius=4.0,
                                      matched_filter_width=4.0, centroid_n_iter=3, cr_detect_contrast=0.25,
                                      de_dup_sources=True, de_dup_radius=6.0, saturation_threshold=0.9999)),
    _task("SourceFilterTask", dict(min_x=0, min_y=0, max_x=NONE, max_y=NONE, min_snr=NONE)),
    _task("ReferenceRetrievalTask", dict(catalog_provider="ATLAS", radius_deg=2.5, min_mag=0.0, max_mag=19.1,
                                         pixscale=NONE, cacheable=True, cache_precision=0.1, cache_max_age_days=7.0)),
    _task("LinearWCSSolverTask", dict(n_stars=150)),
    _task("SourceMatchingTask", dict(n_sigma_clip=5.0, region_shape=[4, 6])),
    _task("TPVWCSSolverTask", dict(tpv_terms=17, niter=3, n_sigma_clip=5.0, use_annealing=True, use_chebyshev=True)),
    _task("LightweightFITSOutputTask", dict(output_dir="./solved")),
    _task("ReprojectionTask", dict(segment_shape=[2048, 2048], grid_resolution=33)),
    _task("CoaddTapTask", dict(background_mode="regional")),
    _task("SFFTSubtractionTask", dict(reference_mode="coadd", kernel_half_width=6, kernel_poly_order=0, background_order=2,
                                      mask_saturated_in_fit=True, mask_no_coverage_in_fit=True, mask_bright_stars=True,
                                      bright_star_threshold=3000.0, mask_dilation_growth=4, variance_conv="direct",
                                      rejection_iters=1, rejection_sigma=3.0, rejection_grow=2, retain_planes=True)),
    _task("DiffDetectionTask", dict(detect_sigma=5.0, bg_region_size=256, max_sources=100000,
                                    matched_filter_width=4.0)),
    _task("TransientOutputTask", dict(fmt="parquet")),
    _task("AlertVettingTask", dict(enabled=True, model_bundle="", detect_sigma=3.0, score_cut=0.5,
                                   emit_all_candidates=False, emit_stamps=True, stamp_size=31, emit_kafka=False,
                                   fmt="parquet")),
    _task("TileImageOutputTask", dict(enabled=True, output_dir="", science_output_dir="", difference_output_dir="",
                                      save_science=True, save_difference=True, save_difference_variance=False,
                                      compress=False, tile_ids=NONE, filename_prefix="epoch")),
]

# Allowed values and why, per (Task, setting). hdps itself validates only the enumerated modes (variance_conv, fmt,
# background_mode, ...: "hdps checks"); every other range is what the setting means, open where nothing limits it.
# Source of each value: the production config line (_H), else "hdps default" (the Task's constructor).
_SIG = "A threshold in sigma: positive."
_PX = "A size in pixels: positive."
_N = "A count: at least 1."
TASK_RULES = {
    ("DataCleanupTask", "target_dtype"): ({"float32", "float64"}, f"{_H}. A float type the later Tasks take."),
    ("DataCleanupTask", "nan_replacement"): (None, f"{_H}. Any value."),
    ("ImageReductionTask", "camera_specs"): (None, "Toy camera: filled at resolve ({serial_number}); no overscan "
                                             "geometry, so nothing is trimmed."),
    ("ImageReductionTask", "bias_path"): (None, "Toy camera: no masters (none)."),
    ("ImageReductionTask", "dark_path"): (None, "Toy camera: no masters (none)."),
    ("ImageReductionTask", "flat_path"): (None, "Toy camera: no masters (none)."),
    ("ImageReductionTask", "overscan_trimming"): ({True, False}, f"{_H} has true; the toy camera has no overscan."),
    ("ImageReductionTask", "bias_subtract"): ({True, False}, f"{_H}"),
    ("ImageReductionTask", "gain"): (None, "none: [camera] gain (the pipeline applies it once, here)."),
    ("ImageReductionTask", "gpu_reduce"): ({True, False}, "hdps default; falls back to the CPU without CuPy."),
    ("ImageReductionTask", "provider"): ({"file", "opsdb"}, f"{_H} uses opsdb (the ops database); file: no "
                                         "database, masters from paths. hdps checks."),
    ("SourceDetectionTask", "detect_sigma"): (POSITIVE, f"{_H}. {_SIG}"),
    ("SourceDetectionTask", "bg_region_size"): ((1, None), f"{_H}. {_PX} The window must be 3 regions across "
                                                "(checked with [field] window)."),
    ("SourceDetectionTask", "max_sources"): ((1, None), f"hdps default. {_N}"),
    ("SourceDetectionTask", "flux_radius"): (POSITIVE, f"hdps default (aperture radius). {_PX}"),
    ("SourceDetectionTask", "matched_filter_width"): (POSITIVE, f"{_H} (FWHM). {_PX}"),
    ("SourceDetectionTask", "centroid_n_iter"): ((1, None), f"hdps default. {_N}"),
    ("SourceDetectionTask", "cr_detect_contrast"): ((0.0, 1.0), "hdps default. A fraction of the peak."),
    ("SourceDetectionTask", "de_dup_sources"): ({True, False}, "hdps default."),
    ("SourceDetectionTask", "de_dup_radius"): (POSITIVE, f"hdps default. {_PX}"),
    ("SourceDetectionTask", "saturation_threshold"): ((0.0, 1.0), "hdps default. A fraction of 65534."),
    ("SourceFilterTask", "min_x"): ((0, None), "hdps default. A pixel index: not negative."),
    ("SourceFilterTask", "min_y"): ((0, None), "hdps default. A pixel index: not negative."),
    ("SourceFilterTask", "max_x"): (None, "hdps default (none: no limit)."),
    ("SourceFilterTask", "max_y"): (None, "hdps default (none: no limit)."),
    ("SourceFilterTask", "min_snr"): (None, "hdps default (none: no cut)."),
    ("ReferenceRetrievalTask", "catalog_provider"): ({"ATLAS"}, f"{_H}. The one provider with a local catalogue "
                                                     "here (Refcat2)."),
    ("ReferenceRetrievalTask", "radius_deg"): ((0.0, 180.0), f"{_H}. An angle on the sky; to cover a whole frame, "
                                               "at least the sensor's half-diagonal (2.0 deg for ArgusSim's)."),
    ("ReferenceRetrievalTask", "min_mag"): (None, "hdps default."),
    ("ReferenceRetrievalTask", "max_mag"): (None, f"{_H}; fainter than min_mag."),
    ("ReferenceRetrievalTask", "pixscale"): (None, "none: ArgusSim's plate scale (telescope pixel_size / "
                                             "focal_length); hdps's config has its prototype camera's 1.42."),
    ("ReferenceRetrievalTask", "cacheable"): ({True, False}, "hdps default."),
    ("ReferenceRetrievalTask", "cache_precision"): (POSITIVE, "hdps default (deg). Positive."),
    ("ReferenceRetrievalTask", "cache_max_age_days"): ((0.0, None), "hdps default. Not negative."),
    ("LinearWCSSolverTask", "n_stars"): ((3, None), f"{_H}. At least 3 stars fix a linear (6-parameter) solution."),
    ("SourceMatchingTask", "n_sigma_clip"): (POSITIVE, f"{_H}. {_SIG}"),
    ("SourceMatchingTask", "region_shape"): (None, f"{_H}: [rows, columns] of matching regions."),
    ("TPVWCSSolverTask", "tpv_terms"): (None, f"{_H}. A TPV term count (hdps's solver decides which it takes)."),
    ("TPVWCSSolverTask", "niter"): ((1, None), f"{_H}. {_N}"),
    ("TPVWCSSolverTask", "n_sigma_clip"): (POSITIVE, f"hdps default. {_SIG}"),
    ("TPVWCSSolverTask", "use_annealing"): ({True, False}, f"{_H}"),
    ("TPVWCSSolverTask", "use_chebyshev"): ({True, False}, f"{_H}"),
    ("ReprojectionTask", "segment_shape"): (None, "hdps default: the 2048 x 2048 px tile of hdps's sky "
                                            "segmentation; fixed by that scheme."),
    ("ReprojectionTask", "grid_resolution"): ((2, None), "hdps default. A grid of at least 2 points per axis."),
    ("CoaddTapTask", "background_mode"): ({"scalar", "regional"}, f"{_H}. hdps checks."),
    ("SFFTSubtractionTask", "reference_mode"): ({"coadd", "first_frame"}, f"{_H}: coadd, the references are earlier "
                                                "coadds (its reference_dir: the run's reference pass, set by "
                                                "mesa_argus.hdps_worker)."),
    ("SFFTSubtractionTask", "kernel_half_width"): ((1, None), "hdps default. hdps checks it is below the tile "
                                                   "overlap minus 2 px."),
    ("SFFTSubtractionTask", "kernel_poly_order"): ((0, None), f"{_H}. A polynomial order: not negative."),
    ("SFFTSubtractionTask", "background_order"): ((0, None), f"{_H} (2 beats 1). A polynomial order: not negative."),
    ("SFFTSubtractionTask", "mask_saturated_in_fit"): ({True, False}, "hdps default."),
    ("SFFTSubtractionTask", "mask_no_coverage_in_fit"): ({True, False}, "true (hdps_full_cpu; hdps main has no such "
                                                         "setting and fits across them): pixels with no reference or "
                                                         "no science data kept out of the kernel fit. Fitting across "
                                                         "them gives the kernel a flat pedestal (a kernel-sized square "
                                                         "around every star) and a background gradient."),
    ("SFFTSubtractionTask", "mask_bright_stars"): ({True, False}, "true (hdps default and production false): "
                                                   "bright-star residual footprints kept out of detection (inf "
                                                   "Var(D)), so with no real/bogus model their residuals are not "
                                                   "alerts. Real variability under the mask is not detected either."),
    ("SFFTSubtractionTask", "bright_star_threshold"): (POSITIVE, "electrons per pixel in the reference (sky-"
                                                       "subtracted coadd) or the science tile (which still holds its "
                                                       "sky, ~1000-1300 e- in rho here): above the science sky, or "
                                                       "the whole science tile is masked. hdps default 5000."),
    ("SFFTSubtractionTask", "mask_dilation_growth"): ((0, None), "hdps default: px added to the mask radius per "
                                                      "magnitude above the threshold."),
    ("SFFTSubtractionTask", "variance_conv"): ({"direct", "fft"}, "hdps default. hdps checks."),
    ("SFFTSubtractionTask", "rejection_iters"): ((0, None), f"{_H}. A count: not negative."),
    ("SFFTSubtractionTask", "rejection_sigma"): (POSITIVE, f"{_H}. {_SIG}"),
    ("SFFTSubtractionTask", "rejection_grow"): ((0, None), f"{_H}. Pixels: not negative."),
    ("SFFTSubtractionTask", "retain_planes"): ({True, False}, "true (hdps default false): AlertVettingTask's ref/sci "
                                               "stamps need the pre-subtraction planes (its emit_stamps note)."),
    ("DiffDetectionTask", "detect_sigma"): (POSITIVE, f"{_H}. {_SIG}"),
    ("DiffDetectionTask", "bg_region_size"): ((1, None), f"{_H}. {_PX}"),
    ("DiffDetectionTask", "max_sources"): ((1, None), f"hdps default. {_N}"),
    ("DiffDetectionTask", "matched_filter_width"): (POSITIVE, f"hdps default (FWHM). {_PX}"),
    ("TransientOutputTask", "fmt"): ({"parquet", "csv"}, f"{_H}. hdps checks."),
    ("AlertVettingTask", "enabled"): ({True, False}, f"{_H} has false (no classifier deployed); true here: the "
                                      "alerts are an output."),
    ("AlertVettingTask", "model_bundle"): (None, "\"\": no real/bogus model exists; every candidate scores 1 "
                                           "(decision 2026-10-20)."),
    ("AlertVettingTask", "detect_sigma"): (POSITIVE, f"{_H}. {_SIG}"),
    ("AlertVettingTask", "score_cut"): ((0.0, 1.0), "A real/bogus score is a probability. With every score 1, any "
                                        "cut accepts every candidate."),
    ("AlertVettingTask", "emit_all_candidates"): ({True, False}, "hdps default."),
    ("AlertVettingTask", "emit_stamps"): ({True, False}, "true (hdps default false): ref/sci/diff stamps with each "
                                          "alert."),
    ("AlertVettingTask", "stamp_size"): (lambda v, r: None if isinstance(v, int) and v > 0 and v % 2 else
                                         "a positive odd integer", "hdps default. Odd, so the candidate sits on the "
                                         "centre pixel."),
    ("AlertVettingTask", "emit_kafka"): ({False}, f"{_H}: no message broker here."),
    ("AlertVettingTask", "fmt"): ({"parquet", "csv"}, "hdps default."),
    ("LightweightFITSOutputTask", "output_dir"): (None, "hdps default; unused in the worker (its product goes to "
                                                  "the data-products sink, fits_image)."),
    ("TileImageOutputTask", "enabled"): ({True, False}, "hdps default false; true here: the field movie's images "
                                         "are hdps's tiles."),
    ("TileImageOutputTask", "output_dir"): (None, "hdps default \"\": the data-products sink."),
    ("TileImageOutputTask", "science_output_dir"): (None, "hdps default (output_dir)."),
    ("TileImageOutputTask", "difference_output_dir"): (None, "hdps default (output_dir)."),
    ("TileImageOutputTask", "save_science"): ({True, False}, "hdps default."),
    ("TileImageOutputTask", "save_difference"): ({True, False}, "hdps default."),
    ("TileImageOutputTask", "save_difference_variance"): ({True, False}, "hdps default."),
    ("TileImageOutputTask", "compress"): ({True, False}, "false (hdps default true): exact float32 with NaN kept; "
                                          "RICE quantizes and maps NaN to 0."),
    ("TileImageOutputTask", "tile_ids"): (None, "hdps default (none: every tile)."),
    ("TileImageOutputTask", "filename_prefix"): (None, "hdps default."),
}

SECTIONS = {"field": FIELD, "night": NIGHT, "stars": STARS, "camera": CAMERA, "lightcurves": LIGHTCURVES,
            "truth": TRUTH, "output": OUTPUT}


def check(where, p, v, resolved=None):
    """None if v is allowed for p, else the error text."""
    if p.optional and v == NONE:
        return None
    a = p.allowed
    if a is None:
        return None
    if a == POSITIVE:
        ok = not isinstance(v, bool) and isinstance(v, (int, float)) and v > 0
        return None if ok else f"{where} = {v!r}: > 0"
    if callable(a):
        err = a(v, resolved)
        return f"{where} = {v!r}: {err}" if err else None
    if isinstance(a, set):
        return None if v in a else f"{where} = {v!r}: one of {sorted(a, key=str)}"
    lo, hi = a
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return f"{where} = {v!r}: a number, {_range(p)}"
    return None if (lo is None or v >= lo) and (hi is None or v <= hi) else f"{where} = {v!r}: {_range(p)}"


def _value(v):
    from mesa_argus.tomlfile import value

    return value(v)


def _range(p):
    a = p.allowed
    if a is None:
        return "any"
    if a == POSITIVE:
        return "> 0"
    if callable(a):
        return p.allowed_text
    if isinstance(a, set):
        return " | ".join(_value(x) for x in sorted(a, key=str))
    lo, hi = a
    if hi is None:
        return f">= {lo} (no upper limit)"
    return f"{lo} .. {hi}"


def template():
    """The documented observation file: every parameter, its default, unit, allowed values and why."""
    import textwrap

    def notes(text):
        return [f"# {line}" for line in textwrap.wrap(text, 116)]

    def block(title, params, header=None):
        out = [f"[{title}]"] + (notes(header) if header else [])
        for p in params:
            default = "\"required\"" if p.default == REQUIRED else _value(NONE if p.default is None else p.default)
            note = (f"[{p.unit}] " if p.unit else "") + f"allowed: {_range(p)}" + (" or \"none\"" if p.optional else "")
            out += notes(f"{note}. {p.why}") + [f"{p.key} = {default}"]
        return out + [""]

    lines = notes("An observation: the one file that defines a run. Written by python -m mesa_argus.schema; every "
                  "parameter with its default, unit, allowed values and the justification.") + [""]
    lines += block("field", FIELD) + block("night", NIGHT) + block("stars", STARS)
    for sub, ps in ARGUSSIM.items():
        lines += block(f"argussim.{sub}", ps, "ArgusSim: how the image is taken. Defaults are read from ArgusSim.")
    lines += block("camera", CAMERA, "The toy camera (mesa_argus/camera.py): electrons to ADU, which ArgusSim does "
                   "not define. none: the default described.")
    lines += block("hdps", HDPS, "hdps: the pipeline the raw frames go through.") + block("hdps.coadd", HDPS_COADD)
    lines += notes("[[hdps.tasks]]: hdps's Task chain, in order, run by hdps.orchestrate's PipelineExecutor; each "
                   "[hdps.tasks.config] holds the Task's constructor parameters.") + [""]
    for t in TASKS:
        lines += ["[[hdps.tasks]]", f"type = \"{t['type']}\"", "[hdps.tasks.config]"]
        for k, v in t["config"].items():
            allowed, why = TASK_RULES[(t["type"], k)]
            text = "a positive odd integer" if callable(allowed) else _range(P(k, v, "", allowed, why))
            lines += notes(f"allowed: {text}. {why}") + [f"{k} = {_value(v)}"]
        lines.append("")
    lines += block("lightcurves", LIGHTCURVES) + block("truth", TRUTH) + block("output", OUTPUT)
    lines += notes("[[place]]: any number of built stars placed in the field (each key below).")
    for p in PLACE:
        lines += notes(f"{p.key}: [{p.unit}] allowed {_range(p)}. {p.why}")
    lines += ["# [[place]]", "# star = \"dwarf_nova_400pc\"", "# x = 60", "# y = 40", "# t0 = \"start+0.05\"", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "examples" / "observations" / "template.toml"
    out.write_text(template())
    print(f"wrote {out}")
