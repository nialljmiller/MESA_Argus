"""The measurement pipeline run on every simulated exposure, treating it as a real image (hdps wherever hdps has it).

One plain module per stage, each with one function per implementation; observe.py runs them in this order and picks
each implementation by name from the observation's [pipeline] section (STAGES below). Swapping a stage for
production code is a new function and a new name here.

    frames       an exposure as an hdps TANSegment (a sky tile cut-out with its WCS); FITS in and out
    extraction   every source in a frame, measured (hdps SourceDetector), with RA/Dec from the tile WCS
    calibration  the frame's zero point against ATLAS-Refcat2 (hdps zeropoint)
    psf          the frame's PSF model (hdps TilePSFTask's Moffat field fit); its FWHM feeds the alert features
    reference    the image frames are subtracted from (a sacrificial night's median), in hdps's reference format
    dia          hdps's SFFTSubtractionTask -> DiffDetectionTask -> TransientOutputTask (hdps's NumPy SFFT)
    alerts       hdps's AlertVettingTask: features, alert catalogue, schema 3.0 packets, stamps
    lightcurves  every frame's detections linked into objects: a catalogue and one light curve per object
    truth        which simulated star each detection / candidate is: for scoring only; no stage above reads it
"""
from . import alerts, calibration, dia, extraction, frames, lightcurves, psf, reference, truth

STAGES = {
    "extraction": {"hdps": extraction.hdps},
    "calibration": {"atlas_refcat2": calibration.atlas_refcat2},
    "psf": {"hdps_moffat": psf.hdps_moffat},
    "reference": {"sacrificial_night": reference.sacrificial_night},
    "dia": {"hdps_sfft": dia.hdps_sfft},
    "alerts": {"hdps_vetting": alerts.hdps_vetting},
    "lightcurves": {"positional": lightcurves.positional},
}
DEFAULTS = {"extraction": "hdps", "calibration": "atlas_refcat2", "psf": "hdps_moffat", "reference": "sacrificial_night",
            "dia": "hdps_sfft", "alerts": "hdps_vetting", "lightcurves": "positional"}
SETTINGS = {**extraction.SETTINGS, **calibration.SETTINGS, **psf.SETTINGS, **dia.SETTINGS, **alerts.SETTINGS,
            **lightcurves.SETTINGS, **truth.SETTINGS}  # every stage's ([measure])


def stage(name, implementation=None):
    """The function for one stage, by implementation name (the default if not given)."""
    implementation = implementation or DEFAULTS[name]
    if implementation not in STAGES[name]:
        raise SystemExit(f"[pipeline] {name} = {implementation!r}: there are {sorted(STAGES[name])}")
    return STAGES[name][implementation]
