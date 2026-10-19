"""The measurement pipeline run on every simulated exposure, treating it as a real image (hdps wherever hdps has it).

One plain module per stage, each with one function per implementation; observe.py runs them in this order and picks
each implementation by name from the observation's [pipeline] section (STAGES below). Swapping a stage for
production code is a new function and a new name here.

    frames       an exposure as an hdps TANSegment (a sky tile cut-out with its WCS); FITS in and out
    extraction   every source in a frame, measured (hdps SourceDetector), with RA/Dec from the tile WCS
    calibration  the frame's zero point against ATLAS-Refcat2 (hdps zeropoint)
    psf          the frame's PSF model (hdps TilePSFTask's Moffat field fit), difference imaging's matched filter
    reference    the image frames are subtracted from (a sacrificial night's median), in hdps's reference format
    dia          ZOGY subtraction and transient candidates (hdps diffimg)
    alerts       alert packets for the candidates (hdps alert schema 3.0, as its Kafka producer sends them)
    truth        which simulated star each detection / candidate is: for scoring only; no stage above reads it
"""
from . import alerts, calibration, dia, extraction, frames, psf, reference, truth

STAGES = {
    "extraction": {"hdps": extraction.hdps},
    "calibration": {"atlas_refcat2": calibration.atlas_refcat2},
    "psf": {"hdps_moffat": psf.hdps_moffat},
    "reference": {"sacrificial_night": reference.sacrificial_night},
    "dia": {"hdps_zogy": dia.hdps_zogy},
    "alerts": {"hdps_v3": alerts.hdps_v3},
}
DEFAULTS = {"extraction": "hdps", "calibration": "atlas_refcat2", "psf": "hdps_moffat", "reference": "sacrificial_night",
            "dia": "hdps_zogy", "alerts": "hdps_v3"}
SETTINGS = {**extraction.SETTINGS, **calibration.SETTINGS, **psf.SETTINGS, **dia.SETTINGS, **truth.SETTINGS}  # every stage's ([measure])


def stage(name, implementation=None):
    """The function for one stage, by implementation name (the default if not given)."""
    implementation = implementation or DEFAULTS[name]
    if implementation not in STAGES[name]:
        raise SystemExit(f"[pipeline] {name} = {implementation!r}: there are {sorted(STAGES[name])}")
    return STAGES[name][implementation]
