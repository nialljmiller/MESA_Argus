"""The measurement pipeline run on every simulated exposure, treating it as a real image (hdps wherever hdps has it).

One plain module per stage, each with one function per implementation; observe.py runs them in this order and picks
each implementation by name from the observation's [pipeline] section (STAGES below). Swapping a stage for
production code is a new function and a new name here.

    frames       an exposure as an hdps TANSegment (a sky tile cut-out with its WCS); FITS in and out
    extraction   every source in a frame, measured (hdps SourceDetector), with RA/Dec from the tile WCS
    calibration  the frame's zero point against ATLAS-Refcat2 (hdps zeropoint)
    truth        which simulated star each detection is: for scoring only; no stage above reads the simulation's truth
"""
from . import calibration, extraction, frames, truth

STAGES = {
    "extraction": {"hdps": extraction.hdps},
    "calibration": {"atlas_refcat2": calibration.atlas_refcat2},
}
DEFAULTS = {"extraction": "hdps", "calibration": "atlas_refcat2"}
SETTINGS = {**extraction.SETTINGS, **calibration.SETTINGS, **truth.SETTINGS}  # every stage's settings ([measure])


def stage(name, implementation=None):
    """The function for one stage, by implementation name (the default if not given)."""
    implementation = implementation or DEFAULTS[name]
    if implementation not in STAGES[name]:
        raise SystemExit(f"[pipeline] {name} = {implementation!r}: there are {sorted(STAGES[name])}")
    return STAGES[name][implementation]
