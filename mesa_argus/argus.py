"""The Argus bands as the movies and plots use them: ArgusSim filter names, symbols, colours, units, plot style."""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

FLAM = u.erg / (u.s * u.cm**2 * u.AA)
BANDS = {"beta": "g", "rho": "rho"}
COLORS = {"beta": "#2078b4", "rho": "#cc5940", "colour": "#7853a6"}
SYMBOLS = {"beta": r"$\beta$", "rho": r"$\rho$"}
MAG_FACTOR = 2.5 / np.log(10.0)


def system_response(tp, band, airmass):
    """Filter x optics x QE x atmosphere^airmass, as in ABPhot.electrons_from_sed (for plotting)."""
    response = np.asarray(tp.filt_wav[band] * tp.throughput_wav * tp.qe_wav, dtype=float)
    return response * np.asarray(tp.filt_wav.atmosphere, dtype=float) ** airmass


def apply_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#86939e",
            "axes.labelcolor": "#253746",
            "xtick.color": "#435766",
            "ytick.color": "#435766",
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
        }
    )
