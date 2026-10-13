"""Interstellar dust: the CCM89 extinction law, applied to every SED the movies see (stars, pulsators, binaries,
transients) with the field's E(B-V)."""

import numpy as np

RV = 3.1


def ccm89(wavelength_angstrom, rv=RV):
    """A_lambda / A_V, Cardelli, Clayton & Mathis (1989), 0.3-10 um^-1 (IR + optical/NIR)."""
    x = 1e4 / np.asarray(wavelength_angstrom, float)
    a, b = np.zeros_like(x), np.zeros_like(x)
    ir = (x >= 0.3) & (x < 1.1)
    a[ir], b[ir] = 0.574 * x[ir] ** 1.61, -0.527 * x[ir] ** 1.61
    opt = (x >= 1.1) & (x <= 3.3)
    y = x[opt] - 1.82
    a[opt] = 1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 + 0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6 + 0.32999 * y**7
    b[opt] = 1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 - 5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7
    out = a + b / rv
    out[(x < 0.3) | (x > 3.3)] = np.nan  # outside the law; the Argus bands lie well inside
    return out


def redden(wavelength_angstrom, flux, ebv, rv=RV):
    """Apply CCM89 extinction; zero outside its validity range is not needed for Argus (360-830 nm)."""
    alam = ccm89(wavelength_angstrom, rv) * rv * ebv
    return flux * 10 ** (-0.4 * np.nan_to_num(alam, nan=0.0))
