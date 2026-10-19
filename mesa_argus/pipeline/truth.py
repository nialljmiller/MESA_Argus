"""Which simulated star each detection is. For scoring the pipeline only: no measuring stage reads this.

The simulated stars in order of expected flux, brightest first, each take the nearest detection not yet taken within
match_radius pixels; a detection belongs to at most one star. A star with no detection was too faint, or merged
into a brighter neighbour's detection.
"""
import numpy as np

SETTINGS = dict(match_radius=2.0)


def associate(det_x, det_y, star_x, star_y, star_flux, radius):
    """For each star, the index of its detection, or -1."""
    match = np.full(len(star_x), -1)
    taken = np.zeros(len(det_x), bool)
    for i in np.argsort(-np.nan_to_num(star_flux, nan=0.0), kind="stable"):
        d = np.hypot(det_x - star_x[i], det_y - star_y[i])
        d[taken] = np.inf
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= radius:
            match[i] = j
            taken[j] = True
    return match


def label(catalogue, star_x, star_y, star_flux, cfg):
    """Adds `star` (the simulated star's index, -1 for none) to the catalogue; returns each star's detection index."""
    radius = cfg.get("match_radius", SETTINGS["match_radius"])
    star_of = associate(np.asarray(catalogue["xcentroid"], float), np.asarray(catalogue["ycentroid"], float),
                        star_x, star_y, star_flux, radius)
    star = np.full(len(catalogue), -1)
    star[star_of[star_of >= 0]] = np.nonzero(star_of >= 0)[0]
    catalogue["star"] = star
    return star_of
