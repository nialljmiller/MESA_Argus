"""Mass grid for an M67 isochrone: solar [M/H], no SED output, to 5 Gyr or the RGB tip.

M67 (NGC 2682) is ~4 Gyr old at [Fe/H] ~ 0. Interpolating these tracks at the cluster age
gives the isochrone from which each real member's initial mass is read off its Gaia
photometry (mesa_argus.m67). The grid is fine near the turnoff (~1.3 Msun) because at a
fixed age a giant's position depends on its mass to ~1e-3 Msun.
"""

STOP_AT_RGB_TIP = ["power_he_burn_upper_limit = 10d0"]
# Nothing in the M67 scene is brighter than ~30 Lsun, so stop at 100 Lsun instead of climbing
# the whole RGB (most of a track's models). Tracks 1.29-1.37 were stopped by hand on the upper RGB
# for the same reason; their histories cover every age the isochrone needs.
STOP_HIGH_ON_RGB = ["log_L_upper_limit = 2.0d0"]
MAX_AGE = 5e9

MASSES = [round(0.45 + 0.05 * i, 2) for i in range(15)] + [round(1.16 + 0.01 * i, 2) for i in range(35)]


def tracks(make):
    return [make(mass, 0.0, make.alpha_sun, max_age=MAX_AGE, population="M67 (NGC 2682)",
                 why="M67 isochrone mass grid", kind="base", tag="m67iso", extra_controls=STOP_AT_RGB_TIP + STOP_HIGH_ON_RGB)
            for mass in MASSES]
