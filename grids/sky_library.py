"""Coverage library for observed-sky modelling, NOT a population probability distribution.

MESA evolution with the existing calibrated composition, physics and atmospheres.
Sample these tracks by initial mass, age/SFH, metallicity, evolutionary dwell time,
distance, extinction and the Argus selection function; NEVER uniformly by model row.
The old-population restriction belongs to the sampler, not stellar evolution:
metal-poor intermediate-mass tracks also cover HB progenitors / satellite populations.
This initial library covers 0.6--20 Msun and [M/H] -2.5--+0.3. It does not cover
cool M dwarfs, white dwarfs, TP-AGB, rotation or interacting binaries. Kurucz grid
membership and bolometric-flux checks in seds.h5 must be applied before use.
Low-mass bulk tracks stop before the costly helium flash. Three separate pilots
attempt the flash and red clump / horizontal branch without that stop.
"""

MASS_BY_MH = {
    0.0: (0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0),
    -0.5: (0.6, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0),
    0.3: (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0),
    -1.0: (0.6, 0.7, 0.8, 0.85, 0.9, 1.0, 1.5, 2.0, 3.0),
    -1.5: (0.6, 0.7, 0.8, 0.85, 0.9, 1.0, 1.5, 2.0),
    -2.0: (0.6, 0.7, 0.8, 0.85, 0.9, 1.0),
    -2.5: (0.6, 0.7, 0.8, 0.85, 0.9, 1.0),
}
PILOT = ((0.8, 0.0), (1.2, 0.0), (2.0, -0.5), (0.8, -1.5), (0.8, 0.3), (8.0, -0.5))
HB_PILOT = ((1.0, 0.0), (0.9, -1.0), (0.8, -1.5))
# Already available as packed SEDs on Newton: include in the manifest, don't recompute.
EXISTING = {(1.0, 0.0), (0.8, -2.0), (3.0, 0.0), (5.0, 0.0), (10.0, 0.0)}


def tracks(make):
    out = []
    for mh, masses in MASS_BY_MH.items():
        for mass in masses:
            old = (mass, mh) in EXISTING
            out.append(make(mass, mh, make.alpha_sun, save_seds=True,
                            tag="sed" if old else "sky_sed",
                            population="coverage only; assign observed-population weights separately",
                            kind="base", why="ordinary-star SED coverage for Argus sky synthesis",
                            extra_controls=["power_he_burn_upper_limit = 10d0"] if mass <= 2.0 else []))
    for mass, mh in HB_PILOT:
        out.append(make(mass, mh, make.alpha_sun, save_seds=True, tag="sky_hb_pilot",
                        population="coverage only; age selection required", kind="HB pilot",
                        why="attempt He flash through red clump/HB; pilot, not guaranteed convergence",
                        extra_controls=[]))
    return out
