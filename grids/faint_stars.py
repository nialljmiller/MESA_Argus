"""Lower-main-sequence optical SED tracks at the supported cool-atmosphere edge.

These tracks extend the existing mass grid toward faint late-K / early-M stars.
The current Kurucz atmosphere table is not a valid spectrum source below about
3500 K, so cooler track rows remain structural coverage until a cool-star
atmosphere grid is integrated.
"""

TRACKS = (
    (0.45, 0.0),
    (0.50, 0.0),
    (0.55, 0.0),
    (0.65, 0.0),
    (0.50, -1.0),
    (0.60, -1.5),
    (0.80, -2.0),
)


def tracks(make):
    out = []
    for mass, feh in TRACKS:
        extra = ["power_he_burn_upper_limit = 10d0"] if mass <= 2.0 else []
        out.append(make(mass, feh, make.alpha_sun, save_seds=True,
                        tag="faint_edge", population="faint lower-main-sequence coverage",
                        kind="base", why="extend ordinary optical SED coverage toward the cool Kurucz edge",
                        extra_controls=extra))
    return out
