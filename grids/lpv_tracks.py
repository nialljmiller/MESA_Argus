"""Checkpointed MESA structural pilots through the luminous AGB.

These are evolutionary progenitor tracks, not LPV pulsation cycles. The current
Kurucz grid is too warm for the coolest Mira phases; the tracks preserve the
stellar parameters while a suitable cool-star / AGB atmosphere is integrated.
"""

TRACKS = (1.5, 2.0, 3.0)


def tracks(make):
    out = []
    for mass in TRACKS:
        out.append(make(mass, 0.0, make.alpha_sun, save_seds=True,
                        tag="lpv_checkpoint", population="AGB / long-period-variable structural pilot",
                        kind="LPV structural pilot", why="continue through core-He burning to log L/Lsun 3.8",
                        continue_past_he_burning=True, accepted_terminations=["log_L_upper_limit"],
                        star_job_extra=["required_termination_code_string(1) = 'log_L_upper_limit'"],
                        extra_controls=["log_L_upper_limit = 3.8d0"],
                        history_interval=10, photo_interval=500,
                        resume_from_photo=True, resume_attempts=4))
    return out
