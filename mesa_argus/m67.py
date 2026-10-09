"""An M67 minipix scene: real Gaia stars, a MESA model per star, SEDs over a common 1 Myr window.

  python -m mesa_argus.m67 scene    pick the stars in the 3x3-minipix block -> data/m67/scene.ecsv
  python -m mesa_argus.m67 fit      cluster age from the m67_isochrone grid; mass for every scene star
  python -m mesa_argus.m67 run      two-stage MESA run per star (resumable); SEDs over the window

Scene: tile 296904, minipixes (42-44, 57-59) of the HDPS tessellation (argus_sim.skymap_shim),
about 84" square in M67's core, plus a margin whose PSF wings reach into it. It holds a red
giant, two subgiants, turnoff and main-sequence members, and field stars, with no red-clump
star or blue straggler (neither is a single-star MESA track).

Every Gaia DR3 source (VizieR I/355) is kept. Members (Hunt & Reffert 2023, P > 0.5) share the
cluster age, distance and reddening; their masses come from the M67 isochrone. Field stars get
a main-sequence model of the mass matching their colour, scaled to their observed G: a template,
not a physical model (their distances and ages are unknown). Binaries are modelled as single stars.

Each star runs twice: evolve to (age - 0.5 Myr) without SED output and save the model, then
reload it and write an SED every step (max 5 kyr) to (age + 0.5 Myr).
"""

import json
import sys

import numpy as np

from . import config as C
from .grids import load
from .runner import prepare, run_dir, termination

DATA = C.DATA / "m67"
SCENE = DATA / "scene.ecsv"
FIT = DATA / "fit.json"

TILE = 296904
BLOCK = (43, 58)  # centre minipix (x, y); the block is the 3x3 around it
MINIPIX_PX = 32
SKY_PX_ARCSEC = 900.0 / 1024.0  # tessellation pixel (skymap_shim.PIXEL_SCALE_ARCSEC)
MARGIN_ARCSEC = 20.0  # sources this far outside the block still spill light into it

WINDOW_YR = 1.0e6
MAX_STEP_YR = 5.0e3
EBV = 0.041  # Taylor (2007)
RV = 3.1
# A_band / E(B-V) for Gaia, solar-type stars (Casagrande & VandenBerg 2018); only for matching
# Gaia photometry to MESA magnitudes. The Argus SEDs are reddened with CCM89 directly.
R_GAIA = {"G": 2.740, "BP": 3.374, "RP": 2.035}
PLX_ZERO_POINT = -0.017  # mas, Lindegren et al. (2021) global offset


# ---------------------------------------------------------------- extinction
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


def redden(wavelength_angstrom, flux, ebv=EBV, rv=RV):
    """Apply CCM89 extinction; zero outside its validity range is not needed for Argus (360-830 nm)."""
    alam = ccm89(wavelength_angstrom, rv) * rv * ebv
    return flux * 10 ** (-0.4 * np.nan_to_num(alam, nan=0.0))


# ---------------------------------------------------------------- scene
def _tile_pixels(ra, dec):
    """Continuous tessellation pixel (x, y) of each position in TILE (skymap_shim conventions)."""
    from argus_sim import skymap_shim as sh

    tra, tdec = (float(v[0]) for v in sh.tile_center(np.array([TILE])))
    cd, sd = np.cos(np.radians(dec)), np.sin(np.radians(dec))
    ctd, std = np.cos(np.radians(tdec)), np.sin(np.radians(tdec))
    dra = np.radians(ra - tra)
    den = std * sd + ctd * cd * np.cos(dra)
    xi = np.degrees(cd * np.sin(dra) / den)
    eta = np.degrees((ctd * sd - std * cd * np.cos(dra)) / den)
    crpix = sh.SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    return -xi / sh.PIXEL_SCALE_DEG + crpix - 0.5, eta / sh.PIXEL_SCALE_DEG + crpix - 0.5


def block_centre_px():
    return (BLOCK[0] + 0.5) * MINIPIX_PX, (BLOCK[1] + 0.5) * MINIPIX_PX


def make_scene():
    """Gaia sources in the block plus margin, with offsets in arcsec from the block centre."""
    from astropy.table import Table

    gaia = Table.read(DATA / "gaia_dr3_core.ecsv")
    members = Table.read(DATA / "hunt2023_members.ecsv")
    prob = dict(zip(np.asarray(members["GaiaDR3"]).astype(np.int64), np.asarray(members["Prob"], float)))
    ra, dec = np.asarray(gaia["RA_ICRS"], float), np.asarray(gaia["DE_ICRS"], float)
    px, py = _tile_pixels(ra, dec)
    cx, cy = block_centre_px()
    dx, dy = (px - cx) * SKY_PX_ARCSEC, (py - cy) * SKY_PX_ARCSEC
    half = 1.5 * MINIPIX_PX * SKY_PX_ARCSEC
    keep = (np.abs(dx) <= half + MARGIN_ARCSEC) & (np.abs(dy) <= half + MARGIN_ARCSEC)
    t = gaia[keep]
    t["P"] = [prob.get(int(s), 0.0) for s in np.asarray(t["Source"]).astype(np.int64)]
    t["member"] = t["P"] > 0.5
    t["dx_arcsec"], t["dy_arcsec"] = dx[keep], dy[keep]
    t["in_block"] = (np.abs(t["dx_arcsec"]) <= half) & (np.abs(t["dy_arcsec"]) <= half)
    t.sort("Gmag")
    t["label"] = [f"S{i + 1:02d}" for i in range(len(t))]
    t.meta.update(tile=TILE, block_centre_minipix=list(BLOCK), block_half_arcsec=half,
                  sky_pixel_arcsec=SKY_PX_ARCSEC, catalogues="Gaia DR3 (VizieR I/355); Hunt & Reffert 2023 members")
    t.write(SCENE, overwrite=True)
    for r in t:
        print(f"  {r['label']}  G {r['Gmag']:6.2f}  BP-RP {float(r['BP-RP']):5.2f}  P {r['P']:.2f}  "
              f"{'member' if r['member'] else 'field '}  {'block ' if r['in_block'] else 'margin'}  "
              f"({r['dx_arcsec']:+6.1f}, {r['dy_arcsec']:+6.1f})\"")
    print(f"{len(t)} sources ({int(np.sum(t['in_block']))} in the block, {int(np.sum(t['member']))} members) -> {SCENE}")
    return t


# ---------------------------------------------------------------- isochrone grid
def grid_histories():
    """{initial mass: history} for the m67_isochrone tracks, finished or still running.

    A running track is usable up to the age it has reached: isochrone_at skips any track
    younger than the requested age, and the ages that matter (~4 Gyr) come long before the
    RGB tip where the slow tracks are still climbing.
    """
    out = {}
    for t in load("m67_isochrone"):
        path = C.RUNS / t["name"] / "LOGS" / "history.data"
        if path.exists():
            # a running track's last line can be half written
            h = np.genfromtxt(path, skip_header=5, names=True, invalid_raise=False)
            if h.size > 10:
                out[t["mass"]] = h
    return out


def _age_at(h, column, value, phase_mask):
    """First age within phase_mask at which column crosses value (linear in age), else nan."""
    x, a = h[column][phase_mask], h["star_age"][phase_mask]
    s = np.sign(x - value)
    hit = np.nonzero(s[:-1] * s[1:] <= 0)[0]
    if not len(hit):
        return np.nan
    i = hit[0]
    f = (value - x[i]) / (x[i + 1] - x[i]) if x[i + 1] != x[i] else 0.0
    return a[i] + f * (a[i + 1] - a[i])


def _phases(h):
    """Boolean masks: main sequence (incl. turnoff), subgiant branch, red giant branch."""
    post = h["center_h1"] < 1e-3
    ms = (~post) & (10.0 ** (h["log_LH"] - h["log_L"]) > 0.5)
    rgb_start = np.argmax(post & (h["log_Teff"] < np.log10(5000.0))) if np.any(post & (h["log_Teff"] < np.log10(5000.0))) else len(h)
    sgb = post.copy()
    sgb[rgb_start:] = False
    rgb = post & ~sgb
    return ms, sgb, rgb


def isochrone_at(hist, age):
    """Absolute Gaia magnitudes of every grid mass still on its track at this age (MS..RGB)."""
    rows = []
    for mass, h in sorted(hist.items()):
        if h["star_age"][-1] < age:
            continue
        i = np.searchsorted(h["star_age"], age)
        f = (age - h["star_age"][i - 1]) / (h["star_age"][i] - h["star_age"][i - 1])
        rows.append([mass] + [h[c][i - 1] + f * (h[c][i] - h[c][i - 1]) for c in ("G", "Gbp", "Grp", "log_Teff", "log_g")])
    return np.array(rows)


def cluster_distance_modulus():
    from astropy.table import Table

    m = Table.read(DATA / "hunt2023_members.ecsv")
    good = (m["Prob"] > 0.7) & (m["Gmag"] < 17) & (m["RUWE"] < 1.4)
    plx = np.median(np.asarray(m["Plx"][good], float)) - PLX_ZERO_POINT
    return 5 * np.log10(100.0 / plx), plx


def observed_to_absolute(g, bp_rp, dm):
    return g - dm - R_GAIA["G"] * EBV, bp_rp - (R_GAIA["BP"] - R_GAIA["RP"]) * EBV


def fit_age(hist, dm, ages=np.arange(3.0e9, 5.01e9, 0.05e9), shifts=np.arange(-0.06, 0.0601, 0.005)):
    """Age (and a small colour shift) that best fit the turnoff, subgiant and lower-RGB members.

    The colour shift absorbs the models' ~0.03 mag turnoff colour offset (reddening, mixing
    length, atmospheres), so the age is set by luminosities: turnoff and subgiant branch.
    Ages whose isochrone does not reach the RGB (turnoff masses not in the grid) are skipped.
    """
    from astropy.table import Table

    m = Table.read(DATA / "hunt2023_members.ecsv")
    sel = (m["Prob"] > 0.7) & (m["RUWE"] < 1.4) & (m["Gmag"] > 11.8) & (m["Gmag"] < 14.5) & (m["BP-RP"] > 0.65)
    g, c = observed_to_absolute(np.asarray(m["Gmag"][sel], float), np.asarray(m["BP-RP"][sel], float), dm)
    scores, best_shift = [], []
    for age in ages:
        iso = isochrone_at(hist, age)
        if len(iso) < 5 or iso[-1, 4] > np.log10(5000.0):
            scores.append(np.inf)
            best_shift.append(np.nan)
            continue
        ig, ic = iso[:, 1], iso[:, 2] - iso[:, 3]
        # densify along the isochrone so the distance is to the curve, not its vertices
        s_ = np.linspace(0, len(ig) - 1, 20 * len(ig))
        fg, fc = np.interp(s_, np.arange(len(ig)), ig), np.interp(s_, np.arange(len(ig)), ic)
        trial = [np.median(np.min(np.hypot((g[:, None] - fg[None]) / 0.1, (c[:, None] - fc[None] - d) / 0.02), axis=1))
                 for d in shifts]
        scores.append(min(trial))
        best_shift.append(float(shifts[int(np.argmin(trial))]))
    k = int(np.argmin(scores))
    return float(ages[k]), best_shift[k], dict(zip(map(float, ages), map(float, scores)))


def assign_mass(hist, age, g_abs, c_abs, member, shift=0.0):
    """Initial mass whose track sits at this (M_G, colour) at the cluster age; and its phase.

    shift is the fitted model-minus-data colour offset: model colour + shift = observed colour.
    """
    c_abs = c_abs - shift
    masses = np.array(sorted(hist))
    ms_g = []
    for mass in masses:
        h = hist[mass]
        ms, sgb, rgb = _phases(h)
        on_ms = h["star_age"][-1] >= age and ms[min(np.searchsorted(h["star_age"], age), len(h) - 1)]
        ms_g.append(np.interp(age, h["star_age"], h["G"]) if on_ms else np.nan)
    ms_g = np.array(ms_g)
    ok = np.isfinite(ms_g)
    if not member:
        # field star: main-sequence template of the matching colour (colour, not G: distance unknown)
        iso = isochrone_at(hist, age)
        col = iso[:, 2] - iso[:, 3]
        on = np.isin(iso[:, 0], masses[ok])
        order = np.argsort(col[on])
        return float(np.interp(c_abs, col[on][order], iso[on, 0][order])), "field (MS template)"
    # main sequence: below the turnoff and blueward of the subgiant branch, or anything fainter
    # than the subgiant branch (red dwarfs are red, not evolved)
    if g_abs >= np.nanmin(ms_g[ok]) - 0.05 and (c_abs < 0.95 or g_abs > 4.0):
        order = np.argsort(ms_g[ok])[::-1]
        return float(np.interp(-g_abs, -ms_g[ok][order], masses[ok][order])), "main sequence / turnoff"
    # evolved: age at which each track reaches this position, solved for age == cluster age
    phase, column, value = ("subgiant", "colour", c_abs) if g_abs > 2.0 and c_abs < 1.1 else ("red giant", "G", g_abs)
    ages, ms_ = [], []
    for mass in masses:
        h = hist[mass]
        _, sgb, rgb = _phases(h)
        mask = sgb if phase == "subgiant" else rgb
        if column == "colour":
            h = {"star_age": h["star_age"], "colour": h["Gbp"] - h["Grp"]}
        a = _age_at(h, column if column == "G" else "colour", value, mask)
        if np.isfinite(a):
            ages.append(a)
            ms_.append(mass)
    ages, ms_ = np.array(ages), np.array(ms_)
    order = np.argsort(ages)
    if not len(ages) or not ages.min() <= age <= ages.max():
        raise ValueError(f"no grid track reaches ({g_abs:.2f}, {c_abs:.2f}) as a {phase} at {age / 1e9:.2f} Gyr")
    return float(np.interp(age, ages[order], ms_[order])), phase


def fit():
    from astropy.table import Table

    hist = grid_histories()
    print(f"{len(hist)} finished isochrone tracks")
    dm, plx = cluster_distance_modulus()
    age, shift, scores = fit_age(hist, dm)
    print(f"distance modulus {dm:.3f} (median member parallax {plx:.4f} mas incl. zero point); "
          f"E(B-V) {EBV}; best age {age / 1e9:.2f} Gyr with colour shift {shift:+.3f} mag")
    t = Table.read(SCENE)
    stars = []
    for r in t:
        c = float(r["BP-RP"]) if np.isfinite(float(r["BP-RP"])) else 1.0
        g_abs, c_abs = observed_to_absolute(float(r["Gmag"]), c, dm)
        mass, phase = assign_mass(hist, age, g_abs, c_abs, bool(r["member"]), shift)
        stars.append(dict(label=r["label"], source=int(r["Source"]), member=bool(r["member"]), mass=round(mass, 5),
                          phase=phase, G=float(r["Gmag"]), bp_rp=c))
        print(f"  {r['label']}  G {r['Gmag']:6.2f}  BP-RP {c:5.2f}  -> {mass:.4f} Msun  {phase}")
    FIT.write_text(json.dumps(dict(age=age, colour_shift=shift, distance_modulus=dm, parallax_mas=plx, ebv=EBV, rv=RV,
                                   age_scores=scores, stars=stars), indent=1) + "\n")
    print(f"-> {FIT}")


# ---------------------------------------------------------------- per-star runs
def star_tracks():
    """One staged track per scene star (grid name m67_stars)."""
    from .grids import _Maker, calibration

    f = json.loads(FIT.read_text())
    make = _Maker(calibration())
    out = []
    for s in f["stars"]:
        t = make(s["mass"], 0.0, make.alpha_sun, max_age=f["age"], population="M67 (NGC 2682)",
                 why=f"{s['label']} G={s['G']:.2f}: {s['phase']}", kind="m67", tag=f"m67_{s['label']}",
                 extra_controls=["power_he_burn_upper_limit = 10d0"])
        t.update(label=s["label"], window=(f["age"] - WINDOW_YR / 2, f["age"] + WINDOW_YR / 2))
        out.append(t)
    return out


def run_star(t):
    """Stage 1: evolve to the window start and save; stage 2: reload and write SEDs across the window."""
    d = C.RUNS / t["name"]
    start, end = t["window"]
    if (d / "SED").is_dir() and termination(d) and (d / "LOGS" / "history.data").exists():
        return t["name"], True, "done"
    if not (d / "window_start.mod").exists():
        stage1 = dict(t, max_age=start, save_seds=False, save_model="window_start.mod",
                      extra_controls=t["extra_controls"] + ["log_directory = 'LOGS_evolve'"])
        ok, minutes, term = run_dir(prepare(stage1))
        (d / "out.txt").rename(d / "out_evolve.txt")
        if not (d / "window_start.mod").exists():
            return t["name"], False, f"stage 1: {term}"
    stage2 = dict(t, max_age=end, save_seds=True, load_model="window_start.mod",
                  extra_controls=t["extra_controls"] + [f"max_years_for_timestep = {MAX_STEP_YR:.1f}d0"])
    ok, minutes, term = run_dir(prepare(stage2))
    return t["name"], ok, term


def run_all(workers=None):
    from concurrent.futures import ThreadPoolExecutor

    tracks = star_tracks()
    workers = workers or C.config()["run"]["workers"]
    with ThreadPoolExecutor(workers) as ex:
        for name, ok, term in ex.map(run_star, tracks):
            print(f"  {'OK ' if ok else 'BAD'} {name:<36s} {term}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "scene":
        make_scene()
    elif cmd == "fit":
        fit()
    elif cmd == "run":
        run_all()
    else:
        raise SystemExit(__doc__)
