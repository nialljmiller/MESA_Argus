"""One night of Argus on any pointing: Gaia field -> chosen variables modelled (MESA RSP / PHOEBE) -> movies.

    python argus_field.py --ra 8.8411 --dec 52.9981 --list                 # what is there, what can be modelled
    python argus_field.py --ra 8.8411 --dec 52.9981 --select 1 --date auto # model star 1, best night, both bands
    python argus_field.py --ra 278.2136 --dec -16.9417 --block 9 --select all --date auto

Runs from any directory (data/, runs/, outputs/ are in the repo). Needs the network (VizieR, IRSA; PHOEBE passbands
on first use), MESA built at mesa_work/build/bin/star, and config.toml (MESA colors data; [run] workers = MESA runs
at once).

1. Field: Gaia DR3 (VizieR) within 0.12 deg (the 8' x 8' simulated area): every source, Gaia's variability
   classification, its RR Lyrae / Cepheid / eclipsing-binary tables and non-single-star eclipsing solutions; ASAS-SN
   variables (HADS periods). Cached in data/<field>/.
2. Block shown: --block x --block HDPS minipixes (odd; 3 = 84", 9 = 253") centred on the minipix holding the pointing.
   Every catalogued variable in it is listed, modellable first, with how it would be modelled or why not.
3. You choose which to model: --select 1,3 / all / none (prompted if run interactively; required otherwise).
   Everything else, and every star that cannot be modelled, is constant at its Gaia brightness and colour.
   - MESA RSP: RRab, RRc, classical Cepheids (F, 1O), BL Her, W Vir, HADS (radiative models).
   - PHOEBE 2 (+ MESA atmosphere SEDs): eclipsing binaries, detached or contact; Gaia's NSS solution if any, else a
     budget-capped fit to Gaia's light-curve model; brightness scaled to Gaia's G.
   - Not modelled: RRd (no stable double-mode RSP solution; --try-rrd to attempt), multi-mode / anomalous Cepheids,
     RV Tau, low-amplitude delta Scuti, double-mode HADS, eccentric binaries without a Gaia solution, evolved
     binaries, and every other class.
   Distances: Gaia parallax if its S/N >= 3; otherwise a period-luminosity relation (RR Lyrae: M_G-[Fe/H],
   Garofalo et al. 2022; Cepheids: Gaia period-Wesenheit, Ripepi et al. 2019; HADS: Ziaali et al. 2019).
   A chosen star is dropped (with the reason) if its inputs are missing: parallax S/N < 3 (binaries), no FLAME
   mass or GSP-Phot Teff (HADS), no GSP-Phot Teff (binaries).
4. Models, one per star, cached in runs/rsp_<field>_G<source> (re-runs reuse them; a recorded failure in its FAILED
   file is final until that directory is deleted). Per-star parameters, the model checks (pulsators: limit cycle at
   Gaia's period, i.e. the star's mode; binaries: Gaia's eclipses reproduced) and every assumption are written to
   outputs/<field>/field_summary.md. MESA runs take ~0.5-2 h per star; PHOEBE ~1.5-3.5 min per binary.
5. Night: --date YYYY-MM-DD (the night starting that local date; the nearest night the field is up >= 3 h if not),
   or --date auto: the night within half a year on which the modelled stars vary most visibly (eclipses covered,
   pulsators through maximum). Exposures run while it is dark (Sun < -18 deg) and the field is at airmass <= 2;
   --start/--end (UT) restrict that. Weather is not modelled (median seeing, photometric sky).
6. Movies (m3_night_movie.py), one per band: ArgusSim scene and noise; SExtractor photometry on every exposure
   (defaults, neighbour masking, per-exposure zero point with a colour term). outputs/<field>/<field>_night_<band>.mp4
   with its .photometry.csv (true and measured magnitudes per exposure). --models-only skips them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

SITES = {"argus": None,  # ArgusSim's configured site
         "sro": dict(latitude=37.07, longitude=-119.41, altitude=1400.0)}  # Sierra Remote Observatories
RADIUS_DEG = 0.12  # Gaia query radius: covers the 8' x 8' simulated area
Z_SUN = 0.0169
MIN_PLX_SNR = 3.0  # parallax S/N for a parallax distance; below it, a period-luminosity distance (pulsators) or dropped (binaries)
# period-Wesenheit relations, Milky Way, W = G - 1.90 (BP - RP) (Ripepi et al. 2019, A&A 625, A14, eq. 2 and Table 3)
PW_RELATIONS = {"DCEP F": (-2.701, -3.320), "DCEP 1O": (-2.976, -4.095), "T2CEP": (-1.194, -2.381)}
TRY_RRD = False  # --try-rrd: attempt the double-mode search anyway (mesa_argus.rsp.accept_double_mode)


def field_name(ra, dec):
    return f"f{ra:07.3f}{'p' if dec >= 0 else 'm'}{abs(dec):06.3f}".replace(".", "_")


def col(t, name):
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)


def fetch(ra, dec, data):
    """Gaia DR3 tables for the field (cached in data/)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from astroquery.vizier import Vizier

    data.mkdir(parents=True, exist_ok=True)
    c = SkyCoord(ra * u.deg, dec * u.deg)
    tables = {"gaia_dr3_core.ecsv": ("I/355/gaiadr3", ["Source", "RA_ICRS", "DE_ICRS", "Plx", "e_Plx", "pmRA", "pmDE",
                                                       "Gmag", "BPmag", "RPmag", "BP-RP", "RUWE"]),
              "gaia_classifier.ecsv": ("I/358/vclassre", ["**"]),
              "gaia_rrlyrae.ecsv": ("I/358/vrrlyr", ["**"]),
              "gaia_cepheid.ecsv": ("I/358/vcep", ["**"]),
              "gaia_eclipsing.ecsv": ("I/358/veb", ["**"]),
              # Gaia's physical eclipsing-binary solutions (non-single stars), when they exist
              "gaia_nss_eb.ecsv": ("I/357/tboeb", ["**"]),
              "gaia_nss_ebsb.ecsv": ("I/357/tboes", ["**"]),
              # periods and classes for delta Scuti stars (Gaia DR3 gives none): ASAS-SN (Jayasinghe et al. 2018-2020)
              "asassn_variables.ecsv": ("II/366/catv2021", ["ASASSN-V", "Per", "Amp", "Type", "HJD", "Vmag", "GaiaDR3",
                                                            "RAJ2000", "DEJ2000"])}
    for name, (cat, columns) in tables.items():
        if (data / name).exists():
            continue
        print(f"  Gaia DR3 {cat} …", flush=True)
        r = Vizier(row_limit=-1, columns=columns, timeout=900).query_region(c, radius=RADIUS_DEG * u.deg, catalog=cat)
        t = r[0] if r else Table(names=["Source"], dtype=[np.int64])
        t.write(data / name, overwrite=True)


def reddening(ra, dec):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.irsa.irsa_dust import IrsaDust

    t = IrsaDust.get_query_table(SkyCoord(ra * u.deg, dec * u.deg), section="ebv")
    return float(t["ext SandF mean"][0])


def inventory(data, ra, dec, block_n=3):
    """Gaia-classified variables in the block_n x block_n block, as rows of dicts (modellable first)."""
    from astropy.table import Table
    from argus_sim import skymap_shim as sh

    import m3_night_movie as movie

    movie.set_block(block_n)
    tid, loc = sh.radec_to_composite_id(np.array([ra]), np.array([dec]))
    tile, block = int(tid[0]), (int(loc[0]) % 64, int(loc[0]) // 64)
    movie.CLUSTERS["_probe"] = dict(name="", centre=(ra, dec), tile=tile, block=block, date="", site=None)
    movie.CLUSTER, movie.DATA = "_probe", data
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    movie.CENTRE, movie.TILE, movie.BLOCK = SkyCoord(ra * u.deg, dec * u.deg), tile, block
    stars = movie.load_scene()
    in_block = np.asarray(stars["in_block"])
    src = np.asarray(stars["Source"]).astype(np.int64)
    gaia = {int(s): i for i, s in enumerate(src)}
    rr = Table.read(data / "gaia_rrlyrae.ecsv")
    rr = {int(r["Source"]): r for r in rr} if len(rr) else {}
    cep_file = data / "gaia_cepheid.ecsv" if (data / "gaia_cepheid.ecsv").exists() else data / "gaia_cepheid_all.ecsv"
    cep = Table.read(cep_file)
    cep = {int(r["Source"]): r for r in cep} if len(cep) else {}
    eb = Table.read(data / "gaia_eclipsing.ecsv")
    eb = {int(r["Source"]): r for r in eb} if len(eb) else {}
    nss = {}
    for name_ in ("gaia_nss_eb.ecsv", "gaia_nss_ebsb.ecsv"):  # EclipsingSpectro (with mass ratio) last: preferred
        if (data / name_).exists():
            t_ = Table.read(data / name_)
            for r in (t_ if len(t_) and "Source" in t_.colnames else []):
                nss[int(r["Source"])] = {c: (float(r[c]) if not np.ma.is_masked(r[c]) and isinstance(r[c], (int, float, np.number))
                                             else np.nan) for c in r.colnames}
    asassn = {}
    if (data / "asassn_variables.ecsv").exists():
        a_t = Table.read(data / "asassn_variables.ecsv")
        if len(a_t) and "GaiaDR3" in a_t.colnames:
            asassn = {int(r["GaiaDR3"]): r for r in a_t if np.isfinite(float(r["GaiaDR3"] or np.nan)) and int(r["GaiaDR3"]) > 0}
    cls = Table.read(data / "gaia_classifier.ecsv")
    gaia_class = {int(r["Source"]): str(r["Class"]) for r in cls}
    rows = []
    for s in sorted(set(gaia_class) | set(asassn) | set(rr) | set(cep) | set(eb)):
        if s not in gaia or not in_block[gaia[s]]:
            continue
        k = gaia[s]
        klass = gaia_class.get(s, f"ASAS-SN {str(asassn[s]['Type']).strip()}" if s in asassn else "")
        row = dict(source=s, gaia_class=klass, G=float(col(stars, "Gmag")[k]), period=np.nan, amp_G=np.nan,
                   kind=klass, model=None, why="no physical model: held constant")
        if s in rr:
            v = rr[s]
            pf, p1 = float(v["PF"]), float(v["P1O"])
            pf, p1 = (pf if np.isfinite(pf) and pf > 0 else np.nan), (p1 if np.isfinite(p1) and p1 > 0 else np.nan)
            row.update(amp_G=float(v["ptpG"]), G_mean=float(v["Gmagavg"]), epoch_bjd=float(v["EpochG"]) + 2455197.5,
                       mh=float(v["[M/H]"]) if "[M/H]" in v.colnames else np.nan)
            if np.isfinite(pf) and np.isfinite(p1):
                row.update(kind="RRd", period=pf, period2=p1, model="MESA RSP (experimental)" if TRY_RRD else None,
                           why="" if TRY_RRD else "double-mode RR Lyrae: RSP gives no stable double-mode solution (tested: "
                               "convection sets A and C decay to F only; B and D cannot be built) - held constant")
            else:
                row.update(kind="RRab" if np.isfinite(pf) else "RRc", period=pf if np.isfinite(pf) else p1)
                if not np.isfinite(row["epoch_bjd"]):
                    row["why"] = "no Gaia epoch of maximum: cannot phase a model"
                else:
                    row.update(model="MESA RSP", why="")
        elif s in cep:
            v = cep[s]
            pf, p1 = float(v["PF"]), float(v["P1O"])
            pf, p1 = (pf if np.isfinite(pf) and pf > 0 else np.nan), (p1 if np.isfinite(p1) and p1 > 0 else np.nan)
            klass, sub, mode = str(v["Class"]), str(v["SubClass"]), str(v["ModeClass"])
            row.update(amp_G=float(v["ptpG"]), G_mean=float(v["Gmagavg"]), epoch_bjd=float(v["EpochG"]) + 2455197.5,
                       mh=float(v["[M/H]"]), period=pf if np.isfinite(pf) else p1,
                       BP_mean=float(v["BPmagavg"]), RP_mean=float(v["RPmagavg"]))
            if klass == "DCEP" and mode in ("FUNDAMENTAL", "FIRST_OVERTONE"):
                row.update(kind="DCEP F" if mode == "FUNDAMENTAL" else "DCEP 1O",
                           period=pf if mode == "FUNDAMENTAL" else p1, model="MESA RSP", why="")
            elif klass == "T2CEP" and sub in ("BL_HER", "W_VIR"):
                row.update(kind="BL Her" if sub == "BL_HER" else "W Vir",
                           model="MESA RSP", why="")
            else:
                row.update(kind=f"{klass} {sub or mode}".strip(),
                           why={"DCEP": "multi-mode Cepheid: not modelled (held constant)",
                                "ACEP": "anomalous Cepheid: mass not calibrated, not modelled (held constant)"}.get(
                               klass, "RV Tau: not modelled (held constant)"))
            if row["model"] and not np.isfinite(row["epoch_bjd"]):
                row.update(model=None, why="no Gaia epoch of maximum: cannot phase a model")
        elif s in asassn and str(asassn[s]["Type"]).strip() in ("HADS", "DSCT", "HADS(B)"):
            v = asassn[s]
            t_ = str(v["Type"]).strip()
            hjd = float(v["HJD"])
            row.update(period=float(v["Per"]), amp_G=float(v["Amp"]), G_mean=row["G"], mh=np.nan,
                       epoch_bjd=hjd + (2400000.0 if hjd < 2400000 else 0.0))  # HJD ~ BJD (differ < 4 s)
            if t_ == "HADS":
                row.update(kind="HADS", model="MESA RSP", why="")
            else:
                row.update(kind="DSCT" if t_ == "DSCT" else "HADS double-mode",
                           why="low-amplitude delta Scuti (mostly non-radial): RSP is radial only (held constant)"
                           if t_ == "DSCT" else "double-mode HADS: not modelled (held constant)")
        elif s in eb:
            v = eb[s]
            f = float(v["Freq"])
            vd = {c: (float(v[c]) if not np.ma.is_masked(v[c]) and isinstance(v[c], (int, float, np.number)) else
                      (np.nan if np.ma.is_masked(v[c]) else v[c])) for c in v.colnames}
            row.update(kind="EB", period=1 / f if f > 0 else np.nan, veb=vd, amp_G=vd.get("DepthE1", np.nan),
                       G_mean=vd.get("magModRef", row["G"]), epoch_bjd=np.nan, nss=nss.get(s))
            ecc_like = (np.isfinite(vd.get("PhaseE2", np.nan))
                        and abs(((vd["PhaseE2"] - vd["PhaseE1"]) % 1.0) - 0.5) > 0.01)
            if not np.isfinite(vd.get("PhaseE1", np.nan)) or not np.isfinite(vd.get("DepthE1", np.nan)):
                row["why"] = "eclipsing binary without Gaia eclipse parameters (held constant)"
            elif ecc_like and s not in nss:
                row["why"] = "eccentric eclipsing binary without a Gaia NSS solution (held constant)"
            else:
                row.update(model="PHOEBE (Gaia NSS)" if s in nss else "PHOEBE (fit to Gaia light curve)", why="")
        rows.append(row)
    rows.sort(key=lambda r: (r["model"] is None, -np.nan_to_num(r["amp_G"], nan=-1), r["G"]))
    return rows, stars, tile, block


def show(rows, block_n=3):
    print(f"\n{len(rows)} catalogued variable(s) in the {block_n} x {block_n} block "
          f"({sum(r['model'] is not None for r in rows)} can be modelled):")
    print(f"  {'#':>2}  {'Gaia DR3':>19}  {'type':<17} {'G':>6} {'ampl G':>7} {'period d':>9}  model")
    for i, r in enumerate(rows, 1):
        amp = f"{r['amp_G']:.2f}" if np.isfinite(r["amp_G"]) else "-"
        per = f"{r['period']:.5f}" if np.isfinite(r["period"]) else "-"
        print(f"  {i:>2}  {r['source']:>19}  {r['kind']:<17} {r['G']:6.2f} {amp:>7} {per:>9}  {r['model'] or r['why']}")


def choose(rows, select):
    ok = [i for i, r in enumerate(rows, 1) if r["model"]]
    if select is None:
        if not sys.stdin.isatty():
            raise SystemExit("\nchoose with --select (e.g. --select 1,2 or --select all)")
        select = input(f"\nSimulate which? numbers from {ok} separated by commas, 'all', or blank for none: ").strip()
    if select in ("", "none"):
        return []
    picks = ok if select == "all" else [int(x) for x in select.split(",")]
    bad = [i for i in picks if i not in ok]
    if bad:
        raise SystemExit(f"cannot model {bad}: " + "; ".join(rows[i - 1]["why"] for i in bad if 0 < i <= len(rows)))
    return [rows[i - 1] for i in picks]


def physical(row, stars, ra, dec):
    """The star's own distance, reddening, composition and mass, with where each came from (None if it cannot be
    modelled, with the reason)."""
    k = int(np.nonzero(np.asarray(stars["Source"]).astype(np.int64) == row["source"])[0][0])
    plx, e_plx = col(stars, "Plx")[k], col(stars, "e_Plx")[k]
    ebv = reddening(float(col(stars, "RA_ICRS")[k]), float(col(stars, "DE_ICRS")[k]))
    if row["kind"] == "EB":
        if not (np.isfinite(plx) and np.isfinite(e_plx) and plx / e_plx >= MIN_PLX_SNR):
            return None, f"parallax S/N {plx / e_plx:.1f} < {MIN_PLX_SNR:g}: no distance"
        from astroquery.vizier import Vizier
        ap = Vizier(columns=["Source", "Teff", "logg"]).query_constraints(catalog="I/355/paramp", Source=str(row["source"]))
        val = lambda c: float(ap[0][c][0]) if ap and len(ap[0]) and not np.ma.is_masked(ap[0][c][0]) else np.nan  # noqa: E731
        d_pc = 1000.0 / (plx + 0.017)
        return dict(dm=round(float(5 * np.log10(d_pc) - 5), 3), ebv=round(ebv, 4), teff_gsp=val("Teff"), logg_gsp=val("logg"),
                    z=Z_SUN, mass=np.nan, kap="gs98",
                    notes=dict(distance=f"parallax {plx:.3f} +/- {e_plx:.3f} mas (+0.017 zero point) -> {d_pc / 1000:.2f} kpc",
                               ebv="Schlafly & Finkbeiner (2011), IRSA",
                               teff=f"Gaia DR3 GSP-Phot Teff {val('Teff'):.0f} K, log g {val('logg'):.2f} (system)",
                               z="assumed solar ([M/H] = 0) for the component atmospheres",
                               method=row["model"])), ""
    good_plx = np.isfinite(plx) and np.isfinite(e_plx) and plx / e_plx >= MIN_PLX_SNR
    snr = plx / e_plx if np.isfinite(plx) and np.isfinite(e_plx) and e_plx > 0 else np.nan
    cepheid = row["kind"].startswith("DCEP") or row["kind"] in ("BL Her", "W Vir")
    hads = row["kind"] == "HADS"
    mh, mh_note = row["mh"], f"Gaia photometric [M/H] = {row['mh']:+.2f}"
    if hads:
        mh, mh_note = 0.0, "assumed solar ([M/H] = 0): HADS are Population I"
    if not np.isfinite(mh) and row["kind"] == "RRd":  # Gaia derives no [M/H] for double-mode RR Lyrae
        mh, mh_note = -1.5, "Gaia gives no [M/H] for RRd: assumed -1.5 (typical of Galactic double-mode RR Lyrae)"
    if not np.isfinite(mh) and row["kind"] in ("RRab", "RRc"):
        mh, mh_note = -1.5, "Gaia gives no [M/H]: assumed -1.5 (old population, as for Type II Cepheids)"
    if not np.isfinite(mh):
        if not cepheid:
            return None, "no Gaia [M/H]"
        mh = 0.0 if row["kind"].startswith("DCEP") else -1.5
        mh_note = f"Gaia gives no [M/H]: assumed {mh:+.1f} ({'young thin disk' if mh == 0 else 'old population'})"
    if good_plx:
        d_pc = 1000.0 / (plx + 0.017)
        dm, how = 5 * np.log10(d_pc) - 5, f"parallax {plx:.3f} +/- {e_plx:.3f} mas (+0.017 zero point) -> {d_pc / 1000:.2f} kpc"
    elif cepheid:
        rel = row["kind"] if row["kind"].startswith("DCEP") else "T2CEP"
        a, b = PW_RELATIONS[rel]
        w_obs = row["G_mean"] - 1.90 * (row["BP_mean"] - row["RP_mean"])
        if not np.isfinite(w_obs):
            return None, f"parallax S/N {snr:.1f} < {MIN_PLX_SNR:g} and no Gaia mean BP/RP for a Wesenheit distance"
        w_abs = a + b * np.log10(row["period"])
        dm = w_obs - w_abs
        how = (f"parallax S/N {snr:.1f} < {MIN_PLX_SNR:g}: period-Wesenheit relation ({rel}, Ripepi et al. 2019 Table 3) "
               f"W = {a} {b:+} log P = {w_abs:.2f}; W = G - 1.90 (BP-RP) = {w_obs:.2f} (reddening-free) -> "
               f"{10 ** (dm / 5 + 1) / 1000:.2f} kpc")
    elif hads:
        bp_rp = float(col(stars, "BP-RP")[k])
        if not np.isfinite(bp_rp):
            return None, f"parallax S/N {snr:.1f} < {MIN_PLX_SNR:g} and no Gaia BP-RP for a P-L distance"
        m_v = -2.94 * np.log10(row["period"]) - 1.34
        g_v = -0.02704 + 0.01424 * bp_rp - 0.2156 * bp_rp ** 2 + 0.01426 * bp_rp ** 3
        dm = row["G_mean"] - g_v - m_v - 3.1 * ebv
        how = (f"parallax S/N {snr:.1f} < {MIN_PLX_SNR:g}: delta Scuti P-L relation M_V = -2.94 log P - 1.34 = {m_v:.2f} "
               f"(Ziaali et al. 2019, fundamental mode); V = G - ({g_v:+.3f}) from BP-RP {bp_rp:.2f} (Riello et al. 2021), "
               f"A_V = 3.1 E(B-V) -> {10 ** (dm / 5 + 1) / 1000:.2f} kpc")
    else:
        m_g = 0.32 * mh + 1.11
        dm = row["G_mean"] - m_g - 2.74 * ebv
        how = (f"parallax S/N {snr:.1f} < {MIN_PLX_SNR:g}: M_G = 0.32 [Fe/H] + 1.11 = {m_g:.2f} (Garofalo et al. 2022), "
               f"A_G = 2.74 E(B-V) -> {10 ** (dm / 5 + 1) / 1000:.2f} kpc")
    z = Z_SUN * 10 ** mh
    if row["kind"].startswith("DCEP"):
        x = 0.73
        y = 1.0 - x - z
        log_l = (4.74 - (row["G_mean"] - dm - 2.74 * ebv)) / 2.5
        mass = 10 ** ((log_l - 0.72 - 1.36 * np.log10(y / 0.28) + 0.34 * np.log10(z / 0.02)) / 3.35)
        mass_note = (f"canonical mass-luminosity relation (Bono et al. 2000b; Caputo et al. 2005 eq. 2) at log L = "
                     f"{log_l:.2f} (G, distance, A_G = 2.74 E(B-V), BC_G = 0), Y = {y:.3f}; overshooting would give "
                     f"~0.2 dex brighter at fixed mass")
    elif cepheid:
        x, mass, mass_note = 0.76, 0.55, "assumed: 0.55 Msun as MESA's rsp_Type_II_Cepheid"
    elif hads:
        from astroquery.vizier import Vizier
        r = Vizier(columns=["Source", "Mass-Flame", "Lum-Flame", "Teff"]).query_constraints(catalog="I/355/paramp",
                                                                                             Source=str(row["source"]))
        val = lambda c: float(r[0][c][0]) if r and len(r[0]) and not np.ma.is_masked(r[0][c][0]) else np.nan  # noqa: E731
        m_fl, teff = val("Mass-Flame"), val("Teff")
        x = 0.70
        if np.isfinite(m_fl):
            mass = m_fl
            mass_note = "Gaia DR3 FLAME (isochrone mass from Gaia L and Teff; assumes a non-variable single star); X = 0.70 as MESA's rsp_Delta_Scuti"
        elif np.isfinite(teff):
            from mesa_argus.binary import ms_mass
            mass = float(ms_mass(teff))
            mass_note = (f"no Gaia FLAME mass: main-sequence mass for GSP-Phot Teff {teff:.0f} K (Pecaut & Mamajek 2013, "
                         f"table 2022.04.16); HADS are slightly evolved, so this may underestimate; X = 0.70 as MESA's rsp_Delta_Scuti")
        else:
            return None, "no Gaia DR3 FLAME mass or GSP-Phot Teff"
    else:
        x, mass = 0.75, float(np.interp(mh, [-2.1, -1.5, 0.0], [0.75, 0.68, 0.55]))
        mass_note = "assumed from [M/H] (0.75 at -2.1, 0.68 at -1.5, 0.55 at 0.0)"
    notes_extra = {}
    if hads:
        notes_extra["convection"] = ("RADIATIVE MODEL: convection off (RSP_alfa = 0). With RSP's turbulent convection "
                                     "HADS modes are only neutrally stable; the radiative model pulsates, but its amplitude "
                                     "is not limited by convection (less physical than the RR Lyrae / Cepheid models)")
    return dict(dm=round(float(dm), 3), ebv=round(ebv, 4), z=round(z, 6), x=x, mass=round(float(mass), 3),
                kap="a09" if mh < -1.0 else "gs98",
                notes=dict(distance=how, ebv="Schlafly & Finkbeiner (2011), IRSA",
                           z=f"{mh_note} -> Z = {Z_SUN} x 10^[M/H]; X = {x}", mass=mass_note,
                           opacities="alpha-enhanced (a09)" if mh < -1.0 else "solar-scaled (gs98)", **notes_extra)), ""


def night_times(ra, dec, date, site, start, end, max_airmass=2.0):
    """BJD-like JD (UTC + light time ignored, < 10 min) of the 5-minute steps the field is observable on a night."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
    from astropy.time import Time

    if site is None:
        from argus_sim import c as cfg
        site = dict(latitude=cfg.observatory.latitude, longitude=cfg.observatory.longitude, altitude=cfg.observatory.altitude)
    loc = EarthLocation(lat=site["latitude"] * u.deg, lon=site["longitude"] * u.deg, height=site["altitude"] * u.m)
    t = Time(f"{date} 23:00:00") + np.arange(0, 14 * 60, 5) * u.min
    frame = AltAz(obstime=t, location=loc)
    alt = SkyCoord(ra * u.deg, dec * u.deg).transform_to(frame).alt.deg
    ok = (get_sun(t).transform_to(frame).alt.deg < -18) & (alt > 0) & (1 / np.sin(np.radians(np.clip(alt, 1e-3, 90))) <= max_airmass)
    clock = np.array([(h - 24 if h >= 20 else h) for h in (t.datetime64.astype("datetime64[m]").astype(int) % 1440) / 60])
    if start:
        ok &= clock >= (lambda h: h - 24 if h >= 20 else h)(int(start[:2]) + int(start[3:5]) / 60)
    if end:
        ok &= clock <= (lambda h: h - 24 if h >= 20 else h)(int(end[:2]) + int(end[3:5]) / 60)
    return t[ok].jd


def variability_score(jd, picks):
    """How visibly each chosen variable varies during a night, from its ephemeris, as a fraction 0-1 summed over the
    stars (so every star counts equally); ties broken by amplitude. Eclipsing binaries: 1 if the night covers a Gaia
    eclipse from -1 to +1 Gaussian sigma, 0.75 if only its centre, 0.5 if it comes within 1.5 sigma. Pulsators: half for the fraction of
    the cycle covered, half if maximum light or the rise to it (phase 0.85-1.05) is covered."""
    if len(jd) == 0:
        return 0.0
    score, amp = 0.0, 0.0
    for r in picks:
        if r["kind"] == "EB":
            v, p = r["veb"], r["period"]
            best = 0.0
            for k in (1, 2):
                ph, sig = v.get(f"PhaseGauss{k}", np.nan), v.get(f"sigPhaseGauss{k}", np.nan)
                if not np.isfinite(ph):
                    continue
                t_ecl = v["TimeRef"] + 2455197.5 + ph * p
                n_c = np.ceil((jd[0] - t_ecl) / p)
                centre = t_ecl + n_c * p
                w = (sig if np.isfinite(sig) else 0.02) * p
                if centre - w >= jd[0] and centre + w <= jd[-1]:  # ingress to egress (+-1 sigma) in the night
                    best = 1.0
                elif centre <= jd[-1]:
                    best = max(best, 0.75)
                elif np.isfinite(sig) and np.any(np.abs(((jd - t_ecl) / p + 0.5) % 1.0 - 0.5) <= 1.5 * sig):
                    best = max(best, 0.5)
            score += best
            amp += r["amp_G"] * best
        else:
            if not np.isfinite(r.get("epoch_bjd", np.nan)):
                continue
            ph = ((jd - r["epoch_bjd"]) / r["period"]) % 1.0
            s_ = 0.5 * min(1.0, (jd[-1] - jd[0]) / r["period"]) + (0.5 if np.any((ph >= 0.85) | (ph <= 0.05)) else 0.0)
            score += s_
            amp += r["amp_G"] * s_
    return score + 1e-3 * amp


def best_night(ra, dec, date, site, start, end, picks, min_minutes=180):
    """The night within half a year of `date` (up >= min_minutes) on which the chosen variables change most."""
    from datetime import date as Date, timedelta

    d0 = Date.fromisoformat(date)
    best = (-1.0, None, None)
    for offset in range(-182, 183):
        d = (d0 + timedelta(days=offset)).isoformat()
        jd = night_times(ra, dec, d, site, start, end)
        if 5 * len(jd) < min_minutes:
            continue
        sc = variability_score(jd, picks)
        if sc > best[0] or (sc == best[0] and abs(offset) < abs((Date.fromisoformat(best[1]) - d0).days)):
            best = (sc, d, jd)
    if best[1] is None:
        raise SystemExit("never observable for 3 h from this site (dark, airmass <= 2)")
    return best[1], best[0]


def nearest_night(ra, dec, date, site, start, end, min_minutes=180):
    """The night nearest `date` (up to half a year away) when the field is up for >= min_minutes."""
    from datetime import date as Date, timedelta

    d0 = Date.fromisoformat(date)
    for offset in sorted(range(-182, 183, 7), key=abs):  # weekly steps (visibility changes slowly)
        d = (d0 + timedelta(days=offset)).isoformat()
        minutes, span = observable(ra, dec, d, site, start, end)
        if minutes >= min_minutes:
            return d, minutes, span
    raise SystemExit("never observable for 3 h from this site (dark, airmass <= 2)")


def observable(ra, dec, date, site, start, end, max_airmass=2.0):
    """Minutes the field is dark (Sun < -18) at airmass <= max_airmass on the night of `date` (within the window)."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
    from astropy.time import Time

    if site is None:
        from argus_sim import c as cfg
        site = dict(latitude=cfg.observatory.latitude, longitude=cfg.observatory.longitude, altitude=cfg.observatory.altitude)
    loc = EarthLocation(lat=site["latitude"] * u.deg, lon=site["longitude"] * u.deg, height=site["altitude"] * u.m)
    t = Time(f"{date} 23:00:00") + np.arange(0, 14 * 60, 5) * u.min
    frame = AltAz(obstime=t, location=loc)
    alt = SkyCoord(ra * u.deg, dec * u.deg).transform_to(frame).alt.deg
    ok = (get_sun(t).transform_to(frame).alt.deg < -18) & (alt > 0) & (1 / np.sin(np.radians(np.clip(alt, 1e-3, 90))) <= max_airmass)
    clock = np.array([(h - 24 if h >= 20 else h) for h in (t.datetime64.astype("datetime64[m]").astype(int) % 1440) / 60])
    if start:
        ok &= clock >= (lambda h: h - 24 if h >= 20 else h)(int(start[:2]) + int(start[3:5]) / 60)
    if end:
        ok &= clock <= (lambda h: h - 24 if h >= 20 else h)(int(end[:2]) + int(end[3:5]) / 60)
    return 5 * int(ok.sum()), (t[ok][0].iso[11:16], t[ok][-1].iso[11:16]) if ok.any() else (None, None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, required=True, help="deg (ICRS)")
    p.add_argument("--dec", type=float, required=True, help="deg (ICRS)")
    p.add_argument("--date", default=None, help="local date the night starts (YYYY-MM-DD); default today. If the field "
                                               "is not up that night, the nearest night it is up is used. 'auto': "
                                               "the night (within half a year) on which the chosen variables change most")
    p.add_argument("--start", default=None, help="UT HH:MM (default: dusk / when the field rises)")
    p.add_argument("--end", default=None, help="UT HH:MM (default: dawn / when the field sets)")
    p.add_argument("--site", choices=list(SITES), default="argus", help="argus = ArgusSim's site; sro = Sierra Remote Obs.")
    p.add_argument("--bands", nargs="+", default=["beta", "rho"], choices=["beta", "rho"])
    p.add_argument("--select", default=None, help="variables to model: numbers from the list, 'all', or 'none'")
    p.add_argument("--name", default=None, help="field name (default from the coordinates)")
    p.add_argument("--list", action="store_true", help="only list the variables in the block")
    p.add_argument("--models-only", action="store_true", help="make the MESA models, do not render")
    p.add_argument("--try-rrd", action="store_true", help="attempt double-mode RR Lyrae (no stable solution found so far)")
    p.add_argument("--block", type=int, default=3, help="minipixes across the block (odd, up to 9; 9 = 253\")")
    args = p.parse_args(argv)
    global TRY_RRD
    TRY_RRD = args.try_rrd

    import os

    os.chdir(Path(__file__).resolve().parent)  # data/, runs/, outputs/ live in the repo, wherever this is run from
    name = args.name or field_name(args.ra, args.dec)
    data, out = Path("data") / name, Path("outputs") / name
    site = SITES[args.site]
    from datetime import date as Date

    auto_date = args.date == "auto"
    asked = Date.today().isoformat() if (auto_date or not args.date) else args.date
    args.date, minutes, (t0, t1) = nearest_night(args.ra, args.dec, asked, site, args.start, args.end)
    if args.date != asked:
        print(f"not up on the night of {asked}: using the nearest night it is, {args.date}")
    print(f"{name}: RA {args.ra:.5f} Dec {args.dec:+.5f}; observable {minutes / 60:.1f} h on the night of {args.date} "
          f"(UT {t0}-{t1}, airmass <= 2) from {args.site}")
    fetch(args.ra, args.dec, data)
    rows, stars, tile, block = inventory(data, args.ra, args.dec, args.block)
    print(f"{len(stars)} Gaia sources in the simulated area, {int(np.sum(stars['in_block']))} in the "
          f"{args.block} x {args.block} block (tile {tile}, centre minipix {block})")
    show(rows, args.block)
    if args.list:
        return
    picks = choose(rows, args.select)
    if not picks:
        print("nothing to model: every star is constant; rendering anyway")

    # the chosen stars' physical parameters, then MESA RSP
    followed = {}
    for i, r in enumerate(picks, 1):
        ph, why = physical(r, stars, args.ra, args.dec)
        if ph is None:
            print(f"  Gaia DR3 {r['source']} ({r['kind']}): cannot be modelled: {why}; held constant")
            r["why"] = why
            continue
        key = f"G{r['source']}"  # models are cached per star (runs/rsp_<field>_G<source>), whatever the selection order
        followed[key] = dict(source=r["source"], kind=r["kind"], period=r["period"], epoch_bjd=r["epoch_bjd"], G=r["G"],
                                 **({"period2": r["period2"]} if "period2" in r else {}),
                                 G_mean=r["G_mean"], amp_G=r["amp_G"], **{k: v for k, v in ph.items() if k != "notes"},
                                 notes=ph["notes"])
        if r["kind"] == "EB":
            followed[key].update(veb=r["veb"], nss=r.get("nss"))
            print(f"  {key} (Gaia DR3 {r['source']}, eclipsing binary, P {r['period']:.5f} d): {r['model']}; distance modulus "
                  f"{ph['dm']:.2f}, E(B-V) {ph['ebv']:.3f}, GSP-Phot Teff {ph['teff_gsp']:.0f} K")
        else:
            print(f"  {key} (Gaia DR3 {r['source']}, {r['kind']}, P {r['period']:.5f} d): distance modulus {ph['dm']:.2f} "
                  f"[{ph['notes']['distance']}], E(B-V) {ph['ebv']:.3f}, Z {ph['z']:.5f}, mass {ph['mass']:.2f} Msun, {ph['kap']}")
    (data / "followed.json").write_text(json.dumps(followed, indent=1, default=float))
    if auto_date and followed:  # the night, for the stars that will actually be modelled
        failed_before = {v["source"] for k_, v in followed.items() if (Path("runs") / f"rsp_{name}_{k_}" / "FAILED").exists()}
        modelled = [r for r in picks if r["source"] in {v["source"] for v in followed.values()} - failed_before]
        args.date, score = best_night(args.ra, args.dec, asked, site, args.start, args.end, modelled)
        minutes, (t0, t1) = observable(args.ra, args.dec, args.date, site, args.start, args.end)
        print(f"night chosen for the variables (--date auto): {args.date}, UT {t0}-{t1} ({minutes / 60:.1f} h); "
              f"predicted visibility score {score:.2f} of {len(modelled)}", flush=True)
    # ephemerides that do not come from Gaia's SOS tables, for the movie
    extra = {str(v["source"]): dict(period=v["period"], epoch_bjd=v["epoch_bjd"], kind=v["kind"], amp=v["amp_G"], G=v["G"])
             for v in followed.values() if v["kind"] == "HADS"}
    (data / "extra_pulsators.json").write_text(json.dumps(extra, indent=1))
    first = next((v for v in followed.values() if v["kind"] != "EB"), dict(dm=0.0, ebv=0.0, z=Z_SUN, mass=0.55, kap="gs98"))
    (data / "field.json").write_text(json.dumps(dict(
        name=args.name or "field", centre=[args.ra, args.dec], tile=tile, block=list(block), block_n=args.block,
        date=args.date, site=site,
        rsp=dict(dm=first["dm"], ebv=first["ebv"], z=first["z"], mass_rr=first["mass"], prefix=f"{name}_", kap=first["kap"])),
        indent=1))
    out.mkdir(parents=True, exist_ok=True)
    failed = {}
    binaries = [k for k, v in followed.items() if v["kind"] == "EB"]
    pulsators_ = [k for k in followed if k not in binaries]
    for label in binaries:  # PHOEBE, one at a time (each capped: mesa_argus.binary.FIT_MAX_EVALS / FIT_MAX_SECONDS)
        from mesa_argus import binary as B
        v = followed[label]
        cached = Path("runs") / f"rsp_{name}_{label}"
        if (cached / "pulsation.h5").exists() and (cached / "binary.json").exists():
            ok, info = True, json.loads((cached / "binary.json").read_text())
            print(f"PHOEBE: {label} (Gaia DR3 {v['source']}): cached model", flush=True)
        else:
            print(f"PHOEBE: {label} (Gaia DR3 {v['source']}) …", flush=True)
            ok, info = B.model_binary(dict(source=v["source"], veb=v["veb"], nss=v.get("nss"), teff_gsp=v["teff_gsp"],
                                           logg_gsp=v["logg_gsp"], G=v["veb"]["magModRef"]),
                                      Path("runs") / f"rsp_{name}_{label}", v["dm"], v["ebv"])
        v["binary"] = info
        print(f"  {label}: " + ("modelled: " + info["route"] + "; " + info["light_curve_check"] + "; " + info["distance_check"]
                                if ok else "NOT modelled: " + info["reason"]), flush=True)
        if ok:  # the movie phases a binary from its primary eclipse (Gaia's; the model's phase 0)
            veb_ = v["veb"]
            extra[str(v["source"])] = dict(period=v["period"], kind="EB", amp=v["amp_G"], G=v["G"],
                                           epoch_bjd=veb_["TimeRef"] + 2455197.5 + veb_["PhaseE1"] * v["period"])
    (data / "extra_pulsators.json").write_text(json.dumps(extra, indent=1))
    if pulsators_:
        todo_ = [k for k in pulsators_ if not (Path("runs") / f"rsp_{name}_{k}" / "pulsation.h5").exists()
                 and not (Path("runs") / f"rsp_{name}_{k}" / "FAILED").exists()]
        print(f"MESA RSP: {len(pulsators_) - len(todo_)} cached, {len(todo_)} to compute"
              + (f" (typically 0.5-2 h each; up to config.toml [run] workers at once); log: {out / 'rsp.log'}" if todo_ else ""),
              flush=True)
        with open(out / "rsp.log", "w") as log:
            subprocess.run([sys.executable, "-m", "mesa_argus.rsp", "--cluster", name, *pulsators_], stdout=log,
                           stderr=subprocess.STDOUT, check=False)
    if followed:
        for label, v in list(followed.items()):
            if not (Path("runs") / f"rsp_{name}_{label}" / "pulsation.h5").exists():
                failed[label] = v
                followed.pop(label)
        for label, v in failed.items():
            print(f"  {label}: {'PHOEBE' if v['kind'] == 'EB' else 'MESA RSP'} FAILED; held constant in the movie")
        (data / "followed.json").write_text(json.dumps(followed, indent=1, default=float))

    movies = []
    for band in ([] if args.models_only else args.bands):
        mp4 = out / f"{name}_night_{band}.mp4"
        cmd = [sys.executable, "m3_night_movie.py", "--cluster", name, "--band", band, "--highlight", str(len(followed)),
               "-o", str(mp4)] + (["--start", args.start] if args.start else []) + (["--end", args.end] if args.end else [])
        print(f"rendering {band} -> {mp4}", flush=True)
        with open(mp4.with_suffix(".log"), "w") as log:
            ok = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode == 0
        print(f"  {'saved' if ok else 'FAILED, see ' + str(mp4.with_suffix('.log'))}", flush=True)
        movies.append((band, mp4, ok))

    lines = [f"# {name}", "", f"RA {args.ra:.5f}, Dec {args.dec:+.5f}; night of {args.date}; site {args.site}; "
             f"UT window {args.start or 'dusk'}-{args.end or 'dawn'}", "",
             f"{len(stars)} Gaia DR3 sources in the 8' x 8' simulated area; {int(np.sum(stars['in_block']))} in the 3 x 3 block.",
             "", "## Gaia-classified variables in the block", "",
             "| # | Gaia DR3 | type | G | ampl. G | period (d) | treatment |", "|---|---|---|---|---|---|---|"]
    chosen = {v["source"]: k for k, v in followed.items()}
    for i, r in enumerate(rows, 1):
        method = "PHOEBE" if r["kind"] == "EB" else "MESA RSP"
        treat = (f"{method} ({chosen[r['source']]})" if r["source"] in chosen else
                 f"{method} failed: constant" if r["source"] in {v["source"] for v in failed.values()} else
                 "not chosen: constant" if r["model"] else r["why"])
        amp = f"{r['amp_G']:.2f}" if np.isfinite(r["amp_G"]) else "-"
        per = f"{r['period']:.5f}" if np.isfinite(r["period"]) else "-"
        lines.append(f"| {i} | {r['source']} | {r['kind']} | {r['G']:.2f} | {amp} | {per} | {treat} |")
    lines += ["", "## Modelled stars: parameters and where they come from", ""]
    for label, v in followed.items():
        lines += [f"**{label}** (Gaia DR3 {v['source']}, {v['kind']}, P = {v['period']:.5f} d)", ""]
        lines += [f"- {key}: {val}" for key, val in v["notes"].items()]
        if v["kind"] == "EB":
            info = v.get("binary", {})
            lines += [f"- {k_.replace('_', ' ')}: {info[k_]}" for k_ in ("route", "light_curve_check", "distance_check", "masses")
                      if k_ in info]
            if "params" in info:
                lines.append(f"- PHOEBE model: {info['params']}; M1 {info.get('M1')} / M2 {info.get('M2')} Msun, "
                             f"R1 {info.get('R1')} / R2 {info.get('R2')} Rsun")
            lines.append("")
            continue
        lines += [f"- values: distance modulus {v['dm']}, E(B-V) {v['ebv']}, Z {v['z']}, mass {v['mass']} Msun, {v['kap']}"]
        pf = Path("runs") / f"rsp_{name}_{label}" / "params.json"
        if pf.exists():  # what MESA RSP settled on
            q = json.loads(pf.read_text())
            lines += [f"- MESA RSP: L = {q['L']:.2f} Lsun (x {q.get('L_factor', 1.0)} the L from the distance), Teff = "
                      f"{q['teff']:.0f} K; linear periods {[round(x, 5) for x in q.get('linear_periods', [])[:2]]} d; "
                      f"non-linear period {q.get('nonlinear_period', float('nan')):.5f} d (Gaia {v['period']:.5f} d)"
                      + (f"; Teff corrected {q['teff_corrections']}x for the non-linear period" if q.get("teff_corrections") else "")]
            lines += [f"- convection: {', '.join(q['extra_controls']) or 'RSP defaults (set A)'}" if q.get("extra_controls")
                      else "- convection: RSP defaults (set A)"]
            for key in ("mode_check", "limit_cycle_note"):
                if q.get(key):
                    lines.append(f"- {key.replace('_', ' ')}: {q[key]}")
        lines.append("")
    lines += ["## Outputs", ""] + [f"- {band}: `{mp4}`" + ("" if ok else " (FAILED)") for band, mp4, ok in movies]
    lines += ["", "Every other star is constant (Gaia G and BP-RP -> colour-matched MESA SED). Argus instrument, PSF, sky, "
              "noise and photometry are as in outputs/m3/ASSUMPTIONS.md."]
    (out / "field_summary.md").write_text("\n".join(lines) + "\n")
    print(f"summary: {out / 'field_summary.md'}")


if __name__ == "__main__":
    main()
