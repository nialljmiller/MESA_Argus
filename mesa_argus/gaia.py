"""Gaia DR3 (and ASAS-SN) for a pointing: the catalogue queries, the variables in a block and how each can be modelled,
and a variable's physical parameters (distance, reddening along its sightline, composition, mass) for MESA RSP or
PHOEBE. Used by make_stars.py (Gaia variables, Gaia field sets).
"""
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from astroquery.vizier import Vizier
import numpy as np

RADIUS_DEG = 0.12  # Gaia query radius: covers the 8' x 8' simulated area
Z_SUN = 0.0169
MIN_PLX_SNR = 3.0  # parallax S/N for a parallax distance; below it, a period-luminosity distance (pulsators) or dropped (binaries)
# period-Wesenheit relations, Milky Way, W = G - 1.90 (BP - RP) (Ripepi et al. 2019, A&A 625, A14, eq. 2 and Table 3)
PW_RELATIONS = {"DCEP F": (-2.701, -3.320), "DCEP 1O": (-2.976, -4.095), "T2CEP": (-1.194, -2.381)}
TRY_RRD = False  # attempt the double-mode search anyway (mesa_argus.rsp.accept_double_mode)


def field_name(ra, dec):
    return f"f{ra:07.3f}{'p' if dec >= 0 else 'm'}{abs(dec):06.3f}".replace(".", "_")


def col(t, name):
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)


def fetch(ra, dec, data):
    """Gaia DR3 tables for the field (cached in data/)."""
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
    """SFD E(B-V) (Schlafly & Finkbeiner 2011 recalibration) at ra, dec, from IRSA."""
    t = IrsaDust.get_query_table(SkyCoord(ra * u.deg, dec * u.deg), section="ebv")
    return float(t["ext SandF mean"][0])


def inventory(data, ra, dec, block_n=3):
    """Gaia-classified variables in the block_n x block_n block, as rows of dicts (modellable first)."""
    from argus_sim import skymap_shim as sh

    import night_movie as movie

    # the engine's star list for this pointing (which stars are in the block), pointed at the cached tables
    tile_id, minipix = sh.radec_to_composite_id(np.array([ra]), np.array([dec]))
    tile, block = int(tile_id[0]), (int(minipix[0]) % 64, int(minipix[0]) // 64)
    movie.set_block(block_n)
    movie.FIELDS["_probe"] = dict(name="", centre=(ra, dec), tile=tile, block=block, date="", site=None)
    movie.FIELD, movie.DATA, movie.TILE, movie.BLOCK = "_probe", data, tile, block
    movie.CENTRE = SkyCoord(ra * u.deg, dec * u.deg)
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


def physical(row, stars, ra, dec):
    """The star's own distance, reddening, composition and mass, with where each came from (None if it cannot be
    modelled, with the reason)."""
    k = int(np.nonzero(np.asarray(stars["Source"]).astype(np.int64) == row["source"])[0][0])
    plx, e_plx = col(stars, "Plx")[k], col(stars, "e_Plx")[k]
    ebv = reddening(float(col(stars, "RA_ICRS")[k]), float(col(stars, "DE_ICRS")[k]))
    if row["kind"] == "EB":
        if not (np.isfinite(plx) and np.isfinite(e_plx) and plx / e_plx >= MIN_PLX_SNR):
            return None, f"parallax S/N {plx / e_plx:.1f} < {MIN_PLX_SNR:g}: no distance"
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
