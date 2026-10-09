"""Find the 9 x 9 minipix window (~4.2') holding the most distinct types of variable that argus_field.py can model.

Anchors are the rare types (Gaia DR3 classical / BL Her / W Vir Cepheids; ASAS-SN HADS). Around each anchor, all
modellable variables within the window are counted from Gaia DR3 (RR Lyrae RRab/RRc, Cepheids, eclipsing binaries with
eclipse parameters) and ASAS-SN (HADS). Ranked by the number of distinct types, then the number of modellable stars.
Observability, parallaxes and crowding are checked afterwards on the winners (argus_field.py --list).

    python grids/find_rich_field.py
"""
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astroquery.vizier import Vizier

warnings.simplefilter("ignore")
HALF_DEG = 4.5 * 32 * (900 / 1024) / 3600  # 9 x 9 minipixes: half-width 126.6"
RADIUS = HALF_DEG * np.sqrt(2) * u.deg      # the query circle around each anchor covers the square window
G_MAX = 16.5


def col(t, c):
    x = t[c]
    return np.asarray(x.filled(np.nan) if hasattr(x, "filled") else x, float)


def anchors():
    cep = Vizier(row_limit=-1, columns=["Source", "Class", "SubClass", "ModeClass", "Gmagavg", "RA_ICRS", "DE_ICRS"],
                 timeout=900).query_constraints(catalog="I/358/vcep", Gmagavg=f"<{G_MAX}")[0]
    ok = (((cep["Class"] == "DCEP") & np.isin(cep["ModeClass"], ["FUNDAMENTAL", "FIRST_OVERTONE"]))
          | ((cep["Class"] == "T2CEP") & np.isin(cep["SubClass"], ["BL_HER", "W_VIR"])))
    cep = cep[ok]
    hads = Vizier(row_limit=-1, columns=["Per", "Amp", "Type", "Vmag", "GaiaDR3", "RAJ2000", "DEJ2000"],
                  timeout=900).query_constraints(catalog="II/366/catv2021", Type="HADS", Vmag=f"<{G_MAX}")[0]
    ra = np.r_[col(cep, "RA_ICRS"), col(hads, "RAJ2000")]
    de = np.r_[col(cep, "DE_ICRS"), col(hads, "DEJ2000")]
    print(f"anchors: {len(cep)} Cepheids (classical F/1O, BL Her, W Vir) and {len(hads)} HADS with G/V < {G_MAX}", flush=True)
    return SkyCoord(ra * u.deg, de * u.deg)


def count(centres):
    """Modellable variables of each type within the window around each centre."""
    kinds = [dict() for _ in range(len(centres))]

    def add(i, kind, src):
        kinds[i].setdefault(kind, set()).add(int(src))

    def inside(t, i, ra_c, de_c):
        c = centres[i]
        dx = (col(t, ra_c) - c.ra.deg) * np.cos(c.dec.rad)
        dy = col(t, de_c) - c.dec.deg
        return (np.abs(dx) <= HALF_DEG) & (np.abs(dy) <= HALF_DEG)

    queries = {"I/358/veb": ["Source", "PhaseE1", "PhaseE2", "DepthE1", "magModRef", "_q", "RA_ICRS", "DE_ICRS"]}
    for i0 in range(0, len(centres), 100):
        chunk = centres[i0:i0 + 100]
        for cat, cols in queries.items():
            r = Vizier(row_limit=-1, columns=cols, timeout=900).query_region(chunk, radius=RADIUS, catalog=cat)
            if not r:
                continue
            t = r[0]
            q = np.asarray(t["_q"]) - 1 + i0
            for i in np.unique(q):
                s = t[q == i]
                if cat == "I/358/vrrlyr":
                    s = s[inside(s, i, "_RAJ2000", "_DEJ2000") & (col(s, "Gmagavg") < G_MAX)]
                    pf, p1 = col(s, "PF"), col(s, "P1O")
                    for r_, a, b in zip(s, np.isfinite(pf) & (pf > 0), np.isfinite(p1) & (p1 > 0)):
                        if a and not b:
                            add(i, "RRab", r_["Source"])
                        elif b and not a:
                            add(i, "RRc", r_["Source"])
                elif cat == "I/358/vcep":
                    s = s[inside(s, i, "RA_ICRS", "DE_ICRS") & (col(s, "Gmagavg") < G_MAX)]
                    for r_ in s:
                        if r_["Class"] == "DCEP" and r_["ModeClass"] in ("FUNDAMENTAL", "FIRST_OVERTONE"):
                            add(i, "DCEP", r_["Source"])
                        elif r_["Class"] == "T2CEP" and r_["SubClass"] in ("BL_HER", "W_VIR"):
                            add(i, "BL Her" if r_["SubClass"] == "BL_HER" else "W Vir", r_["Source"])
                elif cat == "I/358/veb":
                    s = s[inside(s, i, "RA_ICRS", "DE_ICRS") & (col(s, "magModRef") < G_MAX) & np.isfinite(col(s, "DepthE1"))]
                    circ = ~(np.abs(((col(s, "PhaseE2") - col(s, "PhaseE1")) % 1) - 0.5) > 0.01)
                    for r_ in s[circ]:
                        add(i, "EB", r_["Source"])
                else:
                    s = s[inside(s, i, "RAJ2000", "DEJ2000") & (col(s, "Vmag") < G_MAX)]
                    for r_ in s[np.asarray(s["Type"]) == "HADS"]:
                        add(i, "HADS", r_["GaiaDR3"] if r_["GaiaDR3"] else -1)
        print(f"  {min(i0 + 100, len(centres))}/{len(centres)} anchors: eclipsing binaries counted", flush=True)
    return kinds


def bulk(cat, cols, **constraints):
    return Vizier(row_limit=-1, columns=cols, timeout=1800).query_constraints(catalog=cat, **constraints)[0]


def local_counts(centres, tables):
    """Counts per type within the window, from bulk tables: {type: (SkyCoord, ids)}."""
    from astropy.coordinates import search_around_sky

    kinds = [dict() for _ in range(len(centres))]
    for kind, (pos, ids) in tables.items():
        ia, ib, _, _ = search_around_sky(centres, pos, RADIUS)
        for a, b_ in zip(ia, ib):
            c = centres[a]
            dx = (pos[b_].ra.deg - c.ra.deg) * np.cos(c.dec.rad)
            dy = pos[b_].dec.deg - c.dec.deg
            if abs(dx) <= HALF_DEG and abs(dy) <= HALF_DEG:
                kinds[a].setdefault(kind, set()).add(int(ids[b_]))
    return kinds


def parallax_snr(sources):
    """Gaia DR3 parallax / error for source ids (batched)."""
    out = {}
    ids = sorted({int(x) for x in sources if int(x) > 0})
    for i0 in range(0, len(ids), 100):  # longer lists fail silently
        t = Vizier(row_limit=-1, columns=["Source", "Plx", "e_Plx"], timeout=900).query_constraints(
            catalog="I/355/gaiadr3", Source="=" + ",".join(map(str, ids[i0:i0 + 100])))
        for r in (t[0] if t else []):
            p_, e_ = float(r["Plx"] or np.nan), float(r["e_Plx"] or np.nan)
            out[int(r["Source"])] = p_ / e_ if e_ > 0 else 0.0
    print(f"  parallaxes found for {len(out)} of {len(ids)} sources", flush=True)
    return out


if __name__ == "__main__":
    # only what argus_field.py can model: RR Lyrae with a Gaia [M/H]; Cepheids, HADS and eclipsing binaries with
    # parallax S/N >= 5 (eclipsing binaries: circular or with eclipse parameters; GSP-Phot Teff checked afterwards)
    print("bulk downloads …", flush=True)
    rr = bulk("I/358/vrrlyr", ["Source", "PF", "P1O", "Gmagavg", "[M/H]", "_RAJ2000", "_DEJ2000"], Gmagavg=f"<{G_MAX}")
    cep = bulk("I/358/vcep", ["Source", "Class", "SubClass", "ModeClass", "Gmagavg", "RA_ICRS", "DE_ICRS"], Gmagavg=f"<{G_MAX}")
    hads = bulk("II/366/catv2021", ["Type", "Vmag", "GaiaDR3", "RAJ2000", "DEJ2000"], Type="HADS", Vmag=f"<{G_MAX}")
    print(f"  {len(rr)} RR Lyrae, {len(cep)} Cepheids, {len(hads)} HADS (G/V < {G_MAX}); parallaxes …", flush=True)
    snr = parallax_snr(list(cep["Source"]) + [x for x in hads["GaiaDR3"] if x])
    pf, p1, mh = col(rr, "PF"), col(rr, "P1O"), col(rr, "[M/H]")
    pos = lambda t, a, b: SkyCoord(col(t, a) * u.deg, col(t, b) * u.deg)  # noqa: E731
    tables = {}
    for kind, sel in (("RRab", (pf > 0) & ~(p1 > 0) & np.isfinite(mh)), ("RRc", (p1 > 0) & ~(pf > 0) & np.isfinite(mh))):
        tables[kind] = (pos(rr[sel], "_RAJ2000", "_DEJ2000"), np.asarray(rr["Source"][sel]))
    good_c = np.array([snr.get(int(x), 0) >= 5 for x in cep["Source"]])
    for kind, sel in (("DCEP", (cep["Class"] == "DCEP") & np.isin(cep["ModeClass"], ["FUNDAMENTAL", "FIRST_OVERTONE"])),
                      ("BL Her", (cep["Class"] == "T2CEP") & (cep["SubClass"] == "BL_HER")),
                      ("W Vir", (cep["Class"] == "T2CEP") & (cep["SubClass"] == "W_VIR"))):
        tables[kind] = (pos(cep[sel & good_c], "RA_ICRS", "DE_ICRS"), np.asarray(cep["Source"][sel & good_c]))
    good_h = np.array([bool(x) and snr.get(int(x), 0) >= 5 for x in hads["GaiaDR3"]])
    tables["HADS"] = (pos(hads[good_h], "RAJ2000", "DEJ2000"), np.asarray(hads["GaiaDR3"][good_h]).astype(np.int64))
    print("  modellable: " + ", ".join(f"{len(v[1])} {k}" for k, v in tables.items()), flush=True)
    anchors_ = [k_ for k_ in ("DCEP", "BL Her", "W Vir", "HADS") if len(tables[k_][1])]
    centres = SkyCoord(np.concatenate([tables[k_][0].ra.deg for k_ in anchors_]) * u.deg,
                       np.concatenate([tables[k_][0].dec.deg for k_ in anchors_]) * u.deg)
    k = local_counts(centres, tables)
    score = np.array([len(x) * 1000 + sum(len(v) for v in x.values()) for x in k])
    top = np.argsort(-score)[:300]
    print(f"{len(centres)} anchors; adding eclipsing binaries (parallax S/N >= 5) around the best {len(top)}", flush=True)
    keb = count(centres[top])
    eb_snr = parallax_snr([s_ for d in keb for s_ in d.get("EB", ())])
    for j, i in enumerate(top):
        good = {s_ for s_ in keb[j].get("EB", ()) if eb_snr.get(s_, 0) >= 5}
        if good:
            k[i]["EB"] = good
    rows = [(centres[i].ra.deg, centres[i].dec.deg, centres[i].galactic.b.deg, len(k[i]), sum(len(v) for v in k[i].values()),
             ", ".join(f"{n} {t}" for t, n in sorted(((t, len(v)) for t, v in k[i].items()), key=lambda x: -x[1])))
            for i in top]
    out = Table(rows=rows, names=["ra", "dec", "b", "n_types", "n_stars", "types"])
    out.sort(["n_types", "n_stars"], reverse=True)
    out.write("data/rich_fields_modellable.ecsv", overwrite=True)
    for r in out[:15]:
        print(f"RA {r['ra']:9.4f} Dec {r['dec']:+8.4f}  b {r['b']:+6.1f}  {r['n_types']} types, {r['n_stars']} stars: {r['types']}", flush=True)
