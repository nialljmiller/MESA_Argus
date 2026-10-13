"""One-off (2026-10-13): MESA RSP / PHOEBE runs of Gaia variables made by the old argus_field.py lived in
runs/rsp_<field>_G<source>; make_stars.py's Gaia variables use runs/rsp_gaia_G<source> (a star no longer belongs to a
field). Renames them so the models are reused rather than recomputed. Safe to re-run; never overwrites.

    python tools/migrate_runs.py
"""
import re
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
moved, kept = 0, []
for d in sorted(RUNS.glob("rsp_f*_G*")):
    m = re.fullmatch(r"rsp_f\d{3}_\d{3}[pm]\d{2}_\d{3}_(G\d+)", d.name)
    if not m:
        continue
    new = RUNS / f"rsp_gaia_{m[1]}"
    if new.exists():
        kept.append(d.name)
        continue
    d.rename(new)
    moved += 1
print(f"renamed {moved} runs to rsp_gaia_G<source>" + (f"; left {kept} (a run for that star already exists)" if kept else ""))
