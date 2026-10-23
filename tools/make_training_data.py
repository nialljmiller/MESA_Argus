"""Real/bogus training data, as fast as the machine allows: one-night observations of the real Gaia fields (each
stars/<field>_field background, at its real positions) with [inject] library stars scattered over the readout
window, no movies, run several at once. Each writes <name>_night_<band>.alerts.csv (every hdps candidate, its
features, its truth); tools/rb_training_set.py turns them into a training set, tools/train_rb.py into an hdps bundle.

Each observation is written as an observation file in <out>/observations/ and run with observe.py into
outputs/pseudo_real/rb_<field>_<date>/. A field the array cannot observe for 3 h near the date is skipped (observe.py
says so); a run that fails is logged and the batch goes on. Re-running skips observations whose alert tables exist.

    python tools/make_training_data.py --fields all --date 2026-10-23 --inject 200 --parallel 4
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def field_of(name):
    """RA, Dec from a field set's name (f294_784m11_623_field -> 294.784, -11.623)."""
    m = re.match(r"f(\d{3})_(\d{3})([pm])(\d{2})_(\d{3})_field$", name)
    if not m:
        return None
    ra = float(f"{m[1]}.{m[2]}")
    dec = float(f"{m[4]}.{m[5]}") * (1 if m[3] == "p" else -1)
    return ra, dec


def observation(field, ra, dec, date, inject, seed, stride):
    name = f"rb_{field.removesuffix('_field')}_{date.replace('-', '')}"
    return name, f'''# Real/bogus training data (tools/make_training_data.py): {field} for one night, {inject} library stars
# injected over the readout window, no movies.
[field]
name = "{name}"
ra = {ra}
dec = {dec}

[night]
date = "{date}"
nights = 1

[stars]
background = "{field}"
layout = "real"

[inject]
number = {inject}
seed = {seed}

[output]
movie = false
pipeline_movie = false
lightcurves = false
stride = {stride}
'''


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", nargs="+", default=["all"], help="field set names (stars/<name>), or all")
    p.add_argument("--date", default="2026-10-23", help="the night to start from (observe.py takes the nearest "
                                                         "night each field is up >= 3 h)")
    p.add_argument("--inject", type=int, default=200, help="library stars per observation")
    p.add_argument("--stride", type=int, default=1, help="every Nth exposure")
    p.add_argument("--parallel", type=int, default=4, help="observations at once (each runs both bands: 2 processes)")
    p.add_argument("--out", type=Path, default=REPO / "data" / "rb")
    args = p.parse_args(argv)

    fields = sorted(d.name for d in (REPO / "stars").glob("*_field")) if args.fields == ["all"] else args.fields
    (args.out / "observations").mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, field in enumerate(fields):
        where = field_of(field)
        if where is None:
            print(f"{field}: not a field set name; skipped")
            continue
        name, text = observation(field, *where, args.date, args.inject, seed=i + 1, stride=args.stride)
        done = list((REPO / "outputs" / "pseudo_real" / name).glob("*_night_*.alerts.csv"))
        if len(done) >= 2:
            print(f"{name}: done already")
            continue
        path = args.out / "observations" / f"{name}.toml"
        path.write_text(text)
        jobs.append((name, path))
    print(f"{len(jobs)} observations, {args.parallel} at a time", flush=True)
    running, results = [], {}
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)
    while jobs or running:
        while jobs and len(running) < args.parallel:
            name, path = jobs.pop(0)
            log = open(log_dir / f"{name}.log", "w")
            running.append((name, log, subprocess.Popen([sys.executable, "observe.py", str(path)], cwd=REPO,
                                                         stdout=log, stderr=subprocess.STDOUT), time.time()))
            print(f"started {name}", flush=True)
        for item in list(running):
            name, log, proc, t0 = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                results[name] = proc.returncode
                status = "finished" if proc.returncode == 0 else f"FAILED (log {log_dir / name}.log)"
                print(f"{status} {name} ({(time.time() - t0) / 3600:.1f} h)", flush=True)
        time.sleep(20)
    ok = [n for n, rc in results.items() if rc == 0]
    print(f"done: {len(ok)} of {len(results)} observations; build the set with: python tools/rb_training_set.py "
          + " ".join(f"outputs/pseudo_real/{n}" for n in ok) + f" -o {args.out}")


if __name__ == "__main__":
    main()
