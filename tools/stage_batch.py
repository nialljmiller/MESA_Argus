"""Run argus_field.py over a batch: one base run file, and lists of values to iterate over; several runs at once,
sharing the machine's cores; resumable.

    python tools/stage_batch.py batches/overnight.toml                 # run (detach with nohup / setsid on a node)
    python tools/stage_batch.py batches/overnight.toml --check         # expand and validate every run; run nothing
    python tools/stage_batch.py batches/overnight.toml --status        # progress table
    python tools/stage_batch.py batches/overnight.toml --jobs 4 --after 12345   # 4 at once, after PID 12345 exits

A batch file (TOML):

    base = "../examples/synthetic_grb.toml"   # a run file (mesa_argus/runconfig.py); relative to this batch file
    combine = "product"                       # "product": every combination of the lists; "zip": side by side
    jobs = 3                                  # runs at once (--jobs overrides)

    [set]                                     # fixed overrides for every run (any run-file key)
    "movie.bands" = ["rho"]

    [vary]                                    # each key a list; one run per combination
    "field.ra,field.dec" = [[294.78, -11.62], [273.04, -20.71]]   # several keys together: a list of tuples,
    "night.date" = ["2026-07-19", "2026-08-01"]                    # or a CSV file name (columns in the keys' order)
    source = [                                                     # whole [[source]] lists, one per run
      [{generator = "GRB", peak = 9, t0 = "start+0.06"}],
      [{generator = "microlensing", tE = 0.2, u0 = 0.02, base = 19.5, bp_rp = 1.0, t0 = "start+0.15"}],
    ]

Every run is expanded and validated before any starts. Each is written out in full to
outputs/batch_<batch>/runs/<run>.toml and run as `python argus_field.py <that file>`, so it is an ordinary run that
can be repeated on its own. Run names: the field names argus_field.py would give (from the coordinates; + _syn<seed>
for synthetic fields) when those are all different, else <batch>_<number>; a name in the base or [vary] is kept.

Each run gets an equal share of the cores (config.toml [run] max_star_procs, else all) through $ARGUS_CORES, which
mesa_argus.rsp uses to size its core pool; the share is recomputed as runs start, so the last runs get more. A run
whose movies (or, with movie.models_only, field summary) already exist is skipped, so re-running the same command
resumes a batch. Per-run logs and a running summary are written to outputs/batch_<batch>/.
"""
import argparse
import csv
import itertools
import os
import shlex
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def field_name(ra, dec):
    return f"f{ra:07.3f}{'p' if dec >= 0 else 'm'}{abs(dec):06.3f}".replace(".", "_")  # as argus_field.field_name


def axis(keys, values, where):
    """One [vary] entry as a list of {key: value} overrides."""
    names = [k.strip() for k in keys.split(",")]
    if isinstance(values, str):  # a CSV file: one row per run, columns in the keys' order (a header row is skipped)
        path = (where / values) if not Path(values).is_absolute() else Path(values)
        rows = []
        for row in csv.reader(path.read_text().splitlines()):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                rows.append([tomllib.loads(f"v = {c.strip()}")["v"] for c in row[:len(names)]])
            except tomllib.TOMLDecodeError:
                if rows:
                    raise SystemExit(f"{path}: cannot read row {row}")
        values = rows
    if not isinstance(values, list) or not values:
        raise SystemExit(f"[vary] {keys}: give a non-empty list (or a CSV file name)")
    out = []
    for v in values:
        if len(names) == 1:
            out.append({names[0]: v})
        else:
            if not isinstance(v, list) or len(v) != len(names):
                raise SystemExit(f"[vary] {keys}: each item should be a list of {len(names)} values, not {v!r}")
            out.append(dict(zip(names, v)))
    return out


def expand(batch_path):
    """[(name, resolved run)] for every run of the batch, validated."""
    from mesa_argus import runconfig as R

    batch_path = Path(batch_path).resolve()
    b = tomllib.loads(batch_path.read_text())
    unknown = sorted(set(b) - {"base", "combine", "jobs", "set", "vary", "name"})
    if unknown:
        raise SystemExit(f"{batch_path}: unknown keys {unknown}; a batch takes base, combine, jobs, name, [set], [vary]")
    if "base" not in b:
        raise SystemExit(f"{batch_path}: base = (a run file) is required")
    base = R.load(batch_path.parent / b["base"])
    fixed = b.get("set", {})
    axes = [axis(k, v, batch_path.parent) for k, v in b.get("vary", {}).items()]
    combine = b.get("combine", "product")
    if combine == "product":
        combos = list(itertools.product(*axes)) if axes else [()]
    elif combine == "zip":
        if len({len(a) for a in axes}) > 1:
            raise SystemExit(f"{batch_path}: combine = \"zip\" needs every [vary] list the same length, not "
                             f"{[len(a) for a in axes]}")
        combos = list(zip(*axes)) if axes else [()]
    else:
        raise SystemExit(f"{batch_path}: combine = {combine!r}: \"product\" or \"zip\"")
    runs = []
    for combo in combos:
        over = dict(fixed)
        for part in combo:
            over.update(part)
        runs.append(R.resolve(R.merge(base, over)))
    tag = b.get("name", batch_path.stem)

    def default(r):
        f = r["field"]
        return f["name"] or field_name(f["ra"], f["dec"]) + (f"_syn{f['seed']}" if f["mode"] == "synthetic" else "")

    names = [default(r) for r in runs]
    if len(set(names)) < len(names):
        width = len(str(len(runs)))
        names = [r["field"]["name"] or f"{tag}_{i:0{width}d}" for i, r in enumerate(runs, 1)]
        if len(set(names)) < len(names):
            raise SystemExit(f"{batch_path}: runs share a field name ({names}); drop field.name or vary it")
    for n, r in zip(names, runs):
        r["field"]["name"] = n
    return tag, b.get("jobs", 3), list(zip(names, runs))


def finished(name, run):
    out = REPO / "outputs" / name
    if run["movie"]["models_only"]:
        return (out / "field_summary.md").exists()
    return all((out / f"{name}_night_{b}.mp4").exists() for b in run["movie"]["bands"])


def total_cores():
    if os.environ.get("ARGUS_CORES"):
        return int(os.environ["ARGUS_CORES"])
    from mesa_argus import config as C

    return int(C.config()["run"]["max_star_procs"]) or os.cpu_count()


def write_summary(path, runs, state):
    lines = [f"# Batch {path.parent.name}", "", f"updated {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             "| run | RA | Dec | mode | sources | status | start | end | wall (h) | log |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda t: datetime.fromtimestamp(t).strftime("%m-%d %H:%M") if t else ""  # noqa: E731
    for name, r in runs:
        s = state[name]
        wall = f"{(s['end'] - s['start']) / 3600:.2f}" if s.get("end") and s.get("start") else ""
        srcs = ", ".join(e["generator"] for e in r["source"]) or "-"
        lines.append(f"| {name} | {r['field']['ra']:.4f} | {r['field']['dec']:+.4f} | {r['field']['mode']} | {srcs} | "
                     f"{s['status']} | {fmt(s.get('start'))} | {fmt(s.get('end'))} | {wall} | {name}.log |")
    path.write_text("\n".join(lines) + "\n")


def main():
    from mesa_argus import runconfig as R

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch")
    ap.add_argument("--jobs", type=int, help="runs at once (default: the batch's jobs, else 3)")
    ap.add_argument("--after", type=int, help="wait for this PID to exit first")
    ap.add_argument("--status", action="store_true", help="print the summary and exit")
    ap.add_argument("--check", action="store_true", help="expand and validate every run, write the run files; run nothing")
    a = ap.parse_args()
    tag, jobs, runs = expand(a.batch)
    jobs = a.jobs or jobs
    out = REPO / "outputs" / f"batch_{tag}"
    summary = out / "summary.md"
    if a.status:
        print(summary.read_text() if summary.exists() else "not started")
        return
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for name, r in runs:
        (out / "runs" / f"{name}.toml").write_text(R.dumps(r, f"batch {tag}: run {name} ({a.batch})"))
    print(f"{len(runs)} runs, all valid; run files in {out / 'runs'}", flush=True)
    if a.check:
        for name, r in runs:
            print(f"  {name}: RA {r['field']['ra']:.4f} Dec {r['field']['dec']:+.4f} {r['field']['mode']}, night "
                  f"{r['night']['date'] or 'today'}, sources: {', '.join(e['generator'] for e in r['source']) or 'none'}")
        return
    state = {name: dict(status="done (earlier)" if finished(name, r) else "queued") for name, r in runs}
    write_summary(summary, runs, state)
    if a.after:
        print(f"waiting for PID {a.after} …", flush=True)
        while subprocess.run(["kill", "-0", str(a.after)], capture_output=True).returncode == 0:
            time.sleep(30)
    todo = [(n, r) for n, r in runs if state[n]["status"] == "queued"]
    cores = total_cores()
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            name, r = todo.pop(0)
            share = max(1, cores // min(jobs, len(running) + 1 + len(todo)))
            cmd = [sys.executable, str(REPO / "argus_field.py"), str(out / "runs" / f"{name}.toml")]
            log = open(out / f"{name}.log", "w")
            log.write(f"$ ARGUS_CORES={share} {shlex.join(cmd)}\n")
            log.flush()
            p = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, ARGUS_CORES=str(share)))
            running[name] = (p, log)
            state[name].update(status=f"running ({share} cores)", start=time.time())
            print(f"{datetime.now():%H:%M} start {name} ({share} cores)", flush=True)
            write_summary(summary, runs, state)
        time.sleep(20)
        for name, (p, log) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            del running[name]
            state[name].update(status="done" if p.returncode == 0 else f"failed (exit {p.returncode})", end=time.time())
            print(f"{datetime.now():%H:%M} {state[name]['status']} {name}", flush=True)
            write_summary(summary, runs, state)
    print(f"batch finished; summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
