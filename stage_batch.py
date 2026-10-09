"""Run argus_field.py on a list of fields, several at once, sharing the machine's cores; resumable.

    python stage_batch.py batches/overnight.txt                 # run (detach with nohup / setsid on a node)
    python stage_batch.py batches/overnight.txt --status        # progress table
    python stage_batch.py batches/overnight.txt --jobs 4 --after 12345   # 4 fields at once, after PID 12345 exits

Batch file: one field per line, `ra dec` then optional argus_field.py options (default: --block 9 --select all
--date auto); `#` starts a comment. Fields run in file order, --jobs at a time (default 3). Each field process gets
an equal share of the cores (config.toml [run] max_star_procs, else all) through $ARGUS_CORES, which
mesa_argus.rsp uses to size its core pool; the share is recomputed as fields start, so the last fields get more.

Nothing in argus_field.py changes: each field is a normal argus_field.py run (models cached per star in runs/,
outputs/<field>/ as usual). A field whose beta and rho movies already exist is skipped, so re-running the same
command resumes a batch. Per-field logs and a running summary (start, end, wall time, exit status, stars modelled
and failed) are written to outputs/batch_<batch file name>/.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULTS = {"--block": "9", "--select": "all", "--date": "auto"}


def field_name(ra, dec):
    return f"f{ra:07.3f}{'p' if dec >= 0 else 'm'}{abs(dec):06.3f}".replace(".", "_")  # as argus_field.field_name


def read_batch(path):
    fields = []
    for line in Path(path).read_text().splitlines():
        words = shlex.split(line.split("#", 1)[0])
        if not words:
            continue
        ra, dec, opts = float(words[0]), float(words[1]), words[2:]
        for k, v in DEFAULTS.items():
            if k not in opts:
                opts += [k, v]
        name = opts[opts.index("--name") + 1] if "--name" in opts else field_name(ra, dec)
        fields.append(dict(ra=ra, dec=dec, opts=opts, name=name))
    return fields


def finished(f):
    out = REPO / "outputs" / f["name"]
    return all((out / f"{f['name']}_night_{b}.mp4").exists() for b in ("beta", "rho"))


def total_cores():
    if os.environ.get("ARGUS_CORES"):
        return int(os.environ["ARGUS_CORES"])
    sys.path.insert(0, str(REPO))
    from mesa_argus import config as C

    return int(C.config().get("run", {}).get("max_star_procs", 0) or 0) or os.cpu_count() or 1


def outcome(f):
    """(modelled, failed or dropped) counts from the field's summary."""
    p = REPO / "outputs" / f["name"] / "field_summary.md"
    if not p.exists():
        return "-", "-"
    rows = [ln for ln in p.read_text().splitlines() if ln.startswith("| ") and ln.count("|") >= 8 and "Gaia DR3" not in ln]
    treat = [ln.rsplit("|", 2)[-2].strip() for ln in rows]
    modelled = sum(t.startswith(("MESA RSP (", "PHOEBE (")) for t in treat)
    failed = sum(("failed" in t or "cannot be modelled" in t or "dropped" in t) for t in treat)
    return modelled, failed


def write_summary(path, fields, state):
    lines = [f"# Batch {path.parent.name}", "", f"updated {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             "| field | RA | Dec | status | start | end | wall (h) | modelled | failed/dropped | log |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for f in fields:
        s = state[f["name"]]
        wall = f"{(s['end'] - s['start']) / 3600:.2f}" if s.get("end") and s.get("start") else ""
        fmt = lambda t: datetime.fromtimestamp(t).strftime("%m-%d %H:%M") if t else ""  # noqa: E731
        m, x = outcome(f) if s["status"].startswith(("done", "failed")) else ("", "")
        lines.append(f"| {f['name']} | {f['ra']:.4f} | {f['dec']:+.4f} | {s['status']} | {fmt(s.get('start'))} | "
                     f"{fmt(s.get('end'))} | {wall} | {m} | {x} | {f['name']}.log |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch")
    ap.add_argument("--jobs", type=int, default=3, help="fields run at once (default 3)")
    ap.add_argument("--after", type=int, help="wait for this PID to exit first")
    ap.add_argument("--status", action="store_true", help="print the summary and exit")
    a = ap.parse_args()
    out = REPO / "outputs" / f"batch_{Path(a.batch).stem}"
    summary = out / "summary.md"
    if a.status:
        print(summary.read_text() if summary.exists() else "not started")
        return
    out.mkdir(parents=True, exist_ok=True)
    fields = read_batch(a.batch)
    state = {f["name"]: dict(status="done (earlier)" if finished(f) else "queued") for f in fields}
    write_summary(summary, fields, state)
    if a.after:
        print(f"waiting for PID {a.after} …", flush=True)
        while Path(f"/proc/{a.after}").exists() or subprocess.run(["kill", "-0", str(a.after)], capture_output=True).returncode == 0:
            time.sleep(30)
    todo = [f for f in fields if state[f["name"]]["status"] == "queued"]
    cores = total_cores()
    running = {}
    while todo or running:
        while todo and len(running) < a.jobs:
            f = todo.pop(0)
            share = max(1, cores // min(a.jobs, len(running) + 1 + len(todo)))
            cmd = [sys.executable, str(REPO / "argus_field.py"), "--ra", str(f["ra"]), "--dec", str(f["dec"]), *f["opts"]]
            log = open(out / f"{f['name']}.log", "w")
            log.write(f"$ ARGUS_CORES={share} {shlex.join(cmd)}\n")
            log.flush()
            p = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, ARGUS_CORES=str(share)))
            running[f["name"]] = (p, log, f)
            state[f["name"]].update(status=f"running ({share} cores)", start=time.time())
            print(f"{datetime.now():%H:%M} start {f['name']} ({share} cores)", flush=True)
            write_summary(summary, fields, state)
        time.sleep(20)
        for name, (p, log, f) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            del running[name]
            ok = p.returncode == 0 and finished(f)
            state[name].update(status="done" if ok else f"failed (exit {p.returncode})", end=time.time())
            print(f"{datetime.now():%H:%M} {state[name]['status']} {name}", flush=True)
            write_summary(summary, fields, state)
    print(f"batch finished; summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
