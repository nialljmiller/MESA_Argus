"""Write MESA run directories from track specs and run them in parallel.

A track is a dict with at least: name, mass, Z, Y, alpha, max_age (see grids/).
Each track runs in runs/<name>/ with a fully rendered `inlist`. Runs are resumable:
finished tracks are skipped, and a `.running` lock (holding the PID) marks
tracks that are in progress, including ones started by another runner.
"""

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config as C

DEFAULT_MAX_AGE = 13.5e9


def track_name(mass, feh, alpha, max_age=DEFAULT_MAX_AGE):
    m = f"{mass:05.2f}" if abs(mass * 100 - round(mass * 100)) < 1e-9 else f"{mass:.3f}"
    name = f"M{m}_MH{feh:+.1f}_a{alpha:.2f}"
    if abs(max_age - DEFAULT_MAX_AGE) > 1:
        name += f"_t{max_age / 1e9:g}"
    return name


def render_inlist(track, mag_system="Vega"):
    tpl = (C.INLISTS / "physics.inlist").read_text()
    return tpl.format(
        mass=f"{track['mass']:.5f}d0", z=f"{track['Z']:.6e}", y=f"{track['Y']:.6f}d0",
        alpha=f"{track['alpha']:.4f}d0", max_age=f"{track.get('max_age', DEFAULT_MAX_AGE):.4e}",
        history_columns=C.INLISTS / "history_columns.list", instrument=C.INSTRUMENT,
        stellar_atm=C.colors_path("stellar_atm"), vega_sed=C.colors_path("vega_sed"),
        mag_system=mag_system,
    )


# ---------------------------------------------------------------- status
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def termination(d):
    out = d / "out.txt"
    if not out.exists():
        return None
    lines = [ln.strip() for ln in out.read_text(errors="replace").splitlines() if "termination code" in ln]
    return lines[-1].split(":", 1)[-1].strip() if lines else None


def status(name):
    d = C.RUNS / name
    if not d.exists():
        return "pending"
    if termination(d):
        return "done"
    lock = d / ".running"
    if lock.exists() and _pid_alive(int(lock.read_text().split()[0])):
        return "running"
    return "failed" if (d / "out.txt").exists() else "pending"


# ---------------------------------------------------------------- running
_launch_lock = threading.Lock()


def _n_star_procs():
    r = subprocess.run(["pgrep", "-x", "star"], capture_output=True, text=True)
    return len(r.stdout.split())


def _wait_for_slot():
    cap = C.config()["run"].get("max_star_procs", 0)
    while cap and _n_star_procs() >= cap:
        time.sleep(15)


def prepare(track, mag_system="Vega", run_dir=None):
    d = run_dir or C.RUNS / track["name"]
    d.mkdir(parents=True, exist_ok=True)
    for f in ("out.txt", ".running"):
        (d / f).unlink(missing_ok=True)
    (d / "inlist").write_text(render_inlist(track, mag_system))
    return d


def run_dir(d, threads=None, timeout=24 * 3600):
    """Run MESA in directory d (which must contain `inlist`). Returns (ok, minutes, termination)."""
    if not C.STAR_BIN.exists():
        raise SystemExit(f"MESA binary not found: {C.STAR_BIN}\n  build it with: make -C mesa_work")
    threads = threads or C.config()["run"]["omp_threads"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    t0 = time.time()
    with _launch_lock:
        _wait_for_slot()
        out = open(d / "out.txt", "w")
        p = subprocess.Popen([str(C.STAR_BIN)], cwd=d, stdout=out, stderr=subprocess.STDOUT, env=env)
        (d / ".running").write_text(f"{p.pid}\n")
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
    finally:
        out.close()
        (d / ".running").unlink(missing_ok=True)
    term = termination(d)
    return bool(term) and (d / "LOGS" / "history.data").exists(), (time.time() - t0) / 60, term


def run_tracks(tracks, workers=None, threads=None, rerun_failed=False):
    """Run every track that is not done or already running elsewhere."""
    todo = []
    for t in tracks:
        s = status(t["name"])
        if s == "pending" or (s == "failed" and rerun_failed):
            todo.append(t)
    skipped = len(tracks) - len(todo)
    print(f"{len(todo)} to run, {skipped} done/running/failed-skipped", flush=True)
    if not todo:
        return []
    workers = workers or C.config()["run"]["workers"]

    def one(t):
        ok, minutes, term = run_dir(prepare(t), threads)
        print(f"  {'OK ' if ok else 'BAD'} {t['name']:<30s} {minutes:7.1f} min  {term or 'no termination code'}", flush=True)
        return t["name"], ok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, todo))
