"""Run jobs (observations, or any command) one at a time from a queue file. On Linux each job runs in its own
memory-limited systemd unit (systemd-run --user --scope -p MemoryMax=...), so a job the limit kills fails alone and
the queue goes on (the unit stays listed as failed: systemctl --user list-units --state=failed). On macOS jobs run
plainly; there, and with --wait-for-engines on Linux, a job starts only when no other engine (observe.py,
night_movie.py) is running on the machine.

Queue file: one job per line, "<name> <command ...>" ('#' comments and blank lines ignored), run from the repository.
It is read again before each job, so jobs can be added, reordered or removed while the queue runs. Each finished job
gets a line in <queue>.state ("<name> ok|failed <hours> h <peak> GB <swap> GB <start> <end>", the peaks measured on
the unit's cgroup, Linux only); a job named there is not run again. Each job's output: <queue folder>/logs/<name>.log.

    nohup python3.14 tools/rb_queue.py outputs/rb_queue/newton.queue --memory-max 45G \\
        > outputs/rb_queue/newton.log 2>&1 < /dev/null & disown
"""
import argparse
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINUX = sys.platform.startswith("linux")


def stamp():
    return time.strftime("%m-%d %H:%M")


def read_queue(path):
    jobs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, command = line.partition(" ")
            jobs.append((name, command.strip()))
    return jobs


def finished(state):
    return {line.split()[0] for line in state.read_text().splitlines() if line.strip()} if state.exists() else set()


def other_engines():
    out = subprocess.run(["pgrep", "-fl", r"observe\.py|night_movie\.py"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "rb_queue" not in line]


def unit_name(name):
    """A unit name not in use (a failed unit keeps its name until reset)."""
    base = "q-" + re.sub(r"[^A-Za-z0-9_.-]", "-", name)
    unit, k = base, 1
    while subprocess.run(["systemctl", "--user", "status", f"{unit}.scope"], capture_output=True).returncode != 4:
        k += 1
        unit = f"{base}-{k}"
    return unit


class Peak(threading.Thread):
    """The unit's memory and swap peaks, read from its cgroup every 20 s (memory.peak where the kernel has it)."""

    def __init__(self, unit):
        super().__init__(daemon=True)
        self.unit, self.memory, self.swap, self.done = unit, 0, 0, threading.Event()

    def _read(self, folder, name):
        try:
            return int((folder / name).read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return 0

    def run(self):
        folder = None
        while not self.done.wait(20.0):
            if folder is None:
                group = subprocess.run(["systemctl", "--user", "show", "-P", "ControlGroup", f"{self.unit}.scope"],
                                       capture_output=True, text=True).stdout.strip()
                folder = Path("/sys/fs/cgroup" + group) if group else None
                if folder is None:
                    continue
            self.memory = max(self.memory, self._read(folder, "memory.peak"), self._read(folder, "memory.current"))
            self.swap = max(self.swap, self._read(folder, "memory.swap.peak"), self._read(folder, "memory.swap.current"))


def run_job(name, command, logs, memory_max, swap_max):
    log = open(logs / f"{name}.log", "w")
    argv = shlex.split(command)
    peak = None
    if LINUX:
        unit = unit_name(name)
        limits = ["-p", f"MemoryMax={memory_max}"] + (["-p", f"MemorySwapMax={swap_max}"] if swap_max else [])
        argv = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}", *limits, *argv]
        peak = Peak(unit)
        print(f"{stamp()} {name}: unit {unit}.scope (MemoryMax={memory_max})", flush=True)
    t0 = time.time()
    process = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if peak:
        peak.start()
    code = process.wait()
    log.close()
    if peak:
        peak.done.set()
    gb = lambda n: n / 2**30  # noqa: E731
    return code, (time.time() - t0) / 3600, gb(peak.memory) if peak else float("nan"), gb(peak.swap) if peak else float("nan"), t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("queue", type=Path, help="the queue file")
    p.add_argument("--memory-max", default="45G", help="each job's MemoryMax (Linux)")
    p.add_argument("--swap-max", default="", help="each job's MemorySwapMax (Linux; default: systemd's)")
    p.add_argument("--wait-for-engines", action="store_true", help="also on Linux, wait while another engine runs")
    p.add_argument("--poll", type=float, default=60.0, help="seconds between looks at the queue when nothing is to run")
    p.add_argument("--exit-when-empty", action="store_true", help="stop when every job in the queue has run")
    args = p.parse_args(argv)

    queue = args.queue.resolve()
    state = queue.with_suffix(queue.suffix + ".state")
    logs = queue.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"{stamp()} queue {queue} ({'units, MemoryMax=' + args.memory_max if LINUX else 'plain processes'})",
          flush=True)
    waiting_said = False
    while True:
        done = finished(state)
        todo = [(n, c) for n, c in read_queue(queue) if n not in done]
        if not todo:
            if args.exit_when_empty:
                print(f"{stamp()} queue empty: done", flush=True)
                return
            time.sleep(args.poll)
            continue
        if (not LINUX or args.wait_for_engines) and other_engines():
            if not waiting_said:
                print(f"{stamp()} waiting: another engine is running ({other_engines()[0][:120]})", flush=True)
                waiting_said = True
            time.sleep(args.poll)
            continue
        waiting_said = False
        name, command = todo[0]
        print(f"{stamp()} start {name}: {command}", flush=True)
        code, hours, memory, swap, t0 = run_job(name, command, logs, args.memory_max, args.swap_max)
        status = "ok" if code == 0 else "failed"
        with open(state, "a") as f:
            f.write(f"{name} {status} {hours:.2f} h {memory:.1f} GB {swap:.1f} GB "
                    f"{time.strftime('%m-%d_%H:%M', time.localtime(t0))} {time.strftime('%m-%d_%H:%M')}\n")
        print(f"{stamp()} {status} {name} (exit {code}; {hours:.2f} h; peak {memory:.1f} GB, swap {swap:.1f} GB)",
              flush=True)


if __name__ == "__main__":
    main()
