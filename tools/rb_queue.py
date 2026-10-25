"""Run jobs (observations, or any command) from a queue file, one at a time or several within a memory budget. On
Linux each job runs in its own memory-limited systemd unit (systemd-run --user --scope -p MemoryMax=...), so a job
the limit kills fails alone and the queue goes on (the unit stays listed as failed: systemctl --user list-units
--state=failed). On macOS jobs run plainly; there, and with --wait-for-engines on Linux, a job starts only when no
other engine (observe.py, night_movie.py) is running on the machine.

Queue file: one job per line, "<name>[@<MemoryMax>] <command ...>" ('#' comments and blank lines ignored), run from
the repository; the optional @size (e.g. @45G) replaces --memory-max for that job. It is read again before each start,
so jobs can be added, reordered or removed while the queue runs. Each finished job gets a line in <queue>.state
("<name> ok|failed <hours> h <peak> GB <swap> GB <start> <end>", the peaks measured on the unit's cgroup, Linux only);
a job named there is not run again. Each job's output: <queue folder>/logs/<name>.log.

Several at once (--parallel N): a job starts only while the running jobs' MemoryMax plus its own stay within --budget,
and only once every running observation has its scenes built (its log says "observing", as make_training_data.py
waits: building a scene loads the 3D dust map, ~10 GB, twice per observation), or after --setup-timeout.

Safety: an observe.py job whose output folder (outputs/pseudo_real/<name>) already exists is not started (a run is
there: move it aside to run it again); a job whose unit (q-<name>.scope) is still running when the queue starts, left
by an earlier runner, is followed to its end and recorded (ok when it left its two alert tables).

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
UNITS = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def stamp():
    return time.strftime("%m-%d %H:%M")


def size(text):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", str(text).strip().upper())
    if not m:
        raise SystemExit(f"not a memory size: {text}")
    return float(m[1]) * UNITS.get(m[2], 1)


def read_queue(path, default_cap):
    jobs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            head, _, command = line.partition(" ")
            name, _, cap = head.partition("@")
            jobs.append((name, cap or default_cap, command.strip()))
    return jobs


def finished(state):
    return {line.split()[0] for line in state.read_text().splitlines() if line.strip()} if state.exists() else set()


def other_engines():
    out = subprocess.run(["pgrep", "-fl", r"observe\.py|night_movie\.py"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "rb_queue" not in line]


def unit_base(name):
    return "q-" + re.sub(r"[^A-Za-z0-9_.-]", "-", name)


def unit_state(unit):
    """systemctl's code for the scope: 0 active, 3 inactive or failed, 4 no such unit."""
    return subprocess.run(["systemctl", "--user", "status", f"{unit}.scope"], capture_output=True).returncode


def unit_name(name):
    """A unit name not in use (a failed unit keeps its name until reset)."""
    unit, k = unit_base(name), 1
    while unit_state(unit) != 4:
        k += 1
        unit = f"{unit_base(name)}-{k}"
    return unit


def output_folder(command, name):
    return REPO / "outputs" / "pseudo_real" / name if "observe.py" in command else None


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


class Job:
    def __init__(self, name, cap, command, logs, adopt_unit=None):
        self.name, self.cap, self.command, self.log_path = name, cap, command, logs / f"{name}.log"
        self.t0, self.process, self.peak, self.unit = time.time(), None, None, adopt_unit
        if adopt_unit:  # left running by an earlier runner: followed, not started
            self.peak = Peak(adopt_unit)
            self.peak.start()
            return
        argv = shlex.split(command)
        if LINUX:
            self.unit = unit_name(name)
            argv = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={self.unit}", "-p", f"MemoryMax={cap}",
                    *argv]
            self.peak = Peak(self.unit)
            print(f"{stamp()} {name}: unit {self.unit}.scope (MemoryMax={cap})", flush=True)
        self.log = open(self.log_path, "w")
        self.process = subprocess.Popen(argv, cwd=REPO, stdout=self.log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL)
        if self.peak:
            self.peak.start()

    def poll(self):
        """None while running, else the exit code (adopted: 0 if it left both alert tables)."""
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                self.log.close()
            return code
        if unit_state(self.unit) == 0:
            return None
        folder = output_folder(self.command, self.name)
        return 0 if folder and len(list(folder.glob("*_night_*.alerts.csv"))) >= 2 else 1

    def set_up(self, timeout):
        """Whether its scenes are built (its log says "observing"), or it is past the timeout, or not an observation."""
        if "observe.py" not in self.command or time.time() - self.t0 > timeout:
            return True
        try:
            return "observing " in self.log_path.read_text(errors="replace")
        except OSError:
            return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("queue", type=Path, help="the queue file")
    p.add_argument("--memory-max", default="45G", help="each job's MemoryMax (Linux), unless its line gives one")
    p.add_argument("--parallel", type=int, default=1, help="jobs at once, at most")
    p.add_argument("--budget", default="", help="the running jobs' MemoryMax summed stays within this (Linux)")
    p.add_argument("--setup-timeout", type=float, default=1800.0, help="seconds to wait for a job's scenes")
    p.add_argument("--wait-for-engines", action="store_true", help="also on Linux, wait while another engine runs")
    p.add_argument("--poll", type=float, default=30.0, help="seconds between looks")
    p.add_argument("--exit-when-empty", action="store_true", help="stop when every job in the queue has run")
    args = p.parse_args(argv)

    queue = args.queue.resolve()
    state = queue.with_suffix(queue.suffix + ".state")
    logs = queue.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    budget = size(args.budget) if args.budget else float("inf")
    print(f"{stamp()} queue {queue} ({'units, MemoryMax=' + args.memory_max if LINUX else 'plain processes'}; "
          f"{args.parallel} at once{', budget ' + args.budget if args.budget else ''})", flush=True)
    running, said = [], set()
    if LINUX:  # jobs an earlier runner left running
        for name, cap, command in read_queue(queue, args.memory_max):
            if name not in finished(state) and unit_state(unit_base(name)) == 0:
                running.append(Job(name, cap, command, logs, adopt_unit=unit_base(name)))
                print(f"{stamp()} following {name}: its unit {unit_base(name)}.scope is running", flush=True)
    while True:
        for job in list(running):
            code = job.poll()
            if code is None:
                continue
            running.remove(job)
            if job.peak:
                job.peak.done.set()
            hours = (time.time() - job.t0) / 3600
            memory, swap = (job.peak.memory / 2**30, job.peak.swap / 2**30) if job.peak else (float("nan"),) * 2
            status = "ok" if code == 0 else "failed"
            with open(state, "a") as f:
                f.write(f"{job.name} {status} {hours:.2f} h {memory:.1f} GB {swap:.1f} GB "
                        f"{time.strftime('%m-%d_%H:%M', time.localtime(job.t0))} {time.strftime('%m-%d_%H:%M')}\n")
            print(f"{stamp()} {status} {job.name} (exit {code}; {hours:.2f} h; peak {memory:.1f} GB, swap {swap:.1f} GB)",
                  flush=True)
        done = finished(state) | {job.name for job in running}
        todo = []
        for name, cap, command in read_queue(queue, args.memory_max):
            folder = output_folder(command, name)
            if name in done:
                continue
            if folder is not None and folder.exists():
                if name not in said:
                    print(f"{stamp()} {name}: {folder.relative_to(REPO)} exists; not started (move it aside to run it)",
                          flush=True)
                    said.add(name)
                continue
            todo.append((name, cap, command))
        if not todo and not running and args.exit_when_empty:
            print(f"{stamp()} queue empty: done", flush=True)
            return
        can_start = (todo and len(running) < args.parallel
                     and all(job.set_up(args.setup_timeout) for job in running)
                     and (not LINUX or sum(size(job.cap) for job in running) + size(todo[0][1]) <= budget))
        if can_start and (not LINUX or args.wait_for_engines) and other_engines():
            if "engines" not in said:
                print(f"{stamp()} waiting: another engine is running ({other_engines()[0][:120]})", flush=True)
                said.add("engines")
            can_start = False
        if can_start:
            said.discard("engines")
            name, cap, command = todo[0]
            print(f"{stamp()} start {name}: {command}", flush=True)
            running.append(Job(name, cap, command, logs))
            continue
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
