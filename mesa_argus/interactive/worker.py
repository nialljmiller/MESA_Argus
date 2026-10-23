"""Worker processes for the interactive app: the star measured in many frames at once, and the movie's frames drawn
for playback, each worker holding its own copy of the night (night.Night).

Workers are spawned, not forked: a forked Jupyter kernel on macOS can crash in the system's numerical libraries.
Each loads the pack itself once (a few seconds, in the background, while the app is already usable: until the
workers are ready the app measures with threads and draws frames itself).
"""
import io
import logging
import multiprocessing as mp
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

_night = None  # this worker's night, and what it was made from
_key = None
_fig = None


def quiet_argus_sim():
    """ArgusSim announces its default configuration at import, through its own handler unless one is there."""
    log = logging.getLogger("argus_sim")
    if not log.handlers:
        log.addHandler(logging.NullHandler())
        log.propagate = False
        log.setLevel(logging.WARNING)


def _setup():
    import warnings

    import matplotlib

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Jupyter's "interrupt kernel" reaches the workers too: not for them
    matplotlib.use("Agg")
    warnings.simplefilter("ignore", RuntimeWarning)
    quiet_argus_sim()


def _get(key):
    global _night, _key
    if key != _key:
        _night = _key = None  # free the previous pack before loading the next
        from mesa_argus.interactive.night import Night

        _night, _key = Night(*key), key
    return _night


def warm(key):
    _get(key)
    return os.getpid()


def measure(key, state, ns):
    return _get(key).measure_frames(state, ns)


def render(key, state, meas, cands, ns, dpi):
    """[(n, PNG bytes)] of movie frames ns for a state, from its measurements."""
    global _fig
    import matplotlib.pyplot as plt

    night = _get(key)
    if night.state is None or repr(night.state) != repr(state):
        night.assemble(state, meas, cands)
    if _fig is None:
        night.nm.apply_style()
        _fig = plt.figure(figsize=(16, 10))
    out = []
    for n in ns:
        night.draw(_fig, n)
        buf = io.BytesIO()
        _fig.savefig(buf, format="png", dpi=dpi)
        out.append((n, buf.getvalue()))
    return out


class Workers:
    """A pool of worker processes for one night at a time (use)."""

    def __init__(self, n=None):
        self.n = n or max(2, min(6, (os.cpu_count() or 4) - 2))
        self.key = None
        self._warming = []
        self.failures = 0  # loads that failed; after two the app works without workers
        self._start()

    def _start(self):
        self.pool = ProcessPoolExecutor(self.n, mp_context=mp.get_context("spawn"), initializer=_setup)

    def use(self, pack_name, band, model, cut):
        """Load this night in every worker (in the background)."""
        self.key = (pack_name, band, model, cut)
        try:
            self._warming = [self.pool.submit(warm, self.key) for _ in range(self.n)]
        except BrokenProcessPool:  # a worker died (killed, out of memory): a fresh pool
            self._start()
            self._warming = [self.pool.submit(warm, self.key) for _ in range(self.n)]

    def ready(self):
        if any(f.done() and f.exception() is not None for f in self._warming):
            self.failures += 1
            self._warming = []
            if self.failures < 2:
                self.use(*self.key)  # a worker failed: once more, in the background
            return False
        return bool(self._warming) and all(f.done() for f in self._warming)

    def measure(self, state, ns):
        """[(n, (mag, err), candidate)] for frames ns, shared among the workers; None while they are loading or
        if they fail (the caller measures in-process then)."""
        if not self.ready():
            return None
        try:
            chunks = [ns[i::self.n] for i in range(self.n)]
            futures = [self.pool.submit(measure, self.key, state, c) for c in chunks if c]
            return [r for f in futures for r in f.result()]
        except BrokenProcessPool:
            self._start()
            self.use(*self.key)
            return None

    def render(self, state, meas, cands, ns, dpi, chunk=3):
        """[future of [(n, PNG)]] for frames ns in order, a few frames per future (playback starts with the first)."""
        ns = list(ns)
        return [self.pool.submit(render, self.key, state, meas, cands, ns[i:i + chunk], dpi)
                for i in range(0, len(ns), chunk)]

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
