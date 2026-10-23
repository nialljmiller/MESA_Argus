"""One night of a field pack with the edited star in it: everything night_movie.draw reads, for every frame.

The field (its stars, hdps's images, its variables' photometry and its alerts) is the pack's: a real hdps run. The
edited star is added to hdps's cut-outs (compose) and measured there with hdps's own detector, alert features and the
real/bogus model (measure). The field's candidates are scored with the same model and cut, so its alerts are those
the model would raise.

Three steps, each usable on its own so worker processes can share the work (worker.py):
    measure_frames(state, ns)    the star in frames ns: its photometry and its alert candidate's features
    assemble(state, meas, cands) the draw data for the whole night (scores the candidates)
    draw(fig, n)                 movie frame n by night_movie.draw (the frame's images composed when drawn)
A state is (settings, airmass offset, noise seed): the same state gives the same images and numbers anywhere.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mesa_argus.interactive import compose, measure, pack, templates
from mesa_argus.interactive.star import EditableStar, Settings

REPO = Path(__file__).resolve().parents[2]
CUT = "fa10"  # the real/bogus operating point: 10 false alerts per deg2 per exposure on the training set
MAX_SHOWN = 6  # stars the light-curve panel draws (as the movie): the edited star and up to 5 of the field's
LABEL = "E"  # the edited star's label on the panels
THREADS = min(8, os.cpu_count() or 1)  # frames measured at once when no worker processes are given


def bundle():
    """The newest trained real/bogus bundle (data/rb/<name>/bundle), the final one when there."""
    found = sorted((REPO / "data" / "rb").glob("*/bundle/manifest.json"), key=lambda p: p.stat().st_mtime)
    final = [p for p in found if p.parent.parent.name == "final"]
    return str((final or found)[-1].parent) if found else None


class TrueMag:
    """true_mag(band, k, j) as the movie calls it, from tables over the pack's frames."""

    def __init__(self, tables, frames):
        self.tables, self.n_of = tables, {j: n for n, j in enumerate(frames)}

    def __call__(self, band, k, j):
        return float(self.tables[band][k][self.n_of[j]])


class Night:
    def __init__(self, pack_name, band, model=None, cut=CUT):
        import night_movie as nm

        self.nm = nm
        self.pack_name = pack_name
        self.p = p = pack.load(pack_name, band)
        self.band, self.cut = band, cut
        self.model = model if model is not None else bundle()
        self.frames = list(p["frames"])
        self.E = len(p["stars"]["dx"])
        self.args = SimpleNamespace(**p["args"])
        self.ab = templates.photometry()
        self.star = EditableStar()
        self.span = (float(p["bjd"][0]), float(p["bjd"][-1]))
        self.field_alerts = self._score_field()
        self.hdps_fwhm, self.hdps_plane = self._hdps_frame_values()
        sky_q = [(np.nan, np.nan) if c is None else compose.noise_ratio(c) for c in p["cutouts"]]
        self.sky, self.q = np.array([a for a, _ in sky_q]), np.array([b for _, b in sky_q])
        self.mask = self._mask_settings()
        self._stats = {}  # (frame, window, airmass offset) -> the frame's noise and sky there (measure.window_stats)
        self._light = {}  # state -> the star's light at every frame (EditableStar.evaluate)
        self.state = self.d = self.result = None
        self._exposure = {}  # frame -> its composed images, for the current state

    # ---------------------------------------------------------------- the field
    def _hdps_frame_values(self):
        """Per frame: the FWHM (px) hdps's alert features used and the plane rms they recorded (the field's
        candidates'), else hdps's defaults."""
        from hdps.alerts.features import FEATURE_NAMES, fwhm_from_seeing

        i_f, i_p = FEATURE_NAMES.index("fwhm_est"), FEATURE_NAMES.index("plane_rms_norm")
        fwhm, plane = [], []
        for c in self.p["candidates"]:
            F = c["features"]
            fwhm.append(float(np.median(F[:, i_f])) if len(F) else fwhm_from_seeing(None))
            plane.append(float(np.median(F[:, i_p])) if len(F) else np.nan)
        return np.array(fwhm), np.array(plane)

    def _mask_settings(self):
        """(threshold, base px, growth px) of hdps's bright-star detection mask as the run had it (the pack's SFFT
        settings, else the observation defaults), or None when the run did not mask."""
        from mesa_argus import schema

        sfft = self.p.get("sfft") or next(t["config"] for t in schema.TASKS if t["type"] == "SFFTSubtractionTask")
        if not sfft.get("mask_bright_stars", False):
            return None
        base = sfft.get("mask_dilation")
        base = int(base) if isinstance(base, (int, float)) else int(sfft.get("kernel_half_width", 6))
        return float(sfft.get("bright_star_threshold", 5000.0)), base, int(sfft.get("mask_dilation_growth", 4))

    def _score_field(self):
        """Per frame: the field's candidates the model accepts at the cut, as the movie's (key, dx, dy)."""
        from hdps.alerts.features import FEATURE_NAMES

        cands = self.p["candidates"]
        clf = measure.classifier(self.model)
        if clf is None:
            return [list(zip(c["key"], c["dx"], c["dy"])) for c in cands]
        sizes = [len(c["key"]) for c in cands]
        if not sum(sizes):
            return [[] for _ in cands]
        F = np.concatenate([c["features"] for c in cands])
        scores = clf.score_features({name: F[:, i].astype(float) for i, name in enumerate(FEATURE_NAMES)})
        keep = np.asarray(scores) >= clf.cut_for(self.cut)
        out, at = [], 0
        for c, size in zip(cands, sizes):
            k = keep[at:at + size]
            out.append([(key, float(x), float(y)) for key, x, y, ok in zip(c["key"], c["dx"], c["dy"], k) if ok])
            at += size
        return out

    # ---------------------------------------------------------------- the star
    def light(self, state):
        """The star's light at every frame (EditableStar.evaluate) for a state; the last few kept."""
        s, d_airmass, _ = state
        key = (repr(s), round(d_airmass, 3))
        if key not in self._light:
            p = self.p
            if len(self._light) > 8:
                self._light.clear()
            self._light[key] = self.star.evaluate(s, p["bjd"][self.frames], p["airmass"][self.frames] + d_airmass,
                                                  p["args"]["exptime"], span=self.span)
        return self._light[key]

    def compose(self, state, n):
        """Frame n's cut-outs with the star added (compose.frame), or None where hdps made none."""
        s, d_airmass, seed = state
        p, j, c = self.p, self.frames[n], self.p["cutouts"][n]
        if c is None:
            return None
        star = self.light(state)
        return compose.frame(c, self.band, float(p["seeing"][j]), float(p["survey"]["field_angle"][j]),
                             float(p["airmass"][j]), float(star["electrons"][self.band][n]),
                             float(star["reference"][self.band][n]), s.dx, s.dy, d_airmass=d_airmass,
                             seed=seed * 100_003 + n, sky=self.sky[n], q=self.q[n])

    def measure_frames(self, state, ns):
        """[(n, (mag, err, saturated), candidate features or None)] of the star in frames ns, measured as hdps would."""
        s, d_airmass, _ = state
        p = self.p
        out = []
        for n in ns:
            comp = self.compose(state, n)
            if comp is None:
                out.append((n, (np.nan, np.nan, 0.0), None))
                continue
            j, c = self.frames[n], p["cutouts"][n]
            zp = p["zp"][n] + 2.5 * np.log10(comp["keep"])  # hdps recalibrates every frame: extinction is absorbed
            m, e, _, _ = measure.photometry(comp, s.dx, s.dy, zp)
            fwhm = self.hdps_fwhm[n] * (comp["seeing"] / float(p["seeing"][j]))  # broader when observed lower
            ys, xs = measure.window(c["extent"], np.shape(c["difference"]), s.dx, s.dy)
            key = (n, ys.start, xs.start, round(d_airmass, 3))
            if key not in self._stats:
                self._stats[key] = measure.window_stats(c if d_airmass == 0 else comp, s.dx, s.dy)
            out.append((n, (m, e, float(comp["saturated"])), measure.candidate(comp, s.dx, s.dy, fwhm, self.hdps_plane[n], self._stats[key],
                                                     self.mask)))
        return out

    def update(self, s: Settings, d_airmass=0.0, seed=0, workers=None):
        """The night with the edited star at these settings (s.band set to the band shown) and the airmass offset:
        measured (by the worker pool when given, else threads here), then assembled."""
        s.band = self.band
        state = (s, float(d_airmass), int(seed))
        self.light(state)  # the star's template integrated here once (cached on disk), not by every worker or thread
        ns = list(range(len(self.frames)))
        done = workers.measure(state, ns) if workers is not None else None
        if done is None:  # threads: FFTs and numba release the GIL; each frame has its own noise seed
            chunks = [ns[i::THREADS] for i in range(THREADS)]
            with ThreadPoolExecutor(max_workers=THREADS) as pool:
                done = [r for part in pool.map(lambda c: self.measure_frames(state, c), chunks) for r in part]
        done.sort(key=lambda r: r[0])
        return self.assemble(state, np.array([r[1] for r in done], float), [r[2] for r in done])

    # ---------------------------------------------------------------- the draw data
    def assemble(self, state, meas, cands):
        """The draw data for a state from the star's measurements: (mag, err) per frame and its candidates."""
        nm, p, B, E, frames = self.nm, self.p, self.band, self.E, self.frames
        s, d_airmass, _ = state
        star = self.light(state)
        scored = measure.alerts(cands, self.model, self.cut)
        X_all = p["airmass"] + d_airmass
        seeing_all = p["seeing"] * (X_all / p["airmass"]) ** 0.6
        bjd = p["bjd"][frames]
        shown = [E] + list(p["shown"])[:MAX_SHOWN - 1]
        transient = star["timing"] != "periodic"
        palette = [nm.BLUE, nm.RED, "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
        meta = templates.light(star["template"])["meta"]
        kind, engine = meta.get("kind", s.kind), meta.get("engine", "")
        if transient:
            entry = dict(kind=kind, transient=dict(engine=engine, z=float(meta.get("z", np.nan))))
        else:
            entry = dict(kind=kind, placed=dict(engine=engine), period=float(s.period), epoch=self.span[0])
        model_curve = {k: v for k, v in p["model_curve"].items() if k in shown}
        if transient:  # the light-curve panel's model: the star on its own time grid
            model_bjd = bjd[0] + (nm.model_hours(p["ut"]) - p["ut"][frames[0]]) / 24.0
            grid = self.star.evaluate(s, model_bjd, np.interp(model_bjd, bjd, X_all[frames]),
                                      p["args"]["exptime"], span=self.span)
            model_curve[E] = np.where(np.isfinite(grid["mag"][B]), grid["mag"][B], np.nan)
        true = {b: {**{k: p["true"][b][k] for k in shown[1:]},
                    E: np.where(np.isfinite(star["mag"][b]), star["mag"][b], np.nan)} for b in ("beta", "rho")}
        follow_sed = {k: p["follow_sed"][k] for k in shown[1:]}
        sed = np.zeros((len(p["bjd"]), star["sed"].shape[1]), np.float32)
        sed[frames] = star["sed"]
        follow_sed[E] = sed
        models = {**{k: p["models"][k] for k in shown[1:]}, E: SimpleNamespace(wave=star["wave"])}
        follow_cycle = {**{k: p["follow_cycle"][k] for k in shown[1:]}, E: star["track"]}
        rel = {k: p["rel"][k] for k in shown[1:]}
        model = true[B][E]
        mean = np.nanmean if transient else np.mean
        lit = np.isfinite(meas[:, 0]).any()
        meas_rel = np.nanpercentile(meas[:, 0] - np.nanmean(meas[:, 0]), [0.25, 99.75]) if lit else []
        rel[E] = dict(measured=float(np.nanmean(meas[:, 0])) if lit else np.nan, model=float(mean(model)),
                      span=(np.nanmin(np.r_[meas_rel, model - mean(model)]), np.nanmax(np.r_[meas_rel, model - mean(model)])),
                      resid=float(np.nanpercentile(np.abs(meas[:, 0] - model), 99.75)) if lit else np.nan)
        G = np.r_[p["G"], 99.0]
        in_block = np.r_[p["in_block"], False]
        static_mag = {b: np.r_[p["static_mag"][b], np.nan] for b in ("beta", "rho")}
        fixed = np.nonzero(in_block & (G < 20) & ~np.isin(np.arange(len(G)), shown))[0]
        cmd_c = np.r_[static_mag["beta"][fixed] - static_mag["rho"][fixed],
                      np.concatenate([follow_cycle[k]["beta"] - follow_cycle[k]["rho"] for k in shown])]
        cmd_m = np.r_[static_mag[B][fixed], np.concatenate([follow_cycle[k][B] for k in shown])]
        ok = np.isfinite(cmd_c) & np.isfinite(cmd_m)
        sed_values = np.concatenate([follow_sed[k][:, (models[k].wave.to_value("nm") >= 300)
                                                   & (models[k].wave.to_value("nm") <= 1000)].ravel() for k in shown])
        sed_values = sed_values[sed_values > 0]
        if sed_values.size == 0:
            sed_values = np.array([1e-16, 1e-14])
        alerted = [list(field) + ([(E, s.dx, s.dy)] if ok_n else []) for field, (_, ok_n) in zip(self.field_alerts, scored)]
        self.d = dict(stars=dict(dx=np.r_[p["stars"]["dx"], s.dx], dy=np.r_[p["stars"]["dy"], s.dy]), G=G,
                      in_block=in_block, static_mag=static_mag, shown=shown, follow=list(range(p["n_follow"] + 1)),
                      colour_of={k: palette[i % len(palette)] for i, k in enumerate(shown)},
                      labels={**p["labels"], E: LABEL}, scene=None, true_mag=TrueMag(true, frames),
                      pulsators={**p["pulsators"], -1000: entry}, source={**p["source"], E: -1000},
                      follow_cycle=follow_cycle, model_curve=model_curve, ut=p["ut"], bjd=p["bjd"], times=p["times"],
                      rel=rel, alerted=alerted, survey=p["survey"], rgb=p["rgb"], dlim=p["dlim"], ab=self.ab,
                      models=models, follow_sed=follow_sed, airmass=X_all, seeing=seeing_all,
                      sed_ylim=(float(np.percentile(sed_values, 3)) / 1.5, float(sed_values.max()) * 1.5),
                      cmd_c=cmd_c[ok], cmd_m=cmd_m[ok], centre=p["centre"], frame_j=frames, n=0, history={})
        self._series = {k: dict(m=p["measured"][k][:, 0], e=p["measured"][k][:, 1], true=true[B][k]) for k in shown[1:]}
        self._series[E] = dict(m=meas[:, 0], e=meas[:, 1], true=true[B][E])
        self.state, self._exposure = state, {}
        self.result = dict(mag=meas[:, 0], err=meas[:, 1], true=true[B][E], score=np.array([sc for sc, _ in scored]),
                           alerted=np.array([a for _, a in scored]), template=star["template"],
                           n_field_alerts=np.array([len(a) for a in self.field_alerts]), meas=meas, cands=cands,
                           saturated=meas[:, 2] > 0)
        return self.result

    def exposure(self, n):
        """Frame n's images as draw() takes them, for the current state."""
        if n not in self._exposure:
            if len(self._exposure) > 4:
                self._exposure.clear()
            comp = self.compose(self.state, n)
            if comp is None:
                self._exposure[n] = None
            else:
                _, d_airmass, _ = self.state
                j = self.frames[n]
                ratio = (self.p["airmass"][j] + d_airmass) / self.p["airmass"][j]  # the sky grows with the airmass
                self._exposure[n] = dict(image=comp["science"], difference=comp["difference"],
                                         background=float(self.sky[n] * ratio), extent=comp["extent"])
        return self._exposure[n]

    def draw(self, fig, n):
        """Movie frame n (of the pack's frames) into fig, by night_movie.draw itself."""
        nm, d, frames = self.nm, self.d, self.frames
        j = frames[n]
        d["n"] = n
        t = list(np.asarray(d["ut"])[frames[:n + 1]])
        d["history"] = {k: dict(t=t, m=list(v["m"][:n + 1]), e=list(v["e"][:n + 1]), true=list(v["true"][:n + 1]))
                        for k, v in self._series.items()}
        fig.clear()
        nm.draw(fig, self.args, {self.band: self.exposure(n)}, j, d)
