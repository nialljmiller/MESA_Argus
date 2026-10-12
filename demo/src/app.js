// The interactive Argus page: the movie's panels (m3_night_movie.draw) with a control panel.

(async () => {
  const D = await Sim.unpackAll(JSON.parse(document.getElementById("field-data").textContent));
  const CMAP = JSON.parse(document.getElementById("colormaps").textContent);
  const SYMBOL = { beta: "β", rho: "ρ" };
  const TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
  const N = D.night.airmass.length, NA = D.airmass_grid.length;
  const $ = (id) => document.getElementById(id);

  const state = { j: 0, band: "rho", zenith: D.zenith_seeing, moon: 1, playing: false, fps: 8, mode: "select",
                  selected: null, seed: 3 };

  // ---------------------------------------------------------------- the scene
  let stars = [];
  function resetScene() {
    stars = D.stars.source.map((source, i) => ({
      source, gaia: i, x: D.stars.x[i], y: D.stars.y[i], scale: 1, colour: D.stars.colour[i],
      rate: Object.fromEntries(D.bands.map((b) => [b, D.stars.rate[b].subarray(i * NA, (i + 1) * NA)])),
      mag0: Object.fromEntries(D.bands.map((b) => [b, D.stars.mag[b][i]])),
    }));
    D.native.forEach(({ star, model }) => {
      const m = D.models[model];
      Object.assign(stars[star], { model, epoch: m.epoch_bjd - 2460000, period: m.gaia_period });
    });
    labelVariables();
    hidden.clear();
    state.selected = null;
  }
  // V1, V2, ...: the variables in the order they entered the scene
  function labelVariables() {
    stars.filter((s) => s.model !== undefined).forEach((s, i) => { s.label = `V${i + 1}`; s.colourIdx = i % 10; });
  }
  const variables = () => stars.filter((s) => s.model !== undefined);
  // above-atmosphere magnitude of a constant star, or a variable's intensity mean, in band b
  function meanMag(s, b) {
    if (s.model === undefined) return s.mag0[b] - 2.5 * Math.log10(s.scale);
    let f = 0;
    for (let i = 0; i < 200; i++) f += 10 ** (-0.4 * Sim.modelMag(D.models[s.model], b, i / 200));
    return -2.5 * Math.log10(f / 200) - 2.5 * Math.log10(s.scale);
  }
  const trueMag = (s, b, j) => s.model === undefined ? meanMag(s, b)
    : Sim.modelMag(D.models[s.model], b, (D.night.bjd[j] - s.epoch) / s.period) - 2.5 * Math.log10(s.scale);
  const inBlock = (s) => Math.abs((s.x - D.centre) * D.plate) <= D.half_arcsec && Math.abs((s.y - D.centre) * D.plate) <= D.half_arcsec;

  // ---------------------------------------------------------------- the night
  const seeingAt = (j) => Measure.seeingAt(state.zenith, D.night.airmass[j]);
  const cond = (j) => ({ seeing: seeingAt(j), moon: state.moon });
  const exposureSeed = (j) => Measure.exposureSeed(state.seed, j);
  function simulate(j, b = state.band) {
    const mean = Sim.meanImage(D, stars, b, j, cond(j));
    return { mean, ...Sim.observe(D, mean, exposureSeed(j)) };
  }

  // the displayed block: pixel range and its extent in arcsec (as the movie's extent / limits)
  const LIM = D.half_arcsec + 2;
  const p0 = Math.max(0, Math.floor(D.centre - LIM / D.plate) - 1), p1 = Math.min(D.n, Math.ceil(D.centre + LIM / D.plate) + 1);
  const W = p1 - p0;
  const crop = (img) => { const out = new Float32Array(W * W); for (let r = 0; r < W; r++) out.set(img.subarray((p0 + r) * D.n + p0, (p0 + r) * D.n + p1), r * W); return out; };

  // difference reference: median of the best-seeing exposures, each a full simulated exposure (as the movie)
  let reference = null, refTimer = null;
  function buildReference() {
    const order = [...Array(N).keys()].sort((a, b) => seeingAt(a) - seeingAt(b) || a - b);
    const frames = order.slice(0, Math.max(20, Math.floor(N / 5)));
    const stack = frames.map((j) => crop(simulate(j).image));
    const ref = new Float32Array(W * W), col = new Float32Array(frames.length), h = frames.length >> 1;
    for (let i = 0; i < ref.length; i++) {
      for (let k = 0; k < frames.length; k++) col[k] = stack[k][i];
      col.sort();
      ref[i] = frames.length % 2 ? col[h] : 0.5 * (col[h - 1] + col[h]);
    }
    reference = { image: ref, n: frames.length };
  }
  function referenceChanged() {
    $("status").textContent = "rebuilding the difference reference …";
    clearTimeout(refTimer);
    refTimer = setTimeout(() => { buildReference(); $("status").textContent = ""; displayLimits(); draw(); }, 50);
  }

  // photometry of the whole night (SExtractor, as the movie) in a worker: the same exposures, measured from the playhead
  // onward; any change to the scene or conditions starts it again. Playback waits for exposures not yet measured.
  const worker = new Worker(URL.createObjectURL(new Blob(["sim-src", "sep-src", "measure-src"].map((id) => $(id).textContent + "\n"),
                                                        { type: "text/javascript" })));
  worker.postMessage({ field: $("field-data").textContent });
  let phot = null, gen = 0, lcPending = false;
  const hidden = new Set(); // variables left out of the light-curve panel (legend clicks)
  const view = { lc: null, diff: null }; // the light-curve panels' y ranges once zoomed or panned (null: fitted)
  function measureNight() {
    const b = state.band, targets = variables(), ut = D.night.ut;
    if (phot && phot.band !== b) view.lc = view.diff = null;
    // each model over the night (absolute magnitudes, as the true light curve)
    const modelT = Array.from({ length: 800 }, (_, i) => ut[0] + (i / 799) * (ut[N - 1] - ut[0]));
    const models = targets.map((s) => ({
      true: Array.from({ length: N }, (_, j) => trueMag(s, b, j)),
      curve: modelT.map((t) => Sim.modelMag(D.models[s.model], b, (D.night.bjd[0] + (t - ut[0]) / 24 - s.epoch) / s.period)
                                - 2.5 * Math.log10(s.scale)),
    }));
    phot = { gen: ++gen, band: b, targets, modelT, models, n: 0, measured: new Uint8Array(N), complete: !targets.length,
             mag: targets.map(() => new Float64Array(N).fill(NaN)), err: targets.map(() => new Float64Array(N).fill(NaN)) };
    if (targets.length) worker.postMessage({ gen, start: state.j, band: b, zenith: state.zenith, moon: state.moon, seed: state.seed,
                                             stars: stars.map(({ rate, mag0, ...d }) => d) });
    else worker.postMessage({ stop: true });
    $("measuring").textContent = targets.length ? `measuring the night: 0/${N} exposures` : "";
  }
  worker.onmessage = ({ data: r }) => {
    if (!phot || r.gen !== phot.gen) return;
    if (r.done) { phot.complete = true; $("measuring").textContent = `measured ${N}/${N} exposures`; return; }
    r.mag.forEach((m, t) => { phot.mag[t][r.j] = m; phot.err[t][r.j] = r.err[t]; });
    phot.measured[r.j] = 1;
    phot.n++;
    $("measuring").textContent = `measuring the night: ${phot.n}/${N} exposures`;
    if (!lcPending) { lcPending = true; requestAnimationFrame(() => { lcPending = false; drawLightCurves(); }); }
  };

  // display limits fixed per scene (as the movie): vmax from the best-airmass exposure, asinh width from the sky
  let lim = {};
  function displayLimits() {
    const b = state.band;
    let jbest = 0;
    for (let j = 0; j < N; j++) if (D.night.airmass[j] < D.night.airmass[jbest]) jbest = j;
    const best = crop(Sim.meanImage(D, stars, b, jbest, cond(jbest)));
    let peak = 1;
    for (const v of best) peak = Math.max(peak, v);
    const bgs = [...Array(N).keys()].map((j) => Sim.background(D, b, j, cond(j))).sort((a, c) => a - c);
    const lw = Math.sqrt(bgs[N >> 1] + D.readnoise ** 2);
    // difference colour limit: half the largest variable excursion at its peak pixel
    const si = Sim.seeingIndex(D, seeingAt(jbest)), S = D.psf[b].shape[2];
    let dlim = 0;
    for (const s of variables()) {
      const { k } = Sim.placement(D, s.x, s.y);
      let pk = 0;
      const base = (si * D.psf[b].shape[1] + k) * S * S;
      for (let i = 0; i < S * S; i++) pk = Math.max(pk, D.psf[b][base + i]);
      for (let j = 0; j < N; j += 3) {
        const aw = Sim.airmassWeights(D.airmass_grid, D.night.airmass[j]);
        let meanRate = 0;
        for (let i = 0; i < 200; i++) meanRate += Sim.modelRate(D.models[s.model], b, i / 200, aw) / 200;
        const now = Sim.modelRate(D.models[s.model], b, (D.night.bjd[j] - s.epoch) / s.period, aw);
        dlim = Math.max(dlim, 0.5 * Math.abs(now - meanRate) * s.scale * D.exptime * pk);
      }
    }
    lim = { vmax: Math.min(peak, D.fullwell), lw, dlim: dlim || 50 };
    cmdLimits();
  }

  // ---------------------------------------------------------------- canvases
  function canvas(id) {
    const c = $(id), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    return { ctx, w: r.width, h: r.height };
  }

  const asinh = Math.asinh;
  function colourise(values, map, toUnit) {
    const img = new ImageData(W, W);
    for (let r = 0; r < W; r++) {
      for (let c = 0; c < W; c++) {
        const v = values[r * W + c], o = ((W - 1 - r) * W + c) * 4; // origin lower
        if (!Number.isFinite(v)) { img.data.set([191, 191, 191, 255], o); continue; }
        const t = Math.min(Math.max(toUnit(v), 0), 1), rgb = map[Math.min(255, Math.floor(t * 256))];
        img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
      }
    }
    return img;
  }

  // an image panel: square axes in arcsec, the block's minipix grid, the variables circled
  function imagePanel(id, values, map, toUnit, title, colourbar, ylabels) {
    const { ctx, w, h } = canvas(id);
    const side = Math.min(w - 76, h - 120);
    const box = { x: ylabels ? 62 : 14, y: 34, w: side, h: side };
    const ax = Plot.axes(ctx, box, [-LIM, LIM], [-LIM, LIM], { title, xlabel: "Offset (arcsec)",
      ylabel: ylabels ? "Offset (arcsec)" : null, ylabels, grid: false, ylabelOffset: 46 });
    const off = new OffscreenCanvas(W, W);
    off.getContext("2d").putImageData(colourise(values, map, toUnit), 0, 0);
    // pixel p spans [(p - centre) * plate, (p + 1 - centre) * plate]
    const a0 = (p0 - D.centre) * D.plate, a1 = (p1 - D.centre) * D.plate;
    ax.clip(() => {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, ax.X(a0), ax.Y(a1), ax.X(a1) - ax.X(a0), ax.Y(a0) - ax.Y(a1));
      ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 0.5;
      for (let v = -D.half_arcsec; v <= D.half_arcsec + 1e-6; v += D.minipix_arcsec) {
        ctx.beginPath(); ctx.moveTo(ax.X(v), ax.Y(-D.half_arcsec)); ctx.lineTo(ax.X(v), ax.Y(D.half_arcsec)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ax.X(-D.half_arcsec), ax.Y(v)); ctx.lineTo(ax.X(D.half_arcsec), ax.Y(v)); ctx.stroke();
      }
      if (id === "exposure") {
        const s = D.minipix_arcsec / 2;
        ctx.strokeStyle = "#7fd1ff"; ctx.lineWidth = 1.2;
        ctx.strokeRect(ax.X(-s), ax.Y(s), ax.X(s) - ax.X(-s), ax.Y(-s) - ax.Y(s));
      }
      ctx.font = 'bold 10px "DejaVu Sans", sans-serif';
      for (const s of variables()) {
        const dx = (s.x - D.centre) * D.plate, dy = (s.y - D.centre) * D.plate, col = TAB10[s.colourIdx];
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(ax.X(dx), ax.Y(dy), ax.X(3) - ax.X(0), 0, 2 * Math.PI); ctx.stroke();
        ctx.fillText(s.label, ax.X(dx + 3.2), ax.Y(dy + 3.2));
      }
      if (state.selected !== null && stars[state.selected]) {
        const s = stars[state.selected];
        ctx.strokeStyle = "#00e5ff"; ctx.setLineDash([3, 2]); ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(ax.X((s.x - D.centre) * D.plate), ax.Y((s.y - D.centre) * D.plate), ax.X(5) - ax.X(0), 0, 2 * Math.PI); ctx.stroke();
      }
    });
    ax.frame();
    colourbar(ctx, { x: box.x, y: box.y + box.h + 50, w: box.w, h: 12 });
    return ax;
  }

  function colourbarStrip(ctx, box, map, ticks, label) {
    for (let i = 0; i < box.w; i++) {
      const rgb = map[Math.min(255, Math.floor((i / box.w) * 256))];
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(box.x + i, box.y, 1.5, box.h);
    }
    ctx.fillStyle = Plot.STYLE.tick; ctx.font = '10px "DejaVu Sans", sans-serif'; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const [t, text] of ticks) {
      ctx.textAlign = t === 0 ? "left" : t === 1 ? "right" : "center";
      ctx.fillText(text, box.x + t * box.w, box.y + box.h + 3);
    }
    ctx.textAlign = "center";
    ctx.fillStyle = Plot.STYLE.label; ctx.font = '12px "DejaVu Sans", sans-serif';
    ctx.fillText(label, box.x + box.w / 2, box.y + box.h + 17);
  }

  // ---------------------------------------------------------------- panels
  let cmdLim = null;
  function cmdLimits() {
    const b = state.band, cs = [], ms = [];
    for (const s of stars) {
      if (!inBlock(s)) continue;
      if (s.model === undefined) { if (s.gaia !== undefined && D.stars.G[s.gaia] >= 20) continue; cs.push(meanMag(s, "beta") - meanMag(s, "rho")); ms.push(meanMag(s, b)); }
      else for (const [c, m] of cycle(s, b)) { cs.push(c); ms.push(m); }
    }
    const sorted = [...cs].sort((a, c) => a - c), pct = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p / 100 * sorted.length))];
    cmdLim = { x: [pct(0.5) - 0.15, pct(99.5) + 0.15], y: [Math.min(20.2, Math.max(...ms) + 0.3), Math.min(...ms) - 0.5] };
  }
  function cycle(s, b) { // [beta - rho, m_b] over one cycle
    const out = [];
    for (let i = 0; i <= 120; i++) {
      const ph = i / 120, mb = Sim.modelMag(D.models[s.model], "beta", ph), mr = Sim.modelMag(D.models[s.model], "rho", ph);
      out.push([mb - mr, (b === "beta" ? mb : mr) - 2.5 * Math.log10(s.scale)]);
    }
    return out;
  }

  function drawCMD() {
    const { ctx, w, h } = canvas("cmd"), b = state.band, j = state.j;
    const ax = Plot.axes(ctx, { x: 60, y: 40, w: w - 80, h: h - 100 }, cmdLim.x, cmdLim.y,
      { title: "CMD", xlabel: "β − ρ (AB mag)", ylabel: `m${SYMBOL[b]} (AB mag)` });
    ax.frame();
    const cs = [], ms = [];
    for (const s of stars) {
      if (s.model !== undefined || !inBlock(s) || (s.gaia !== undefined && D.stars.G[s.gaia] >= 20)) continue;
      cs.push(meanMag(s, "beta") - meanMag(s, "rho")); ms.push(meanMag(s, b));
    }
    ax.points(cs, ms, { color: "#b9c4cc", r: 2 });
    for (const s of variables()) {
      if (!inBlock(s)) continue;
      const c = cycle(s, b), col = TAB10[s.colourIdx];
      ax.line(c.map((p) => p[0]), c.map((p) => p[1]), { color: col, lw: 1.3, alpha: 0.9 });
      ax.points([trueMag(s, "beta", j) - trueMag(s, "rho", j)], [trueMag(s, b, j)], { color: col, r: 6, edge: "#2b2b2b" });
    }
    ctx.fillStyle = "#b9c4cc"; ctx.beginPath(); ctx.arc(ax.box.x + 16, ax.box.y + ax.box.h - 16, 2.5, 0, 7); ctx.fill();
    ctx.fillStyle = Plot.STYLE.label; ctx.font = '11px "DejaVu Sans", sans-serif'; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText("Constant stars", ax.box.x + 26, ax.box.y + ax.box.h - 16);
  }

  // light curves of the variables (SExtractor MAG_AUTO): measured magnitudes up to this exposure with the model's true
  // magnitudes behind, absolute (calibrated as photometry.py does); below, measured - true. Ranges fitted to the stars
  // shown unless zoomed or panned (wheel, drag; double-click fits again).
  let lcLayout = null;
  function drawLightCurves() {
    const { ctx, w, h } = canvas("lc"), j = state.j, ut = D.night.ut, xl = [ut[0] - 0.1, ut[N - 1] + 0.1], B = SYMBOL[state.band];
    const P = phot && phot.band === state.band ? phot : { targets: [], models: [], mag: [], err: [], modelT: [] };
    const shown = P.targets.map((s) => !hidden.has(s));
    let lo = Infinity, hi = -Infinity, dlo = 0, dhi = 0;
    P.targets.forEach((_, t) => {
      if (!shown[t]) return;
      P.mag[t].forEach((v, q) => {
        if (!Number.isFinite(v)) return;
        lo = Math.min(lo, v); hi = Math.max(hi, v);
        const d = v - P.models[t].true[q];
        dlo = Math.min(dlo, d); dhi = Math.max(dhi, d);
      });
      for (const v of P.models[t].true) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    });
    if (!(hi >= lo)) { lo = 14.7; hi = 15.3; }
    const pad = 0.08 * Math.max(hi - lo, 0.05), dm = 0.05 * Math.max(dhi - dlo, 0.02);
    const fit = { lc: [hi + pad, lo - 2.5 * pad], diff: [dhi + dm, dlo - dm] };
    const top = 36, gap = 10, H = h - 92 - gap, box = { x: 76, y: top, w: w - 96, h: H * 2 / 3 };
    const box2 = { x: box.x, y: top + box.h + gap, w: box.w, h: H / 3 };
    const ax = Plot.axes(ctx, box, xl, view.lc || fit.lc,
      { title: "Light curves (SExtractor MAG_AUTO)", titleRight: `UT ${hhmm(ut[j])} · airmass ${D.night.airmass[j].toFixed(2)} · seeing ${seeingAt(j).toFixed(2)}″`,
        ylabel: `m${B} (AB mag)`, xlabels: false, ylabelOffset: 52 });
    ax.frame();
    const ax2 = Plot.axes(ctx, box2, xl, view.diff || fit.diff, { xlabel: `UT hours (${D.date} night)`, ylabel: "measured − true\n(mag)", ylabelOffset: 52 });
    ax2.frame();
    for (const a of [ax, ax2]) a.line([ut[j], ut[j]], [-100, 100], { color: "#455868", lw: 0.8, alpha: 0.35 });
    ax2.line(xl, [0, 0], { color: "#808080", lw: 0.6, dash: [1, 2] });
    P.targets.forEach((s, t) => {
      if (!shown[t]) return;
      const col = TAB10[s.colourIdx];
      ax.line(P.modelT, P.models[t].curve, { color: col, lw: 1.0, alpha: 0.35 });
      const xs = [], ys = [], d = [];
      for (let q = 0; q <= j; q++) {
        d.push(P.mag[t][q] - P.models[t].true[q]);
        if (!Number.isFinite(P.mag[t][q])) continue;
        const y = P.mag[t][q], e = P.err[t][q];
        xs.push(ut[q]); ys.push(y);
        ax.line([ut[q], ut[q]], [y - e, y + e], { color: col, lw: 0.5 });
      }
      ax.points(xs, ys, { color: col, r: 1.8 });
      ax2.line(ut.slice(0, j + 1), d, { color: col, lw: 0.8 });
    });
    // legend (as the movie: upper left, four columns); click an entry to show or hide that star
    const kinds = new Set(P.targets.map((s) => D.models[s.model].kind));
    const modelName = [...(([...kinds].some((k) => k !== "EB")) ? ["MESA RSP"] : []), ...(kinds.has("EB") ? ["PHOEBE"] : [])].join(" / ");
    const entries = P.targets.map((s, t) => ({ text: `${s.label} ${D.models[s.model].kind} ${s.period.toFixed(3)} d`, col: TAB10[s.colourIdx],
                                              point: true, star: s, off: !shown[t] }));
    if (entries.length) entries.push({ text: `${modelName} model`, col: "#808080", point: false });
    lcLayout = { axes: { lc: ax, diff: ax2 }, fit, legend: legend(ctx, box, entries, 4) };
  }
  function legend(ctx, box, entries, ncol) {
    if (!entries.length) return [];
    ctx.save();
    ctx.font = '10px "DejaVu Sans", sans-serif';
    const colW = [], rows = Math.ceil(entries.length / ncol), lh = 14;
    entries.forEach((e, i) => { const c = Math.floor(i / rows); colW[c] = Math.max(colW[c] || 0, 22 + ctx.measureText(e.text).width); });
    const W = colW.reduce((a, v) => a + v + 10, 0) + 4, H = rows * lh + 6, x0 = box.x + 6, y0 = box.y + 6;
    ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(x0, y0, W, H);
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    const rects = [];
    entries.forEach((e, i) => {
      const c = Math.floor(i / rows), r = i % rows, x = x0 + 4 + colW.slice(0, c).reduce((a, v) => a + v + 10, 0), y = y0 + 3 + r * lh + lh / 2;
      rects.push({ x, y: y - lh / 2, w: colW[c], h: lh, star: e.star });
      ctx.globalAlpha = e.off ? 0.3 : 1;
      ctx.strokeStyle = e.col; ctx.fillStyle = e.col;
      if (e.point) { ctx.beginPath(); ctx.arc(x + 8, y, 2.5, 0, 2 * Math.PI); ctx.fill(); }
      else { ctx.globalAlpha = 0.5; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 16, y); ctx.stroke(); ctx.globalAlpha = 1; }
      ctx.fillStyle = Plot.STYLE.label; ctx.fillText(e.text, x + 20, y);
      ctx.globalAlpha = 1;
    });
    ctx.restore();
    return rects;
  }

  // light-curve panel: legend clicks, wheel zoom and drag pan of each y axis, double-click to fit again
  function lcHit(ev) {
    const r = $("lc").getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
    if (!lcLayout) return { px, py };
    const entry = lcLayout.legend.find((e) => e.star && px >= e.x && px <= e.x + e.w && py >= e.y && py <= e.y + e.h);
    const key = Object.keys(lcLayout.axes).find((k) => { const b = lcLayout.axes[k].box; return px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h; });
    return { px, py, entry, key };
  }
  const yRange = (key) => view[key] || lcLayout.fit[key];
  $("lc").addEventListener("wheel", (ev) => {
    const { py, key } = lcHit(ev);
    if (!key) return;
    ev.preventDefault();
    const b = lcLayout.axes[key].box, [y0, y1] = yRange(key), v = y0 + ((b.y + b.h - py) / b.h) * (y1 - y0), f = Math.exp(ev.deltaY * 0.002);
    view[key] = [v + (y0 - v) * f, v + (y1 - v) * f];
    drawLightCurves();
  }, { passive: false });
  let lcDrag = null;
  $("lc").addEventListener("mousedown", (ev) => {
    const { py, entry, key } = lcHit(ev);
    if (entry) { if (hidden.has(entry.star)) hidden.delete(entry.star); else hidden.add(entry.star); drawLightCurves(); return; }
    if (key) lcDrag = { key, py, range: yRange(key) };
  });
  window.addEventListener("mousemove", (ev) => {
    if (!lcDrag) return;
    const r = $("lc").getBoundingClientRect(), b = lcLayout.axes[lcDrag.key].box, [y0, y1] = lcDrag.range;
    const shift = ((ev.clientY - r.top - lcDrag.py) / b.h) * (y1 - y0);
    view[lcDrag.key] = [y0 + shift, y1 + shift];
    drawLightCurves();
  });
  window.addEventListener("mouseup", () => { lcDrag = null; });
  $("lc").addEventListener("dblclick", (ev) => { const { key } = lcHit(ev); if (key) { view[key] = null; drawLightCurves(); } });
  $("lc").addEventListener("mousemove", (ev) => { const { entry, key } = lcHit(ev); $("lc").style.cursor = entry ? "pointer" : key ? "ns-resize" : ""; });

  function drawSED() {
    const { ctx, w, h } = canvas("sed"), j = state.j, X = D.night.airmass[j];
    const vars = variables();
    const curves = vars.map((s) => sedAt(s, j));
    let hi = 1e-30, lo = 1e30;
    for (const c of curves) for (const v of c.f) if (v > 0) { hi = Math.max(hi, v); lo = Math.min(lo, v); }
    if (!curves.length) { hi = 1e-14; lo = 1e-16; }
    const ax = Plot.axes(ctx, { x: 70, y: 36, w: w - 130, h: h - 92 }, [300, 1000], [lo / 1.5, hi * 1.5],
      { logy: true, title: "SED", xlabel: "Wavelength (nm)", ylabel: "f_λ at Argus (erg s⁻¹ cm⁻² Å⁻¹)", ylabelOffset: 52 });
    // detected response of both bands on a twin axis (0-1)
    const rw = D.response_wave_nm, resp = {};
    for (const b of D.bands) {
      const base = D.response[b].base, atm = D.response[b].atmosphere;
      resp[b] = Array.from(base, (v, i) => v * atm[i] ** X);
      ax.clip(() => {
        ctx.fillStyle = b === "beta" ? "rgba(32,120,180,0.10)" : "rgba(204,89,64,0.10)";
        ctx.beginPath(); ctx.moveTo(ax.X(rw[0]), ax.box.y + ax.box.h);
        for (let i = 0; i < rw.length; i++) ctx.lineTo(ax.X(rw[i]), ax.box.y + ax.box.h * (1 - resp[b][i]));
        ctx.lineTo(ax.X(rw[rw.length - 1]), ax.box.y + ax.box.h); ctx.fill();
      });
    }
    ax.frame();
    vars.forEach((s, i) => {
      const col = TAB10[s.colourIdx], c = curves[i];
      ax.line(c.w, c.f, { color: col, lw: 0.9 });
      for (const b of D.bands) { // the SED as each band records it
        const seen = Array.from(rw, (wv, k) => interp(c.w, c.f, wv) * resp[b][k]);
        ax.line(rw, seen.map((v) => (v > 0 ? v : NaN)), { color: col, lw: 0.9, dash: [1.5, 2] });
      }
    });
    // right axis: detected response
    ctx.fillStyle = Plot.STYLE.tick; ctx.font = '11px "DejaVu Sans", sans-serif'; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) ctx.fillText(t.toFixed(1), ax.box.x + ax.box.w + 6, ax.box.y + ax.box.h * (1 - t));
    ctx.save(); ctx.translate(ax.box.x + ax.box.w + 44, ax.box.y + ax.box.h / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.fillStyle = Plot.STYLE.label; ctx.font = '12px "DejaVu Sans", sans-serif'; ctx.fillText("Detected response", 0, 0); ctx.restore();
    ctx.strokeStyle = Plot.STYLE.edge; ctx.beginPath(); ctx.moveTo(ax.box.x + ax.box.w, ax.box.y); ctx.lineTo(ax.box.x + ax.box.w, ax.box.y + ax.box.h); ctx.stroke();
  }
  // a variable's SED at exposure j: linear between the stored SED steps (display only), times its scale
  function sedAt(s, j) {
    const m = D.models[s.model], ph = (((D.night.bjd[j] - s.epoch) / s.period) % 1 + 1) % 1, P = m.sed_phase, nw = m.sed_wave_nm.length;
    let i = P.findIndex((p) => p > ph);
    let ia, ib, pa, pb;
    if (i === -1) { ia = P.length - 1; ib = 0; pa = P[ia]; pb = P[0] + 1; }
    else if (i === 0) { ia = P.length - 1; ib = 0; pa = P[ia] - 1; pb = P[0]; }
    else { ia = i - 1; ib = i; pa = P[ia]; pb = P[ib]; }
    const t = pb === pa ? 0 : (ph - pa) / (pb - pa);
    const f = new Float64Array(nw);
    for (let k = 0; k < nw; k++) f[k] = s.scale * ((1 - t) * m.sed[ia * nw + k] + t * m.sed[ib * nw + k]);
    return { w: m.sed_wave_nm, f };
  }
  function interp(xs, ys, x) {
    if (x < xs[0] || x > xs[xs.length - 1]) return 0;
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] > x) hi = mid; else lo = mid; }
    const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }
  const hhmm = (h) => { const m = Math.round(h * 60); return `${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`; };

  let exposureAxes = null;
  function draw() {
    const j = state.j, b = state.band, obs = simulate(j);
    const img = crop(obs.image);
    const den = asinh(lim.vmax / lim.lw);
    exposureAxes = imagePanel("exposure", img, CMAP.magma, (v) => asinh(v / lim.lw) / den, `Exposure (${SYMBOL[b]})`,
      (ctx, box) => colourbarStrip(ctx, box, CMAP.magma, [0, 0.25, 0.5, 0.75, 1].map((t) => [t, Math.round(lim.lw * Math.sinh(t * den)).toString()]), "electrons / pixel"), true);
    const diff = img.map((v, i) => v - reference.image[i]);
    imagePanel("difference", diff, CMAP.RdBu_r, (v) => 0.5 + v / (2 * lim.dlim), "Difference (image − reference)",
      (ctx, box) => colourbarStrip(ctx, box, CMAP.RdBu_r, [0, 0.25, 0.5, 0.75, 1].map((t) => [t, Math.round((2 * t - 1) * lim.dlim).toString().replace("-", "−")]), "difference (electrons / pixel)"), false);
    drawCMD();
    drawLightCurves();
    drawSED();
    $("clock").textContent = `exposure ${j + 1}/${N} · UT ${hhmm(D.night.ut[j])}`;
    $("conditions").textContent = `airmass ${D.night.airmass[j].toFixed(2)} · seeing ${seeingAt(j).toFixed(2)}″ · ` +
      `Moon ${Math.round(D.night.moon_illum[j] * 100)}% lit, altitude ${Math.round(D.night.moon_alt[j])}° · ` +
      `sky ${(Sim.background(D, b, j, cond(j)) / D.exptime - D.dark).toFixed(2)} e/s/pixel (${SYMBOL[b]})`;
    $("exposure-slider").value = j;
    showSelected();
    document.getElementById("title").textContent = title();
  }

  function title() {
    const kinds = variables().map((s) => D.models[s.model].kind);
    const puls = kinds.filter((k) => k !== "EB"), eb = kinds.length - puls.length, parts = [];
    if (puls.length) parts.push(`${puls.length} pulsator${puls.length !== 1 ? "s" : ""} (${[...new Set(puls)].join(", ")}): MESA RSP`);
    if (eb) parts.push(`${eb} eclipsing binar${eb !== 1 ? "ies" : "y"}: PHOEBE`);
    return `ARGUS — ${D.centre_label} (${D.name}) — ${D.block_n} × ${D.block_n} minipixes (${Math.round(2 * D.half_arcsec)}″) — ${D.date} — 60 s exposures — ${parts.join(" · ") || "no modelled variables"}`;
  }

  // ---------------------------------------------------------------- interaction
  function showSelected() {
    const s = stars[state.selected];
    if (!s) { $("selected").innerHTML = "click a star in the exposure"; return; }
    const b = state.band, kind = s.model !== undefined ? D.models[s.model].kind : "constant";
    const name = s.gaia !== undefined ? `Gaia DR3 ${s.source}` : "added star";
    const magLabel = s.model !== undefined ? "mean ρ" : "G";
    const magValue = s.model !== undefined ? meanMag(s, "rho") : (s.gaia !== undefined ? D.stars.G[s.gaia] : s.G) - 2.5 * Math.log10(s.scale);
    $("selected").innerHTML = `${s.label ? s.label + " · " : ""}${name} · ${kind}${s.model !== undefined ? ` · P ${s.period.toFixed(5)} d` : ""}<br>` +
      `m${SYMBOL[b]} ${trueMag(s, b, state.j).toFixed(3)} now · BP−RP ${Number.isFinite(s.colour) ? s.colour.toFixed(2) : "–"}<br>` +
      `<label>${magLabel} <input type="number" id="sel-mag" step="0.1" value="${magValue.toFixed(2)}"><button id="sel-delete">Delete</button></label>`;
    $("sel-mag").onchange = (e) => {
      s.scale *= 10 ** (-0.4 * (Number(e.target.value) - magValue));
      sceneChanged();
    };
    $("sel-delete").onclick = () => { stars.splice(state.selected, 1); state.selected = null; labelVariables(); sceneChanged(); };
  }

  function sceneChanged() { displayLimits(); referenceChanged(); measureNight(); draw(); }

  function toPixel(ev) {
    const r = $("exposure").getBoundingClientRect(), ax = exposureAxes;
    const px = ev.clientX - r.left, py = ev.clientY - r.top;
    const dx = ax.xlim[0] + ((px - ax.box.x) / ax.box.w) * (ax.xlim[1] - ax.xlim[0]);
    const dy = ax.ylim[0] + ((ax.box.y + ax.box.h - py) / ax.box.h) * (ax.ylim[1] - ax.ylim[0]);
    return { x: D.centre + dx / D.plate, y: D.centre + dy / D.plate, inside: Math.abs(dx) <= LIM && Math.abs(dy) <= LIM };
  }
  let dragging = null;
  $("exposure").addEventListener("mousedown", (ev) => {
    const p = toPixel(ev);
    if (!p.inside) return;
    if (state.mode === "select") {
      let best = null, bd = 4 / D.plate; // within 4"
      stars.forEach((s, i) => { const d = Math.hypot(s.x - p.x, s.y - p.y); if (d < bd) { bd = d; best = i; } });
      state.selected = best;
      dragging = best;
      draw();
    } else if (state.mode === "constant") {
      addConstant(p.x, p.y, Number($("new-g").value), Number($("new-colour").value));
    } else {
      addVariable(p.x, p.y, Number($("new-model").value), Number($("new-mag").value));
    }
  });
  window.addEventListener("mousemove", (ev) => {
    if (dragging === null) return;
    const p = toPixel(ev);
    stars[dragging].x = p.x; stars[dragging].y = p.y;
    draw();
  });
  window.addEventListener("mouseup", () => { if (dragging !== null) { dragging = null; sceneChanged(); } });

  function addConstant(x, y, G, colour) {
    const t = D.templates, i = nearestIndex(t.bp_rp, colour);
    stars.push({ x, y, G, template: i, colour: t.bp_rp[i], scale: 10 ** (-0.4 * (G - t.G)),
      rate: Object.fromEntries(D.bands.map((b) => [b, t.rate[b].subarray(i * NA, (i + 1) * NA)])),
      mag0: Object.fromEntries(D.bands.map((b) => [b, t.mag[b][i]])) });
    state.selected = stars.length - 1;
    sceneChanged();
  }
  function addVariable(x, y, model, meanRho) {
    const m = D.models[model], s = { x, y, model, scale: 1, period: m.period, epoch: D.night.bjd[0] - Math.random() * m.period, colour: NaN };
    s.scale = 10 ** (-0.4 * (meanRho - meanMag(s, "rho")));
    stars.push(s);
    labelVariables();
    state.selected = stars.length - 1;
    sceneChanged();
  }
  const nearestIndex = (arr, v) => arr.reduce((best, a, i) => (Math.abs(a - v) < Math.abs(arr[best] - v) ? i : best), 0);

  // ---------------------------------------------------------------- controls
  $("exposure-slider").max = N - 1;
  $("exposure-slider").oninput = (e) => {
    state.j = Number(e.target.value);
    if (phot && !phot.complete) worker.postMessage({ gen: phot.gen, from: state.j });
    draw();
  };
  $("band").onchange = (e) => { state.band = e.target.value; sceneChanged(); };
  $("seeing").value = state.zenith;
  const seeingOut = () => { $("seeing-out").textContent = `${state.zenith.toFixed(2)}″`; };
  $("seeing").oninput = (e) => { state.zenith = Number(e.target.value); seeingOut(); };
  $("seeing").onchange = () => sceneChanged();
  const moonOut = () => { $("moon-out").textContent = `${state.moon}`; };
  $("moon").oninput = (e) => { state.moon = Number(e.target.value); moonOut(); };
  $("moon").onchange = () => sceneChanged();
  const speedOut = () => { $("speed-out").textContent = `${state.fps} exp/s`; };
  $("speed").oninput = (e) => { state.fps = Number(e.target.value); speedOut(); };
  for (const mode of ["select", "constant", "variable"]) {
    $(`mode-${mode}`).onclick = () => {
      state.mode = mode;
      for (const m of ["select", "constant", "variable"]) $(`mode-${m}`).classList.toggle("on", m === mode);
    };
  }
  D.models.forEach((m, i) => { if (m.library) $("new-model").add(new Option(`${m.kind} (P ${m.period.toFixed(3)} d)`, i)); });
  $("reset").onclick = () => { resetScene(); sceneChanged(); };
  let last = 0;
  function tick(t) {
    if (!state.playing) return;
    const next = (state.j + 1) % N;
    if (t - last >= 1000 / state.fps && (!phot || phot.complete || phot.measured[next])) { last = t; state.j = next; draw(); }
    requestAnimationFrame(tick);
  }
  $("play").onclick = () => {
    state.playing = !state.playing;
    $("play").textContent = state.playing ? "⏸ Pause" : "▶ Play";
    if (state.playing) requestAnimationFrame(tick);
  };
  window.addEventListener("resize", () => draw());

  resetScene();
  seeingOut(); moonOut(); speedOut();
  displayLimits();
  buildReference();
  measureNight();
  draw();
})();
