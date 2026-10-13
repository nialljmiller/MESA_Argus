// The Argus exposure simulator of the interactive page: a port of night_movie.Scene.observe.
// Runs in the browser and in node (demo/validate/*.mjs). All physics tables come from demo/export_demo.py.

const Sim = (() => {
  // ---------------------------------------------------------------- data
  async function unpack(p) {
    const bytes = Uint8Array.from(atob(p.data), (c) => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    const buf = await new Response(stream).arrayBuffer();
    const a = new Float32Array(buf);
    a.shape = p.shape;
    return a;
  }

  // every {shape, data} object in the export -> Float32Array (with .shape), in place
  async function unpackAll(obj) {
    for (const [k, v] of Object.entries(obj)) {
      if (v && typeof v === "object" && "shape" in v && "data" in v) obj[k] = await unpack(v);
      else if (v && typeof v === "object") await unpackAll(v);
    }
    return obj;
  }

  // ---------------------------------------------------------------- random numbers (seeded per exposure)
  // xoshiro128** seeded with splitmix32
  function rng(seed) {
    let s = seed >>> 0;
    const split = () => {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    let a = split(), b = split(), c = split(), d = split();
    const next = () => {
      const r = Math.imul(rotl(Math.imul(b, 5), 7), 9) >>> 0;
      const t = b << 9;
      c ^= a; d ^= b; b ^= c; a ^= d; c ^= t; d = rotl(d, 11);
      return r;
    };
    const uniform = () => (next() + 0.5) / 4294967296; // (0, 1)
    let spare = null;
    const normal = () => { // Box-Muller
      if (spare !== null) { const v = spare; spare = null; return v; }
      const u = uniform(), v = uniform(), r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return { uniform, normal, poisson: (mean) => poisson(mean, uniform) };
  }
  const rotl = (x, k) => ((x << k) | (x >>> (32 - k))) >>> 0;

  // Poisson deviates: multiplication method below 10, PTRS above (Hormann 1993, transformed rejection; exact)
  function poisson(mean, uniform) {
    if (mean <= 0) return 0;
    if (mean < 10) {
      const limit = Math.exp(-mean);
      let k = 0, p = uniform();
      while (p > limit) { k++; p *= uniform(); }
      return k;
    }
    const slam = Math.sqrt(mean), loglam = Math.log(mean);
    const b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
    const invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
      const U = uniform() - 0.5, V = uniform();
      const us = 0.5 - Math.abs(U);
      const k = Math.floor((2 * a / us + b) * U + mean + 0.43);
      if (us >= 0.07 && V <= vr) return k;
      if (k < 0 || (us < 0.013 && V > us)) continue;
      if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -mean + k * loglam - logFactorial(k)) return k;
    }
  }
  function logFactorial(k) {
    if (k < 2) return 0;
    if (k < 256) { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; }
    const x = k + 1; // Stirling series for log Gamma(k + 1)
    return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI) + 1 / (12 * x) - 1 / (360 * x ** 3);
  }

  // ---------------------------------------------------------------- the scene
  // linear interpolation on the airmass grid, clipped to it (as night_movie.main rates_at)
  function airmassWeights(grid, x) {
    x = Math.min(Math.max(x, grid[0]), grid[grid.length - 1]);
    let i1 = grid.findIndex((g) => g >= x);
    i1 = Math.min(Math.max(i1, 1), grid.length - 1);
    const f = (x - grid[i1 - 1]) / (grid[i1] - grid[i1 - 1]);
    return [i1 - 1, i1, f];
  }

  // a model's band flux table (steps x airmass) at phase: linear between the bracketing steps (as RSPStar.sed)
  function modelRate(model, band, phase, aw) {
    const ph = model.phase, n = ph.length, nA = model.rate[band].shape[1], r = model.rate[band];
    phase = ((phase % 1) + 1) % 1;
    let lo = 0, hi = n; // first step with ph >= phase
    while (lo < hi) { const mid = (lo + hi) >> 1; if (ph[mid] < phase) lo = mid + 1; else hi = mid; }
    let ia, ib, pa, pb;
    if (lo === 0) { ia = n - 1; ib = 0; pa = ph[n - 1] - 1; pb = ph[0]; }
    else if (lo === n) { ia = n - 1; ib = 0; pa = ph[n - 1]; pb = ph[0] + 1; }
    else { ia = lo - 1; ib = lo; pa = ph[lo - 1]; pb = ph[lo]; }
    const w = pb === pa ? 0 : (phase - pa) / (pb - pa);
    const at = (i) => r[i * nA + aw[0]] * (1 - aw[2]) + r[i * nA + aw[1]] * aw[2];
    return (1 - w) * at(ia) + w * at(ib);
  }

  // the same interpolation on the above-atmosphere magnitudes (in flux), for the true light curves
  function modelMag(model, band, phase) {
    const ph = model.phase, n = ph.length, m = model.mag[band];
    phase = ((phase % 1) + 1) % 1;
    let lo = 0, hi = n;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (ph[mid] < phase) lo = mid + 1; else hi = mid; }
    let ia, ib, pa, pb;
    if (lo === 0) { ia = n - 1; ib = 0; pa = ph[n - 1] - 1; pb = ph[0]; }
    else if (lo === n) { ia = n - 1; ib = 0; pa = ph[n - 1]; pb = ph[0] + 1; }
    else { ia = lo - 1; ib = lo; pa = ph[lo - 1]; pb = ph[lo]; }
    const w = pb === pa ? 0 : (phase - pa) / (pb - pa);
    return -2.5 * Math.log10((1 - w) * 10 ** (-0.4 * m[ia]) + w * 10 ** (-0.4 * m[ib]));
  }

  // star position (pixels) -> PSF stamp origin and sub-pixel phase (as Scene.__init__)
  function placement(D, x, y) {
    const P = D.psf_phases, [c0y, c0x] = D.c0;
    const oy = Math.ceil(y - c0y), ox = Math.ceil(x - c0x);
    const clip = (v) => Math.min(Math.max(v, 0), P - 1);
    const py = clip(Math.round((oy + c0y - y) * P)), px = clip(Math.round((ox + c0x - x) * P));
    return { oy, ox, k: py * P + px };
  }

  // nearest seeing on the PSF grid
  function seeingIndex(D, seeing) {
    let best = 0;
    D.seeing_grid.forEach((s, i) => { if (Math.abs(s - seeing) < Math.abs(D.seeing_grid[best] - seeing)) best = i; });
    return best;
  }

  // electrons/s of every star in exposure j: stars = [{x, y, rate: {band: Float32Array(airmass)} | model, ...}]
  function starRate(D, star, band, j, airmass) {
    const aw = airmassWeights(D.airmass_grid, airmass);
    if (star.model !== undefined) {
      const m = D.models[star.model];
      const phase = (D.night.bjd[j] - star.epoch) / star.period;
      return star.scale * modelRate(m, band, phase, aw);
    }
    const r = star.rate[band];
    return star.scale * (r[aw[0]] * (1 - aw[2]) + r[aw[1]] * aw[2]);
  }

  // the noiseless mean image (electrons) of exposure j: sky + dark + every star's PSF (as Scene.observe's mean)
  function meanImage(D, stars, band, j, cond) {
    const n = D.n, out = new Float64Array(n * n);
    out.fill(background(D, band, j, cond));
    const si = seeingIndex(D, cond.seeing), psf = D.psf[band], S = psf.shape[2], stampSize = S * S;
    const airmass = D.night.airmass[j];
    for (const st of stars) {
      const flux = starRate(D, st, band, j, airmass) * D.exptime;
      if (!(flux > 0)) continue;
      const { oy, ox, k } = placement(D, st.x, st.y);
      const base = (si * psf.shape[1] + k) * stampSize;
      for (let r = 0; r < S; r++) {
        const yy = oy + r;
        if (yy < 0 || yy >= n) continue;
        for (let c = 0; c < S; c++) {
          const xx = ox + c, p = psf[base + r * S + c];
          if (xx < 0 || xx >= n || !(p > 0)) continue;
          out[yy * n + xx] += flux * p;
        }
      }
    }
    return out;
  }

  // sky + dark electrons per pixel in exposure j; cond.moon scales ArgusSim's Moon term
  function background(D, band, j, cond) {
    const s = D.night.sky[band];
    const sky = s.airglow[j] + s.zodiacal[j] + s.starlight[j] + cond.moon * s.moon[j] + s.artif[j];
    return (sky + D.dark) * D.exptime;
  }

  // one exposure: Poisson(mean) clipped at full well, plus read noise (as Scene.observe)
  function observe(D, mean, seed) {
    const g = rng(seed), image = new Float32Array(mean.length), full = new Uint8Array(mean.length);
    for (let i = 0; i < mean.length; i++) {
      const charge = Math.min(g.poisson(mean[i]), D.fullwell);
      full[i] = charge >= D.fullwell;
      image[i] = charge + g.normal() * D.readnoise;
    }
    return { image, full };
  }

  return { unpack, unpackAll, rng, poisson, airmassWeights, modelRate, modelMag, placement, seeingIndex, starRate,
           meanImage, background, observe };
})();

if (typeof module !== "undefined") module.exports = Sim;
