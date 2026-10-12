// The page's night photometry, as m3_night_movie measures it: the catalogue a pipeline would calibrate against, the
// FWHM it matches with, and each exposure through SEP.Photometry. The page runs it in a Web Worker (from the inlined
// sim.js, sep.js and this file); demo/validate/sep.mjs uses the same catalogue and FWHM.

const Measure = (() => {
  // ArgusSim seeing_at_airmass, and each exposure's noise seed (the page and the worker draw identical exposures)
  const seeingAt = (zenith, airmass) => zenith * airmass ** 0.6;
  const exposureSeed = (seed, j) => (Math.imul(seed, 1000003) + j) >>> 0;

  // the page's stars without their table views ({x, y, scale, colour, gaia | template | model, epoch, period})
  // -> stars as Sim.meanImage takes them
  function stars(D, list) {
    const NA = D.airmass_grid.length;
    return list.map((d) => {
      const s = { ...d };
      if (d.model !== undefined) return s;
      const src = d.gaia !== undefined ? D.stars : D.templates, i = d.gaia ?? d.template;
      s.rate = Object.fromEntries(D.bands.map((b) => [b, src.rate[b].subarray(i * NA, (i + 1) * NA)]));
      s.mag0 = Object.fromEntries(D.bands.map((b) => [b, src.mag[b][i]]));
      return s;
    });
  }

  // catalogue inputs as the movie builds them: magnitude (variables: intensity mean), flux at airmass 1 (variables: mean
  // over the night's exposures), Gaia BP-RP; the variables are the targets
  function catalogue(D, stars, band) {
    const mag = [], flux = [], colour = [], targets = [];
    stars.forEach((s, i) => {
      colour.push(s.colour);
      if (s.model === undefined) {
        mag.push(s.mag0[band] - 2.5 * Math.log10(s.scale));
        flux.push(s.scale * s.rate[band][0] * D.exptime);
      } else {
        const m = D.models[s.model];
        let fm = 0;
        for (let q = 0; q < 200; q++) fm += 10 ** (-0.4 * Sim.modelMag(m, band, q / 200)) / 200;
        mag.push(-2.5 * Math.log10(fm) - 2.5 * Math.log10(s.scale));
        let r = 0;
        for (let j = 0; j < D.night.airmass.length; j++) {
          const aw = Sim.airmassWeights(D.airmass_grid, D.night.airmass[j]);
          for (let q = 0; q < 200; q++) r += Sim.modelRate(m, band, q / 200, aw) / 200;
        }
        flux.push(s.scale * r / D.night.airmass.length * D.exptime);
        targets.push(i);
      }
    });
    return { mag, flux, colour, targets };
  }

  // the movie's FWHM: second moment of the sub-pixel-phase-0 PSF stamp at the night's best seeing
  function fwhm(D, band, zenith) {
    let best = Infinity;
    for (const X of D.night.airmass) best = Math.min(best, seeingAt(zenith, X));
    const si = Sim.seeingIndex(D, best), S = D.psf[band].shape[2], base = si * D.psf[band].shape[1] * S * S;
    let cy = 0, cx = 0, m2 = 0;
    for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) { const p = D.psf[band][base + r * S + c]; cy += p * r; cx += p * c; }
    for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) m2 += D.psf[band][base + r * S + c] * ((r - cy) ** 2 + (c - cx) ** 2);
    return 2.3548 * Math.sqrt(0.5 * m2);
  }

  // the worker: {field: JSON text} once, then jobs {gen, stars, band, zenith, moon, seed}; posts {gen, j, mag, err}
  // for every exposure in order (NaN where the movie drops a point: no match or flux/sigma < 5), then {gen, done}.
  // A newer job stops an older one between exposures.
  function worker() {
    let ready = null, latest = -1;
    self.onmessage = async ({ data: msg }) => {
      if (msg.field) { ready = Sim.unpackAll(JSON.parse(msg.field)); return; }
      const gen = (latest = msg.gen), D = await ready;
      if (latest !== gen) return;
      const st = stars(D, msg.stars), cat = catalogue(D, st, msg.band);
      const phot = new SEP.Photometry(st.map((s) => s.x), st.map((s) => s.y), cat.mag, cat.flux, D.n, D.fullwell,
                                      cat.targets, cat.colour, fwhm(D, msg.band, msg.zenith));
      for (let j = 0; j < D.night.airmass.length; j++) {
        await new Promise((r) => setTimeout(r, 0));
        if (latest !== gen) return;
        const cond = { seeing: seeingAt(msg.zenith, D.night.airmass[j]), moon: msg.moon };
        const { image } = Sim.observe(D, Sim.meanImage(D, st, msg.band, j, cond), exposureSeed(msg.seed, j));
        const r = phot.measure(image), ok = r.map((t) => Number.isFinite(t.mag) && t.flux / t.sigma >= 5);
        self.postMessage({ gen, j, mag: r.map((t, i) => (ok[i] ? t.mag : NaN)), err: r.map((t, i) => (ok[i] ? t.mag_err : NaN)) });
      }
      self.postMessage({ gen, done: true });
    };
  }

  return { seeingAt, exposureSeed, stars, catalogue, fwhm, worker };
})();

if (typeof module !== "undefined") module.exports = Measure;
else if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) Measure.worker();
