// The page's SExtractor port (demo/src/sep.js) vs SEP itself (mesa_argus/photometry.py) on the same exposures.
//   node demo/validate/sep.mjs make      -> demo/build/validate_sep/<case>.{f64,json}: page exposures + catalogue inputs
//   python demo/validate/sep_reference.py -> <case>.ref.json, <case>.{back,seg}: SEP's results on them
//   node demo/validate/sep.mjs compare
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const Sim = require("../src/sim.js"), SEP = require("../src/sep.js");

const root = new URL("../build/", import.meta.url), dir = new URL("validate_sep/", root);
mkdirSync(dir, { recursive: true });
const D = await Sim.unpackAll(JSON.parse(readFileSync(new URL("f294_784m11_623.json", root))));
const NA = D.airmass_grid.length;

// the scene as the page builds it, plus `extra` random constant stars and `vars` library variables (seeded)
function scene(extra, vars, seed) {
  const stars = D.stars.source.map((_, i) => ({
    x: D.stars.x[i], y: D.stars.y[i], scale: 1, colour: D.stars.colour[i], mag0: D.stars.mag.rho[i],
    rate: Object.fromEntries(D.bands.map((b) => [b, D.stars.rate[b].subarray(i * NA, (i + 1) * NA)])),
    magb: Object.fromEntries(D.bands.map((b) => [b, D.stars.mag[b][i]])),
  }));
  for (const { star, model } of D.native) Object.assign(stars[star], { model, epoch: D.models[model].epoch_bjd - 2460000, period: D.models[model].gaia_period });
  const g = Sim.rng(seed), t = D.templates;
  for (let k = 0; k < extra; k++) {
    const i = Math.floor(g.uniform() * t.bp_rp.length), G = 13 + 7 * g.uniform(), scale = 10 ** (-0.4 * (G - t.G));
    stars.push({ x: 10 + (D.n - 20) * g.uniform(), y: 10 + (D.n - 20) * g.uniform(), scale, colour: t.bp_rp[i],
      rate: Object.fromEntries(D.bands.map((b) => [b, t.rate[b].subarray(i * NA, (i + 1) * NA)])),
      magb: Object.fromEntries(D.bands.map((b) => [b, t.mag[b][i] - 2.5 * Math.log10(scale)])) });
  }
  const lib = D.models.map((m, i) => i).filter((i) => D.models[i].library);
  for (let k = 0; k < vars; k++) {
    const model = lib[k % lib.length], m = D.models[model];
    stars.push({ x: D.centre + (g.uniform() - 0.5) * 200, y: D.centre + (g.uniform() - 0.5) * 200, model, period: m.period,
      epoch: D.night.bjd[0] - g.uniform() * m.period, scale: 10 ** (-0.4 * (13.5 + 3 * g.uniform() - 15)), colour: NaN });
  }
  return stars;
}

// catalogue inputs as the movie builds them: magnitude (variables: intensity mean), flux at airmass 1 (variables: mean
// over the night's exposures), Gaia BP-RP
function catalogue(stars, band) {
  const mag = [], flux = [], colour = [], targets = [];
  stars.forEach((s, i) => {
    colour.push(s.colour);
    if (s.model === undefined) {
      mag.push(s.magb[band]);
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

// the movie's FWHM: second moment of the sub-pixel-phase-0 PSF stamp at the reference seeing
function fwhmAt(band, seeing) {
  const si = Sim.seeingIndex(D, seeing), S = D.psf[band].shape[2], base = si * D.psf[band].shape[1] * S * S;
  let cy = 0, cx = 0, m2 = 0;
  for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) { const p = D.psf[band][base + r * S + c]; cy += p * r; cx += p * c; }
  for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) m2 += D.psf[band][base + r * S + c] * ((r - cy) ** 2 + (c - cx) ** 2);
  return 2.3548 * Math.sqrt(0.5 * m2);
}

const CASES = [
  { name: "native_rho_165", extra: 0, vars: 0, band: "rho", j: 165 },
  { name: "native_rho_0", extra: 0, vars: 0, band: "rho", j: 0 },
  { name: "native_beta_330", extra: 0, vars: 0, band: "beta", j: 330 },
  { name: "crowded_rho_165", extra: 1500, vars: 7, band: "rho", j: 165 },
  { name: "packed_rho_165", extra: 6000, vars: 7, band: "rho", j: 165 },
];
const seeingAt = (j) => D.zenith_seeing * D.night.airmass[j] ** 0.6;
let bestSeeing = Infinity;
for (let j = 0; j < D.night.airmass.length; j++) bestSeeing = Math.min(bestSeeing, seeingAt(j));

const mode = process.argv[2];
for (const c of CASES) {
  const stars = scene(c.extra, c.vars, 7 + c.extra);
  const cat = catalogue(stars, c.band), fwhm = fwhmAt(c.band, bestSeeing), x = stars.map((s) => s.x), y = stars.map((s) => s.y);
  if (mode === "make") {
    const mean = Sim.meanImage(D, stars, c.band, c.j, { seeing: seeingAt(c.j), moon: 1 });
    const { image } = Sim.observe(D, mean, 1000 + c.j);
    writeFileSync(new URL(`${c.name}.f64`, dir), Buffer.from(Float64Array.from(image).buffer));
    writeFileSync(new URL(`${c.name}.json`, dir), JSON.stringify({ n: D.n, fullwell: D.fullwell, fwhm, x, y, ...cat }));
    console.log(`${c.name}: ${stars.length} stars, ${cat.targets.length} variables, FWHM ${fwhm.toFixed(3)} px`);
    continue;
  }
  // compare
  const read = (ext, T) => { const b = readFileSync(new URL(`${c.name}.${ext}`, dir)); return new T(b.buffer, b.byteOffset, b.length / T.BYTES_PER_ELEMENT); };
  const image = read("f64", Float64Array), ref = JSON.parse(readFileSync(new URL(`${c.name}.ref.json`, dir)));
  const t0 = performance.now();
  const ex = SEP.photometryExtract(image, D.n, D.n, fwhm);
  const t1 = performance.now();
  const backRef = read("back", Float64Array), segRef = read("seg", Int32Array);
  let dBack = 0, nSeg = 0;
  for (let i = 0; i < backRef.length; i++) { dBack = Math.max(dBack, Math.abs(ex.back[i] - backRef[i])); nSeg += ex.segmap[i] !== segRef[i]; }
  const nObj = Math.min(ex.objects.length, ref.obj.x.length);
  const worst = (key, a, b) => { let w = 0; for (let i = 0; i < nObj; i++) w = Math.max(w, Math.abs(a(i) - b[key][i])); return w; };
  const relFlux = (() => { let w = 0; for (let i = 0; i < nObj; i++) w = Math.max(w, Math.abs(ex.detections[i].flux - ref.det.flux[i]) / Math.max(Math.abs(ref.det.flux[i]), 1)); return w; })();
  const phot = new SEP.Photometry(x, y, cat.mag, cat.flux, D.n, D.fullwell, cat.targets, cat.colour, fwhm);
  const calSame = phot.cal.length === ref.cal.length && phot.cal.every((v, i) => v === ref.cal[i]);
  const t2 = performance.now();
  const m = phot.measure(image);
  const t3 = performance.now();
  let dMag = 0;
  m.forEach((r, i) => { const a = r.mag, b = ref.measure.mag[i]; if (Number.isFinite(a) || Number.isFinite(b)) dMag = Math.max(dMag, Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) : Infinity); });
  console.log(`${c.name}: globalrms ${ex.globalrms.toFixed(5)} vs ${ref.globalrms.toFixed(5)} · back max |Δ| ${dBack.toExponential(1)} e · ` +
    `detections ${ex.objects.length} vs ${ref.obj.x.length} · segmap pixels differing ${nSeg} · ` +
    `max |Δx| ${worst("x", (i) => ex.objects[i].x, ref.obj).toExponential(1)} |Δa| ${worst("a", (i) => ex.objects[i].a, ref.obj).toExponential(1)} · ` +
    `windowed |Δx| ${worst("x", (i) => ex.detections[i].x, ref.det).toExponential(1)} · Kron |Δ| ${worst("kron", (i) => ex.detections[i].kron, ref.det).toExponential(1)} · ` +
    `MAG_AUTO flux max rel |Δ| ${relFlux.toExponential(1)} · calibration stars ${calSame ? "same" : `DIFFER (${phot.cal.length} vs ${ref.cal.length})`} · ` +
    `targets' mag max |Δ| ${dMag.toExponential(1)} · time: extract ${(t1 - t0).toFixed(0)} ms, measure ${(t3 - t2).toFixed(0)} ms`);
}
