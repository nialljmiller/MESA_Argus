// The page's noiseless image vs the movie's (demo/validate/mean_image.py): node demo/validate/mean_image.mjs
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const Sim = createRequire(import.meta.url)("../src/sim.js");

const root = new URL("../build/", import.meta.url);
const D = await Sim.unpackAll(JSON.parse(readFileSync(new URL(`${process.env.DEMO_FIELD || "f294"}.json`, root))));
const read = (k) => { const b = readFileSync(new URL(`validate_mean/${k}.f64`, root)); return new Float64Array(b.buffer, b.byteOffset, b.length / 8); };

// the scene as the page builds it: every Gaia star from its template rates, the field's variables from their models
const stars = D.stars.source.map((_, i) => ({
  x: D.stars.x[i], y: D.stars.y[i], scale: 1,
  rate: Object.fromEntries(D.bands.map((b) => [b, D.stars.rate[b].subarray(i * 9, i * 9 + 9)])),
}));
for (const { star, model } of D.native) {
  Object.assign(stars[star], { model, epoch: D.models[model].epoch_bjd - 2460000, period: D.models[model].gaia_period });
}
for (const j of [0, 150, 298]) {
  const seeing = read(`seeing_${j}`)[0];
  for (const b of D.bands) {
    const ref = read(`${b}_${j}`), mine = Sim.meanImage(D, stars, b, j, { seeing, moon: 1 });
    let worst = 0, worstAbs = 0;
    for (let i = 0; i < ref.length; i++) {
      worstAbs = Math.max(worstAbs, Math.abs(mine[i] - ref[i]));
      worst = Math.max(worst, Math.abs(mine[i] - ref[i]) / ref[i]);
    }
    console.log(`exposure ${j} ${b}: seeing ${seeing.toFixed(3)}"  max |page - movie| ${worstAbs.toExponential(2)} e (${(worst * 100).toExponential(2)}% of the pixel)`);
  }
}
