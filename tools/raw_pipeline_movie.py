"""A look at the rebuilt chain as it stands (2026-10-21): one night of a field as ArgusSim's survey takes it, each
exposure rendered on its telescope's sensor (mesa_argus.sensor: ArgusSim geometry, PSF at each star's field angle,
seeing) in the smallest readout window, written as a raw camera frame (mesa_argus.camera, the toy), and processed by
hdps's production Tasks up to the TPV WCS solve (hdps.orchestrate build_task_list).

The stars: the field's Gaia DR3 stars (a cached core table), all constant, rho ~ G with a zero point of 25.77 per 60 s
frame (the hdps calibration of the f294 runs); sky 16 e-/s/px. Not yet: our MESA variables, ArgusSim's spectral sky.

    python tools/raw_pipeline_movie.py --gaia GAIA_CORE.ecsv --ra 294.7844 --dec -11.6235 --date 2026-07-30 \
        --window 28 -o outputs/pseudo_real/raw_pipeline_f294/raw_pipeline_f294.mp4
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.ERROR)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gaia", type=Path, required=True)
    p.add_argument("--ra", type=float, required=True)
    p.add_argument("--dec", type=float, required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--window", type=int, default=28, help="readout window side (minipixes)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("-o", "--output", type=Path, required=True)
    a = p.parse_args()

    import matplotlib.pyplot as plt
    from astropy.table import Table
    from astropy.time import Time
    from argus_sim import c
    from argus_sim.coverage_exact import Layout
    from argus_sim.observatory import seeing_at_airmass
    from hdps.core.images import FullFrameImage
    from hdps.orchestrate.pipeline import build_task_list
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import AsinhNorm
    from matplotlib.patches import Rectangle

    from mesa_argus import camera, night, runconfig, schema as S, sensor, survey
    from mesa_argus.argus import BLUE, COLORS, NAVY, RED, add_logo, apply_style

    g = Table.read(a.gaia)
    G = np.nan_to_num(np.asarray(g["Gmag"], float), nan=30.0)
    ex = survey.exposures(a.ra, a.dec, a.date, nights=1)
    ex.sort("time")
    if a.max_frames:
        ex = ex[: a.max_frames]
    layout = Layout.from_config()
    t = c.telescope
    psfs = sensor.PSFs(c)
    obs = runconfig.resolve({"field": {"ra": a.ra, "dec": a.dec, "window": a.window}, "stars": {"background": "x"}})
    wanted = ("DataCleanupTask", "ImageReductionTask", "SourceDetectionTask", "SourceFilterTask", "ReferenceRetrievalTask",
              "LinearWCSSolverTask", "SourceMatchingTask", "TPVWCSSolverTask")
    cfg = [dict(type=k["type"], config={key: (None if v == S.NONE else v) for key, v in k["config"].items()})
           for k in obs["hdps"]["tasks"] if k["type"] in wanted]
    tasks = build_task_list(cfg)
    side_px = int(round(a.window * S.minipix_arcsec() / t.plate_scale))
    raw_dir = a.output.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    apply_style()
    fig = plt.figure(figsize=(16, 9))
    writer = FFMpegWriter(fps=a.fps, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    history = dict(t=[], med=[], band=[], ndet=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(ex)} exposures through {len(set(ex['tel_id']))} telescopes -> {a.output}", flush=True)
    with writer.saving(fig, str(a.output), dpi=100):
        for n, row in enumerate(ex):
            when = Time(row["time"])
            dt = when.jyear - 2016.0
            ra = np.asarray(g["RA_ICRS"]) + np.nan_to_num(np.asarray(g["pmRA"], float)) * dt / 3.6e6 / np.cos(np.radians(a.dec))
            dec = np.asarray(g["DE_ICRS"]) + np.nan_to_num(np.asarray(g["pmDE"], float)) * dt / 3.6e6
            at = sensor.array_frame(np.r_[a.ra, ra], np.r_[a.dec, dec], [row["time"]])
            alt, az = at(np.array([row["theta_deg"]]))
            u, v, ang = sensor.focal_plane(alt[0], az[0], int(row["ota"]), layout)
            x, y = sensor.to_pixels(u, v)
            win = sensor.window(x[0], y[0], side_px)
            airmass = float(night.sky_position(Time([row["time"]]), a.ra, a.dec, night.location(c))[0][0])
            seeing = float(seeing_at_airmass(row["seeing_zenith"], airmass))
            electrons = 60 * 10 ** (-0.4 * (G - 25.77 + 2.5 * np.log10(60)))
            img = sensor.render(x[1:], y[1:], electrons, ang[1:], row["band"], seeing, 60 * (16.0 + t.dark_current), win,
                                psfs, rng, t.readnoise, t.fullwell)
            path = raw_dir / f"{camera.camera_id(row['tel_id'])}_{when.datetime:%Y%m%d_%H%M%S}.fits"
            camera.write_raw(path, img, dict(utc_start=(when - 30 / 86400).isot, utc_end=(when + 30 / 86400).isot,
                                             exptime=60.0, ratchet=f"{when.datetime:%Y%m%d}_{int(row['ratchet'])}",
                                             tel_id=int(row["tel_id"]), target_ra=a.ra, target_dec=a.dec, band=row["band"],
                                             x_origin=win[0], y_origin=win[1]))
            image = FullFrameImage.from_fits(str(path))
            t0, status = time.time(), "WCS solved"
            for task in tasks:
                try:
                    image = task.run(image)
                except Exception as e:
                    status = f"{task.name}: {type(e).__name__}"
                    break
            elapsed = time.time() - t0
            det = image.sources.table if getattr(image, "sources", None) is not None else None
            med = np.nan
            if status == "WCS solved" and image.wcs is not None:
                xs, ys = x[1:] - win[0], y[1:] - win[1]
                sel = (xs > 20) & (xs < win[2] - 20) & (ys > 20) & (ys < win[3] - 20) & (G < 17)
                r2, d2 = image.wcs.all_pix2world(xs[sel], ys[sel], 0)
                med = float(np.median(np.hypot((r2 - ra[sel]) * np.cos(np.radians(a.dec)), d2 - dec[sel]) * 3600))
            history["t"].append((when.mjd % 1.0) * 24)  # UT clock hours
            history["med"].append(med)
            history["band"].append(row["band"])
            history["ndet"].append(len(det) if det is not None else 0)

            fig.clear()
            grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.6, 1.25], height_ratios=[1, 1], wspace=0.28, hspace=0.38,
                                    left=0.05, right=0.97, bottom=0.08, top=0.86)
            ax_s = fig.add_subplot(grid[0, 0])
            ax_s.add_patch(Rectangle((0, 0), t.n_pixels_x, t.n_pixels_y, fill=False, ec=NAVY, lw=1.2))
            ax_s.add_patch(Rectangle(win[:2], win[2], win[3], fc=COLORS[row["band"]], ec=COLORS[row["band"]], alpha=0.8))
            ax_s.plot(x[0], y[0], "+", color=NAVY, ms=10)
            ax_s.plot((t.n_pixels_x - 1) / 2, (t.n_pixels_y - 1) / 2, "x", color="0.6", ms=6)
            ax_s.set(xlim=(-300, t.n_pixels_x + 300), ylim=(-300, t.n_pixels_y + 300), aspect="equal",
                     xlabel="sensor x (px)", ylabel="sensor y (px)")
            ax_s.set_title(f"Telescope {int(row['tel_id'])}: where the window falls", loc="left", weight="bold", fontsize=11)
            ax_s.text(0.02, 0.97, f"field angle {float(row['field_angle']):.2f}°", transform=ax_s.transAxes, va="top", fontsize=9)

            ax_i = fig.add_subplot(grid[:, 1])
            norm = AsinhNorm(linear_width=40, vmin=np.percentile(img, 5), vmax=np.percentile(img, 99.8))
            ax_i.imshow(img, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
            if det is not None and len(det):
                ax_i.scatter(det["xcentroid"], det["ycentroid"], s=26, facecolors="none", edgecolors=RED, linewidths=0.6)
            ax_i.set_title(f"Raw frame, {win[2]} x {win[3]} px readout window ({row['band']})", loc="left", weight="bold",
                           fontsize=11)
            ax_i.set(xlabel="window x (px)", ylabel="window y (px)")
            ax_i.text(0.02, 0.98, f"hdps: {history['ndet'][-1]} detections (red)\n{status} in {elapsed:.1f} s",
                      transform=ax_i.transAxes, va="top", fontsize=9, color="white",
                      bbox=dict(facecolor=NAVY, alpha=0.75, edgecolor="none", pad=3))

            ax_w = fig.add_subplot(grid[0, 2])
            th, mh, bh = np.array(history["t"]), np.array(history["med"]), np.array(history["band"])
            for band in ("beta", "rho"):
                m = bh == band
                ax_w.plot(th[m], mh[m], "o", ms=3, color=COLORS[band], label=band)
            ax_w.set(xlabel="UT (hours)", ylabel="median |WCS - truth| (\")", ylim=(0, max(0.5, np.nanmax(mh) * 1.2 if np.isfinite(mh).any() else 0.5)))
            ax_w.set_title("hdps WCS solve against truth", loc="left", weight="bold", fontsize=11)
            ax_w.legend(fontsize=8, frameon=False)
            ax_n = fig.add_subplot(grid[1, 2])
            nh = np.array(history["ndet"])
            for band in ("beta", "rho"):
                m = bh == band
                ax_n.plot(th[m], nh[m], "o", ms=3, color=COLORS[band])
            ax_n.set(xlabel="UT (hours)", ylabel="detections per frame")
            ax_n.set_title("hdps source detection", loc="left", weight="bold", fontsize=11)

            ax_t = fig.add_subplot(grid[1, 0])
            ax_t.axis("off")
            ax_t.text(0, 1, "\n".join([f"exposure {n + 1} of {len(ex)}", f"UTC {row['time'][:19]}",
                                       f"telescope {int(row['tel_id'])}, ratchet {int(row['ratchet'])}",
                                       f"band {row['band']}", f"seeing {seeing:.2f}\"  airmass {airmass:.2f}",
                                       f"window origin ({win[0]}, {win[1]})", "",
                                       "ArgusSim: survey, telescope, filter, time,", "tracking angle, seeing, geometry, PSF",
                                       "toy camera: gain 0.373 e-/ADU, no bias",
                                       "hdps: cleanup -> reduction -> detection ->", "Refcat2 -> linear WCS -> matching -> TPV"]),
                      va="top", fontsize=10, family="monospace")
            add_logo(fig, (0.012, 0.9, 0.14, 0.08))
            fig.suptitle(f"Raw frames through hdps — RA {a.ra:.4f}, Dec {a.dec:+.4f} — night of {a.date}", x=0.58, fontsize=14)
            writer.grab_frame()
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(ex)}; last frame hdps {elapsed:.1f} s, median WCS error {med:.3f}\"", flush=True)
    plt.close(fig)
    ok = np.isfinite(np.array(history["med"]))
    print(f"done: {a.output}; WCS solved in {ok.sum()}/{len(ok)} frames; median error {np.nanmedian(history['med']):.3f}\"")


if __name__ == "__main__":
    main()
