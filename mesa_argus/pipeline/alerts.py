"""Alerts from a frame's difference image: hdps's AlertVettingTask (hdps.orchestrate.tasks), mirrored on the CPU.

hdps_vetting, as AlertVettingTask._run_impl and _maybe_kafka:
- detection and features: hdps.alerts.features.detect_and_featurize on D, Var(D) and the valid pixels (finite, Var(D)
  finite and > 0), at the task's permissive detect_sigma 3, at the frame's PSF FWHM (the psf stage's measured model;
  hdps derives it from the header SEEING, fwhm_from_seeing);
- the alert catalogue row per candidate (camera, ratchet, tile, x, y, RA, Dec, PSF flux and magnitude, score,
  accepted, model version, time, every feature as feat_*): <out>/catalogue/alerts_<camera>_<ratchet>_<frame>.parquet;
- one hdps.alerts.models.Alert (schema 3.0) per accepted candidate, its Candidate filled field for field as
  _maybe_kafka fills it (UUID alertId; minipix index of the tile's 32 px block; HEALPix order 16 id; PSF flux,
  magnitude, chi2, S/N, zero point; FWHM, elongation, pixels, PSF FWHM and beta, PSF-over-aperture; the fields that path
  has no producer for sent blank, ALERT_UNPRODUCED_FIELDS), serialised as hdps's Kafka producer sends it
  (orjson of Alert.to_dict()), one JSON packet per line: <out>/<field>_<band>_NNNN.jsonl;
- ref / sci / diff cutouts (31 px) of the accepted candidates, as the task's emit_stamps:
  <out>/stamps/alerts_stamps_<camera>_<ratchet>_<frame>.npz.
hdps.orchestrate cannot be installed without CUDA (it needs hdps-core[gpu] and Ray), so its three task helpers used
here are copied with their names: ALERT_HEALPIX_ORDER, ALERT_UNPRODUCED_FIELDS, _cutout.

Departures, each because the simulation lacks what production has:
- no real/bogus model bundle exists, so there is no score: a candidate is accepted when its PSF S/N (feature
  snr_psf) is at least alert_min_snr (rb_score NaN, rbModelVersion "none"); hdps accepts score >= the bundle's cut;
- the packet's timestamp (build time; hdps: the wall clock) is the end of the exposure;
- x, y are tile pixels (the frame is a cut-out of the tile; its offset in the tile is metadata tile_x0, tile_y0).
"""
from pathlib import Path

import numpy as np

SETTINGS = dict(alert_detect_sigma=3.0, alert_min_snr=8.0, alert_stamp_size=31)
ALERT_HEALPIX_ORDER = 16  # AlertVettingTask.ALERT_HEALPIX_ORDER
ALERT_UNPRODUCED_FIELDS = {  # AlertVettingTask.ALERT_UNPRODUCED_FIELDS (tasks.py)
    "filter": 0, "cameraIds": [], "otaIds": [], "filterIds": [], "focuserIds": [], "subarrayIds": [],
    "referenceId": -1, "referenceStart": 0.0, "referenceEnd": 0.0, "referenceDepth": 0.0,
}
ORIGIN = "MESA_ARGUS pipeline.alerts.hdps_vetting (hdps AlertVettingTask, CPU mirror)"


def _cutout(plane, y, x, size):
    """AlertVettingTask._cutout: (size, size) float32 cut-out centred on (y, x), NaN-padded at the edges."""
    out = np.full((size, size), np.nan, dtype=np.float32)
    if plane is None:
        return out
    h = size // 2
    yc, xc = int(round(y)), int(round(x))
    y0, y1 = max(0, yc - h), min(plane.shape[0], yc + h + 1)
    x0, x1 = max(0, xc - h), min(plane.shape[1], xc + h + 1)
    if y1 <= y0 or x1 <= x0:
        return out
    out[y0 - (yc - h):y1 - (yc - h), x0 - (xc - h):x1 - (xc - h)] = plane[y0:y1, x0:x1]
    return out


def hdps_vetting(difference, frame, fwhm_px, out_dir, cfg):
    """The frame's alert catalogue (astropy Table, one row per candidate, with alert_id: '' unless accepted); writes
    the catalogue, the packets and the stamps into out_dir (None: write nothing)."""
    import uuid

    import astropy.units as u
    import orjson
    from astropy.table import Table
    from astropy.time import Time
    from astropy_healpix import HEALPix
    from hdps.alerts.features import FEATURE_NAMES, MOFFAT_BETA, detect_and_featurize
    from hdps.alerts.models import SCHEMA_VERSION, Alert, AlertFlags, Candidate
    from hdps.skymap import xy_to_local_idx

    c = {**SETTINGS, **cfg}
    m = frame.metadata
    empty = Table(names=("x", "y", "ra_deg", "dec_deg", "accepted", "alert_id"), dtype=(float, float, float, float, bool, str))
    if difference is None or difference.get("solve_failed"):
        return empty
    D, var_d = difference["D"], difference["var"]
    valid = np.isfinite(D) & np.isfinite(var_d) & (var_d > 0)
    if valid.mean() < 0.05:
        return empty
    feats, n_pos, _ = detect_and_featurize(D, var_d, valid, fwhm_px, detect_sigma=c["alert_detect_sigma"])
    if feats is None:
        return empty
    n = len(feats["x"])
    ra, dec = frame.full_wcs.all_pix2world(feats["x"], feats["y"], 0)
    flux_psf = np.asarray(feats["flux_psf"], np.float64)
    magzp = m.get("magzp")
    mag = np.full(n, np.nan)
    if magzp is not None and np.isfinite(magzp):
        pos_flux = flux_psf > 0
        mag[pos_flux] = float(magzp) - 2.5 * np.log10(flux_psf[pos_flux])
    snr_psf = np.asarray(feats["snr_psf"], float)
    accept = np.isfinite(snr_psf) & (snr_psf >= c["alert_min_snr"])  # in place of score >= cut (see the notes)
    camera_id, ratchet = str(m.get("tel_id", "unknown")), int(m.get("ratchet", 0))
    frame_id = f"{m['field']}_{m['band']}_{m['exposure']:04d}"
    x_tile = np.asarray(feats["x"], float) + float(m.get("tile_x0", 0.0))
    y_tile = np.asarray(feats["y"], float) + float(m.get("tile_y0", 0.0))

    catalogue = Table(dict(camera_id=[camera_id] * n, ratchet_num=np.full(n, ratchet),
                           tanseg_id=np.full(n, int(frame.tile_id)), x=np.asarray(feats["x"], float),
                           y=np.asarray(feats["y"], float), x_tile=x_tile, y_tile=y_tile, ra_deg=ra, dec_deg=dec,
                           flux_psf=flux_psf, mag_psf=mag, rb_score=np.full(n, np.nan), accepted=accept,
                           model_version=["none"] * n, obs_time=[m["date_obs"]] * n))
    for name in FEATURE_NAMES:
        catalogue[f"feat_{name}"] = np.asarray(feats[name], float)
    catalogue["alert_id"] = np.full(n, "", dtype="U36")

    idxs = np.flatnonzero(accept)
    if idxs.size:
        t_end = Time(m["date_obs"]) + float(m["exptime"]) / 2 * u.s
        hp = HEALPix(nside=2**ALERT_HEALPIX_ORDER, order="nested")
        healpix = np.asarray(hp.lonlat_to_healpix(ra[idxs] * u.deg, dec[idxs] * u.deg), np.int64)
        lines, stamps = [], dict(row_index=[], ref=[], sci=[], diff=[])
        size = int(c["alert_stamp_size"])

        def feat(name, i):
            v = float(feats[name][i])
            return v if np.isfinite(v) else None

        for j, i in enumerate(idxs):
            ix_block, iy_block = int(x_tile[i]) // 32, int(y_tile[i]) // 32
            minipix = xy_to_local_idx(ix_block, iy_block) if 0 <= ix_block < 64 and 0 <= iy_block < 64 else None
            alert_id = str(uuid.uuid4())
            candidate = Candidate(
                alertId=alert_id, obsStartTime=m["date_obs"], obsEndTime=m["date_obs"],
                obsWeightedMidpoint=m["date_obs"], integrationTime=float(m["exptime"]), coadd=False,
                epochId=frame_id, timestamp=int(t_end.unix * 1000), tansegId=int(frame.tile_id), minipixIdx=minipix,
                healpixId=int(healpix[j]), healpixOrder=ALERT_HEALPIX_ORDER, ra=float(ra[i]), dec=float(dec[i]),
                x=float(x_tile[i]), y=float(y_tile[i]), flags=int(AlertFlags.NONE), psfFluxDif=float(flux_psf[i]),
                psfMagnitudeDif=float(mag[i]) if np.isfinite(mag[i]) else None, psfChi2=feat("chi2_psf", i),
                psfSnr=feat("snr_psf", i), isFluxPositive=bool(flux_psf[i] > 0),
                psfZeropoint=float(magzp) if magzp is not None and np.isfinite(magzp) else None,
                fwhm=float(feats["fwhm_est"][i]), ellipticity=float(feats["elong"][i]), nPixels=int(feats["npix"][i]),
                psfFwhm=float(fwhm_px), psfBeta=float(MOFFAT_BETA), psfOverAp=feat("psf_over_ap", i),
                realBogusScore=None, rbModelVersion="none", realBogusConfidence=None, **ALERT_UNPRODUCED_FIELDS)
            alert = Alert(schemaVersion=SCHEMA_VERSION, alertOrigin=ORIGIN, alertCadence=0, alertId=alert_id,
                          candidate=candidate)
            lines.append(orjson.dumps(alert.to_dict()).decode())
            catalogue["alert_id"][i] = alert_id
            yi, xi = float(feats["y"][i]), float(feats["x"][i])
            stamps["row_index"].append(int(i))
            stamps["ref"].append(_cutout(difference["reference"], yi, xi, size))
            stamps["sci"].append(_cutout(difference["science"], yi, xi, size))
            stamps["diff"].append(_cutout(D, yi, xi, size))
        if out_dir is not None:
            out = Path(out_dir)
            (out / "stamps").mkdir(parents=True, exist_ok=True)
            (out / f"{frame_id}.jsonl").write_text("\n".join(lines) + "\n")
            np.savez_compressed(out / "stamps" / f"alerts_stamps_{camera_id}_{ratchet:04d}_{frame_id}.npz",
                                row_index=np.asarray(stamps["row_index"], np.int32),
                                ref=np.stack(stamps["ref"]), sci=np.stack(stamps["sci"]), diff=np.stack(stamps["diff"]))
    if out_dir is not None:
        (Path(out_dir) / "catalogue").mkdir(parents=True, exist_ok=True)
        catalogue.write(Path(out_dir) / "catalogue" / f"alerts_{camera_id}_{ratchet:04d}_{frame_id}.parquet",
                        overwrite=True)
    return catalogue
