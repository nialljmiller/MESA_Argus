"""Alert packets from a frame's difference-image candidates.

hdps_v3: one hdps.alerts.models.Alert (schema 3.0) per positive candidate, serialised exactly as hdps's Kafka producer
sends it (orjson.dumps(Alert.to_dict())), one JSON packet per line, one file per frame. Negative candidates (something
fainter than in the reference) are kept in the frame's candidate table but not alerted. No real/bogus score (hdps's
classifier needs a trained model bundle; none exists yet: rbModelVersion "none"). Cutouts are not filled. timestamp
(packet build time) is the end of the exposure, as if the packet were built at once.
"""
from pathlib import Path

import numpy as np

SCHEMA_ORIGIN = "MESA_ARGUS simulation"
FILTER_ID = {"beta": 1, "rho": 2}  # the schema's filter codes (argus.alert.candidate.avsc: 1 = b, 2 = rho)


def hdps_v3(candidates, frame, reference, out_dir):
    """Writes <out_dir>/<field>_<band>_<exposure>.jsonl; returns the alert IDs (one per alerted candidate, else '')."""
    import astropy.units as u
    from astropy.time import Time
    from hdps.alerts.models import SCHEMA_VERSION, Alert, Candidate
    import orjson

    m = frame.metadata
    t_mid = Time(m["date_obs"])
    t_start, t_end = (t_mid + s * m["exptime"] / 2 * u.s for s in (-1, 1))
    ids, lines = [], []
    for i, c in enumerate(candidates):
        if c["is_negative"] or not np.isfinite(c["flux"]) or c["flux"] <= 0:
            ids.append("")
            continue
        alert_id = f"{m['field']}-{m['band']}-{m['exposure']:05d}-{i:04d}"
        mag = m["magzp"] - 2.5 * np.log10(c["flux"]) if np.isfinite(m.get("magzp", np.nan)) else None
        candidate = Candidate(
            alertId=alert_id, obsStartTime=t_start.isot, obsEndTime=t_end.isot, obsWeightedMidpoint=t_mid.isot,
            integrationTime=float(m["exptime"]), coadd=False, filter=FILTER_ID[m["band"]],
            epochId=str(m["exposure"]), cameraIds=[int(m.get("tel_id", 0))], otaIds=[int(m.get("tel_id", 0))],
            filterIds=[FILTER_ID[m["band"]]], focuserIds=[], subarrayIds=[],
            timestamp=int(t_end.unix * 1000), ra=float(c["ra"]), dec=float(c["dec"]), x=float(c["x"]), y=float(c["y"]),
            flags=0, tansegId=int(frame.tile_id), isFluxPositive=True, rbModelVersion="none",
            referenceId=int(reference["tile_id"]), referenceStart=reference["tstart"], referenceEnd=reference["tstop"],
            referenceDepth=reference["depth5"], psfFluxDif=float(c["flux"]), psfFluxDifErr=float(c["flux_err"]),
            psfMagnitudeDif=mag)
        alert = Alert(schemaVersion=SCHEMA_VERSION, alertOrigin=SCHEMA_ORIGIN, alertCadence=int(m["exptime"]),
                      alertId=alert_id, candidate=candidate)
        lines.append(orjson.dumps(alert.to_dict()).decode())
        ids.append(alert_id)
    if lines:
        out = Path(out_dir) / f"{m['field']}_{m['band']}_{m['exposure']:04d}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
    return ids
