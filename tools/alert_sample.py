"""An alert sample for an alert broker: hdps's packets exactly as hdps wrote them, and beside them what the
simulation put at each one.

For each observation folder (outputs/pseudo_real/<name>) and band:
    <out>/<name>/<band>/packets/<epochId>.avro   hdps's packets for that exposure (an Avro object container file, hdps's
                                                 alert schema embedded: hdps_alerts/schemas): copied byte for byte, or,
                                                 with --cut, only those whose realBogusScore is at or above the cut,
                                                 each record unchanged
    <out>/<name>/<band>/truth.parquet            one row per packet, keyed by alertId (and the packet's epochId, ra,
                                                 dec): what is there (truth_class: transient / variable / constant /
                                                 none), which simulated source (truth_label, its kind and model), its
                                                 true magnitude and its true flux change since the reference, and the
                                                 real/bogus label as tools/rb_training_set.py defines it
    <out>/manifest.json                          per night and band: date, Moon illumination, the hdps settings that
                                                 decide which alerts exist (bright-star mask, real/bogus model and
                                                 cut), the schema version, packet counts by truth class

The packets are the stream Argus would send (mesa_argus.alert_files: hdps's AlertVettingTask builds every packet,
the file stands in for Kafka); nothing in them is changed or added. The truth comes from the engine's
<name>_night_<band>.alerts.csv (every hdps candidate with its features and its truth, alertId from the packets).

The cut: hdps sends a packet only for a candidate whose score is at or above AlertVettingTask's score_cut
(emit_all_candidates adds below-cut rows to the alert catalogue, never packets), and no packet field depends on the
cut. So a night run with score_cut = 0 (a packet for every candidate, the real/bogus model loaded) and selected here
with --cut C holds exactly the packets a run at score_cut = C sends (checked on hdps's v4.0 AlertVettingTask: every
field equal but alertId and the build timestamp, which differ between any two runs), and the cut can change without
running the night again. --cut takes a number or an operating point of --bundle (e.g. fa10).

    python tools/alert_sample.py outputs/pseudo_real/rb_f299_911m08_104_20261023 -o data/alert_sample
    python tools/alert_sample.py <folders> --cut fa10 --bundle data/rb/<bundle> -o data/alert_sample
"""
import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# What each truth row carries from the simulated source (scene/sources.json, by label), when it is a placed star.
SOURCE_KEYS = ("kind", "generator", "engine", "model", "params", "z", "t0_bjd", "peak_days", "period", "ebv",
               "distance_pc", "star")


def moon_illumination(time_utc):
    """Fraction of the Moon's disc lit at this time (astropy: Sun-Moon elongation)."""
    from astropy.coordinates import get_body, get_sun
    from astropy.time import Time

    t = Time(time_utc)
    return float((1.0 - np.cos(get_sun(t).separation(get_body("moon", t)).rad)) / 2.0)


def window_deg2(folder, band, obs):
    """The readout window's area (deg2): its side in sensor pixels from the engine's log, times the plate scale."""
    import re

    from mesa_argus.runconfig import plate_scale

    log = folder / f"{folder.name}_night_{band}.log"
    m = re.search(r"\((\d+) px window\)", log.read_text(errors="replace")) if log.exists() else None
    return (int(m[1]) * plate_scale(obs) / 3600.0) ** 2 if m else None


def night_settings(folder, band):
    """The resolved observation's settings that decide which alerts exist."""
    obs = tomllib.loads((folder / "observation.toml").read_text())
    tasks = {t["type"]: t.get("config", {}) for t in obs["hdps"]["tasks"]}
    sfft, vet = tasks.get("SFFTSubtractionTask", {}), tasks.get("AlertVettingTask", {})
    area = window_deg2(folder, band, obs)
    inject = len(sources_by_label(folder))  # the library stars placed over the window (scene/sources.json)
    return dict(field=obs["stars"].get("background", ""), date=obs["night"]["date"],
                mask_bright_stars=sfft.get("mask_bright_stars"),
                bright_star_threshold=sfft.get("bright_star_threshold"), model_bundle=vet.get("model_bundle") or "",
                score_cut=vet.get("score_cut"), emit_all_candidates=vet.get("emit_all_candidates"),
                stride=obs["output"].get("stride"), injected=inject, window_deg2=round(area, 4) if area else None,
                injected_per_deg2=round(inject / area, 1) if area else None)


def sources_by_label(folder):
    path = folder / "scene" / "sources.json"
    if not path.exists():
        return {}
    return {s["label"]: s for s in json.loads(path.read_text()) if s.get("label")}


def resolve_cut(cut, bundle):
    """A cut as a number: a literal, or an operating point from the bundle's manifest.json (hdps's cut_for)."""
    if cut is None:
        return None
    try:
        return float(cut)
    except ValueError:
        if bundle is None:
            raise SystemExit(f"--cut {cut}: an operating point needs --bundle")
        points = json.loads((Path(bundle) / "manifest.json").read_text()).get("operating_points", {})
        if cut not in points:
            raise SystemExit(f"--cut {cut}: {bundle} offers {sorted(points)}")
        return float(points[cut])


def one_band(folder, band, out, min_sigma, cut=None):
    """Copy a night's packets for one band (with a cut: those at or above it) and write their truth; returns the
    manifest entry (None: no packets)."""
    from fastavro import writer

    from mesa_argus import alert_files
    from rb_training_set import label

    name = folder.name
    packet_dir = folder / "hdps" / band / "alerts"
    files = sorted(packet_dir.glob("*.avro"))
    table_path = folder / f"{name}_night_{band}.alerts.csv"
    if not files or not table_path.exists():
        return None
    table = pd.read_csv(table_path, low_memory=False)
    table["alertId"] = table["alertId"].fillna("").astype(str)
    table["label"] = label(table, min_sigma)
    by_id = table[table["alertId"] != ""].drop_duplicates("alertId").set_index("alertId")
    sources = sources_by_label(folder)
    dest = out / name / band / "packets"
    dest.mkdir(parents=True, exist_ok=True)
    rows, versions, n_read = [], set(), 0
    for path in files:
        packets = alert_files.read(path)
        n_read += len(packets)
        if cut is None:
            shutil.copy2(path, dest / path.name)
        else:
            packets = [k for k in packets if (k["candidate"]["realBogusScore"] or 0.0) >= cut]
            if not packets:
                continue
            with open(path, "rb") as f:  # the file's own embedded schema, the records as they were
                from fastavro import reader

                schema = reader(f).writer_schema
            with open(dest / path.name, "wb") as f:
                writer(f, schema, packets, codec="deflate")
        for packet in packets:
            versions.add(packet["schemaVersion"])
            c = packet["candidate"]
            alert_id = str(packet["alertId"])
            row = dict(alertId=alert_id, epochId=c["epochId"], ra=c["ra"], dec=c["dec"],
                       packet_file=f"packets/{path.name}")
            if alert_id in by_id.index:
                t = by_id.loc[alert_id]
                row.update(exposure=int(t["exposure"]), time_utc=t["time_utc"], truth_class=t["truth_class"],
                           truth_label=t["truth_label"] if isinstance(t["truth_label"], str) else "",
                           truth_sep_arcsec=t["truth_sep_arcsec"], truth_mag=t["truth_mag"],
                           truth_flux_now=t["truth_flux_now"], truth_flux_reference=t["truth_flux_reference"],
                           truth_diff_flux=t["truth_diff_flux"], label=int(t["label"]))
                source = sources.get(row["truth_label"], {})
                row.update({f"source_{k}": source.get(k) for k in SOURCE_KEYS})
            rows.append(row)
    truth = pd.DataFrame(rows)
    missing = int(truth["truth_class"].isna().sum()) if "truth_class" in truth else len(truth)
    truth.to_parquet(out / name / band / "truth.parquet", index=False)
    settings = night_settings(folder, band)
    mid = table["time_utc"].iloc[len(table) // 2] if len(table) else None
    models = sorted({str(m) for m in table.get("model_version", pd.Series(dtype=str)).dropna().unique()})
    frames = int(table["exposure"].nunique()) if len(table) else 0
    return dict(observation=name, band=band, **settings, sample_cut=cut, rb_model_versions=models, exposures=frames,
                moon_illumination=round(moon_illumination(mid), 3) if mid else None,
                schema_versions=sorted(versions), packets=len(truth), packets_before_cut=n_read,
                packet_files=len(files),
                packets_without_truth=missing,
                by_truth_class={k: int(v) for k, v in truth["truth_class"].value_counts().items()}
                if "truth_class" in truth else {},
                by_label={str(k): int(v) for k, v in truth["label"].value_counts().items()} if "label" in truth else {})


KNOWN_GAPS = """\
Known gaps of hdps's schema v4.0 producer (hdps PR #474, under review: its schema may still change):
- psfFluxDifErr is -1.0 (not produced) until hdps exposes the flux error from extract_features (hdps#476);
- coadd is true on every alert (hdps#477);
- no cutouts in the packets (cutoutScience/Difference/Template null: cutout extraction is deferred to a later PR);
- fields with no producer yet are explicit blanks: filter 0, cameraIds/otaIds/filterIds/focuserIds/subarrayIds [],
  referenceId -1 and the reference times 0.0, alertCadence 0; no solar-system association runs (ssoAssociation null);
- JD(TT) is computed from hdps's Argus-epoch anchor inline (until hdps#410's argus_epoch_to_jd);
- candidates without a sky position or a finite difference flux are not sent;
- on Kafka each message would be one schemaless Avro record (topic and framing still to be decided upstream); here
  each exposure's packets are one Avro object container file with the schema embedded.
"""


def write_readme(out, manifest):
    """README.md for the sample, from its manifest."""
    nights = manifest["nights"]
    cuts = sorted({f"{n['score_cut']} (= {n['score_cut_value']:.4g})" if n.get("score_cut_value") is not None
                   else str(n["score_cut"]) for n in nights})
    models = sorted({m for n in nights for m in n["rb_model_versions"]})
    bundles = sorted({n["model_bundle"] for n in nights})
    versions = sorted({v for n in nights for v in n["schema_versions"]})
    total = sum(n["packets"] for n in nights)
    real = sum(n["by_label"].get("1", 0) for n in nights)
    lines = [f"# Simulated Argus alert sample{(' ' + manifest['name']) if manifest.get('name') else ''}", "",
             "Alerts from simulated Argus Array observations of real Gaia fields (MESA_ARGUS: ArgusSim images, hdps, "
             "the Argus pipeline, run as production runs it), for training and testing an alert broker.", "",
             f"- packets: {total} in {sum(n['packet_files'] for n in nights)} files, hdps alert schema "
             f"{', '.join(versions)} (argus.alert; the .avsc is embedded in every file)",
             f"- real/bogus model: {', '.join(models)} ({', '.join(bundles)}), loaded in hdps's AlertVettingTask",
             f"- cut: score_cut {', '.join(cuts)}: only candidates at or above it are sent, as production sends "
             "them (\"fa10\": the score at which 10 false alerts per square degree per exposure pass, measured on "
             "simulated nights the model did not train on)",
             f"- of the packets, {real} are real by the truth rule below, {total - real} bogus or below it", "",
             "## Nights", "",
             "| observation | field | night | Moon lit | band | exposures | window deg2 | injected (per deg2) | "
             "packets | real | by truth class |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in nights:
        lines.append(f"| {n['observation']} | {n['field']} | {n['date']} | {n['moon_illumination']} | {n['band']} | "
                     f"{n['exposures']} | {n['window_deg2']} | {n['injected']} ({n['injected_per_deg2']}) | "
                     f"{n['packets']} | {n['by_label'].get('1', 0)} | "
                     + ", ".join(f"{k} {v}" for k, v in n["by_truth_class"].items()) + " |")
    lines += ["", "## Injection", "",
              "Each night's readout window holds the field's real Gaia stars (constant unless injected) plus the "
              "injected library stars in the table: supernovae, dwarf novae, microlensing events, RR Lyrae, eclipsing "
              "binaries, modelled light curves placed at random over the window. That density is far above what "
              "Argus would see in such a window; it gives the sample many real alerts. The bogus alerts (subtraction "
              "residuals of constant stars, noise, artefacts) come at the rate the cut allows.", "",
              "## Files", "",
              "- `<observation>/<band>/packets/<epochId>.avro`: one exposure's packets, exactly as hdps built them "
              "(read with any Avro reader, e.g. `fastavro.reader`).",
              "- `<observation>/<band>/truth.parquet`: one row per packet, joined on `alertId` (also `epochId`, "
              "`ra`, `dec`): `truth_class` (transient, variable: an injected source; constant: a real star that did "
              "not change; none: nothing simulated there), `truth_label` and `source_*` (the injected source: kind, "
              "model, redshift, t0, ...), `truth_mag`, `truth_diff_flux` (its true flux change since the reference, "
              "electrons), and `label`: 1 real (an injected source whose true change is at least 3 times the "
              "candidate's PSF flux error), 0 bogus (a constant star or nothing), -1 an injected source changing less "
              "than that.",
              "- `manifest.json`: everything in the table above, per night and band.", "", KNOWN_GAPS]
    (out / "README.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("observations", nargs="+", type=Path, help="observation folders (outputs/pseudo_real/<name>)")
    p.add_argument("-o", "--out", type=Path, default=REPO / "data" / "alert_sample")
    p.add_argument("--bands", nargs="+", default=["beta", "rho"])
    p.add_argument("--min-sigma", type=float, default=3.0, help="the real/bogus label's rule (tools/rb_training_set.py)")
    p.add_argument("--cut", help="keep packets with realBogusScore >= this: a number, or an operating point of --bundle")
    p.add_argument("--bundle", type=Path, help="the real/bogus bundle the nights were scored with (for a --cut label)")
    p.add_argument("--name", help="the sample's name, for the README")
    args = p.parse_args(argv)
    cut = resolve_cut(args.cut, args.bundle)
    sys.path.insert(0, str(REPO / "tools"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"nights": []}
    for folder in args.observations:
        for band in args.bands:
            entry = one_band(folder.resolve(), band, args.out, args.min_sigma, cut)
            if entry is None:
                print(f"{folder.name} {band}: no packets or no alerts.csv; skipped")
                continue
            manifest["nights"] = [n for n in manifest["nights"] if (n["observation"], n["band"]) != (folder.name, band)]
            manifest["nights"].append(entry)
            print(f"{folder.name} {band}: {entry['packets']} packets in {entry['packet_files']} files, schema "
                  f"{','.join(entry['schema_versions'])}, {entry['packets_without_truth']} without truth; "
                  f"{entry['by_truth_class']}", flush=True)
    if args.name:
        manifest["name"] = args.name
    for n in manifest["nights"]:  # the cut each night ran at, as a number (a label resolved in its bundle)
        bundle = Path(n["model_bundle"]) if n["model_bundle"] else None
        if bundle is not None and not bundle.is_absolute():
            bundle = REPO / bundle
        try:
            n["score_cut_value"] = resolve_cut(str(n["score_cut"]), bundle) if n["model_bundle"] else None
        except SystemExit:
            n["score_cut_value"] = None
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
    write_readme(args.out, manifest)


if __name__ == "__main__":
    main()
