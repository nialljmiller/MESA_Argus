"""Train a real/bogus random forest on a training set (tools/rb_training_set.py) and write it as an hdps model bundle,
the directory hdps's AlertVettingTask loads (model_bundle; hdps.alerts.classifier.RFBundleClassifier):

    manifest.json   model_file, feature_names (hdps's FEATURE_NAMES order), feature_spec, model_version,
                    operating_points {label: score cut}, metrics
    model.joblib    a scikit-learn RandomForestClassifier: predict_proba(X)[:, 1] is the real score

Held-out test: whole observations when there are several (--test-fraction of them), else whole exposures, so the
test never shares a frame (or a field) with training. Operating points: "faN" is the score cut that leaves N false
alerts per square degree per exposure on the test set (hdps's convention, e.g. score_cut = "fa10"), from the bogus
candidates above the cut over the test frames' readout-window area; "p50" etc. keep that fraction of real ones.

    python tools/train_rb.py data/rb/training_set.parquet -o data/rb/bundle_v1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def frame_area_deg2(observation_dirs):
    """Readout-window area (deg^2) of each observation (its resolved [field] window, hdps minipixes of 28.125")."""
    from mesa_argus import runconfig

    out = {}
    for d in observation_dirs:
        obs = runconfig.load(Path(d) / "observation.toml")
        side_deg = obs["field"]["window"] * 32 * (900.0 / 1024.0) / 3600.0
        out[d] = side_deg ** 2
    return out


def main(argv=None):
    from hdps.alerts.features import FEATURE_NAMES, FEATURE_SPEC_VERSION
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    import joblib

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("training_set", type=Path, help="training_set.parquet (tools/rb_training_set.py)")
    p.add_argument("-o", "--out", type=Path, required=True, help="the bundle directory to write")
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--min-leaf", type=int, default=5, help="min_samples_leaf: smooths the score")
    p.add_argument("--test-fraction", type=float, default=0.25, help="held-out share (observations, else exposures)")
    p.add_argument("--fa", type=float, nargs="+", default=[1.0, 10.0, 100.0], help="false alerts / deg^2 / exposure")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    t = pd.read_parquet(args.training_set)
    manifest_in = json.loads(args.training_set.with_suffix(".json").read_text())
    if manifest_in["feature_spec"] != FEATURE_SPEC_VERSION:
        sys.exit(f"training set feature spec {manifest_in['feature_spec']} != hdps's {FEATURE_SPEC_VERSION}")
    names = list(FEATURE_NAMES)
    t["frame"] = t["observation_dir"] + "|" + t["band"] + "|" + t["exposure"].astype(str)
    rng = np.random.default_rng(args.seed)
    observations = sorted(t["observation_dir"].unique())
    if len(observations) >= 4:  # hold out whole observations (fields / nights the model never saw)
        held = set(rng.choice(observations, max(1, round(args.test_fraction * len(observations))), replace=False))
        test = t["observation_dir"].isin(held).to_numpy()
        split = f"by observation: {len(held)} of {len(observations)} held out"
    else:  # one or a few observations: hold out whole exposures
        frames = t["frame"].unique()
        held = set(rng.choice(frames, max(1, round(args.test_fraction * len(frames))), replace=False))
        test = t["frame"].isin(held).to_numpy()
        split = f"by exposure: {len(held)} of {len(frames)} held out (fewer than 4 observations)"
    X, y = t[names].to_numpy(float), t["label"].to_numpy(int)
    print(f"{len(t)} rows ({y.sum()} real, {(1 - y).sum()} bogus); test split {split}", flush=True)
    if y[~test].sum() == 0 or y[test].sum() == 0:
        sys.exit("no real rows in the training or the test part: more (injected) observations are needed")

    t0 = time.time()
    model = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=args.min_leaf, class_weight="balanced",
                                   n_jobs=-1, random_state=args.seed)
    model.fit(X[~test], y[~test])
    score = model.predict_proba(X[test])[:, 1]
    yt = y[test]
    area = frame_area_deg2(t.loc[test, "observation_dir"].unique())
    test_frames = t.loc[test, ["frame", "observation_dir"]].drop_duplicates()
    sky = float(sum(area[d] for d in test_frames["observation_dir"]))  # deg^2 x exposures on the test set
    bogus_scores, real_scores = np.sort(score[yt == 0])[::-1], score[yt == 1]
    points = {}
    for fa in args.fa:  # the cut leaving fa false alerts per deg^2 per exposure
        n_allowed = int(np.floor(fa * sky))
        cut = float(bogus_scores[n_allowed]) if n_allowed < len(bogus_scores) else 0.0
        points[f"fa{fa:g}"] = cut
    for keep in (0.5, 0.9):
        points[f"p{round(keep * 100)}"] = float(np.quantile(real_scores, 1 - keep))
    metrics = dict(train_rows=int((~test).sum()), test_rows=int(test.sum()), test_real=int(yt.sum()),
                   test_bogus=int((1 - yt).sum()), test_deg2_exposures=sky, split=split,
                   roc_auc=float(roc_auc_score(yt, score)), average_precision=float(average_precision_score(yt, score)),
                   at_points={k: dict(cut=c, real_kept=float(np.mean(real_scores >= c)),
                                      false_per_deg2_exposure=float(np.sum(bogus_scores >= c) / sky))
                              for k, c in points.items()},
                   feature_importance={n: float(v) for n, v in sorted(zip(names, model.feature_importances_),
                                                                       key=lambda kv: -kv[1])},
                   fit_seconds=round(time.time() - t0, 1))
    # the bundle is trained on everything (the test split above only measures it)
    model.fit(X, y)
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out / "model.joblib", compress=3)
    version = f"mesa-argus-rf-{time.strftime('%Y%m%d')}-{len(t)}"
    manifest = dict(model_file="model.joblib", feature_names=names, feature_spec=FEATURE_SPEC_VERSION,
                    model_version=version, operating_points=points, metrics=metrics,
                    training=dict(set=str(args.training_set), label_rule=manifest_in["label_rule"],
                                  counts=manifest_in["counts"], trees=args.trees, min_samples_leaf=args.min_leaf,
                                  class_weight="balanced", note="trained on simulated data (MESA_ARGUS injection); "
                                                                "measure it on real data before trusting its cuts"))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"ROC AUC {metrics['roc_auc']:.4f}, average precision {metrics['average_precision']:.4f}")
    for k, v in metrics["at_points"].items():
        print(f"  {k}: cut {v['cut']:.3f}: real kept {v['real_kept']:.3f}, false {v['false_per_deg2_exposure']:.2f} /deg2/exposure")
    print(f"bundle -> {args.out} (model_version {version})")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
