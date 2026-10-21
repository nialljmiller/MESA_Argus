"""Running hdps on raw camera frames, as its worker does (hdps.orchestrate.worker.GPUWorker), with hdps's own classes:

- the Task chain: build_task_list on the observation's [[hdps.tasks]], run by PipelineExecutor.execute per frame,
  with a product deque the Tasks emit into;
- each frame: FullFrameImage.from_fits, then the worker's _prepare_frame stamps (CAMID, RATCHNUM, coadd_camera_id,
  coadd_ratchet_num, coadd_field_id);
- the coadd: a CoaddAccumulator ([hdps.coadd]) with its weight strategy injected into CoaddTapTask
  (set_accumulator), finalized at each camera's ratchet change and at the end (finalize_ratchet), written with
  coadd_writer.write_segment_fits;
- SFFT's references: SFFTSubtractionTask configured with reference_dir (its CoaddReferenceProvider);
- products: hdps's DiskWriterBackend, into products/<product type>/.

The one toy: AlertVettingTask's real/bogus classifier. No trained model bundle exists, so the task gets AllReal, which
scores every candidate 1 (real/bogus not run; decision 2026-10-20).
"""
from collections import deque
from pathlib import Path

import numpy as np

from mesa_argus import schema as S


class AllReal:
    """Toy stand-in for hdps.alerts.classifier.RFBundleClassifier: every candidate scores 1 (no model exists)."""

    model_version = "none"

    def __init__(self):
        from hdps.alerts.features import FEATURE_NAMES

        self.feature_names = list(FEATURE_NAMES)

    def cut_for(self, label_or_value):
        return float(label_or_value) if not isinstance(label_or_value, str) else 0.5

    def score_features(self, feats):
        return np.ones(len(feats["x"]), dtype=float)

    def score_features_with_confidence(self, feats):
        return self.score_features(feats), None


def task_configs(obs):
    """[[hdps.tasks]] as build_task_list takes it ("none" -> None)."""
    return [dict(type=t["type"], config={k: (None if v == S.NONE else v) for k, v in t["config"].items()})
            for t in obs["hdps"]["tasks"]]


class Runner:
    """hdps's pipeline over raw frames. stop_after: the last Task type to run (None: the whole chain).
    reference_dir: SFFT's references (None: SFFT finds none and skips, as hdps's bootstrap pass 1)."""

    def __init__(self, obs, out_dir, reference_dir=None, stop_after=None, field_id="mesa_argus"):
        from hdps.orchestrate.coadd import CoaddAccumulator, InverseVarianceStrategy, UniformStrategy
        from hdps.orchestrate.pipeline import PipelineExecutor, build_task_list

        configs = task_configs(obs)
        if stop_after is not None:
            configs = configs[: [c["type"] for c in configs].index(stop_after) + 1]
        for c in configs:
            if c["type"] == "SFFTSubtractionTask" and reference_dir is not None:
                c["config"]["reference_dir"] = str(reference_dir)
        self.tasks = build_task_list(configs)
        self.executor = PipelineExecutor(self.tasks)
        self.out_dir = Path(out_dir)
        self.coadd_dir = self.out_dir / "coadds"
        self.field_id = field_id
        co = obs["hdps"]["coadd"]
        self.coadd = CoaddAccumulator(worker_id="mesa_argus", checkpoint_dir=None, checkpoint_interval=0,
                                      accumulation_dtype=co["accumulation_dtype"], clip_kappa=co["clip_kappa"],
                                      clip_min_frames=co["clip_min_frames"])
        self.coadd.current_output_dir = str(self.coadd_dir)
        strategy = InverseVarianceStrategy() if co["strategy"] == "inverse_variance" else UniformStrategy()
        for task in self.tasks:
            if hasattr(task, "set_accumulator"):
                task.set_accumulator(self.coadd, strategy)
            if type(task).__name__ == "AlertVettingTask":
                task._classifier = AllReal()
        self._ratchet = {}  # camera -> current ratchet
        self.coadds_written = []

    def _finalize(self, camera_id, ratchet):
        from hdps.orchestrate.coadd_writer import write_segment_fits

        for seg in self.coadd.finalize_ratchet(ratchet, camera_id=camera_id):
            self.coadds_written.append(write_segment_fits(seg, str(self.coadd_dir)))

    def process(self, path):
        """One raw frame through the chain. Returns (the frame after the frame-level Tasks: sources, WCS; the
        chain's output, None if a Task failed (the reason in the frame's metadata mesa_argus_failed); the products
        emitted)."""
        from hdps.core.images import FullFrameImage
        from hdps.orchestrate.data_products import DiskWriterBackend

        image = FullFrameImage.from_fits(str(path))
        camera_id, ratchet = str(image.metadata["CAMID"]), image.metadata["RATCHNUM"]
        ratchet_num = _ratchet_number(ratchet)
        if camera_id in self._ratchet and self._ratchet[camera_id] != ratchet_num:  # the camera's ratchet changed
            self._finalize(camera_id, self._ratchet[camera_id])
        self._ratchet[camera_id] = ratchet_num
        image.metadata.setdefault("CAMID", camera_id)  # the worker's _prepare_frame stamps
        image.metadata.setdefault("RATCHNUM", ratchet)
        image.metadata["coadd_camera_id"] = camera_id
        image.metadata["coadd_ratchet_num"] = ratchet_num
        image.metadata["coadd_field_id"] = self.field_id
        products = deque()
        try:
            result = self.executor.execute(image, product_deque=products)
        except Exception as e:  # the worker counts a failed frame and moves on (a skip, not a fault)
            image.metadata["mesa_argus_failed"] = f"{type(e).__name__}: {e}"
            result = None
        for product in products:
            DiskWriterBackend(str(self.out_dir / "products" / product.product_type)).write(product)
        return image, result, list(products)

    def close(self):
        """Finalize every open coadd (the end of the run)."""
        for camera_id, ratchet in list(self._ratchet.items()):
            self._finalize(camera_id, ratchet)
        self._ratchet.clear()


def _ratchet_number(ratchet):
    """RATCHNUM as an integer (the camera writes <date>_<n>; hdps keys coadds on an int)."""
    text = str(ratchet)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits[-12:]) if digits else 0
