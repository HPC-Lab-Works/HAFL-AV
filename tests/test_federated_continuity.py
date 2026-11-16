"""
Regression test for the most important (and least obvious) correctness
requirement of the whole suite: a vehicle's local_train() must actually
CONTINUE training from the global weights it was just given, not silently
rebuild a fresh randomly-initialized model.

This is not hypothetical -- it was a real bug caught while building this
suite. ultralytics' `Model.train()` only seeds the trainer's model from the
in-memory `self.model` when `self.model.ckpt` is truthy (see
ultralytics.engine.model.Model.train); a YOLO object built from a bare
`.yaml` (as hafl.models.YOLODetector does) starts with `ckpt = {}`, which is
falsy, so without `YOLODetector.load_state_dict`'s `self.model.ckpt = {...}`
line, every local_train() call rebuilt the model from scratch and the
"federated" part of federated learning silently never happened. A second,
related bug: if the global model isn't built at the dataset's actual class
count (`nc`) from the start, ultralytics rebuilds just the detection HEAD at
training time and only transfers the backbone -- the part of the network
that actually makes predictions keeps getting reinitialized every round.

Both are covered here by loading a known marker value into every parameter
and asserting the *entire* state_dict (not just one tensor) is still exactly
that marker value at the moment training starts, on a synthetic 3-class
dataset (so the nc=3-vs-yolov8n.yaml's-default-nc=80 case is exercised, not
masked by accidentally matching COCO's 80 classes the way ultralytics'
bundled "coco8" sample dataset would).

Marked slow/integration: builds a real (tiny) YOLOv8 model and runs one real
training step on CPU. Takes ~5-10s.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from PIL import Image

from hafl.models import YOLODetector


def _build_tiny_dataset(root: Path, num_images: int = 8, num_classes: int = 3) -> Path:
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    rng = random.Random(0)
    for i in range(num_images):
        Image.new("RGB", (64, 64), color=(i * 10 % 255, 0, 0)).save(img_dir / f"img{i}.jpg")
        cls = rng.randrange(num_classes)
        (lbl_dir / f"img{i}.txt").write_text(f"{cls} 0.5 0.5 0.3 0.3\n")

    yaml_path = root / "data.yaml"
    names = "\n".join(f"  {i}: class{i}" for i in range(num_classes))
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/train\nnames:\n{names}\n"
    )
    return yaml_path


@pytest.mark.slow
def test_local_train_continues_from_loaded_weights_not_reinit(tmp_path):
    data_yaml = _build_tiny_dataset(tmp_path, num_classes=3)

    detector = YOLODetector(model_yaml="yolov8n.yaml", device="cpu", nc=3)
    original_sd = detector.state_dict()
    marker_sd = {k: (torch.full_like(v, 777.0) if v.is_floating_point() else v)
                 for k, v in original_sd.items()}
    detector.load_state_dict(marker_sd)

    captured = {}

    def snapshot(trainer):
        captured["sd_at_start"] = {k: v.clone() for k, v in trainer.model.state_dict().items()}

    detector.model.add_callback("on_train_start", snapshot)
    detector.local_train(
        data_yaml=str(data_yaml), epochs=1, batch=2, lr0=0.01, momentum=0.9,
        project=str(tmp_path / "run"), name="v0", workers=0,
    )

    assert "sd_at_start" in captured, "on_train_start callback never fired"
    sd_at_start = captured["sd_at_start"]

    assert sd_at_start.keys() == marker_sd.keys(), (
        "the rebuilt training-time model has different parameter names than the "
        "global model -- architecture mismatch between global model and client model"
    )
    mismatched = [
        k for k in marker_sd
        if marker_sd[k].is_floating_point() and not torch.allclose(sd_at_start[k], marker_sd[k])
    ]
    assert not mismatched, (
        f"{len(mismatched)}/{len(marker_sd)} parameter tensors were NOT the marker value at "
        f"train start -- these were silently reinitialized instead of continuing from the "
        f"global model's weights: {mismatched[:5]}{'...' if len(mismatched) > 5 else ''}"
    )
