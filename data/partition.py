#!/usr/bin/env python3
"""
Split a prepared YOLO-format dataset (output of prepare_kitti.py /
prepare_bdd100k.py) across `--num_vehicles` simulated vehicles using the
Dirichlet(alpha) label-skew partition in hafl.topology, and write one
data-config yaml per vehicle plus a shared validation yaml -- the inputs
hafl.methods.*.run() expects (ExperimentConfig.vehicle_data_yamls / val_data_yaml).

Each vehicle gets its own private directory of symlinks --
<out_dir>/vehicle{vid}/images/train/<file> and .../labels/train/<file> --
pointing at that vehicle's image/label subset in the shared prepared dataset,
rather than a single shared `images/train`+`labels/train` pair referenced by
a per-vehicle text file. No image bytes are copied (symlinks only, a few KB
of inode overhead per vehicle), but each vehicle's `train:` now resolves to
a directory it does not share with any other vehicle.

This was NOT cosmetic -- it fixes a real race/corruption bug hit when
actually running the Table-1 sweep on 30 concurrently-training vehicles.
Ultralytics' YOLODataset.get_labels() derives its on-disk label-cache path
from the *parent directory of the first label file* alone:
    cache_path = label_files[0].parent.with_suffix(".cache")
When every vehicle's `train:` pointed at a text file listing a subset of
images that all lived under the one shared `images/train` directory,
img2label_paths() mapped every vehicle's labels back into the one shared
`labels/train` directory too -- so EVERY vehicle, regardless of its actual
(different) image subset, computed the identical cache_path
`labels/train.cache`. Each vehicle's subset hashes differently, so one
vehicle's cache write made every other vehicle's next cache *read* fail its
hash check and trigger a rebuild-and-overwrite of that same shared file --
and when two vehicles' rebuilds landed at the same instant (common with 30
workers hammering the same path every round), one process's
`path.unlink()` ran after another process had already removed the file,
crashing the whole round with FileNotFoundError /
concurrent.futures.process.BrokenProcessPool. Confirmed against a real run:
it trained cleanly through many rounds before two rebuilds happened to
collide. Giving each vehicle its own labels directory makes its cache_path
unique (<out_dir>/vehicle{vid}/labels/train.cache), eliminating the
collision entirely -- and as a side effect, each vehicle's cache now
persists validly across rounds instead of being rebuilt every time.

Usage:
    python data/partition.py \\
        --data_yaml /data/kitti/kitti.yaml \\
        --out_dir   /data/kitti/partitions/alpha0.5_n30 \\
        --num_vehicles 30 --alpha 0.5 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hafl.topology import dirichlet_label_partition  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_yaml", required=True, help="kitti.yaml / bdd100k.yaml from the prepare_* scripts")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_vehicles", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.5,
                    help="Dirichlet concentration; smaller = more severe non-IID skew. "
                         "Use a large value (e.g. 100) to approximate IID.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_images_per_vehicle", type=int, default=20)
    return p.parse_args()


def dominant_class_per_image(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return 0
    classes = [int(l.split()[0]) for l in lines]
    return max(set(classes), key=classes.count)  # mode


def link_vehicle_shard(img_paths, idx, labels_dir: Path, vehicle_dir: Path) -> None:
    """Populate vehicle_dir/{images,labels}/train with symlinks to this
    vehicle's image/label subset (see module docstring for why this needs
    to be a *private* directory per vehicle, not a shared one)."""
    img_link_dir = vehicle_dir / "images" / "train"
    lbl_link_dir = vehicle_dir / "labels" / "train"
    img_link_dir.mkdir(parents=True, exist_ok=True)
    lbl_link_dir.mkdir(parents=True, exist_ok=True)

    for i in idx:
        img_path = img_paths[i]
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        img_link = img_link_dir / img_path.name
        lbl_link = lbl_link_dir / f"{img_path.stem}.txt"
        if not img_link.exists():
            img_link.symlink_to(img_path.resolve())
        if not lbl_link.exists():
            if lbl_path.exists():
                lbl_link.symlink_to(lbl_path.resolve())
            else:
                # prepare_kitti.py / prepare_bdd100k.py always write a label
                # file (possibly empty) for every image, so this is just a
                # defensive fallback, not the expected path.
                lbl_link.write_text("")


def main():
    args = parse_args()
    data_cfg = yaml.safe_load(Path(args.data_yaml).read_text())
    root = Path(data_cfg["path"])
    train_img_dir = root / data_cfg["train"]

    img_paths = sorted(p for p in train_img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    assert img_paths, f"No images found under {train_img_dir}"

    labels_dir = Path(str(train_img_dir).replace("/images/", "/labels/"))
    dominant_classes = np.array([
        dominant_class_per_image(labels_dir / f"{p.stem}.txt") for p in img_paths
    ])

    partitions = dirichlet_label_partition(
        dominant_classes, num_clients=args.num_vehicles, alpha=args.alpha, seed=args.seed,
        min_samples_per_client=args.min_images_per_vehicle,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Absolute: vehicle yaml's `path:` is each vehicle's own private
    # directory (see link_vehicle_shard), not `root`, so `val:` must be
    # given as an absolute path rather than relative to that `path:`.
    val_yaml_path = (root / data_cfg["val"]).resolve()

    vehicle_yaml_paths = []
    sample_counts = []
    for vid, idx in enumerate(partitions):
        vehicle_dir = out_dir / f"vehicle{vid}"
        link_vehicle_shard(img_paths, idx, labels_dir, vehicle_dir)

        yaml_path = out_dir / f"vehicle{vid}.yaml"
        yaml_path.write_text(
            f"path: {vehicle_dir.resolve()}\n"
            f"train: images/train\n"
            f"val: {val_yaml_path}\n"
            "names:\n" + "\n".join(f"  {k}: {v}" for k, v in data_cfg["names"].items()) + "\n"
        )
        vehicle_yaml_paths.append(str(yaml_path.resolve()))
        sample_counts.append(len(idx))

    manifest = {
        "data_yaml": str(Path(args.data_yaml).resolve()),
        "num_vehicles": args.num_vehicles,
        "alpha": args.alpha,
        "seed": args.seed,
        "vehicle_data_yamls": vehicle_yaml_paths,
        "vehicle_sample_counts": sample_counts,
        "val_data_yaml": str(Path(args.data_yaml).resolve()),
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    print(f"Wrote {args.num_vehicles} vehicle configs to {out_dir}")
    print(f"Sample counts: min={min(sample_counts)} max={max(sample_counts)} "
          f"mean={sum(sample_counts)/len(sample_counts):.1f}")
    print(f"Manifest: {out_dir / 'manifest.yaml'} (pass this to scripts/run_experiment.py --manifest)")


if __name__ == "__main__":
    main()
