#!/usr/bin/env python3
"""
Convert raw KITTI 2D object-detection annotations (data/download_kitti.sh's
output: <raw_dir>/training/{image_2,label_2}) into a standard YOLO-format
dataset (images/{train,val}, labels/{train,val}, one data.yaml) that
hafl.models.YOLODetector can train/evaluate on directly.

KITTI label format per line (space-separated):
    type truncated occluded alpha bbox_x1 bbox_y1 bbox_x2 bbox_y2
    h w l x y z rotation_y

We keep the three classes most AV-detection papers evaluate on by default
(Car, Pedestrian, Cyclist) and drop DontCare / Misc / Van / Truck /
Person_sitting / Tram; pass --classes to use a different subset or --classes
all to keep every non-DontCare KITTI class.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image

DEFAULT_CLASSES = ["Car", "Pedestrian", "Cyclist"]
ALL_KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_dir", required=True, help="output dir of download_kitti.sh")
    p.add_argument("--out_dir", required=True, help="where to write the YOLO-format dataset")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="KITTI class names to keep (or pass 'all'); others are dropped")
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--symlink", action="store_true",
                    help="symlink images instead of copying (saves disk + time on the DGX)")
    return p.parse_args()


def kitti_to_yolo_box(x1, y1, x2, y2, img_w, img_h):
    x1, x2 = sorted((max(0.0, x1), min(float(img_w), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(img_h), y2)))
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def convert_split(image_ids: List[str], img_src: Path, lbl_src: Path, out_dir: Path,
                   split: str, class_to_idx: Dict[str, int], symlink: bool) -> int:
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_boxes = 0
    for stem in image_ids:
        src_img = img_src / f"{stem}.png"
        if not src_img.exists():
            continue
        dst_img = img_out / f"{stem}.png"
        if not dst_img.exists():
            (dst_img.symlink_to(src_img.resolve()) if symlink else shutil.copy2(src_img, dst_img))

        with Image.open(src_img) as im:
            img_w, img_h = im.size

        lines = []
        label_file = lbl_src / f"{stem}.txt"
        if label_file.exists():
            for raw_line in label_file.read_text().splitlines():
                parts = raw_line.strip().split()
                if len(parts) < 15:
                    continue
                cls_name = parts[0]
                if cls_name not in class_to_idx:
                    continue
                x1, y1, x2, y2 = map(float, parts[4:8])
                cx, cy, w, h = kitti_to_yolo_box(x1, y1, x2, y2, img_w, img_h)
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{class_to_idx[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                n_boxes += 1
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
    return n_boxes


def main():
    args = parse_args()
    classes = ALL_KITTI_CLASSES if args.classes == ["all"] else args.classes
    class_to_idx = {c: i for i, c in enumerate(classes)}

    raw_dir = Path(args.raw_dir)
    img_src = raw_dir / "training" / "image_2"
    lbl_src = raw_dir / "training" / "label_2"
    assert img_src.exists(), f"{img_src} not found -- run data/download_kitti.sh first"

    stems = sorted(p.stem for p in img_src.glob("*.png"))
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * args.val_fraction))
    val_ids, train_ids = stems[:n_val], stems[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train_boxes = convert_split(train_ids, img_src, lbl_src, out_dir, "train", class_to_idx, args.symlink)
    n_val_boxes = convert_split(val_ids, img_src, lbl_src, out_dir, "val", class_to_idx, args.symlink)

    yaml_path = out_dir / "kitti.yaml"
    yaml_path.write_text(
        "path: {}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n".format(out_dir.resolve())
        + "\n".join(f"  {i}: {c}" for c, i in class_to_idx.items())
        + "\n"
    )

    print(f"train: {len(train_ids)} images, {n_train_boxes} boxes")
    print(f"val:   {len(val_ids)} images, {n_val_boxes} boxes")
    print(f"classes: {classes}")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
