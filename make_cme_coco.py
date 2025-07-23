#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert CME polar‑coordinate labels → COCO boxes (y_max = H, 태양 포함).
Usage:
  python make_cme_coco.py \
      --img-dir   datasets/CME/train/diff_coordinate_image/c2_coordinate_lev1 \
      --label-dir datasets/CME/train_labels \
      --out-json  datasets/CME/annotations/instances_train.json
"""
import json, os, re, math, argparse, cv2
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------------------------
def polar_to_boxes(obj, W, H):
    """θ,r → list[[x0,y0,x1,y1]], y1 = H (태양까지)"""
    t0, t1 = obj["theta_start"] % 360, obj["theta_end"] % 360
    r_max  = obj["radius_end"]
    px_per_deg = W / 360.0
    y_top, y_bot = max(H - r_max, 0), H

    if t0 <= t1:
        return [[t0*px_per_deg, y_top, t1*px_per_deg, y_bot]]
    # wrap → 두 박스
    return [[0,             y_top, t1*px_per_deg, y_bot],
            [t0*px_per_deg, y_top, W,             y_bot]]

# -----------------------------------------------------------
def build(args):
    img_dir   = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    assert img_dir.exists() and label_dir.exists()

    coco = {"images": [], "annotations": [], "categories":[{"id":1,"name":"CME"}]}
    ann_id = 1

    lbl_files = sorted(f for f in label_dir.iterdir() if f.suffix==".json")
    for lbl_path in tqdm(lbl_files, desc="converting"):
        # -------- derive image path --------
        m = re.match(r"labels_(\d{8})_(\d{6})\.json", lbl_path.name)
        if not m: continue
        yyyymmdd, hhmmss = m.groups()
        year = yyyymmdd[:4]
        img_rel = Path(year) / yyyymmdd / f"coordinate_{yyyymmdd}_{hhmmss}.png"
        img_path = img_dir / img_rel
        if not img_path.exists():
            tqdm.write(f"[warn] image missing: {img_path}")
            continue

        # -------- read image size --------
        H, W = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).shape

        # -------- add image entry --------
        coco["images"].append({
            "id": img_rel.as_posix(), "file_name": img_rel.as_posix(),
            "width": W, "height": H
        })

        # -------- read label, convert --------
        with lbl_path.open() as f: js = json.load(f)
        for obj in js["image"]["labels"]:
            for x0,y0,x1,y1 in polar_to_boxes(obj, W, H):
                w,h = x1-x0, y1-y0
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_rel.as_posix(),
                    "category_id": 1, "bbox":[x0,y0,w,h],
                    "area": w*h, "iscrowd":0
                })
                ann_id += 1

    # -------- save --------
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(coco, f, indent=2)
    print(f"COCO saved → {args.out_json}  (images={len(coco['images'])}, ann={ann_id-1})")

# -----------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir",   required=True, type=Path)
    p.add_argument("--label-dir", required=True, type=Path)
    p.add_argument("--out-json",  required=True, type=Path)
    build(p.parse_args())
