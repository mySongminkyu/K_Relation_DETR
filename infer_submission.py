#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation‑DETR CME inference → submission.csv
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from accelerate import Accelerator
from tqdm import tqdm

# 프로젝트 유틸(리포지토리 내부 경로에 맞게 조정)
from util.lazy_load import Config
from util.logger import setup_logger
from util.utils import load_checkpoint, load_state_dict
from test import create_test_data_loader  # Relation‑DETR 원본 코드에 있음


# ----------------------------------------------------------------------
# 1. Argument parser
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Inference & CSV submission generator")
    # data
    p.add_argument("--image-dir", required=True,
                   help="Root folder with test coordinate PNGs")
    p.add_argument("--workers", type=int, default=4)
    # model
    p.add_argument("--model-config", required=True,
                   help=".py config used for training")
    p.add_argument("--checkpoint", required=True,
                   help="Path to best .pth")
    # output
    p.add_argument("--result", default="./submission.csv",
                   help="Where to save the submission CSV")
    # misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------------
# 2. Dataset
# ----------------------------------------------------------------------
class InferenceDataset(data.Dataset):
    """
    Recursively load every *.png under `root`.
    Returns CHW uint8 torch tensors.
    """
    def __init__(self, root: str):
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"[InferenceDataset] dir not found: {root}")

        import glob
        self.images = sorted(glob.glob(os.path.join(root, "**", "*.png"),
                                       recursive=True))
        if not self.images:
            raise RuntimeError(f"[InferenceDataset] no PNGs under {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img_bytes = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)  # CHW
        return torch.from_numpy(img.copy())


# ----------------------------------------------------------------------
# 3. Helper: xyxy → polar
# ----------------------------------------------------------------------
def xyxy_to_polar(box: np.ndarray, rad: float) -> Tuple[float, float, float, float]:
    """Convert normalized [x1,y1,x2,y2] → (θ_s, θ_e, r_s, r_e)."""
    x1, y1, x2, y2 = box.tolist()
    theta_s = x1 * 360.0
    theta_e = x2 * 360.0
    r_s = (1.0 - y2) * rad
    r_e = (1.0 - y1) * rad
    return theta_s, theta_e, r_s, r_e


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    accelerator = Accelerator()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    setup_logger(distributed_rank=accelerator.local_process_index)

    # data
    ds = InferenceDataset(args.image_dir)
    dl = create_test_data_loader(
        ds, accelerator=accelerator,
        batch_size=1, num_workers=args.workers
    )

    # model
    cfg = Config(args.model_config)
    model = cfg.model.eval()
    ckpt = load_checkpoint(args.checkpoint)
    if isinstance(ckpt, Dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    load_state_dict(model, ckpt)
    model = accelerator.prepare(model)

    # inference loop
    rows_dict = defaultdict(list)  # image_id → list[pred_str]

    with torch.inference_mode():
        for idx, imgs in enumerate(tqdm(dl, disable=not accelerator.is_main_process)):
            out = model(imgs)[0]
            boxes = out["boxes"].cpu().numpy()      # (N,4) normalized
            scores = out["scores"].cpu().numpy()    # (N,)

            fname = os.path.basename(ds.images[idx])
            image_id = fname.replace("coordinate_", "").replace(".png", "")
            h = imgs.shape[-2]
            rad = h / 2.0

            for b, s in zip(boxes, scores):
                th_s, th_e, r_s, r_e = xyxy_to_polar(b, rad)
                rows_dict[image_id].append(
                    f"{s:.4f} {th_s:.2f} {th_e:.2f} {r_s:.2f} {r_e:.2f}"
                )

    # rank‑0: save CSV
    if accelerator.is_main_process:
        rows = [{"image_id": k, "prediction": " ".join(v)}
                for k, v in rows_dict.items()]
        pd.DataFrame(rows).to_csv(args.result, index=False)
        print(f"[✓] submission saved → {args.result}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
