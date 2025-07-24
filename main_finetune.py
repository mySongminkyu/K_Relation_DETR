#!/usr/bin/env python3
# ---------------------------------------------
# CME Relation‑DETR  ▸  Fine‑tune Script
# ---------------------------------------------
import os, datetime, pprint, re, time, argparse
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"

import torch, accelerate
from torch.utils import data
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from accelerate.tracking import TensorBoardTracker

from util.collate_fn import collate_fn
from util.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from util.lazy_load import Config
from util.misc import default_setup
from util.engine import train_one_epoch_acc, evaluate_acc
from util.utils import load_state_dict, HighestCheckpoint

# ------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/finetune_config.py")
    p.add_argument("--mixed-precision", choices=["no","fp16","bf16"], default="fp16")
    p.add_argument("--accumulate-steps", type=int, default=1)
    return p.parse_args()
# ------------------------------------------------------ #
def main():
    args = parse_args()

    # ── Load config (lazy) ──────────────────────────────
    lazy = ("lr_scheduler","optimizer","param_dicts")
    cfg  = Config(file_path=args.config_file, partials=lazy)

    # ── Accelerator & trackers ─────────────────────────
    project_cfg = ProjectConfiguration(project_dir=cfg.output_dir, total_limit=5)
    accelerator = Accelerator(
        log_with="wandb",
        project_config=project_cfg,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_steps,
        kwargs_handlers=[DistributedDataParallelKwargs()],
    )
    accelerator.init_trackers("rel_detr_finetune",
                              config={"output_dir": cfg.output_dir})

    logger = get_logger("finetune")

    # ── Dataloader ─────────────────────────────────────
    params = dict(num_workers=cfg.num_workers,
                  pin_memory=cfg.pin_memory,
                  persistent_workers=True,
                  collate_fn=collate_fn)
    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)
    train_loader = data.DataLoader(
        cfg.train_dataset,
        batch_sampler=GroupedBatchSampler(data.RandomSampler(cfg.train_dataset),
                                          group_ids, cfg.batch_size),
        **params)
    val_loader = data.DataLoader(cfg.test_dataset, 1, shuffle=False, **params)

    # ── Model / Opt / Scheduler ────────────────────────
    model = Config(cfg.model_path).model
    optimizer   = cfg.optimizer(cfg.param_dicts(model))
    lr_scheduler = cfg.lr_scheduler(optimizer)

    # ★  Pretrained weight → load_state_dict (strict=False OK)
    if cfg.pretrained_weight:
        sd = torch.load(cfg.pretrained_weight, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded pre‑trained weights: {cfg.pretrained_weight}")

    # ── DDP prepare ────────────────────────────────────
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler)

    best_ckpt = HighestCheckpoint(accelerator, model)
    logger.info("Start fine‑tuning !")

    for epoch in range(cfg.starting_epoch, cfg.num_epochs):
        train_stats = train_one_epoch_acc(
            model, optimizer, train_loader, epoch,
            print_freq=cfg.print_freq, max_grad_norm=cfg.max_norm,
            accelerator=accelerator)
        lr_scheduler.step()

        # ─ log (train) ─
        if accelerator.is_main_process and hasattr(train_stats, "meters"):
            log_dict = {f"train_{k}": v.global_avg for k,v in train_stats.meters.items()}
            log_dict.update({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})
            accelerator.log(log_dict)

        # ─ evaluate ─
        coco_eval = evaluate_acc(model, val_loader, epoch, accelerator)
        ap, ap50 = coco_eval.coco_eval["bbox"].stats[:2]
        if accelerator.is_main_process:
            accelerator.log({"val_AP": ap, "val_AP50": ap50})

        # save
        accelerator.save_state(safe_serialization=False)
        best_ckpt.update(ap=ap, ap50=ap50)

    accelerator.end_training()

# ------------------------------------------------------ #
if __name__ == "__main__":
    main()
