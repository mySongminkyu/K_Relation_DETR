#!/usr/bin/env python3
# -----------------------------------------------------------
# CME Relation‑DETR Training Script  (ConvNeXt‑Base backbone)
# -----------------------------------------------------------
import argparse, datetime, os, pprint, re, time
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
from torch.utils import data
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import ProjectConfiguration

# --- project modules
from util.collate_fn import collate_fn
from util.engine import evaluate_acc, train_one_epoch_acc
from util.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from util.lazy_load import Config
from util.misc import default_setup, encode_labels, fixed_generator, seed_worker
from util.utils import HighestCheckpoint, load_checkpoint, load_state_dict

# ------------------------------------------------------------------
# 0. argument parser
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train CME detector")
    parser.add_argument("--config-file", default="configs/train_config.py")
    parser.add_argument("--mixed-precision", type=str, default=None,
                        choices=["no", "fp16", "bf16", "fp8"])
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    dynamo_backend = ["no", "eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser",
                      "cudagraphs", "ofi", "fx2trt", "onnxrt", "tensorrt", "ipex", "tvm"]
    parser.add_argument("--dynamo-backend", type=str, default="no", choices=dynamo_backend)
    return parser.parse_args()

# ------------------------------------------------------------------
# 1. checkpoint / output‑dir helper
# ------------------------------------------------------------------
def update_checkpoint_path(cfg: Config):
    weight_path = getattr(cfg, "resume_from_checkpoint", None)

    # output_dir 결정
    if weight_path and os.path.isdir(weight_path):
        cfg.output_dir = weight_path
    elif getattr(cfg, "output_dir", None) is None:
        accelerate.utils.wait_for_everyone()
        cfg.output_dir = os.path.join(
            "checkpoints",
            os.path.basename(cfg.model_path).split(".")[0],
            "train",
            datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"),
        )

    # resume checkpoint 디렉터리 처리
    if weight_path and os.path.isdir(weight_path):
        if "checkpoints" in os.listdir(weight_path):         # 최신 checkpoint 찾기
            ckpts = sorted(
                (os.path.join(weight_path, "checkpoints", d)
                 for d in os.listdir(os.path.join(weight_path, "checkpoints"))),
                key=lambda p: int(re.findall(r"([0-9]+)$", p)[0])
            )
            cfg.resume_from_checkpoint = ckpts[-1]
        else:
            cfg.resume_from_checkpoint = None
    return cfg

# ------------------------------------------------------------------
# 2. main training routine
# ------------------------------------------------------------------
def train():
    args = parse_args()

    # lazy load config (모델·옵티마이저 빌드 지연)
    lazy = ("lr_scheduler", "optimizer", "param_dicts")
    cfg = Config(file_path=args.config_file, partials=lazy)
    cfg = update_checkpoint_path(cfg)

    # --- Accelerator 초기화 (wandb 로그 포함) -----------------------
    project_cfg = ProjectConfiguration(project_dir=cfg.output_dir,
                                       total_limit=5,
                                       automatic_checkpoint_naming=True)
    tb_tracker = TensorBoardTracker(run_name="tf_log", logging_dir=cfg.output_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_parameters)

    accelerator = Accelerator(
        log_with="wandb",                 # wandb tracker만 즉시 등록
        project_config=project_cfg,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_steps,
        dynamo_backend=args.dynamo_backend,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    # TensorBoard tracker는 수동으로 추가
    tb_tracker = TensorBoardTracker(run_name="tf_log", logging_dir=cfg.output_dir)
    accelerator.trackers.append(tb_tracker)

    # wandb run 생성
    accelerator.init_trackers(
        project_name="rel_detr_cme",
        config=vars(args) | {"output_dir": cfg.output_dir},
    )


    # logger
    default_setup(args, cfg, accelerator)
    logger = get_logger(os.path.basename(os.getcwd()) + "." + __name__)

    # ----------------------------------------------------------------
    # 2‑A. Dataset & Dataloader
    # ----------------------------------------------------------------
    params = dict(num_workers=cfg.num_workers,
                  pin_memory=cfg.pin_memory,
                  persistent_workers=True,
                  collate_fn=collate_fn)
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        params.update({"worker_init_fn": seed_worker,
                       "generator": fixed_generator()})

    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)
    train_batch_sampler = GroupedBatchSampler(
        data.RandomSampler(cfg.train_dataset), group_ids, cfg.batch_size
    )
    train_loader = data.DataLoader(cfg.train_dataset, batch_sampler=train_batch_sampler, **params)
    val_loader   = data.DataLoader(cfg.test_dataset, 1, shuffle=False, **params)


    # ----------------------------------------------------------------
    # 2‑B. Model, Optimizer, LR scheduler
    # ----------------------------------------------------------------
    model = Config(cfg.model_path).model
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = cfg.optimizer(cfg.param_dicts(model))
    lr_scheduler = cfg.lr_scheduler(optimizer)

    # load pretrained weight (fine‑tune) or resume training
    if (wp := getattr(cfg, "resume_from_checkpoint", None)) and os.path.isfile(wp):
        load_state_dict(model, load_checkpoint(wp))
        logger.info(f"loaded pretrained weights from {wp}")

    # register label names
    cat_ids = list(range(max(cfg.train_dataset.coco.cats.keys()) + 1))
    classes = tuple(cfg.train_dataset.coco.cats.get(c, {"name": "none"})["name"]
                    for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    # Accelerate prepare (DDP wrap)
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # # ---------- 모델·로더 준비 이후 디버그 스니펫 ----------
    # images, _ = next(iter(train_loader))
    # images = [im.to(next(model.parameters()).device) for im in images]

    # model.eval()

    # # ① postprocessor 우회용 더미 모듈
    # class _IdentityPP(torch.nn.Module):
    #     def forward(self, x, y):     # 인자 signature 맞춰줌
    #         return x                 # 그대로 출력

    # # ② 원본 백업 후 교체
    # orig_pp = model.postprocessor
    # model.postprocessor = _IdentityPP()

    # with torch.no_grad():
    #     out = model(images)          # pred_logits 포함 원본 dict 반환

    # # ③ postprocessor 복구
    # model.postprocessor = orig_pp
    # # --------------------------------------------------------

    # logits = out["pred_logits"]      # [B, Q, C]
    # print("logits shape:", logits.shape)
    # print("sample logits[0,0] =", logits[0, 0])

    # prob = logits.sigmoid()
    # top_val, top_idx = torch.topk(prob.view(logits.shape[0], -1), 10)
    # labels = (top_idx % logits.shape[2]).cpu()
    # print("labels in top‑10 scores:", labels[0].tolist())

    # resume state (if directory)
    if (wp := getattr(cfg, "resume_from_checkpoint", None)) and os.path.isdir(wp):
        accelerator.load_state(wp)
        ep_start = int(os.path.basename(wp).split("_")[-1]) + 1
        cfg.starting_epoch = ep_start
        accelerator.project_configuration.iteration = ep_start
        logger.info(f"resume training from {wp} (epoch {ep_start})")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"model params: {n_params}")
        logger.info(f"optimizer   : {optimizer}")
        logger.info(f"lr_scheduler: {pprint.pformat(lr_scheduler.state_dict())}")

    # save label map
    if accelerator.is_main_process:
        with open(os.path.join(cfg.output_dir, "label_names.txt"), "w") as f:
            for k, v in cfg.train_dataset.coco.cats.items():
                f.write(f"{k} {v['name']}\n")

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    start_time = time.perf_counter()
    best_ckpt = HighestCheckpoint(accelerator, model)

    logger.info("Start training  ✨")
    for epoch in range(cfg.starting_epoch, cfg.num_epochs):
        train_stats = train_one_epoch_acc(
            model, optimizer, train_loader, epoch,
            print_freq=cfg.print_freq, max_grad_norm=cfg.max_norm,
            accelerator=accelerator,
        )
        lr_scheduler.step()

        # ── wandb 로그: train
        # if accelerator.is_main_process:
        #     accelerator.log({
        #         "epoch": epoch,
        #         **{f"train_{k}": v for k, v in train_stats.items()},
        #         "lr": optimizer.param_groups[0]["lr"],
        #     })
        if accelerator.is_main_process:
            # MetricLogger → epoch 평균값 dict 로 변환
            if hasattr(train_stats, "meters"):
                train_stats = {k: v.global_avg for k, v in train_stats.meters.items()}

            accelerator.log({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                "lr": optimizer.param_groups[0]["lr"],
            })

        # --- 평가
        coco_eval = evaluate_acc(model, val_loader, epoch, accelerator)
        ap, ap50 = coco_eval.coco_eval["bbox"].stats[:2]

        # ── wandb 로그: val
        if accelerator.is_main_process:
            accelerator.log({"val_AP": ap, "val_AP50": ap50})

        # save checkpoint & best
        accelerator.save_state(safe_serialization=False)
        best_ckpt.update(ap=ap, ap50=ap50)

    total_t = str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))
    logger.info(f"Training finished in {total_t}")
    accelerator.end_training()

# ------------------------------------------------------------------
if __name__ == "__main__":
    train()
