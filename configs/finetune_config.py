# configs/finetune_config.py
from torch import optim
from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# ---------- 기본 학습 파라미터 ----------
num_epochs       = 30          # 짧게 fine‑tune
batch_size       = 4
num_workers      = 4
pin_memory       = True
print_freq       = 50
starting_epoch   = 0
max_norm         = 0.1

# ---------- 출력 디렉터리 ----------
output_dir = "experiments/cme_finetune_q20_bbox10"

# ---------- 데이터셋 (새 데이터 경로) ----------
coco_path = "datasets/CME512_1024_split"
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/train",
    ann_file=f"{coco_path}/annotations/instances_train_0.json",
    transforms=presets.cme_mosaic_pd(),      # ← 앞서 만든 강‑증강 preset
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/val",
    ann_file=f"{coco_path}/annotations/instances_val_0.json",
    transforms=presets.cme_eval(),           # val 은 증강 없이
)

# ---------- 모델 설정 ----------
model_path = "configs/relation_detr/relation_detr_convnext_b_512_1024.py"

# ---------- pretrained 가중치 (fine‑tune) ----------
pretrained_weight      = "experiments/cme_convnext_b_q20_bbox10/best_epoch07_ap50.pth"
resume_from_checkpoint = None     # 이어 학습 아님

# ---------- 옵티마이저 / LR 스케줄 ----------
learning_rate = 1e-5             # 파인튜닝이면 LR ↓
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[15, 25], gamma=0.1)

param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
find_unused_parameters = False
