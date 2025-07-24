# --------------------------------------------------
# ❶ 공통 하이퍼 파라미터
# --------------------------------------------------
num_epochs       = 30        # fine‑tune이므로 짧게
batch_size       = 4
num_workers      = 4
pin_memory       = True
print_freq       = 50
starting_epoch   = 0         # 항상 0부터
max_norm         = 0.1

# --------------------------------------------------
# ❷ 출력 디렉터리
#   (기존과 다른 폴더명이어야 기존 체크포인트와 충돌 X)
# --------------------------------------------------
output_dir = "experiments/cme_finetune_aug_mosaic"

# --------------------------------------------------
# ❸ 데이터셋 (새 train/val 경로 사용)
# --------------------------------------------------
from datasets.coco import CocoDetection
from transforms import presets   # ← 기존 presets 모듈

coco_path = "datasets/MK_data_v2/CME_split"

train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train",
    ann_file=f"{coco_path}/annotations/instances_train.json",
    transforms=presets.cme_ft(),     # 증강 preset
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val",
    ann_file=f"{coco_path}/annotations/instances_val.json",
)

# --------------------------------------------------
# ❹ 모델 설정 (원래 쓰던 스크립트 경로)
# --------------------------------------------------
model_path = "configs/relation_detr/relation_detr_convnext_b_500_1000.py"

# --------------------------------------------------
# ❺ [중요] pretrained_weight  만 지정
#     *directory* 가 아닌 **.pth 파일** 경로!!
# --------------------------------------------------
pretrained_weight      = "experiments/cme_convnext_b_q10_bbox5_augplus/best_ap50.pth"
resume_from_checkpoint = None         # ← 반드시 None (이어 학습 X)

# --------------------------------------------------
# ❻ Optimizer / LR (미세 조정용 ↓)
# --------------------------------------------------
from torch import optim
from optimizer import param_dict

learning_rate = 1e-5                  # backbone 미세 튜닝이면 LR ↓
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[15, 25], gamma=0.1)

param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)

find_unused_parameters = False
