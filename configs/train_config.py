from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 150   # train epochs
batch_size = 4    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm

output_dir = "experiments/cme_convnext_b_q10_bbox5_aug"  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train
coco_path = "datasets/CME/CME_split"  # /PATH/TO/YOUR/COCODIR
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/train",
    ann_file=f"{coco_path}/annotations/instances_train.json",
    transforms=presets.cme_mosaic_pd(),  # see transforms/presets to choose a transform
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/val",
    ann_file=f"{coco_path}/annotations/instances_val.json"
)

# --- 모델 config 지정 ---
model_path = "configs/relation_detr/relation_detr_convnext_b_500_1000.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
# checkpoints/relation_detr_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/relation_detr_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
resume_from_checkpoint = None

learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)
