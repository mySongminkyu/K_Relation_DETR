# CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$(pwd) python models/backbones/convnext_classifier_amp.py

# -------------------------------------------------
# 1. 라이브러리
# -------------------------------------------------
import os, json, random, math
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# ★ AMP
from torch.cuda.amp import autocast, GradScaler      # ← 추가


# -------------------------------------------------
# 0. Seed 설정
# -------------------------------------------------
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# 2. Backbone: ConvNeXt‑Large (scratch)
# -------------------------------------------------
from models.backbones.convnext import ConvNeXtBackbone

set_seed(42)

backbone = ConvNeXtBackbone(
    arch="conv_s",
    weights=None,
    return_indices=(1, 2, 3),
    freeze_indices=()
)

class ConvNeXtClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            x = feats[-1]
        elif isinstance(feats, dict):
            x = next(reversed(feats.values()))
        else:
            x = feats
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = ConvNeXtClassifier(backbone)

# -------------------------------------------------
# 3. 초기화 & 장치
# -------------------------------------------------
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)

model.apply(init_weights)

device = "cuda"
model.to(device)

# -------------------------------------------------
# 4. Dataset
# -------------------------------------------------
class CocoBinaryClassificationDataset(Dataset):
    def __init__(self, label_dir, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        txt_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])
        self.samples = []
        for txt_file in txt_files:
            basename = os.path.splitext(txt_file)[0]
            try:
                _, yyyymmdd, hhmmss = basename.split('_')
                year = yyyymmdd[:4]
                with open(os.path.join(label_dir, txt_file), "r", encoding="utf-8") as f:
                    label = 0 if json.load(f).get("isEvent", "") == "N" else 1
                img_path = os.path.join(image_dir, year, yyyymmdd,
                                        f"coordinate_{yyyymmdd}_{hhmmss}.png")
                self.samples.append((img_path, label))
            except Exception as e:
                print(f"Skip {txt_file}: {e}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize((423, 936)),
    transforms.ToTensor(),
])

dataset = CocoBinaryClassificationDataset(
    label_dir="/data/minkyu/Kaist/Relation-DETR/datasets/final-dataset/train/labels/",
    image_dir="/data/minkyu/Kaist/Relation-DETR/datasets/final-dataset/train/diff_coordinate_image/c2_coordinate_lev1/",
    transform=transform,
)
print("Dataset size:", len(dataset))

# -------------------------------------------------
# 5. Train / Validation split (90 : 10)
# -------------------------------------------------
seed = 42
g = torch.Generator().manual_seed(seed)
train_size = int(0.9 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

def count_labels(ds, name):
    cnt = Counter(l for _, l in ds)
    print(f"[{name}] 0: {cnt[0]}, 1: {cnt[1]}")
count_labels(train_ds, "Train"); count_labels(val_ds, "Val")

# -------------------------------------------------
# 6. Optimizer, 스케줄러, 손실
# -------------------------------------------------
# criterion = nn.BCEWithLogitsLoss()
neg = 12303
pos =  5840
pos_weight = torch.tensor([neg / pos]).to(device)   # ≈ 2.107

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

total_epochs = 100
def lr_lambda(epoch):
    warmup = 5
    if epoch < warmup:
        return epoch / float(max(1, warmup))
    t = (epoch - warmup) / (total_epochs - warmup)
    return 0.5 * (1 + math.cos(math.pi * t))
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# ★ AMP
scaler = GradScaler()                       # ← 스케일러 선언

# -------------------------------------------------
# 7. DataLoader
# -------------------------------------------------
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                          num_workers=4, pin_memory=True)

# DataLoader 만든 뒤
with torch.no_grad(), autocast():
    x, y = next(iter(val_loader))
    logits = model(x.to(device)).squeeze(1)
    print("Before training  ⇒  sigmoid mean:", torch.sigmoid(logits).mean().item())


# -------------------------------------------------
# 8. 학습 & 평가 루프 (AMP 적용)
# -------------------------------------------------
def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad(), autocast():       # ★ AMP
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            pred = torch.sigmoid(model(x).squeeze(1)) > 0.5
            correct += (pred == y.bool()).sum().item()
            total   += y.size(0)
    return correct / total

import wandb
wandb.init(project="convnext_binary_cls", name="ConvNeXt-S_sampler_amp", config={
    "backbone": "convnext_small",
    "batch_size": 32,
    "epochs": total_epochs,
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "pos_weight": float(pos_weight.item()),
    "augmentation": "no",
    "sampler": "WeightedRandomSampler"
})

# 모델 저장 디렉토리 및 Top-K 리스트
save_dir = "./convnext_s_backbone"
os.makedirs(save_dir, exist_ok=True)
top_k = 10
best_models = []  # (val_acc, path) 저장 리스트



for epoch in range(total_epochs):
    model.train(); epoch_loss = correct = total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device).float()
        optimizer.zero_grad()

        with autocast():                    # ★ FP16/bf16 구간
            out = model(x).squeeze(1)
            loss = criterion(out, y)

        scaler.scale(loss).backward()       # ★ 스케일된 backward
        scaler.step(optimizer)              # ★ 스케일 고려한 step
        scaler.update()                     # ★ 스케일 값 갱신

        epoch_loss += loss.item() * x.size(0)
        correct    += ((torch.sigmoid(out) > 0.5) == y.bool()).sum().item()
        total      += y.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    scheduler.step()

    train_acc = correct / total
    # val_acc   = evaluate(model, val_loader)
    # print(f"[{epoch+1}] Train ACC {train_acc:.4f} | Val ACC {val_acc:.4f} "
    #       f"| LR {scheduler.get_last_lr()[0]:.6f}")
    val_acc = evaluate(model, val_loader)

#     with torch.no_grad(), autocast():
#         x, _ = next(iter(val_loader))
#         p_mean = torch.sigmoid(model(x.to(device)).squeeze(1)).mean().item()
#     print(f"[{epoch+1}] Train ACC {train_acc:.4f} | Val ACC {val_acc:.4f} "
#         f"| prob_mean {p_mean:.3f} | LR {scheduler.get_last_lr()[0]:.6f}")


# # -------------------------------------------------
# # 9. backbone weight 저장
# # -------------------------------------------------
# os.makedirs("./convnext_l_backbone", exist_ok=True)
# torch.save(model.backbone.state_dict(), "./convnext_l_backbone/cls_trained_amp.pth")
# print("Backbone weights saved.")
# wandb 로깅
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_loss / total,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lr": scheduler.get_last_lr()[0]
    })

    print(f"[{epoch+1}] Train ACC {train_acc:.4f} | Val ACC {val_acc:.4f} "
          f"| LR {scheduler.get_last_lr()[0]:.6f}")

    # Top-K 저장
    save_path = os.path.join(save_dir, f"not_small_aug_epoch{epoch+1:02d}_val{val_acc:.4f}.pth")
    torch.save(model.backbone.state_dict(), save_path)
    best_models.append((val_acc, save_path))
    best_models.sort(reverse=True)  # 높은 acc 순
    if len(best_models) > top_k:
        _, path_to_remove = best_models.pop()
        if os.path.exists(path_to_remove):
            os.remove(path_to_remove)