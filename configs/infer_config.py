# configs/infer_config.py
# -----------------------------------------------------------
# 목적: inference 때 lazy_load.Config 가 import → model 만 필요
# -----------------------------------------------------------

from models.backbones.convnext import ConvNeXtBackbone
from models.relation_detr import build_relation_detr   # ← 실제 함수 이름에 맞춰 수정

NUM_CLASSES = 1           # CME 하나만 탐지한다면 1
CLASSES = ("cme",)        # id 0 ↔︎ "cme"

# ---- backbone ----------------------------------------------------------
backbone = ConvNeXtBackbone(
    arch="conv_b",        # convnext‑base
    weights=None,         # 가중치는 체크포인트에서 로드
    return_indices=(1, 2, 3),
)

# ---- Relation‑DETR -----------------------------------------------------
model = build_relation_detr(
    num_classes=NUM_CLASSES,
    backbone=backbone,
    num_queries=20,          # train_config.py에 쓰인 값과 같게
    hidden_dim=256,           # "
    # 다른 hyper‑param 도 원래 학습 설정과 동일하게 넣어주세요
)
