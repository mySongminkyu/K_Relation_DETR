#!/usr/bin/env python3
# run_infer_vis_verbose.py
# -----------------------------------------------------------
#  * Relation‑DETR 체크포인트 로드
#  * test 이미지 전부 추론 → 신뢰도 ≥ conf_thresh 박스 표시
#  * vis_out/ 에 시각화 PNG 저장 + detections.csv 생성
# -----------------------------------------------------------
import os, sys, glob, pathlib, csv, cv2, numpy as np, torch
from tqdm import tqdm

# ───────────── 사용자 설정 ──────────────
ckpt_path = "/data/minkyu/Kaist/Relation-DETR/experiments/cme_convnext_b_q10_bbox5_augplus/best_ap50.pth"
cfg_path  = "/data/minkyu/Kaist/Relation-DETR/configs/relation_detr/relation_detr_convnext_b_500_1000.py"
img_root  = "/data/minkyu/Kaist/Relation-DETR/datasets/final-dataset/test/diff_coordinate_image/c2_coordinate_lev1"
vis_dir   = "vis_out_0.5"  
csv_path   = "detections_0.5.csv"      # 박스 정보 저장
conf_thresh = 0.50
device      = "cuda"
font_scale  = 0.45
# ───────────────────────────────────────

# 0. PYTHONPATH
proj_root = pathlib.Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.append(str(proj_root))

# 1. 모델 로드
from util.lazy_load import Config
from util.utils import load_state_dict

cfg   = Config(cfg_path)
model = cfg.model
state = torch.load(ckpt_path, map_location="cpu")
load_state_dict(model, state.get("model", state))
model.to(device).eval()
print(f"[✓] 모델 로드: {ckpt_path}")

# 2. 이미지 수집
pngs = glob.glob(os.path.join(img_root, "**", "*.png"), recursive=True)
assert pngs, f"PNG 없음: {img_root}"
print(f"[✓] 이미지 {len(pngs)}장")

# 3. CSV 준비
csv_head = ["image_path", "score", "x1", "y1", "x2", "y2"]
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_head)

# 4. 추론 & 시각화
for p in tqdm(pngs):
    im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    print(f"[✓] 원본 이미지 shape (HWC): {im_bgr.shape}")
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    
    inp    = torch.from_numpy(im_rgb.transpose(2, 0, 1)).float() / 255.
    print(f"[✓] 입력 텐서 shape (CHW): {inp.shape}")

    with torch.no_grad():
        out = model([inp.to(device)])[0]

    boxes  = out["boxes"].cpu().numpy()      # 절대 xyxy
    scores = out["scores"].cpu().numpy()

    drawn = 0
    for box, s in zip(boxes, scores):
        if s < conf_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int)
        csv_writer.writerow([p, f"{s:.4f}", x1, y1, x2, y2])

        # 그리기
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"CME {s:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(im_bgr, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(im_bgr, text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        drawn += 1

    # 시각화 PNG 저장
    save_path = pathlib.Path(vis_dir) / pathlib.Path(p).relative_to(img_root)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), im_bgr)

    print(f"{p}  →  {drawn} boxes")

csv_file.close()
print(f"[✓] 시각화: {vis_dir}/,  CSV: {csv_path}")



# #!/usr/bin/env python3
# # run_infer_vis_single.py
# # -----------------------------------------------
# # 1장 이미지만 추론, bbox 시각화 및 shape 확인용
# # -----------------------------------------------
# import os, sys, pathlib, cv2, numpy as np, torch
# from util.lazy_load import Config
# from util.utils import load_state_dict

# # ───────────── 사용자 설정 ──────────────
# ckpt_path = "/data/minkyu/Kaist/Relation-DETR/experiments/cme_convnext_b_q10_bbox5_augplus/best_ap50.pth"
# cfg_path  = "/data/minkyu/Kaist/Relation-DETR/configs/relation_detr/relation_detr_convnext_b_500_1000.py"
# img_path  = "/data/minkyu/Kaist/Relation-DETR/datasets/final-dataset/test/diff_coordinate_image/c2_coordinate_lev1/2000/20000101/coordinate_20000101_013035.png"
# vis_path  = "/data/minkyu/Kaist/Relation-DETR/datasets/final-dataset/test_one.png"
# conf_thresh = 0.50
# device      = "cuda"
# font_scale  = 0.5
# # ───────────────────────────────────────

# # PYTHONPATH 설정
# proj_root = pathlib.Path(__file__).resolve().parents[1]
# if str(proj_root) not in sys.path:
#     sys.path.append(str(proj_root))

# # 모델 로드
# cfg   = Config(cfg_path)
# model = cfg.model
# state = torch.load(ckpt_path, map_location="cpu")
# load_state_dict(model, state.get("model", state))
# model.to(device).eval()
# print(f"[✓] 모델 로드 완료")

# # 이미지 로드
# im_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
# print(f"[✓] 원본 이미지 shape (HWC): {im_bgr.shape}")  # (H, W, C)

# im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
# inp    = torch.from_numpy(im_rgb.transpose(2, 0, 1)).float() / 255.
# print(f"[✓] 입력 텐서 shape (CHW): {inp.shape}")

# # 추론
# with torch.no_grad():
#     out = model([inp.to(device)])[0]

# boxes  = out["boxes"].cpu().numpy()
# scores = out["scores"].cpu().numpy()

# drawn = 0
# for box, s in zip(boxes, scores):
#     if s < conf_thresh:
#         continue
#     x1, y1, x2, y2 = box.astype(int)
#     cv2.rectangle(im_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     text = f"CME {s:.2f}"
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
#     cv2.rectangle(im_bgr, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
#     cv2.putText(im_bgr, text, (x1, y1 - 2),
#                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
#     print(f"[✓] Box {drawn+1}: score={s:.4f}, coords=({x1}, {y1}, {x2}, {y2})")
#     drawn += 1

# # 결과 저장
# os.makedirs(os.path.dirname(vis_path), exist_ok=True)
# cv2.imwrite(vis_path, im_bgr)
# print(f"[✓] 박스 {drawn}개 시각화 저장 완료 → {vis_path}")
