import json, sys, pathlib

def convert(json_path: str):
    path = pathlib.Path(json_path)
    coco = json.load(open(path))

    # 1) 모든 annotation 의 category_id → 0
    for ann in coco["annotations"]:
        ann["category_id"] = 0

    # 2) categories 리스트 단일 클래스(cme)로 교체
    coco["categories"] = [{"id": 0, "name": "cme"}]

    # 3) 새 파일 저장 (원본은 보존)
    out = path.with_name(path.stem + "_0.json")
    json.dump(coco, open(out, "w"))
    print(f"✅  saved single‑class file → {out}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
