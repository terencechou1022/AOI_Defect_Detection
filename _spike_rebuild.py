"""Spike step 1b: 把 HF parquet 還原成 anomalib 認得的 MVTec AD 目錄結構。

官方下載點（mydrive.ch）實測只有 ~70 kB/s，5.26 GB 要 20 小時以上，不可用。
改走 HF 鏡像 TheoM55/mvtec_all_objects_split，per-category parquet，實測 ~7 MB/s。

目標結構（anomalib MVTecAD 預期）：
  datasets/MVTecAD/<category>/train/good/*.png
  datasets/MVTecAD/<category>/test/good/*.png
  datasets/MVTecAD/<category>/test/<defect>/*.png
  datasets/MVTecAD/<category>/ground_truth/<defect>/*_mask.png
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CATEGORY = "transistor"
ROOT = Path("datasets/MVTecAD") / CATEGORY
counts = {"image": 0, "mask": 0}

for split in ("train", "test"):
    table = pq.read_table(f"hf_cache/data/{CATEGORY}.{split}-00000-of-00001.parquet")
    data = table.to_pydict()

    for i in range(table.num_rows):
        defect = data["defect"][i]
        img = data["image_path"][i]
        mask = data["mask_path"][i]

        dst = ROOT / split / defect / img["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(img["bytes"])
        counts["image"] += 1

        if mask and mask.get("bytes"):
            m = ROOT / "ground_truth" / defect / mask["path"]
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_bytes(mask["bytes"])
            counts["mask"] += 1

    print(f"[{split}] {table.num_rows} rows 寫出完成")

print(f"\n共 {counts['image']} 張影像、{counts['mask']} 張 mask")
print("目錄結構：")
for p in sorted(ROOT.rglob("*")):
    if p.is_dir():
        n = len(list(p.glob("*.png")))
        if n:
            print(f"  {p.relative_to(ROOT.parent)}  {n} 張")
