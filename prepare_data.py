"""取得 MVTec AD 並還原成 anomalib 預期的目錄結構。

不要用 anomalib 內建的 download。官方下載點（mydrive.ch）實測只有約 70 kB/s，
5.26 GB 的單一壓縮包要 20 小時以上，而且沒有分類別下載。
改走 HF 鏡像 TheoM55/mvtec_all_objects_split，per-category parquet，實測約 7 MB/s。

資料集是 CC BY-NC-SA 4.0（非商業），不進版控。見 README 的授權段。
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "TheoM55/mvtec_all_objects_split"
CATEGORIES = ("transistor", "screw", "metal_nut")
CACHE = Path("hf_cache")
ROOT = Path("datasets/MVTecAD")


def fetch(category):
    paths = {}
    for split in ("train", "test"):
        paths[split] = hf_hub_download(
            REPO,
            f"data/{category}.{split}-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=CACHE,
        )
    return paths


def rebuild(category, paths):
    """parquet 欄位：image_path{bytes,path}、mask_path{bytes,path}、
    split、object、defect、label。攤成 MVTec 的原生目錄佈局。"""
    out = ROOT / category
    images = masks = 0

    for split, path in paths.items():
        data = pq.read_table(path).to_pydict()
        for i in range(len(data["defect"])):
            defect = data["defect"][i]
            img = data["image_path"][i]
            mask = data["mask_path"][i]

            dst = out / split / defect / img["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(img["bytes"])
            images += 1

            if mask and mask.get("bytes"):
                m = out / "ground_truth" / defect / mask["path"]
                m.parent.mkdir(parents=True, exist_ok=True)
                m.write_bytes(mask["bytes"])
                masks += 1

    return images, masks


def main():
    for category in CATEGORIES:
        target = ROOT / category / "train" / "good"
        if target.exists() and len(list(target.glob("*.png"))) > 100:
            n = len(list((ROOT / category).rglob("*.png")))
            print(f"[{category}] 已存在，{n} 張，跳過")
            continue

        print(f"[{category}] 下載中")
        paths = fetch(category)
        images, masks = rebuild(category, paths)
        print(f"[{category}] {images} 張影像、{masks} 張 mask")

    print("\n各類別內容:")
    for category in CATEGORIES:
        base = ROOT / category
        if not base.exists():
            continue
        train = len(list((base / "train" / "good").glob("*.png")))
        test = len(list((base / "test").rglob("*.png")))
        gt = len(list((base / "ground_truth").rglob("*.png")))
        defects = sorted(p.name for p in (base / "test").iterdir() if p.is_dir())
        print(f"  {category:<12} train {train:>3}　test {test:>3}　mask {gt:>3}　{defects}")


if __name__ == "__main__":
    main()
