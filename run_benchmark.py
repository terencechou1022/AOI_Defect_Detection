"""3 類別 × 2 方法的指標與延遲對照，外加解析度取捨掃描。

驗收標準 P0 要的三張表都由這支產出：
  1. 三類別的 image AUROC / pixel AUROC / pixel F1 / AUPRO
  2. PatchCore vs PaDiM 的指標與單張延遲對照
  3. 1024 / 512 / 256 三個輸入尺寸的 AUROC vs 延遲取捨

AUPRO 是 P0 而非 P1，因為 spike 實測 transistor 的 pixel AUROC 0.9735
但 pixel F1 只有 0.6146，pixel AUROC 被大量正常像素稀釋而虛高，單報它會誤導。

每一輪跑完就寫一個 JSON，5 小時的工作中途掛掉不會全丟。
"""

import json
import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")

import torch
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.transforms.v2.functional import InterpolationMode
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.metrics import AUPRO, AUROC, Evaluator, F1Score
from anomalib.models import Padim, Patchcore
from anomalib.pre_processing import PreProcessor

CATEGORIES = ("transistor", "screw", "metal_nut")
RESULTS = Path("results/benchmark")


def build_evaluator():
    """把 AUPRO 加進預設指標。pixel F1 與 AUPRO 一起看才誠實。"""
    return Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
            AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
        ]
    )


def make_transform(size):
    """跟 anomalib 預設 pre_processor 同樣的組成，只換尺寸。

    重要：PatchCore 與 PaDiM 的預設 pre_processor 本來就
    Resize 到 256 × 256。所以「不設定」不等於「原生解析度」，
    而是等於 256。解析度掃描一定要從這裡改，
    改 datamodule 的 augmentations 沒有用，會被 pre_processor 再壓回 256。
    """
    return Compose([
        Resize((size, size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_model(method, size, coreset=0.01):
    """coreset 預設 1% 而不是 anomalib 的 10%。

    成本 ∝ k × n，而 k = coreset × n，所以 1% 比 10% 快十倍。
    這不是偷工：PatchCore 論文的主要回報配置就是 1%（PatchCore-1%），
    10% 與 25% 是附帶的變體。實測 transistor 10% 要 55 分鐘，1% 約 6 分鐘。
    """
    kwargs = {
        "pre_processor": PreProcessor(transform=make_transform(size)),
        "evaluator": build_evaluator(),
    }
    if method == "padim":
        return Padim(**kwargs)
    return Patchcore(coreset_sampling_ratio=coreset, **kwargs)


def make_datamodule(category):
    return MVTecAD(
        root="datasets/MVTecAD", category=category,
        train_batch_size=8, eval_batch_size=1, num_workers=0,
    )


def footprint_mb(model):
    """PatchCore 的 memory bank 是要進 repo 的東西，PaDiM 存的是統計量。"""
    total = 0
    for _, buf in model.named_buffers():
        total += buf.numel() * buf.element_size()
    return total / 1024**2


def measure_latency(model, datamodule, size, n=15):
    """延遲一定要在模型實際運作的解析度上量。

    第一版直接把 dataloader 給的原生 1024 × 1024 張量餵進 model()，
    繞過了 pre_processor，量到 921 ms，被放大了十幾倍。
    這裡自己套一次同樣的 transform，讓輸入尺寸與訓練時一致。
    回傳值含 effective_hw，用來事後驗證解析度真的有換。
    """
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))
    image = make_transform(size)(batch.image)

    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(image)
        lat = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(image)
            lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return {
        "effective_hw": list(image.shape[-2:]),
        "ms_min": round(lat[0]),
        "ms_p50": round(lat[n // 2]),
        "ms_p90": round(lat[int(n * 0.9)]),
    }


def run(category, method, size, coreset=0.01):
    tag = f"{category}_{method}_{size}"
    if method == "patchcore":
        tag += f"_cs{int(coreset * 100)}"
    out = RESULTS / f"{tag}.json"
    if out.exists():
        print(f"[{tag}] 已完成，跳過")
        return json.loads(out.read_text(encoding="utf-8"))

    print(f"[{tag}] 開始")
    datamodule = make_datamodule(category)
    model = make_model(method, size, coreset)
    engine = Engine(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        default_root_dir=f"results/runs/{tag}",
    )

    t0 = time.time()
    engine.fit(datamodule=datamodule, model=model)
    train_s = round(time.time() - t0, 1)

    metrics = engine.test(datamodule=datamodule, model=model)[0]
    record = {
        "category": category, "method": method, "image_size": size,
        "coreset_ratio": coreset if method == "patchcore" else None,
        "train_seconds": train_s,
        "buffer_mb": round(footprint_mb(model), 1),
        **{k: round(float(v), 4) for k, v in metrics.items()},
        **measure_latency(model, datamodule, size),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] 完成 {train_s}s  " +
          "  ".join(f"{k}={v}" for k, v in record.items() if "AUROC" in k or k == "ms_p50"))
    return record


# 三張表，依成本排序。中途停掉也已經拿到前面的結果，
# 而且完成的輪次會寫 JSON，下次執行自動跳過，可以分次跑完。
#
# 解析度掃描用 PaDiM 不用 PatchCore，理由是成本。PatchCore 的 coreset
# 成本 ∝ n²，而 n = 訓練張數 × (尺寸/8)²，所以 512 是 256 的 16 倍
# （實測外推 14.6 小時）、1024 是 256 倍（200 小時以上）。
# PaDiM 只算 mean 與 covariance 沒有 coreset，任何解析度都便宜。
# 「PatchCore 在 CPU 上無法擴展到高解析度」本身就是報告該寫的發現。
TIERS = {
    "main": {
        "why": "P0 主表：3 類別 × 2 方法 @ 256",
        "runs": [(c, m, 256, 0.01) for c in CATEGORIES for m in ("patchcore", "padim")],
        "estimate": "約 55 分鐘",
    },
    "resolution": {
        "why": "解析度取捨：PaDiM @ 384／512（256 已在主表）",
        "runs": [("transistor", "padim", s, None) for s in (384, 512)],
        "estimate": "約 15 分鐘",
    },
    "coreset": {
        "why": "coreset 成本取捨：transistor PatchCore 10% vs 主表的 1%",
        "runs": [("transistor", "patchcore", 256, 0.10)],
        "estimate": "約 55 分鐘（可略，spike 已有 10% 的參考數字）",
    },
}


def main():
    import sys as _sys

    wanted = _sys.argv[1:] or ["main", "resolution"]
    plan = []
    for tier in wanted:
        if tier not in TIERS:
            print(f"未知的 tier: {tier}。可選 {list(TIERS)}")
            return
        print(f"{tier:<12} {TIERS[tier]['why']}　{TIERS[tier]['estimate']}")
        plan += TIERS[tier]["runs"]

    print(f"\n共 {len(plan)} 輪\n")
    records = []
    for category, method, size, coreset in plan:
        try:
            records.append(run(category, method, size, coreset or 0.01))
        except Exception as exc:                       # 一輪失敗不要中斷整批
            print(f"[{category}_{method}_{size}] 失敗: {type(exc).__name__}: {exc}")

    existing = sorted(RESULTS.glob("*.json"))
    all_records = [
        json.loads(f.read_text(encoding="utf-8"))
        for f in existing if f.name != "summary.json"
    ]
    (RESULTS / "summary.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n本次 {len(records)}/{len(plan)} 輪，"
          f"累計 {len(all_records)} 輪 -> {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
