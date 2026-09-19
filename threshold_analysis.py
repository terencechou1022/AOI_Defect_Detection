"""漏檢與誤殺的成本分析，以及閾值該怎麼選。

這是整個專案的主軸。AUROC 高不高在產線上不是重點，
重點是「這條線要漏檢還是要誤殺，以及願意付多少代價換」。

兩種錯誤的代價在產線上完全不對稱：
  漏檢 FN　瑕疵品流到客戶端。退貨、賠償、信譽損失，而且可能整批召回
  誤殺 FP　良品被判為不良。重工或報廢，成本是一顆料加上人工

所以不能用預設的 0.5 閾值，也不能用讓 F1 最大的閾值，
那兩種做法都隱含「兩種錯誤一樣貴」。

本支不替產線決定成本比，因為那是業務數字不是技術數字。
它做的是**掃描成本比並顯示最佳閾值怎麼移動**，讓讀者代入自己的數字。
"""

import json
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")

import anomalib
import torch
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.transforms.v2.functional import InterpolationMode
from anomalib.data import MVTecAD
from anomalib.models import Patchcore

# PyTorch 2.6 起 torch.load 預設 weights_only=True。這些 checkpoint 是
# run_benchmark.py 在本機剛寫出來的，來源可信，所以明確關掉。
#
# 不走 add_safe_globals 的白名單路線，是因為 checkpoint 夾帶了自訂的
# pre_processor 與 evaluator，追下去最後連 getattr 都要允許，
# 那時白名單已經形同虛設，不如誠實寫明「這是本機產物，我信任它」。
WEIGHTS_ONLY = False

CATEGORIES = ("transistor", "screw", "metal_nut")
SIZE = 256
OUT = Path("results/threshold")

# 漏檢成本 / 誤殺成本。不替使用者決定，掃描讓他代入自己的數字。
# 1 代表兩種錯誤一樣貴（等於只看準確率），50 代表電子業常見的嚴格端。
COST_RATIOS = (1, 5, 10, 50)


def ckpt_path(category):
    base = Path(f"results/runs/{category}_patchcore_256_cs1/Patchcore/MVTecAD/{category}")
    hits = sorted(base.glob("*/weights/lightning/model.ckpt"))
    return hits[-1] if hits else None


def transform():
    return Compose([
        Resize((SIZE, SIZE), interpolation=InterpolationMode.BILINEAR, antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collect_scores(category):
    """跑完測試集，回傳 (每張的異常分數, 每張的真實標籤)。"""
    model = Patchcore.load_from_checkpoint(
        ckpt_path(category), map_location="cpu", weights_only=WEIGHTS_ONLY
    )
    model.eval()
    tf = transform()

    dm = MVTecAD(root="datasets/MVTecAD", category=category,
                 eval_batch_size=1, num_workers=0)
    dm.setup("test")

    # 要的是 model.model（PatchcoreModel）而不是 model（AnomalibModule）。
    #
    # 走外層會經過 post_processor 的正規化，實測整批 pred_score 全部飽和成 1.0，
    # 閾值掃描就變成一條直線。內層給的是原始距離分數，
    # 與 checkpoint 裡學到的 image_threshold 同一個尺度（transistor 是 44.87）。
    #
    # 這是 anomalib 的 pre／post processor 第三次咬我：它們是 Lightning hook，
    # 直接呼叫 module 與走 Engine 的行為不一樣。量任何數字之前先印出來確認尺度。
    inner = model.model
    scores, labels = [], []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            out = inner(tf(batch.image))
            score = out.pred_score if hasattr(out, "pred_score") else out["pred_score"]
            scores.append(float(score.reshape(-1)[0]))
            labels.append(int(batch.gt_label.reshape(-1)[0]))

    assert len(set(scores)) > 1, "所有分數相同，八成又拿到正規化後的值"
    return scores, labels


def confusion_at(scores, labels, threshold):
    """異常分數 >= threshold 判為異常（正類）。"""
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = s >= threshold
        if y == 1 and pred:
            tp += 1
        elif y == 1 and not pred:
            fn += 1          # 漏檢：瑕疵品被放行
        elif y == 0 and pred:
            fp += 1          # 誤殺：良品被攔下
        else:
            tn += 1
    return tp, fp, tn, fn


def analyse(category):
    scores, labels = collect_scores(category)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    # 候選閾值取所有分數的中點，保證掃到每一種可能的切法
    cand = sorted(set(scores))
    thresholds = [cand[0] - 1e-6]
    thresholds += [(a + b) / 2 for a, b in zip(cand, cand[1:])]
    thresholds += [cand[-1] + 1e-6]

    rows = []
    for t in thresholds:
        tp, fp, tn, fn = confusion_at(scores, labels, t)
        rows.append({"threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn})

    best = {}
    for ratio in COST_RATIOS:
        pick = min(rows, key=lambda r: r["fn"] * ratio + r["fp"])
        best[ratio] = {
            "threshold": round(pick["threshold"], 4),
            "fn": pick["fn"], "fp": pick["fp"],
            "cost": pick["fn"] * ratio + pick["fp"],
            "recall": round(pick["tp"] / n_pos, 4) if n_pos else None,
            "fpr": round(pick["fp"] / n_neg, 4) if n_neg else None,
        }

    # 零漏檢需要付出多少誤殺，這是產線最常問的一句
    zero_fn = min((r for r in rows if r["fn"] == 0), key=lambda r: r["fp"], default=None)

    return {
        "category": category,
        "n_normal": n_neg, "n_defect": n_pos,
        "score_min": round(min(scores), 4), "score_max": round(max(scores), 4),
        "best_by_cost_ratio": best,
        "zero_miss": None if zero_fn is None else {
            "threshold": round(zero_fn["threshold"], 4),
            "fp": zero_fn["fp"],
            "fp_rate": round(zero_fn["fp"] / n_neg, 4),
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for category in CATEGORIES:
        if ckpt_path(category) is None:
            print(f"[{category}] 找不到 checkpoint，先跑 run_benchmark.py")
            continue
        print(f"[{category}] 分析中")
        r = analyse(category)
        results.append(r)
        (OUT / f"{category}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("成本最佳閾值隨「漏檢成本 / 誤殺成本」的變化（PatchCore 1% coreset @ 256）")
    print("=" * 78)
    head = f"{'類別':<12}{'比值':>5}{'閾值':>9}{'漏檢FN':>8}{'誤殺FP':>8}{'召回率':>9}{'誤殺率':>9}"
    print(head)
    print("-" * 78)
    for r in results:
        for ratio, b in r["best_by_cost_ratio"].items():
            print(f"{r['category']:<12}{ratio:>5}{b['threshold']:>9.3f}"
                  f"{b['fn']:>8}{b['fp']:>8}{b['recall']:>9.3f}{b['fpr']:>9.3f}")
        print("-" * 78)

    print("\n零漏檢要付的代價（FN = 0 時最少要誤殺幾顆良品）")
    for r in results:
        z = r["zero_miss"]
        if z:
            print(f"  {r['category']:<12} 閾值 {z['threshold']:>8.3f}　"
                  f"誤殺 {z['fp']:>3} / {r['n_normal']} 顆良品（{z['fp_rate']:.1%}）")
        else:
            print(f"  {r['category']:<12} 無法達到零漏檢")


if __name__ == "__main__":
    main()
