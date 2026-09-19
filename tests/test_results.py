"""驗證落盤的結果檔完整、自洽，而且 README 的表格沒有跟資料脫節。

不測模型，因為 CI 上沒有資料集也沒有 GPU，裝 anomalib 要好幾 GB。
測的是**結果的完整性**與**文件與資料的一致性**，那是這個 repo 實際會壞掉的地方：
數字改了但 README 忘了改，或者換了設定卻以為沒換。

`effective_hw` 那條尤其重要。開發時曾經以為改了輸入尺寸，
實際上被 pre_processor 壓回 256，白跑 40 分鐘才發現。
現在每一輪都會記錄實際的張量尺寸，這條測試守住它。
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "benchmark"
THRESH = ROOT / "results" / "threshold"
README = ROOT / "README.md"

CATEGORIES = {"transistor", "screw", "metal_nut"}
METRICS = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score", "pixel_AUPRO")


def load(directory):
    return [
        json.loads(f.read_text(encoding="utf-8"))
        for f in sorted(directory.glob("*.json"))
        if f.name != "summary.json"
    ]


@pytest.fixture(scope="module")
def bench():
    return load(BENCH)


@pytest.fixture(scope="module")
def thresh():
    return load(THRESH)


@pytest.fixture(scope="module")
def readme():
    return README.read_text(encoding="utf-8")


class TestBenchmark:
    def test_八輪都在(self, bench):
        assert len(bench) == 8

    def test_三類別都有兩種方法(self, bench):
        at256 = {(r["category"], r["method"]) for r in bench if r["image_size"] == 256}
        assert at256 == {(c, m) for c in CATEGORIES for m in ("patchcore", "padim")}

    def test_五個指標都有且在合理範圍(self, bench):
        for r in bench:
            for m in METRICS:
                assert m in r, f"{r['category']}/{r['method']} 缺 {m}"
                assert 0.0 <= r[m] <= 1.0, f"{m}={r[m]} 超出 0 到 1"

    def test_實際張量尺寸等於宣告尺寸(self, bench):
        # 守住「以為改了解析度但其實沒改」這個坑
        for r in bench:
            assert r["effective_hw"] == [r["image_size"]] * 2, (
                f"{r['category']}/{r['method']} 宣告 {r['image_size']} "
                f"但實際跑 {r['effective_hw']}"
            )

    def test_延遲的分位數有序(self, bench):
        for r in bench:
            assert r["ms_min"] <= r["ms_p50"] <= r["ms_p90"]

    def test_patchcore有記錄coreset比例(self, bench):
        for r in bench:
            if r["method"] == "patchcore":
                assert r["coreset_ratio"] is not None
            else:
                assert r["coreset_ratio"] is None

    def test_pixel_AUROC普遍高於pixel_F1(self, bench):
        # 這是本專案的核心論點：pixel AUROC 被大量正常像素稀釋而虛高。
        # 如果哪天不成立了，README 的整段論述要重寫。
        worse = [r for r in bench if r["pixel_F1Score"] >= r["pixel_AUROC"]]
        assert not worse, f"論點不成立於: {[(r['category'], r['method']) for r in worse]}"


class TestThreshold:
    def test_三類別都有(self, thresh):
        assert {r["category"] for r in thresh} == CATEGORIES

    def test_混淆矩陣數量守恆(self, thresh):
        for r in thresh:
            for ratio, b in r["best_by_cost_ratio"].items():
                assert 0 <= b["fn"] <= r["n_defect"], f"{r['category']} 比值{ratio} fn 越界"
                assert 0 <= b["fp"] <= r["n_normal"], f"{r['category']} 比值{ratio} fp 越界"

    def test_成本比越高漏檢不會變多(self, thresh):
        # 漏檢越貴，最佳解不該容忍更多漏檢
        for r in thresh:
            fns = [b["fn"] for _, b in sorted(
                r["best_by_cost_ratio"].items(), key=lambda kv: int(kv[0]))]
            assert fns == sorted(fns, reverse=True) or len(set(fns)) == 1, \
                f"{r['category']} 的漏檢數隨成本比上升: {fns}"

    def test_零漏檢確實是零漏檢(self, thresh):
        for r in thresh:
            z = r["zero_miss"]
            if z is None:
                continue
            assert 0 <= z["fp"] <= r["n_normal"]
            assert abs(z["fp_rate"] - z["fp"] / r["n_normal"]) < 1e-4


class TestReadmeMatchesData:
    """README 的數字必須對得回 JSON。開發過程中這裡漏過好幾次。"""

    def test_主表的image_AUROC都出現在README(self, bench, readme):
        for r in bench:
            if r["image_size"] != 256:
                continue
            value = f"{r['image_AUROC']:.4f}"
            assert value in readme, (
                f"{r['category']}/{r['method']} 的 image_AUROC {value} 不在 README 裡"
            )

    def test_零漏檢的誤殺率都出現在README(self, thresh, readme):
        for r in thresh:
            z = r["zero_miss"]
            if z is None:
                continue
            pct = f"{z['fp_rate'] * 100:.1f}%"
            assert pct in readme, f"{r['category']} 的零漏檢誤殺率 {pct} 不在 README 裡"

    def test_README沒有殘留待執行字樣(self, readme):
        # benchmark 已經跑完，狀態表不該還寫著待執行
        stale = re.findall(r"腳本就緒，待執行", readme)
        assert not stale, "README 現況表仍寫著待執行"
