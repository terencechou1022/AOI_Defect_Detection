# 工業產品表面瑕疵檢測（AOI）

[![CI](https://github.com/terencechou1022/AOI_Defect_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/terencechou1022/AOI_Defect_Detection/actions/workflows/ci.yml)

以無監督異常偵測對產品影像輸出正常／異常判定與異常熱區圖，模擬產線 AOI 場景。
訓練資料只有正常品，不需要瑕疵標註，這是產線導入時最現實的前提：
瑕疵樣本本來就稀少，而且新的瑕疵型態隨時會出現。

重點不在刷 AUROC。PatchCore 在 MVTec 的公開 image AUROC 平均已經約 0.99，
達標本身不構成賣點。這個專案要回答的是導入時真正會被問的三件事：
**漏檢與誤殺的代價差多少、閾值該怎麼定、CPU 上跑得動嗎。**

## 現況

實作中。資料管線與三張指標表已完成，報告尚未開始。

| 項目 | 狀態 |
|---|---|
| 資料取得與還原（3 類別） | 完成，筆數與 MVTec 官方一致 |
| PatchCore CPU 可行性驗證 | 完成，見下方 spike 結果 |
| 3 類別 × 2 方法 benchmark | **完成，見下方結果** |
| 解析度取捨表（256／384／512） | **完成，見下方結果** |
| 漏檢/誤殺成本分析 | **完成，見下方** |
| pytest | 未開始 |

## Spike 結果（transistor / PatchCore / CPU / 256 × 256）

先驗證「CPU 跑得動」再往下做，這一步過了才寫實作計畫。

| 指標 | 實測 |
|---|---|
| image AUROC | 0.9921 |
| image F1 | 0.9620 |
| pixel AUROC | 0.9735 |
| pixel F1 | **0.6146** |
| CPU 訓練耗時 | 2963 秒（49 分鐘） |
| memory bank | (21811, 1536) float32 = 127.8 MB |
| 環境 | torch 2.14.0+cpu，4 threads |

**pixel AUROC 0.9735 但 pixel F1 只有 0.6146**，這個落差是本專案指標設計的起點。
瑕疵區域只佔整張圖極小比例，pixel AUROC 的分母被大量正常像素撐大，
所以它會虛高。只報 pixel AUROC 等於誤導，因此指標表一律加上 **AUPRO**
與 pixel F1 並列。這一點原本被列為 nice to have，是 spike 的數字把它推成必要項。

## 結果

2026-09-20 03:12 到 03:48 實測，8 輪，CPU，總耗時 36 分鐘。原始資料在 `results/benchmark/*.json`。

### 表一　三類別 × 兩方法 @ 256

| 類別 | 方法 | image AUROC | pixel AUROC | pixel F1 | AUPRO | 訓練 | p50 延遲 | 權重 |
|---|---|---|---|---|---|---|---|---|
| transistor | PatchCore | **0.9896** | 0.9672 | 0.6053 | 0.9348 | 494s | 411ms | 12.9 MB |
| transistor | PaDiM | 0.9383 | 0.9707 | 0.6129 | 0.8917 | 63s | 143ms | 157.8 MB |
| screw | PatchCore | **0.9660** | 0.9809 | 0.3869 | 0.9199 | 836s | 428ms | 19.4 MB |
| screw | PaDiM | 0.8627 | 0.9801 | 0.2263 | 0.9177 | 85s | 137ms | 157.8 MB |
| metal_nut | PatchCore | **0.9990** | 0.9870 | 0.8441 | 0.9421 | 342s | 359ms | 13.4 MB |
| metal_nut | PaDiM | 0.9130 | 0.9488 | 0.6658 | 0.8458 | 48s | 130ms | 157.8 MB |

PatchCore 用 1% coreset（PatchCore 論文的主要回報配置），PaDiM 為預設。

**PaDiM 三個類別的 image AUROC 全部低於 0.95**，PatchCore 三個全過。
這不是失敗，正是方法比較要得到的結果。

### 表二　方法取捨

| | PatchCore | PaDiM |
|---|---|---|
| image AUROC | **贏，三類別皆是**（+0.05 到 +0.10） | 輸 |
| 單張延遲 | 359 到 428 ms | **贏，130 到 143 ms（約 3 倍快）** |
| 訓練耗時 | 342 到 836 s | **贏，48 到 85 s（5 到 10 倍快）** |
| 權重體積 | **贏，13 到 19 MB** | 輸，157.8 MB（約 10 倍大） |

**結論：PatchCore 在準確度與體積上都贏，只輸在速度。**

體積這一項與直覺相反，值得解釋：coreset 從 10% 降到 1% 之後，
PatchCore 的 memory bank 跟著縮小十倍（127.8 MB → 13 MB），
而 PaDiM 存的是每個 patch 位置的完整共變異數矩陣，
**體積由解析度決定，不隨取樣率變動**。
所以「PatchCore 很肥、PaDiM 很輕」只在預設的 10% coreset 下成立。

### 表三　解析度取捨（PaDiM，transistor）

| 尺寸 | image AUROC | AUPRO | p50 延遲 | 權重 |
|---|---|---|---|---|
| **256** | **0.9383** | **0.8917** | 143ms | 157.8 MB |
| 384 | 0.9092 | 0.8800 | 295ms | 355.1 MB |
| 512 | 0.8817 | 0.8437 | 484ms | 631.3 MB |

**解析度越高越差，而且成本每階約翻倍。** 這與「高解析度較準但較慢」的直覺相反。

推測原因：backbone 在 ImageNet 上是 224 到 256 的尺度預訓練，
拉高解析度改變了感受野與瑕疵尺寸的相對關係；
而且 PaDiM 要為每個空間位置估高斯分布，位置數隨解析度平方成長，
訓練資料仍只有 220 張，估計品質因此下降。

實務意涵：**產線提高相機解析度不會自動換到更好的檢測率**，
反而要重新驗證。這是這張表最值得寫進報告的一句。

### pixel AUROC 是會騙人的，screw 最明顯

| screw | pixel AUROC | pixel F1 | AUPRO |
|---|---|---|---|
| PatchCore | 0.9809 | **0.3869** | 0.9199 |
| PaDiM | 0.9801 | **0.2263** | 0.9177 |

pixel AUROC 說 0.98，pixel F1 說 0.23 到 0.39。螺絲的瑕疵是細小刮痕，
正常像素佔絕大多數，把 AUROC 的分母撐大。

一個容易被忽略的細節：**AUPRO 也沒有完全揭露這個落差**（0.92，接近 AUROC）。
因為 AUPRO 與 AUROC 同樣不綁閾值，而 F1 綁。
所以三個指標講的是三件不同的事，**光加 AUPRO 不夠，pixel F1 才是最誠實的那個**。
報告三個都要報。

## 漏檢與誤殺：閾值該怎麼選

這是整個專案的主軸。上面的 AUROC 表回答「模型分得開嗎」，這一段回答
**「這條線要漏檢還是要誤殺，以及願意付多少代價換」**，後者才是導入時真正被問的。

兩種錯誤的代價完全不對稱：

- **漏檢 FN**　瑕疵品流到客戶端。退貨、賠償、信譽損失，可能整批召回
- **誤殺 FP**　良品被判不良。重工或報廢，成本是一顆料加人工

`threshold_analysis.py` 不替產線決定成本比（那是業務數字不是技術數字），
它掃描成本比並顯示最佳閾值怎麼移動，讓讀者代入自己的數字。

### 最佳閾值隨成本比的移動（PatchCore 1% coreset @ 256）

| 類別 | 漏檢/誤殺成本比 | 閾值 | 漏檢 | 誤殺 | 召回率 | 誤殺率 |
|---|---|---|---|---|---|---|
| transistor | 1 | 43.23 | 3 | 1 | 0.925 | 0.017 |
| transistor | 5 / 10 / 50 | 39.76 | 0 | 10 | 1.000 | 0.167 |
| screw | 1 | 35.39 | 9 | 3 | 0.924 | 0.073 |
| screw | 5 / 10 / 50 | 33.36 | 0 | 21 | 1.000 | 0.512 |
| metal_nut | 1 / 5 / 10 / 50 | 43.50 | 0 | 1 | 1.000 | 0.045 |

**閾值只在成本比 1 到 5 之間移動一次，之後就不動了。**
意思是決策其實只有一個分水嶺：兩種錯誤是否等價。一旦漏檢比誤殺貴到 5 倍以上，
就一律推到零漏檢，再貴也不會改變選擇。這對產線談判很有用：
不需要精算成本比，只要確認「漏檢是不是貴 5 倍以上」。

### 零漏檢的代價，三個類別差十倍

| 類別 | image AUROC | 零漏檢要誤殺 | 產線可用嗎 |
|---|---|---|---|
| metal_nut | 0.9990 | 1 / 22 = **4.5%** | 可以 |
| transistor | 0.9896 | 10 / 60 = **16.7%** | 勉強，要算帳 |
| screw | 0.9660 | 21 / 41 = **51.2%** | **不可用** |

**這張表是整個專案最重要的一張。** AUROC 從 0.9990 掉到 0.9660 看起來只差 3 個百分點，
換算成零漏檢的代價卻是 4.5% 對 51.2%，差了十倍以上。

screw 的意思是：**要做到一顆瑕疵都不漏，就得報廢一半的良品。**
那條線不可能這樣開。所以 screw 這個類別的正確結論不是「AUROC 0.966 還行」，
而是「以現有方法不具備零漏檢能力，必須放寬到接受漏檢，或換方法、改打光」。

這正是「AUROC 高不高在產線上不是重點」的實證。

### 順帶發現：函式庫的預設閾值對 AOI 是錯的方向

anomalib 為 transistor 學到的 `image_threshold` 是 **44.87**，
比成本比 1 的最佳解（43.23）還高。閾值越高越保守、越不願判為異常，
所以預設值不只隱含「兩種錯誤一樣貴」，還**略微偏向避免誤殺**。
AOI 要的是相反方向。直接用預設閾值上線，等於默默選了一個對產線不利的取捨。

## 資料集與授權

使用 **MVTec AD**，先做與電子製造相關的三類：`transistor`、`screw`、`metal_nut`。

| 類別 | train（正常品） | test | ground truth mask |
|---|---|---|---|
| transistor | 213 | 100 | 40 |
| screw | 320 | 160 | 119 |
| metal_nut | 220 | 115 | 93 |

**授權：MVTec AD 採 CC BY-NC-SA 4.0，僅限非商業使用。**
因此本 repo 的處理方式是：

- **資料集不進版控**（`.gitignore` 排除 `datasets/` 與 `hf_cache/`），
  使用者自行執行 `prepare_data.py` 取得
- demo 站只放少量範例圖並標註來源與授權，不重新散布整份資料集
- 任何衍生指標與圖表都註明資料來源

引用時請依原作者要求標註 MVTec AD 資料集。

### 取得方式

不要用 anomalib 內建的 download。實測結果：

| 來源 | 實測 | 結論 |
|---|---|---|
| 官方 mydrive.ch（anomalib 的 `DOWNLOAD_INFO`） | 約 70 kB/s，單一壓縮包 5.26 GB，ETA 20 小時以上，無分類別下載 | 不可用 |
| HF `BrachioLab/mvtec-ad` | 檔案結構正確但為 gated repo（401） | 不可用 |
| HF `TheoM55/mvtec_all_objects_split` | per-category parquet，約 7 MB/s，含 ground truth mask | **採用** |

`prepare_data.py` 走第三條，並把 parquet 還原成 anomalib 預期的目錄佈局。

```bash
pip install "anomalib[cpu]" pyarrow
python prepare_data.py

# 分層執行。預設跑 main + resolution，實測約 40 分鐘。
python run_benchmark.py                    # main + resolution
python run_benchmark.py main               # 只跑 P0 主表，實測約 35 分鐘
python run_benchmark.py coreset            # 選配，約 55 分鐘（10% coreset 對照）
```

`pyarrow` 不在 `anomalib[cpu]` 的相依裡，要自己補。

**每一輪跑完就寫一個 JSON，重複執行會自動跳過已完成的輪次**，
所以可以分次跑完，中途停掉不會白費。

### 三張表與成本

| tier | 內容 | 實測 |
|---|---|---|
| `main` | 3 類別 × {PatchCore, PaDiM} @ 256 | 訓練 1868s，含測試與延遲量測約 35 分鐘 |
| `resolution` | PaDiM @ 384／512（256 已在主表） | 訓練 252s，約 5 分鐘 |
| `coreset` | transistor PatchCore 10%，對照主表的 1% | 尚未執行，估約 55 分鐘，可略 |

兩個把成本壓下來的決定，都寫在腳本註解裡：

**coreset 取樣率用 1% 而不是 anomalib 預設的 10%。**
成本 ∝ k × n，而 k = ratio × n，所以 1% 比 10% 快十倍
（transistor 實測：10% 要 2963 秒，1% 是 494 秒，約 6 倍差距，
比理論的 10 倍小，因為特徵抽取那一段的成本不隨取樣率變動）。
這不是偷工：PatchCore 論文的主要回報配置就是 PatchCore-1%，
10% 與 25% 是附帶變體。`coreset` tier 保留兩者的對照。

**解析度掃描用 PaDiM 而不是 PatchCore。**
PatchCore 的 coreset 成本 ∝ n²，而 n = 訓練張數 × (尺寸/8)²，
所以 512 是 256 的 16 倍（外推 14.6 小時）、1024 是 256 倍（200 小時以上）。
PaDiM 只算 mean 與 covariance，成本只隨位置數線性成長，任何解析度都便宜。

**但表三的結果讓這個取捨變得沒有意義：解析度拉高反而更差。**
所以「PatchCore 在 CPU 上擴展不到高解析度」不再是缺點，
因為高解析度本來就不該用。這一條原本預期會寫成
「想提高解析度就得換 PaDiM」，實測後改成
「先驗證提高解析度有沒有好處，本案例是沒有」。

## 部署

部署 Hugging Face Spaces 免費 CPU 版。兩件事要先知道：

- **免費 CPU Space 閒置 48 小時會休眠**，喚醒後 PyTorch 冷啟動約 1 到 2 分鐘。
  要拿 demo 給人看之前先自己開一次把它喚醒。實測秒數待部署後補上。
- **權重體積的結論與規劃時相反，實測見表二。**
  規劃階段我以為 PatchCore 肥（10% coreset 下每類別 127.8 MB）而 PaDiM 輕。
  改用 1% coreset 之後 PatchCore 降到 **每類別 13 到 19 MB，三類別合計約 46 MB**，
  而 PaDiM 是 **每類別 157.8 MB，三類別合計約 473 MB**。
  所以要部署的是 PatchCore，**不需要 git lfs**；
  若改部署 PaDiM 才需要。把 PaDiM 納入比較的理由因此改成
  「它在速度上明顯較快」，而不是原先以為的體積。

## 執行環境

- Python 3.12，Windows 11 開發
- anomalib 2.6.1，PyTorch 2.14.0+cpu
- **訓練與推論全程 CPU 可完成**，有 GPU 則加速
- 禁止從頭訓練 backbone，只用 anomalib 的現成方法

「CPU 可完成」與「CPU 上很快」是兩件事，而差多少取決於 coreset 取樣率：

| 配置 | transistor 單輪訓練 | 主表 6 輪 |
|---|---|---|
| PatchCore 10% coreset（anomalib 預設） | 2963s，約 49 分鐘（實測） | 估約 5 小時 |
| PatchCore 1% coreset（本專案採用） | 494s，約 8 分鐘（實測） | 1868s，約 31 分鐘（實測） |

三個類別的 PatchCore 訓練時間差異很大（transistor 494s、metal_nut 342s、
screw 836s），差別來自訓練張數（213／220／320），coreset 成本隨 n² 成長。

這在作品集情境下都可接受，在產線情境下該用 GPU，報告會把這個界線講清楚。

## 檔案

| 檔案 | 用途 |
|---|---|
| `prepare_data.py` | 取得 MVTec AD 並還原成 anomalib 目錄結構 |
| `run_benchmark.py` | 3 類別 × 2 方法的指標與延遲，加解析度掃描 |
| `threshold_analysis.py` | 漏檢/誤殺成本掃描與零漏檢代價 |
| `_spike_*.py` | 可行性驗證用的一次性腳本，保留作為紀錄 |
| `results/benchmark/*.json` | 每一輪的結果，逐輪落盤 |

## 不做什麼

不自行訓練或微調 backbone、不做即時視訊流、不做多類別共用單一模型、
第一版不做資料增強實驗。
