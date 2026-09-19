"""Spike step 3: 從 checkpoint 補量單張推論延遲，不重跑 49 分鐘的訓練。

上一輪失敗的原因：anomalib 的 dataset 回傳 ImageItem dataclass，
不能餵給 torch 的 default_collate。要用 datamodule 自己的 dataloader。
"""

import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")

import anomalib
import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore

CKPT = Path("results/spike/Patchcore/MVTecAD/transistor/v1/weights/lightning/model.ckpt")

# PyTorch 2.6 起 torch.load 預設 weights_only=True，會擋掉 anomalib.PrecisionType。
# 這個 checkpoint 是 anomalib 本機剛寫出來的，來源可信。
torch.serialization.add_safe_globals([anomalib.PrecisionType])

model = Patchcore.load_from_checkpoint(CKPT, map_location="cpu")
model.eval()
torch.set_grad_enabled(False)

bank = model.model.memory_bank
print(f"memory bank: {tuple(bank.shape)}  "
      f"{bank.numel() * bank.element_size() / 1024**2:.1f} MB  {bank.dtype}")
print(f"checkpoint 檔案大小: {CKPT.stat().st_size / 1024**2:.1f} MB")

datamodule = MVTecAD(
    root="datasets/MVTecAD", category="transistor",
    eval_batch_size=1, num_workers=0,
)
datamodule.setup("test")
batch = next(iter(datamodule.test_dataloader()))
image = batch.image
print(f"輸入張量: {tuple(image.shape)}")

for _ in range(3):                       # warm up
    model(image)

N = 20
lat = []
for _ in range(N):
    t0 = time.perf_counter()
    model(image)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

print("\n" + "=" * 58)
print("SPIKE 延遲結果　MVTec AD / transistor / PatchCore / CPU")
print("=" * 58)
print(f"  {'單張延遲 最小 (ms)':<30} {lat[0]:.0f}")
print(f"  {'單張延遲 中位數 (ms)':<30} {lat[N // 2]:.0f}")
print(f"  {'單張延遲 P90 (ms)':<30} {lat[int(N * 0.9)]:.0f}")
print(f"  {'torch 版本 / 執行緒':<30} {torch.__version__} / {torch.get_num_threads()}")
print("=" * 58)
