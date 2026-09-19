"""Spike step 2: PatchCore on MVTec AD transistor, CPU only.

驗證三件事，其他都不做：
  1. CPU 訓練跑得完
  2. image-level AUROC（順便收 pixel-level）
  3. 單張推論延遲 ms

順便量 memory bank 體積，那是 A-1.4 要的數字。
"""

import sys, time, warnings

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

CATEGORY = "transistor"
t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


# num_workers=0：Windows 的 multiprocessing 在這裡只會拖慢，不會加速
datamodule = MVTecAD(
    root="datasets/MVTecAD",
    category=CATEGORY,
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)
model = Patchcore()          # wide_resnet50_2, layer2+layer3, coreset 10%
engine = Engine(
    accelerator="cpu",
    devices=1,
    max_epochs=1,            # PatchCore 是 memory bank，本來就只走一輪
    logger=False,
    # 不要傳 enable_checkpointing=False：anomalib 會自己注入 ModelCheckpoint，
    # 兩者衝突會丟 MisconfigurationException
    default_root_dir="results/spike",
)

log(f"開始訓練　category={CATEGORY}　backbone=wide_resnet50_2　CPU")
t0 = time.time()
engine.fit(datamodule=datamodule, model=model)
train_s = time.time() - t0
log(f"訓練完成，耗時 {train_s:.1f}s")

log("開始測試")
t0 = time.time()
results = engine.test(datamodule=datamodule, model=model)
test_s = time.time() - t0
log(f"測試完成，耗時 {test_s:.1f}s")

# ---- memory bank 體積 ----
bank_mb = None
for name, buf in model.named_buffers():
    if "memory_bank" in name and buf.numel():
        bank_mb = buf.numel() * buf.element_size() / 1024**2
        log(f"memory bank: {tuple(buf.shape)}  {bank_mb:.1f} MB  dtype={buf.dtype}")

# ---- 單張推論延遲 ----
log("量測單張推論延遲")
datamodule.setup("test")
loader = torch.utils.data.DataLoader(datamodule.test_data, batch_size=1, num_workers=0)
batch = next(iter(loader))
image = batch.image if hasattr(batch, "image") else batch["image"]

model.eval()
torch.set_grad_enabled(False)
for _ in range(3):                       # warm up
    model(image)

N = 20
lat = []
for _ in range(N):
    t0 = time.perf_counter()
    model(image)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

print("\n" + "=" * 64)
print(f"SPIKE 結果　MVTec AD / {CATEGORY} / PatchCore / CPU")
print("=" * 64)
for r in results:
    for k, v in r.items():
        print(f"  {k:<32} {v:.4f}")
print(f"  {'訓練耗時 (s)':<32} {train_s:.1f}")
print(f"  {'測試耗時 (s)':<32} {test_s:.1f}")
if bank_mb:
    print(f"  {'memory bank (MB)':<32} {bank_mb:.1f}")
print(f"  {'單張延遲 中位數 (ms)':<32} {lat[N // 2]:.1f}")
print(f"  {'單張延遲 最小 (ms)':<32} {lat[0]:.1f}")
print(f"  {'單張延遲 P90 (ms)':<32} {lat[int(N * 0.9)]:.1f}")
print(f"  {'torch / threads':<32} {torch.__version__} / {torch.get_num_threads()}")
print("=" * 64)
