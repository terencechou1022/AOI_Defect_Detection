"""Spike step 1: fetch MVTec AD (official archive, hash-verified by anomalib)."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from anomalib.data import MVTecAD

t0 = time.time()
print("[download] start, 4.9 GiB, official mydrive.ch archive", flush=True)
dm = MVTecAD(root="datasets/MVTecAD", category="transistor")
dm.prepare_data()
print(f"[download] done in {time.time()-t0:.0f}s", flush=True)
