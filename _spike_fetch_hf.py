import sys, time, os
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from huggingface_hub import hf_hub_download
REPO = "TheoM55/mvtec_all_objects_split"
t0 = time.time()
paths = {}
for split in ("train", "test"):
    f = f"data/transistor.{split}-00000-of-00001.parquet"
    p = hf_hub_download(REPO, f, repo_type="dataset", local_dir="hf_cache")
    paths[split] = p
    print(f"[{time.time()-t0:6.1f}s] {split}: {os.path.getsize(p)/1024**2:7.1f} MB  {p}", flush=True)
import pyarrow.parquet as pq
for split, p in paths.items():
    sc = pq.read_schema(p)
    md = pq.read_metadata(p)
    print(f"\n== {split}: {md.num_rows} rows")
    for n, t in zip(sc.names, sc.types):
        print(f"   {n:<18} {t}")
