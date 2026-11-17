# 同一個受記憶體上限約束的分析內，平行估計同一賽制與切分的兩種模型。
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby

os.environ.setdefault("RAYON_NUM_THREADS", "6")


def run_job(job):
    speed, fold, kind = job
    longer = speed == "Classical" or job in [("Bullet", "Oct-Dec", "slope"), ("Blitz", "Oct-Dec", "slope")]
    env = os.environ | {"HIERARCHICAL_SPEED": speed, "HIERARCHICAL_FOLD": fold, "HIERARCHICAL_KIND": kind,
                        "HIERARCHICAL_DRAWS": "4000" if longer else "2000",
                        "HIERARCHICAL_TUNE": "2000" if longer else "1000"}
    print("fit", speed, fold, kind, flush=True)
    subprocess.run([sys.executable, "scripts/hierarchical_prediction.py"], env=env, check=True)
    subprocess.run([sys.executable, "scripts/hierarchical_evaluate.py"], env=env, check=True)

jobs = [(speed, fold, kind) for speed in ["Classical", "Bullet", "Blitz"]
        for fold in ["Jul-Sep", "Oct-Dec"] for kind in ["slope", "intercept"]]
# 診斷失敗的配適單獨處理後，可明確指定下一件工作，避免重跑已驗證的配適。
start = os.environ.get("HIERARCHICAL_START")
if start:
    jobs = jobs[jobs.index(tuple(start.split(","))):]
workers = int(os.environ.get("HIERARCHICAL_WORKERS", "2"))
with ThreadPoolExecutor(max_workers=workers) as pool:
    # 不重疊不同時間切分，避免兩個最大的 random-slope 配適同時到達記憶體峰值。
    for _, group in groupby(jobs, key=lambda job: job[:2]):
        pending = [pool.submit(run_job, job) for job in group]
        for future in pending:
            future.result()
subprocess.run([sys.executable, "scripts/hierarchical_summary.py"], check=True)
