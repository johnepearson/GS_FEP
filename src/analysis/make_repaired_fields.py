"""
Write repaired label images for the archived master-trajectory fields, into a NEW folder (master_fields_repaired/).

Why: master_trajectory_progress.npz (in code_repaired) holds the repaired per-organism rows, but master_fields/frame_XXXX.npz
still holds the OLD (unrepaired) label images, so any script that pairs labels with rows (sweep_from_master.py,
check_local_jacobian.py, run_random_disks.py ...) was mixing the two. This script runs the repaired production pipeline
(run_master_trajectory.process_frame with the patched gray_scott) on the archived u, v of every frame and archives
u, v, labels, raw_labels exactly as the original run did, then checks each frame against the repaired rows.

Run from code_repaired (reads master_fields/, never writes to it):
    python3 make_repaired_fields.py --limit 3 --workers 3      # quick test first
    python3 make_repaired_fields.py --workers 8
Afterwards pass  --fields-dir master_fields_repaired  to the scripts that read labels.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np

import run_master_trajectory as M
from gray_scott import GrayScottSim

CHECKPOINT = "checkpoint_1024_settled.npz"
OUT = "master_fields_repaired"
_P = None


def params():
    global _P
    if _P is None:
        d = np.load(CHECKPOINT)
        _P = dict(n=int(d["n"]), L=float(d["L"]), Du=float(d["Du"]), Dv=float(d["Dv"]),
                  F=float(d["F"]), k=float(d["k"]), dt=float(d["dt"]))
    return _P


def work(path):
    frame_id = int(os.path.basename(path).split("_")[1].split(".")[0])
    d = np.load(path)
    proxy = GrayScottSim(**params())
    proxy.u, proxy.v = d["u"].astype(np.float64), d["v"].astype(np.float64)
    res = M.process_frame(proxy, frame_id, float(d["time"]), OUT, True)
    if res is None:
        return frame_id, None
    return frame_id, res[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields-dir", default="master_fields")
    ap.add_argument("--rows", default="master_trajectory_progress.npz", help="the repaired rows to check against")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.fields_dir, "frame_*.npz")))
    if a.limit:
        paths = paths[:a.limit]
    ref = np.load(a.rows)
    fid = ref["frame_id"]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    bad = 0
    with Pool(a.workers) as pool:
        for k, (f, rows) in enumerate(pool.imap(work, paths), 1):
            sel = np.where(fid == f)[0]
            if rows is None:
                ok = len(sel) == 0
                msg = "no organisms"
            else:
                same_n = len(rows["I"]) == len(sel)
                dI = np.nanmax(np.abs(rows["I"] - ref["I"][sel])) if same_n and len(sel) else np.inf
                ok = same_n and dI < 1e-6
                msg = f"n={len(rows['I'])} vs {len(sel)}  max|dI|={dI:.2e}"
            if not ok:
                bad += 1
                print(f"frame {f}: MISMATCH with the repaired rows ({msg})", flush=True)
            if k % 50 == 0 or k == len(paths):
                print(f"{k}/{len(paths)} frames  ({time.time() - t0:.0f} s)  mismatches so far: {bad}", flush=True)
    print(f"done: {len(paths)} frames written to {OUT}/, {bad} mismatching the repaired rows")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
