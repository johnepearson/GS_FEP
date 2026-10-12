"""
Depth-fraction sweep (main text S2.2 / ESM Figure S1, the r(d) curve for core depth fractions
d = 0.5 ... 0.9) computed OFFLINE from the master single-trajectory run: no simulation.

For every selected frame it reads the archived v field and merged labels (master_fields/frame_XXXX.npz) and the
per-organism centroids, fluxes and merge_size already stored in master_trajectory_progress.npz, and recomputes
the interior/exterior states at each depth fraction with the same function the production runs use
(run_confound_checks.organism_interior_exterior_states_fast, crop 90, exterior ring 24 px).

Writes sweep_merged_progress.npz in the format plot_depth_sensitivity_pc1.py reads (keys '0.5_I', '0.5_E', '0.5_fu',
'0.5_fv', ..., merge_size), so afterwards:
    python3 plot_depth_sensitivity_pc1.py --data sweep_merged_progress.npz

As a check it also reports how closely the d=0.5 values reproduce the I, E stored by the master run (they should agree
to ~1e-6: the archive keeps v in float32).

Usage:
    python3 sweep_from_master.py                  # every 10th frame
    python3 sweep_from_master.py --stride 1       # all archived frames
"""
import argparse
import os
import types
import numpy as np

from run_confound_checks import organism_interior_exterior_states_fast

FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90]
EXTERIOR_RING_PX = 24
CROP_IE = 90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--fields-dir", default="master_fields")
    ap.add_argument("--stride", type=int, default=10, help="use every Nth frame (default 10)")
    ap.add_argument("--out", default="sweep_merged_progress.npz")
    args = ap.parse_args()

    d = np.load(args.data)
    fid = d["frame_id"]
    frames = [f for f in np.unique(fid) if f % args.stride == 0]
    store = {f: {k: [] for k in ("I", "E", "fu", "fv")} for f in FRACTIONS}
    ms_all, chk = [], []
    n_used = 0
    for f in frames:
        path = os.path.join(args.fields_dir, f"frame_{int(f):04d}.npz")
        if not os.path.exists(path):
            continue
        a = np.load(path)
        labels = a["labels"].astype(np.int64)
        sel = np.where(fid == f)[0]
        sel = sel[np.argsort(d["org_idx"][sel])]
        n = len(sel)
        if labels.max() != n:
            print(f"frame {int(f)}: label count {labels.max()} != rows {n}, skipped")
            continue
        sim = types.SimpleNamespace(n=labels.shape[0], v=a["v"].astype(np.float64))
        cent = np.column_stack([d["cy"][sel], d["cx"][sel]])
        for fr in FRACTIONS:
            I, E = organism_interior_exterior_states_fast(sim, labels, n, cent, depth_fraction=fr,
                                                          exterior_ring_px=EXTERIOR_RING_PX, crop=CROP_IE)
            store[fr]["I"].append(I); store[fr]["E"].append(E)
            store[fr]["fu"].append(d["fu"][sel]); store[fr]["fv"].append(d["fv"][sel])
            if fr == 0.5:
                chk.append(np.nanmax(np.abs(I - d["I"][sel])))
        ms_all.append(d["merge_size"][sel])
        n_used += 1
        if n_used % 10 == 0:
            print(f"{n_used} frames done", flush=True)

    if not n_used:
        raise SystemExit("no archived frames found")
    payload = {"frames_done": n_used, "merge_size": np.concatenate(ms_all)}
    for fr in FRACTIONS:
        for k in ("I", "E", "fu", "fv"):
            payload[f"{fr}_{k}"] = np.concatenate(store[fr][k])
    np.savez(args.out, **payload)
    N = len(payload["merge_size"])
    print(f"\n{n_used} frames, N={N:,} organism-instances per depth fraction, saved to {args.out}")
    print(f"consistency of d=0.5 with the master run's stored I: max |difference| = {max(chk):.2e}")


if __name__ == "__main__":
    main()
