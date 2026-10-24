"""Level-set (non-watershed) individuation check, run on the archived fields of the master trajectory (marker spacing 24).

Level-set individuation applied to the stored master trajectory. No new simulation: for every
STRIDE-th archived frame (master_fields/frame_XXXX.npz) it finds fixed-threshold connected components of v (v > 0.1356), keeps
the single-maximum components (peaks found with peak_local_max, min_distance=24), and also recovers the multi-maximum
components with a local watershed so the selection effect can be checked. Reports the PC1-flux partial correlation (the paper's
convention) and, for reference, the joint two-variable value, for: single-peak only, recovered multi-peak only, pooled.

    python3 level_set_from_master.py            # default --stride 5 (100 frames)
"""
import argparse
import os
import numpy as np
from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
import run_level_set_individuation as ls

MIN_PEAK_DISTANCE = 24


def resid(y, B):
    return y - B @ np.linalg.lstsq(B, y, rcond=None)[0]


def pc1(fu, fv):
    Z = np.column_stack([(fu - fu.mean()) / fu.std(), (fv - fv.mean()) / fv.std()])
    w, V = np.linalg.eigh(np.cov(Z.T))
    return Z @ V[:, np.argmax(w)]


def stats(I, E, fu, fv):
    g = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    I, E, fu, fv = I[g], E[g], fu[g], fv[g]
    B1 = np.column_stack([np.ones(len(I)), pc1(fu, fv)])
    B2 = np.column_stack([np.ones(len(I)), fu, fv])
    return (len(I), np.corrcoef(I, E)[0, 1], np.corrcoef(resid(I, B1), resid(E, B1))[0, 1],
            np.corrcoef(resid(I, B2), resid(E, B2))[0, 1])


ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=5)
ap.add_argument("--fields-dir", default="master_fields")
args = ap.parse_args()

sim = PseudoSpectralGS.from_checkpoint(ls.CHECKPOINT_FILE)
proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)
files = sorted(f for f in os.listdir(args.fields_dir) if f.startswith("frame_"))[::args.stride]
keys = ["I", "E", "fu", "fv"]
single = {k: [] for k in keys}
multi = {k: [] for k in keys}
n_comp = n_single = n_zero = n_multi_peaks = 0
for i, f in enumerate(files):
    a = np.load(os.path.join(args.fields_dir, f))
    sim.u, sim.v = a["u"].astype(np.float64), a["v"].astype(np.float64)
    proxy.u, proxy.v = sim.u, sim.v
    proxy.Lu = laplacian_periodic(sim.u, sim.dx)
    proxy.Lv = laplacian_periodic(sim.v, sim.dx)
    labeled, nc = ls.level_set_components(sim.v)
    peaks = ls.find_all_peaks(sim.v, min_peak_distance=MIN_PEAK_DISTANCE)
    n_comp += nc
    if nc == 0:
        continue
    counts = ls.assign_peaks_to_components(labeled, peaks)
    n_zero += int((counts == 0).sum())
    lab_s, ns = ls.relabel_subset(labeled, np.where(counts == 1)[0] + 1)
    for k, x in zip(keys, ls.compute_states(sim, proxy, lab_s, ns)):
        single[k].append(x)
    n_single += ns
    split, nm = ls.split_multi_peak_components(labeled, counts, peaks, sim.v)
    n_multi_peaks += nm
    if nm > 0:
        for k, x in zip(keys, ls.compute_states(sim, proxy, split, nm)):
            multi[k].append(x)
    if (i + 1) % 10 == 0:
        print(f"{i+1}/{len(files)} frames; single-peak so far {n_single}, recovered multi-peak {n_multi_peaks}", flush=True)

S = {k: np.concatenate(v) for k, v in single.items()}
M = {k: np.concatenate(v) if v else np.array([]) for k, v in multi.items()}
print(f"\n{len(files)} frames; components {n_comp}: single-maximum {n_single} ({100*n_single/n_comp:.1f}%), "
      f"zero-maximum {n_zero} ({100*n_zero/n_comp:.1f}%), multi-maximum peaks recovered {n_multi_peaks} "
      f"({100*n_multi_peaks/(n_single+n_multi_peaks):.1f}% of the pooled organisms)")
for name, d in (("single-peak only", S), ("recovered multi-peak only", M),
                ("pooled (single + recovered)", {k: np.concatenate([S[k], M[k]]) for k in keys})):
    if len(d["I"]) == 0:
        continue
    N, rm, rp, rj = stats(d["I"], d["E"], d["fu"], d["fv"])
    print(f"{name:30s} N={N:,}  marginal r={rm:+.4f}  PC1 partial r={rp:+.4f}  (joint {rj:+.4f})")

# save the per-organism states so the nonlinear CMI check (run_levelset_cmi.py) can use exactly these organisms
np.savez("levelset_states.npz", **{f"single_{k}": S[k] for k in keys}, **{f"multi_{k}": M[k] for k in keys})
print("wrote levelset_states.npz")
