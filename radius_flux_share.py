"""
How much of an organism's radius is shared with its blanket flux?

The main text (§3.3.3) says: "radius shares 70% of its variance with the first principal component of flux, so
the adjusted flux is nearly a readout of I itself."  This script computes that number.

For the single trajectory (master_trajectory_progress.npz) and, if present, the independent ensembles
(ensemble_merged_seed1000_worker0..11.npz, seed9000), it prints:
  - R^2 of radius_px regressed on PC1 of the standardized flux pair (the number quoted in the paper),
  - R^2 of radius_px regressed on both fluxes (standardized),
  - the correlation of radius with I.
PC1 is computed as in partial_corr_utils.pc1_partial_corr (leading principal component of the standardized (Phi_u, Phi_v)).

Usage (from the code folder):  python3 radius_flux_share.py
"""
import glob
import numpy as np


def load_ensemble(base):
    files = sorted(glob.glob(f"ensemble_merged_seed{base}_worker*.npz"))
    if not files:
        return None
    out = {k: [] for k in ("I", "fu", "fv", "radius_px")}
    for f in files:
        d = np.load(f)
        for k in out:
            if k not in d.files:
                return None
            out[k].append(np.asarray(d[k]).ravel())
    return {k: np.concatenate(v) for k, v in out.items()}


def report(name, I, fu, fv, rad):
    g = np.isfinite(I) & np.isfinite(fu) & np.isfinite(fv) & np.isfinite(rad)
    I, fu, fv, rad = I[g], fu[g], fv[g], rad[g]
    zu, zv = (fu - fu.mean()) / fu.std(), (fv - fv.mean()) / fv.std()
    Z = np.column_stack([zu, zv])
    w, V = np.linalg.eigh(np.cov(Z.T))
    pc1 = Z @ V[:, np.argmax(w)]
    r_pc1 = np.corrcoef(pc1, rad)[0, 1]
    X = np.column_stack([np.ones(len(rad)), zu, zv])
    beta = np.linalg.lstsq(X, rad, rcond=None)[0]
    r2_both = 1 - np.sum((rad - X @ beta) ** 2) / np.sum((rad - rad.mean()) ** 2)
    print(f"{name:34s} N={len(rad):>8,}  R^2(radius ~ PC1) = {r_pc1**2:.3f}   R^2(radius ~ both fluxes) = {r2_both:.3f}   "
          f"corr(radius, I) = {np.corrcoef(rad, I)[0, 1]:+.3f}")


d = np.load("master_trajectory_progress.npz")
report("single trajectory", d["I"], d["fu"], d["fv"], d["radius_px"])
for base in (1000, 9000, 20000):
    e = load_ensemble(base)
    if e is not None:
        report(f"ensemble seed {base}", e["I"], e["fu"], e["fv"], e["radius_px"])
