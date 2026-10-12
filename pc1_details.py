"""
Details of the PC1 flux regressor, for the ESM subsection "The PC1 flux regressor".

For each dataset it prints:
  N, r(Phi_u, Phi_v), fraction of the standardized flux pair's variance carried by PC1, the condition number of
  the joint-regression design matrix [1, Phi_u, Phi_v] as it stands (raw units) and after standardizing the
  fluxes, and the I-E partial correlation four ways: conditioning on PC1 (the paper's convention), on both fluxes
  jointly, on Phi_u alone and on Phi_v alone. The stability criterion is that the joint value must lie between the
  Phi_u-only and Phi_v-only values.

Datasets (in local_package/): master_trajectory_progress.npz (single trajectory) and the independent ensembles
ensemble_merged_seed1000_worker0..11.npz and, if present, ensemble_merged_seed9000_worker0..11.npz.
Usage:  python3 pc1_details.py
"""
import glob
import numpy as np
from partial_corr_utils import pc1_partial_corr, check_partial_corr_stability


def report(name, I, E, fu, fv):
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    I, E, fu, fv = I[good], E[good], fu[good], fv[good]
    r_flux = np.corrcoef(fu, fv)[0, 1]
    r_pc1, n, expl = pc1_partial_corr(I, E, fu, fv)
    r_both, r_fu, r_fv, cond_raw, suspect = check_partial_corr_stability(I, E, fu, fv)
    Z = np.column_stack([np.ones(n), (fu - fu.mean()) / fu.std(), (fv - fv.mean()) / fv.std()])
    cond_std = np.linalg.cond(Z)
    lo, hi = sorted([r_fu, r_fv])
    print(f"\n{name}   N={n:,}")
    print(f"  r(Phi_u, Phi_v) = {r_flux:+.4f};  PC1 carries {100*expl:.2f}% of the standardized flux variance")
    print(f"  design condition number: raw units {cond_raw:.3g};  standardized {cond_std:.3g}")
    print(f"  partial r (I,E | .):  PC1 {r_pc1:+.4f}   joint {r_both:+.4f}   Phi_u only {r_fu:+.4f}   Phi_v only {r_fv:+.4f}")
    print(f"  joint lies between the single-flux values [{lo:+.4f}, {hi:+.4f}]: {'yes' if not suspect else 'NO'}")


def load_ensemble(seed_base):
    files = sorted(glob.glob(f"ensemble_merged_seed{seed_base}_worker*.npz"))
    if not files:
        return None
    parts = [np.load(f) for f in files]
    return tuple(np.concatenate([p[k] for p in parts]) for k in ("I", "E", "fu", "fv"))


def main():
    d = np.load("master_trajectory_progress.npz")
    report("Single trajectory (master)", d["I"], d["E"], d["fu"], d["fv"])
    for sb in (1000, 9000, 20000):
        e = load_ensemble(sb)
        if e is None:
            print(f"\n(no ensemble files for seed base {sb})")
        else:
            report(f"Independent ensemble, seed {sb}", *e)


if __name__ == "__main__":
    main()
