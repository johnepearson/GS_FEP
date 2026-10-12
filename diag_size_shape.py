"""
Why does adding radius and eccentricity change the I-E partial correlation so much? Diagnostic on the master data.

    python3 diag_size_shape.py

For the full population and for clean / crowded organisms separately: marginal correlations of radius and
eccentricity with I, E and the fluxes, and the PC1-flux partial correlation of I and E with each of radius,
eccentricity, area added alone and together, plus within-radius-bin partial correlations.
"""
import numpy as np
from partial_corr_utils import pc1_partial_corr


def resid(y, B):
    return y - B @ np.linalg.lstsq(B, y, rcond=None)[0]


def pc1_of(fu, fv):
    Z = np.column_stack([(fu - fu.mean()) / fu.std(), (fv - fv.mean()) / fv.std()])
    w, V = np.linalg.eigh(np.cov(Z.T))
    return Z @ V[:, np.argmax(w)]


def pr(I, E, fu, fv, *covs):
    B = np.column_stack([np.ones(len(I)), pc1_of(fu, fv)] + list(covs))
    return np.corrcoef(resid(I, B), resid(E, B))[0, 1]


d = np.load("master_trajectory_progress.npz")
I, E, fu, fv, ms = d["I"], d["E"], d["fu"], d["fv"], d["merge_size"]
rad, ecc, area = d["radius_px"], d["eccentricity"], d["area_px"]
good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv) & np.isfinite(rad) & np.isfinite(ecc)
print(f"radius (px): median {np.median(rad[good]):.1f}, 5-95% [{np.percentile(rad[good],5):.1f}, {np.percentile(rad[good],95):.1f}];"
      f"  eccentricity median {np.median(ecc[good]):.2f}\n")
for name, m in (("full", good), ("clean (merge_size == 1)", good & (ms == 1)), ("crowded (merge_size > 1)", good & (ms > 1))):
    i, e, u, v, r, c, a = I[m], E[m], fu[m], fv[m], rad[m], ecc[m], area[m]
    print(f"{name}: N={m.sum():,}   median radius {np.median(r):.1f} px, median eccentricity {np.median(c):.2f}")
    print("  marginal correlations:  r(I,E) = %+.3f | radius with I %+.3f, E %+.3f, Phi_u %+.3f | eccentricity with I %+.3f, E %+.3f"
          % (np.corrcoef(i, e)[0, 1], np.corrcoef(r, i)[0, 1], np.corrcoef(r, e)[0, 1], np.corrcoef(r, u)[0, 1],
             np.corrcoef(c, i)[0, 1], np.corrcoef(c, e)[0, 1]))
    print("  PC1-flux partial r(I,E):  flux only %+.4f | + radius %+.4f | + eccentricity %+.4f | + area %+.4f | + radius & eccentricity %+.4f"
          % (pr(i, e, u, v), pr(i, e, u, v, r), pr(i, e, u, v, c), pr(i, e, u, v, a), pr(i, e, u, v, r, c)))
    qs = np.percentile(r, [0, 20, 40, 60, 80, 100])
    print("  partial r(I,E | flux) within radius quintiles:  " +
          "  ".join(f"[{qs[k]:.0f}-{qs[k+1]:.0f}px] {pr(i[(r>=qs[k])&(r<=qs[k+1])], e[(r>=qs[k])&(r<=qs[k+1])], u[(r>=qs[k])&(r<=qs[k+1])], v[(r>=qs[k])&(r<=qs[k+1])]):+.3f}" for k in range(5)))
    print()
