"""
Cross-check of our conditional-mutual-information estimator (conditional_mi.cmi_ksg) against Tigramite's CMIknn (Runge 2018),
on the real master-trajectory data.

For the same z-scored variables and the same N=8000 subsamples used in the paper, it computes I(I; E_w | Z_w) with both
estimators (k = 5) for a few blanket thicknesses w (w = 0 is the flux-only blanket of the headline test), and, for flux only,
also the null distribution from Tigramite's own local-permutation test (10 neighbours) against ours.

Setup (once):  pip install tigramite
Usage (from local_package, with master_trajectory_progress.npz present):
    python3 run_cmi_crosscheck.py --quick
    python3 run_cmi_crosscheck.py

Notes: Tigramite's CMIknn applies a RANK transform by default. The main comparison therefore uses transform="standardize"
(z-scoring, as in our estimator) so that only the estimator differs; a third column shows Tigramite with its default
rank transform. Takes a few minutes.
"""
import argparse
import numpy as np

from conditional_mi import cmi_ksg
from run_u_enriched_cmi import zscore, local_null
from run_band_sweep_cmi import build

try:
    from tigramite.independence_tests.cmiknn import CMIknn
except ImportError as exc:
    raise SystemExit(f"could not import tigramite ({exc}). If it names numba, run:  python3 -m pip install numba")

WIDTHS = [0, 8, 24]


def tig_cmi(t, x, y, Z):
    arr = np.vstack([x, y, Z.T])
    xyz = np.array([0, 1] + [2] * Z.shape[1])
    return t.get_dependence_measure(arr, xyz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_sub, a.seeds, a.n_null = min(a.n_sub, 2000), 1, 10
    d = np.load(a.data)
    I, fu, fv = d["I"], d["fu"], d["fv"]
    built = {w: build(d, w) for w in WIDTHS}
    good = np.isfinite(I) & np.isfinite(fu) & np.isfinite(fv)
    for w, (E, B) in built.items():
        good &= np.isfinite(E) & np.isfinite(B).all(1)
    idx_all = np.where(good)[0]
    t = CMIknn(knn=5, shuffle_neighbors=10, transform="standardize")
    t_rank = CMIknn(knn=5, shuffle_neighbors=10)
    print(f"{len(idx_all):,} usable organism-instances; N={a.n_sub} per subsample, {a.seeds} subsample(s)")
    print(f"{'w':>3s} {'seed':>4s} {'ours':>8s} {'tig(std)':>9s} {'ratio':>6s} {'tig(rank)':>10s}")
    summary = {w: [] for w in WIDTHS}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx_all, a.n_sub, replace=False)
        flux = np.column_stack([fu[idx], fv[idx]])
        for w in WIDTHS:
            E, B = built[w]
            Ii, Ei = zscore(I[idx])[:, 0], zscore(E[idx])[:, 0]
            Z = zscore(np.column_stack([flux, B[idx]]))
            ours = cmi_ksg(Ii, Ei, Z)
            tig = tig_cmi(t, Ii, Ei, Z)
            tig_r = tig_cmi(t_rank, Ii, Ei, Z)
            summary[w].append((ours, tig, tig_r))
            print(f"{w:3d} {seed:4d} {ours:8.4f} {tig:9.4f} {ours/tig if abs(tig) > 0.005 else float('nan'):6.2f} {tig_r:10.4f}", flush=True)
            if w == 0 and seed == 0:
                Zf = Z
                Ii0, Ei0 = Ii, Ei
                obs_o, obs_t = ours, tig
    print("\nMeans over subsamples")
    for w in WIDTHS:
        o, g, gr = np.array(summary[w]).mean(0)
        print(f"  w={w:3d}: ours {o:.4f}  tigramite(std) {g:.4f}  difference {o-g:+.4f}   tigramite(rank) {gr:.4f}")

    print(f"\nFlux-only null ({a.n_null} trials each), first subsample")
    nl_ours = local_null(Ii0, Ei0, Zf, a.n_null, 0)
    arr = np.vstack([Ii0, Ei0, Zf.T])
    xyz = np.array([0, 1, 2, 2])
    out = t.get_shuffle_significance(arr, xyz, obs_t, return_null_dist=True)
    nl_tig = np.asarray(out[1] if isinstance(out, tuple) else out)
    print(f"  ours:      observed {obs_o:.4f}, null mean {nl_ours.mean():.4f}, SD {nl_ours.std():.4f}, "
          f"excess {(obs_o - nl_ours.mean()) / nl_ours.std():.1f} SD")
    print(f"  tigramite: observed {obs_t:.4f}, null mean {nl_tig.mean():.4f}, SD {nl_tig.std():.4f}, "
          f"excess {(obs_t - nl_tig.mean()) / nl_tig.std():.1f} SD")


if __name__ == "__main__":
    main()
