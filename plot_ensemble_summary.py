"""
Two-panel summary of the independent-ensemble result (paper: Figure of the ensemble CMI test, ensemble_1024.png).

Left : interior-exterior mutual information MI(I;E) inside each of 20 equal-count slices of the PC1 flux regressor
       (3000 points per draw, mean of 8 draws, +-2 SE) against the same estimator on random 3000-point draws from the
       whole population (grey band, dashed mean).  Shows that conditioning on flux RAISES the dependence.
Right: conditional MI I(I;E | Phi_u, Phi_v) on an N=8000 subsample (all variables z-scored, k_NN = 5) against a null
       built by local permutation among the 5 nearest neighbours in flux space (20 random swaps per point), 300 trials.

Reads ensemble_merged_seed<BASE>_worker*.npz (written by run_ensemble_merged.py).  Takes a few minutes; prints the
numbers quoted in the main text (observed CMI, null mean and SD, excess in null SDs).

Usage:  python3 plot_ensemble_summary.py
        python3 plot_ensemble_summary.py --seed-base 1000 --out ensemble_1024.png
        python3 plot_ensemble_summary.py --quick
"""
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from conditional_mi import cmi_ksg


def z(x):
    return (x - x.mean(0)) / x.std(0)


def local_shuffle(E, nbr, rng, swaps_per_point=20):
    n, k = len(E), nbr.shape[1]
    out = E.copy()
    ii = rng.integers(0, n, size=n * swaps_per_point)
    jj = nbr[ii, rng.integers(0, k, size=n * swaps_per_point)]
    for i, j in zip(ii, jj):
        out[i], out[j] = out[j], out[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--n-cmi", type=int, default=8000)
    ap.add_argument("--n-null", type=int, default=300)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ensemble_1024.png")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_cmi, a.n_null, a.draws = 2000, 20, 2
    files = sorted(glob.glob(f"ensemble_merged_seed{a.seed_base}_worker*.npz"))
    if not files:
        raise SystemExit(f"no files ensemble_merged_seed{a.seed_base}_worker*.npz in this folder")
    ds = [np.load(f) for f in files]
    I, E, fu, fv = (np.concatenate([d[k] for d in ds]) for k in ("I", "E", "fu", "fv"))
    ok = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    I, E, fu, fv = I[ok], E[ok], fu[ok], fv[ok]
    N = len(I)
    print(f"{len(files)} files, N = {N:,}")
    rng = np.random.default_rng(a.seed)

    # ---- left panel: MI inside PC1 slices
    S = z(np.column_stack([fu, fv]))
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    pc1 = U[:, 0] * sv[0]
    zI, zE = z(I), z(E)
    nsl, m3 = 20, 3000 if not a.quick else 500
    edges = np.quantile(pc1, np.linspace(0, 1, nsl + 1))
    idx = np.clip(np.searchsorted(edges, pc1, side="right") - 1, 0, nsl - 1)
    cen, mi, se = [], [], []
    for b in range(nsl):
        m = np.where(idx == b)[0]
        cen.append(np.median(pc1[m]))
        est = [cmi_ksg(zI[p], zE[p], np.zeros((len(p), 1)), k=5)
               for p in (rng.choice(m, size=min(m3, len(m)), replace=False) for _ in range(a.draws))]
        mi.append(np.mean(est)); se.append(np.std(est, ddof=1) / np.sqrt(a.draws))
    mi, se = np.array(mi), np.array(se)
    pop = np.array([cmi_ksg(zI[p], zE[p], np.zeros((m3, 1)), k=5)
                    for p in (rng.choice(N, size=m3, replace=False) for _ in range(40 if not a.quick else 8))])
    band = 2 * pop.std() / np.sqrt(a.draws)
    print(f"MI(I;E) whole population at n={m3}: {pop.mean():.4f}; within flux slices: mean {mi.mean():.4f}, "
          f"min {mi.min():.4f}, max {mi.max():.4f}; {(mi > pop.mean() + band).sum()} of {nsl} slices above the band")
    peak = int(np.argmax(mi))
    print(f"peak slice: number {peak + 1} of {nsl} counting from the lowest flux, MI {mi[peak]:.4f} = {mi[peak] / pop.mean():.2f} times the whole-population value")
    print("slice-by-slice MI / whole population: " + " ".join(f"{x / pop.mean():.2f}" for x in mi))

    # ---- right panel: CMI against the local-permutation null
    # The subsample is drawn from the workers in numeric order (0, 1, 2, ...) with its own generator, exactly as in
    # rebuild_fig5_and_fe_model.py, so that both scripts report the same CMI and null.
    num = sorted(files, key=lambda f: int(f.rsplit("worker", 1)[1].split(".")[0]))
    dn = [np.load(f) for f in num]
    I2, E2, fu2, fv2 = (np.concatenate([d[k] for d in dn]) for k in ("I", "E", "fu", "fv"))
    ok2 = np.isfinite(I2) & np.isfinite(E2) & np.isfinite(fu2) & np.isfinite(fv2)
    I2, E2, fu2, fv2 = I2[ok2], E2[ok2], fu2[ok2], fv2[ok2]
    sub = np.random.default_rng(a.seed).choice(len(I2), a.n_cmi, replace=False)
    Is, Es = z(I2[sub]), z(E2[sub])
    B = z(np.column_stack([fu2[sub], fv2[sub]]))
    obs = cmi_ksg(Is, Es, B, k=5)
    nbr = NearestNeighbors(n_neighbors=6).fit(B).kneighbors(B)[1][:, 1:]
    null = np.array([cmi_ksg(Is, local_shuffle(Es, nbr, np.random.default_rng(1000 + t)), B, k=5)
                     for t in range(a.n_null)])
    n_sd = (obs - null.mean()) / null.std()
    print(f"CMI I(I;E | flux) at N={a.n_cmi}: observed {obs:.4f}, null mean {null.mean():.4f}, SD {null.std():.4f}, "
          f"{n_sd:.1f} null SDs ({a.n_null} trials)")
    print(f"unconditioned MI(I;E) on the same subsample: {cmi_ksg(Is, Es, np.zeros((len(Is), 1)), k=5):.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.errorbar(cen, mi, yerr=2 * se, fmt="o-", color="#1f4e8c", capsize=2,
                 label=f"within each flux slice (mean of {a.draws} draws, $\\pm$2 SE)")
    ax1.axhspan(pop.mean() - band, pop.mean() + band, color="0.6", alpha=0.4,
                label=f"whole population (mean of {a.draws} draws, $\\pm$2 SE)")
    ax1.axhline(pop.mean(), color="k", ls="--", lw=1)
    ax1.set_xlabel("flux regressor (PC1), slice median")
    ax1.set_ylabel("mutual information MI$(I;E)$ (nats)")
    ax1.set_title("Interior-exterior dependence within slices of fixed flux")
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.hist(null, bins=30, color="steelblue", alpha=0.7, label="nearest-neighbor null")
    ax2.axvline(obs, color="crimson", lw=2, label=f"observed = {obs:.3f} nats")
    ax2.set_xlabel("conditional mutual information (nats)")
    ax2.set_title(f"{n_sd:.1f} null-standard-deviations above null mean")
    ax2.set_yticklabels([])
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
