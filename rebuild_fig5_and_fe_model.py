"""
Figure 3 (ensemble flux-slice and CMI-null panels; written as ensemble_1024_rebuilt.png) and the free-energy generative-model fit of main text §5.2.

Loads the 12 worker files ensemble_merged_seed1000_worker{0..11}.npz (the same files analyze_ensemble_merged.py merges), so
run it from local_package.  Prints: N, the marginal and PC1 partial correlations; the eight parameters of the generative
model (prior mu_p, sigma_p^2; likelihood A, b, Sigma; posterior slope, intercept, sigma_q^2); the KL term; and an N=8000
subsample CMI with a nearest-neighbour local-shuffle null (300 trials).  Saves ensemble_1024_rebuilt.png.

Usage:
    python3 rebuild_fig5_and_fe_model.py
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

N_WORKERS = 12
SEED_BASE = 1000
N_NULL = 300
K = 5


def residualize(y, B):
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ c


def pc1_of(x1, x2):
    B_std = np.column_stack([(x1 - x1.mean()) / x1.std(), (x2 - x2.mean()) / x2.std()])
    cov = np.cov(B_std.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return B_std @ eigvecs[:, np.argmax(eigvals)]


def load_and_merge():
    I_list, E_list, fu_list, fv_list, ms_list = [], [], [], [], []
    for w in range(N_WORKERS):
        path = f"ensemble_merged_seed{SEED_BASE}_worker{w}.npz"
        try:
            d = np.load(path)
        except FileNotFoundError:
            print(f"  WARNING: {path} not found, skipping")
            continue
        I_list.append(d["I"]); E_list.append(d["E"])
        fu_list.append(d["fu"]); fv_list.append(d["fv"])
        ms_list.append(d["merge_size"])
        print(f"  {path}: {len(d['I'])} organism-instances")
    I, E = np.concatenate(I_list), np.concatenate(E_list)
    fu, fv = np.concatenate(fu_list), np.concatenate(fv_list)
    ms = np.concatenate(ms_list)
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    return I[good], E[good], fu[good], fv[good], ms[good]


def main():
    I, E, fu, fv, ms = load_and_merge()
    N = len(I)
    print(f"\nmerged N={N}")

    # --- sanity check against already-reported numbers ---
    r_marginal = np.corrcoef(I, E)[0, 1]
    pc1 = pc1_of(fu, fv)
    B = np.column_stack([np.ones(N), pc1])
    r_pc1 = np.corrcoef(residualize(I, B), residualize(E, B))[0, 1]
    print(f"marginal r(I,E) = {r_marginal:.4f}")
    print(f"PC1 partial r   = {r_pc1:.4f}")

    # --- generative model fit (main text \S5.2) ---
    mu_p, sigma_p2 = E.mean(), E.var()
    print(f"\nPrior p(s)=N(mu_p,sigma_p^2):  mu_p={mu_p:.4f}  sigma_p^2={sigma_p2:.3e}")

    # Likelihood: (fu,fv) = A*E + b + noise(Sigma)
    Bo = np.column_stack([np.ones(N), E])
    coef_u, *_ = np.linalg.lstsq(Bo, fu, rcond=None)
    coef_v, *_ = np.linalg.lstsq(Bo, fv, rcond=None)
    resid_u = fu - Bo @ coef_u
    resid_v = fv - Bo @ coef_v
    Sigma = np.cov(np.column_stack([resid_u, resid_v]).T)
    A = np.array([coef_u[1], coef_v[1]])
    b = np.array([coef_u[0], coef_v[0]])
    print(f"Likelihood p(o|s)=N(As+b,Sigma):  A=({A[0]:.3e},{A[1]:.3e})  "
          f"b=({b[0]:.3e},{b[1]:.3e})")
    print(f"  Sigma = [[{Sigma[0,0]:.3e},{Sigma[0,1]:.3e}],[{Sigma[1,0]:.3e},{Sigma[1,1]:.3e}]]")

    # Posterior: E ~ slope*I + intercept, residual var sigma_q^2
    Bi = np.column_stack([np.ones(N), I])
    coef_q, *_ = np.linalg.lstsq(Bi, E, rcond=None)
    resid_q = E - Bi @ coef_q
    sigma_q2 = resid_q.var()
    print(f"Posterior q(s|I)=N(mu_q(I),sigma_q^2):  slope={coef_q[1]:.4f}  "
          f"intercept={coef_q[0]:.4f}  sigma_q^2={sigma_q2:.3e}")

    # --- KL term distribution (for the \S5.2 "mean/minimum" sentence) ---
    def kl(mu_q, sig_q2, mu_p_, sig_p2_):
        return np.log(np.sqrt(sig_p2_) / np.sqrt(sig_q2)) + (sig_q2 + (mu_q - mu_p_) ** 2) / (2 * sig_p2_) - 0.5
    mu_q_all = Bi @ coef_q
    kl_vals = kl(mu_q_all, sigma_q2, mu_p, sigma_p2)
    print(f"\nKL(q||p) over population: mean={kl_vals.mean():.4f}  min={kl_vals.min():.4f}")

    # --- NN-null CMI sanity check on an N=8000 subsample ---
    rng = np.random.default_rng(0)
    sub = rng.choice(N, size=min(8000, N), replace=False)
    I_s, E_s = I[sub], E[sub]
    fu_s, fv_s = fu[sub], fv[sub]
    from conditional_mi import cmi_ksg
    Iz, Ez = (I_s - I_s.mean()) / I_s.std(), (E_s - E_s.mean()) / E_s.std()
    Bz = np.column_stack([(fu_s - fu_s.mean()) / fu_s.std(), (fv_s - fv_s.mean()) / fv_s.std()])
    observed = cmi_ksg(Iz, Ez, Bz, k=K)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(Bz)
    _, neighbor_idx = nn.kneighbors(Bz)
    neighbor_idx = neighbor_idx[:, 1:]
    n_sub = len(I_s)
    null_vals = np.empty(N_NULL)
    for t in range(N_NULL):
        r = np.random.default_rng(1000 + t)
        out = Ez.copy()
        n_swaps = n_sub * 20
        i_picks = r.integers(0, n_sub, n_swaps)
        k_picks = r.integers(0, K, n_swaps)
        j_picks = neighbor_idx[i_picks, k_picks]
        for i, j in zip(i_picks, j_picks):
            out[i], out[j] = out[j], out[i]
        null_vals[t] = cmi_ksg(Iz, out, Bz, k=K)
        if (t + 1) % 50 == 0:
            print(f"  null trial {t+1}/{N_NULL}")
    null_mean, null_std = null_vals.mean(), null_vals.std()
    n_sd = (observed - null_mean) / null_std
    print(f"\nCMI sanity check: observed={observed:.4f}  null_mean={null_mean:.4f}  "
          f"null_std={null_std:.4f}  n_sd={n_sd:.1f}")

    # --- regenerate Figure 5 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    sub2 = rng.choice(N, size=min(20000, N), replace=False)
    I_res_full, E_res_full = residualize(I, B), residualize(E, B)
    ax.scatter(I_res_full[sub2], E_res_full[sub2], s=2, alpha=0.15, color="steelblue", rasterized=True)
    ax.set_xlabel("residual interior state $I$ (PC1 blanket flux regressed out)")
    ax.set_ylabel("residual exterior state $E$ (PC1 blanket flux regressed out)")
    ax.set_title(f"Partial correlation $r={r_pc1:.3f}$  ($N={N:,}$)")
    ax.axhline(0, color="gray", linewidth=0.8, zorder=0)
    ax.axvline(0, color="gray", linewidth=0.8, zorder=0)

    ax = axes[1]
    ax.hist(null_vals, bins=30, color="steelblue", alpha=0.7, label="nearest-neighbor null")
    ax.axvline(observed, color="crimson", linewidth=2, label=f"observed = {observed:.3f} nats")
    ax.set_xlabel("conditional mutual information (nats)")
    ax.set_ylabel("count (null trials)")
    ax.set_title(f"{n_sd:.1f} null-standard-deviations above null mean")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig("ensemble_1024_rebuilt.png", dpi=130)
    print("\nsaved ensemble_1024_rebuilt.png ")


if __name__ == "__main__":
    main()
