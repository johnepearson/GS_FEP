"""
Held-out-half validation of the free-energy generative model (main text §5.2): fit on the whole primary
ensemble, fit again on a random half, and correlate the two free energies on the other half.  Merges the 12 per-worker files
ensemble_merged_seed1000_worker{0..11}.npz and uses fit_prior, fit_likelihood, fit_posterior and compute_F from
run_friston_fe_fit.py unchanged.

Usage (from local_package):
    python3 rebuild_splithalf_check.py
"""
import numpy as np

from run_friston_fe_fit import fit_prior, fit_likelihood, fit_posterior, compute_F

N_WORKERS = 12
SEED_BASE = 1000
SEED = 0  # matches run_friston_fe_fit.py's --seed default


def load_and_merge():
    I_list, E_list, fu_list, fv_list = [], [], [], []
    for w in range(N_WORKERS):
        path = f"ensemble_merged_seed{SEED_BASE}_worker{w}.npz"
        try:
            d = np.load(path)
        except FileNotFoundError:
            print(f"  WARNING: {path} not found, skipping")
            continue
        I_list.append(d["I"]); E_list.append(d["E"])
        fu_list.append(d["fu"]); fv_list.append(d["fv"])
        print(f"  {path}: {len(d['I'])} organism-instances")
    I, E = np.concatenate(I_list), np.concatenate(E_list)
    fu, fv = np.concatenate(fu_list), np.concatenate(fv_list)
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    return I[good], E[good], fu[good], fv[good]


def main():
    I, E, fu, fv = load_and_merge()
    N = len(I)
    print(f"\nmerged N={N}")

    mu_p, sigma_p2 = fit_prior(E)
    A, b, Sigma = fit_likelihood(E, fu, fv)
    slope, intercept, sigma_q2, r_marg = fit_posterior(I, E)
    print(f"\nFitted generative model :")
    print(f"  prior:      mu_p={mu_p:.4f}  sigma_p^2={sigma_p2:.4e}")
    print(f"  likelihood: A=({A[0]:.4e}, {A[1]:.4e})  b=({b[0]:.4e}, {b[1]:.4e})")
    print(f"  posterior:  slope={slope:.4f}  intercept={intercept:.4f}  sigma_q^2={sigma_q2:.4e}")

    # --- held-out-half validation, identical procedure to
    # run_friston_fe_fit.py's own main() ---
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(N)
    half = N // 2
    idx_fit, idx_eval = idx[:half], idx[half:]

    def fit_all(ii):
        mp, sp2 = fit_prior(E[ii])
        Ai, bi, Si = fit_likelihood(E[ii], fu[ii], fv[ii])
        sl, ic, sq2, _ = fit_posterior(I[ii], E[ii])
        return mp, sp2, Ai, bi, Si, sl, ic, sq2

    full_params = (mu_p, sigma_p2, A, b, Sigma, slope, intercept, sigma_q2)
    heldout_params = fit_all(idx_fit)

    F_full, _, _ = compute_F(I[idx_eval], E[idx_eval], fu[idx_eval], fv[idx_eval], *full_params[2:5],
                              *full_params[5:8], full_params[0], full_params[1])
    F_held, _, _ = compute_F(I[idx_eval], E[idx_eval], fu[idx_eval], fv[idx_eval], *heldout_params[2:5],
                              *heldout_params[5:8], heldout_params[0], heldout_params[1])
    r_check = np.corrcoef(F_full, F_held)[0, 1]
    print(f"\nHeld-out-half validation:")
    print(f"  full-dataset-fit F vs. held-out-half-fit F on the other half: r={r_check:.6f}")


if __name__ == "__main__":
    main()
