"""
Fits the Friston free-energy generative model (main text Section 5.2) from an independent-ensemble dataset
(I, E, fu, fv) and computes the closed-form KL and accuracy terms.

  Hidden state s := E
  Observation  o := (Phi_u, Phi_v)
  Prior        p(s) = N(mu_p, sigma_p^2), fit to the marginal distribution of E across the ensemble population
  Likelihood   p(o|s) = N(A*s + b, Sigma), fit by linear regression of (fu, fv) on E
  Posterior    q(s|I) = N(mu_q(I), sigma_q^2), with mu_q(I) a linear regression of E on I and sigma_q^2 the
               residual variance

KL(q||p) and the expected negative log-likelihood are both available in closed form for Gaussians.

It also runs the held-out-half validation: fit all three ingredients on a random half of the ensemble, evaluate F on the
other half with both the held-out-fit and the full-dataset-fit parameters, and correlate the two.

fit_prior, fit_likelihood, fit_posterior and compute_F are also imported by rebuild_splithalf_check.py.

Usage (from the folder holding ensemble_merged_seed<BASE>_worker*.npz):
    python3 run_friston_fe_fit.py                      # seed base 1000
    python3 run_friston_fe_fit.py --seed-base 9000
"""
import argparse
import glob
import numpy as np


def fit_prior(E):
    return E.mean(), E.var(ddof=1)


def fit_likelihood(E, fu, fv):
    """Linear regression of (fu, fv) on E: o = A*s + b + noise."""
    n = len(E)
    X = np.column_stack([np.ones(n), E])
    O = np.column_stack([fu, fv])
    coef, *_ = np.linalg.lstsq(X, O, rcond=None)  # shape (2, 2): [intercept; slope] x [fu, fv]
    b = coef[0, :]
    A = coef[1, :]
    resid = O - X @ coef
    Sigma = np.cov(resid.T, ddof=1)
    return A, b, Sigma


def fit_posterior(I, E):
    """Linear regression of E on I: mu_q(I) = slope*I + intercept."""
    n = len(I)
    X = np.column_stack([np.ones(n), I])
    coef, *_ = np.linalg.lstsq(X, E, rcond=None)
    intercept, slope = coef
    resid = E - X @ coef
    sigma_q2 = np.var(resid, ddof=1)
    r = np.corrcoef(I, E)[0, 1]
    return slope, intercept, sigma_q2, r


def kl_term(mu_q, sigma_q2, mu_p, sigma_p2):
    sigma_q, sigma_p = np.sqrt(sigma_q2), np.sqrt(sigma_p2)
    return np.log(sigma_p / sigma_q) + (sigma_q2 + (mu_q - mu_p) ** 2) / (2 * sigma_p2) - 0.5


def accuracy_term(o, mu_q, sigma_q2, A, b, Sigma):
    """-E_q[log p(o|s)], vectorized over instances. o: (N,2), mu_q: (N,)."""
    Sigma_inv = np.linalg.inv(Sigma)
    sign, logdet = np.linalg.slogdet(Sigma)
    const = 0.5 * (2 * np.log(2 * np.pi) + logdet)
    pred = np.outer(mu_q, A) + b  # (N, 2)
    resid = o - pred
    quad = np.einsum('ni,ij,nj->n', resid, Sigma_inv, resid)
    extra = sigma_q2 * (A @ Sigma_inv @ A)
    return const + 0.5 * (quad + extra)


def compute_F(I, E, fu, fv, A, b, Sigma, slope, intercept, sigma_q2, mu_p, sigma_p2):
    mu_q = slope * I + intercept
    KL = kl_term(mu_q, sigma_q2, mu_p, sigma_p2)
    o = np.column_stack([fu, fv])
    acc = accuracy_term(o, mu_q, sigma_q2, A, b, Sigma)
    F = KL + acc
    return F, KL, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", type=int, default=1000, help="reads ensemble_merged_seed<BASE>_worker*.npz")
    ap.add_argument("--seed", type=int, default=0, help="random seed for the held-out split")
    args = ap.parse_args()

    files = sorted(glob.glob(f"ensemble_merged_seed{args.seed_base}_worker*.npz"))
    if not files:
        raise SystemExit(f"no files match ensemble_merged_seed{args.seed_base}_worker*.npz")
    ds = [np.load(f) for f in files]
    I_all, E_all, fu_all, fv_all = (np.concatenate([d[k] for d in ds]) for k in ("I", "E", "fu", "fv"))
    print(f"{len(files)} files")
    good = np.isfinite(I_all) & np.isfinite(E_all) & np.isfinite(fu_all) & np.isfinite(fv_all)
    I, E, fu, fv = I_all[good], E_all[good], fu_all[good], fv_all[good]
    N = len(I)
    print(f"N={N} organism-instances")

    mu_p, sigma_p2 = fit_prior(E)
    A, b, Sigma = fit_likelihood(E, fu, fv)
    slope, intercept, sigma_q2, r_marg = fit_posterior(I, E)

    print(f"\nFitted generative model:")
    print(f"  prior:      mu_p={mu_p:.4f}  sigma_p^2={sigma_p2:.4e}")
    print(f"  likelihood: A=({A[0]:.4e}, {A[1]:.4e})  b=({b[0]:.4e}, {b[1]:.4e})")
    print(f"              Sigma=\n{Sigma}")
    print(f"  posterior:  slope={slope:.4f}  intercept={intercept:.4f}  sigma_q^2={sigma_q2:.4e}")
    print(f"              (marginal corr(I,E) = {r_marg:.4f})")

    F, KL, acc = compute_F(I, E, fu, fv, A, b, Sigma, slope, intercept, sigma_q2, mu_p, sigma_p2)
    print(f"\nKL term (interior-informed divergence from population prior):")
    print(f"  mean={KL.mean():.4f}  min={KL.min():.4f}  max={KL.max():.4f}")
    print(f"\nFull F = KL + accuracy:")
    print(f"  mean={F.mean():.4f}  min={F.min():.4f}  max={F.max():.4f}")
    print(f"  (accuracy term mean={acc.mean():.4f} -- dominated by an arbitrary additive")
    print(f"   constant from flux's physical units, per the paper's own caveat; only")
    print(f"   relative changes in F are meaningful, not this absolute value)")

    # Held-out-half validation, matching the paper's own check exactly
    rng = np.random.default_rng(args.seed)
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
