"""
Follow-up to the "are touching organisms recently-divided siblings?"
check (analyze_touching_vs_age.py -- answer: no, not predominantly).
That rules out the easy explanation for why flux screens I-E dependence
almost completely for touching organisms (Table tab:gap_split: 99.7%
screened touching vs. 44.3% isolated). The remaining question, stated
directly by the user rather than either review: is that near-complete
screening because flux genuinely is close to the *only* interaction
channel when organisms touch (the paper's current informal claim, sec
5.2/sec:mechanism_scope), or because flux becomes nearly REDUNDANT with
(I,E) once organisms are pressed together -- i.e. flux is close to a
deterministic function of I and E for touching organisms specifically,
which would make "controlling for flux" mechanically similar to
"controlling for (I,E) itself" and produce near-complete screening as a
near-tautology rather than as evidence about a real information
channel?

This is directly checkable: if flux (fu, fv) is highly predictable from
(I, E) for touching organisms but not for isolated ones, that is the
redundancy/near-tautology signature. If flux is comparably (weakly)
predictable from (I, E) in both regimes, the geometric-redundancy
explanation does not hold up, and the "flux is the dominant channel of
a real interaction" reading is on firmer ground.

Uses positive_gap_progress_v2.npz (same file, same touching/isolated
split by sign of edge_gap, as Table tab:gap_split in sec:mechanism_scope
-- the pc1_partial_corr sanity check below should reproduce that
table's numbers closely as a check we're using the right data/split).

Reports, for touching (edge_gap<0) and isolated (edge_gap>0) separately:
  1. Sanity check: r_marginal(I,E) and r_partial(I,E|flux PC1), to
     confirm this reproduces Table tab:gap_split.
  2. R^2 of I ~ (fu,fv) jointly, and E ~ (fu,fv) jointly (how well flux
     predicts each side of the blanket).
  3. R^2 of fu ~ (I,E) jointly, and fv ~ (I,E) jointly (the more
     relevant direction for the redundancy question: how well I,E
     jointly predict flux).
  4. First canonical correlation between (fu,fv) and (I,E) -- a single
     number summarizing the maximum possible linear redundancy between
     the two 2-variable sets, invariant to which side you regress on.
     Ridge-regularized (fu,fv are collinear, r~-0.98) for stability.
All with frame-cluster bootstrap CIs (resampling whole frames, matching
this section's existing convention, since organism-instances within a
frame are not independent).

Usage:
    python run_flux_redundancy_check.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from partial_corr_utils import pc1_partial_corr  # noqa: E402

DATA_FILE = "positive_gap_progress_v2.npz"
N_BOOT = 500
RNG_SEED = 0
RIDGE = 1e-6  # relative ridge regularization for the canonical-correlation
              # covariance inversion, given fu/fv's near-total collinearity


def load(path):
    d = np.load(path)
    I, E, fu, fv, edge_gap, frame_id = (
        d["I"], d["E"], d["fu"], d["fv"], d["edge_gap"], d["frame_id"])
    good = (np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
            & np.isfinite(edge_gap))
    return I[good], E[good], fu[good], fv[good], edge_gap[good], frame_id[good]


def r2_multivariate(y, X):
    """R^2 of OLS regression of y on X (X: N x k, no intercept column needed)."""
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    yhat = Xc @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def first_canonical_correlation(X, Y, ridge=RIDGE):
    """First canonical correlation between column-sets X and Y (each N x 2
    here), ridge-regularized for stability given X's near-collinearity.
    Returns the largest correlation achievable between any linear
    combination of X's columns and any linear combination of Y's columns."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    n = len(Xc)
    Sxx = (Xc.T @ Xc) / n
    Syy = (Yc.T @ Yc) / n
    Sxy = (Xc.T @ Yc) / n
    Sxx = Sxx + ridge * np.trace(Sxx) / Sxx.shape[0] * np.eye(Sxx.shape[0])
    Syy = Syy + ridge * np.trace(Syy) / Syy.shape[0] * np.eye(Syy.shape[0])
    Sxx_inv = np.linalg.inv(Sxx)
    Syy_inv = np.linalg.inv(Syy)
    M = Sxx_inv @ Sxy @ Syy_inv @ Sxy.T
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.clip(eigvals, 0, 1)  # guard tiny negative numerical noise
    return np.sqrt(eigvals.max())


def bootstrap_ci_by_frame(fn, frame_id, *arrays, n_boot=N_BOOT, seed=RNG_SEED):
    """Cluster bootstrap: resample whole frames with replacement, matching
    this section's existing frame-level-clustering convention."""
    rng = np.random.default_rng(seed)
    unique_frames = np.unique(frame_id)
    idx_of = {}
    for i, f in enumerate(frame_id):
        idx_of.setdefault(f, []).append(i)
    idx_lists = [np.array(idx_of[f]) for f in unique_frames]

    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique_frames), size=len(unique_frames))
        rows = np.concatenate([idx_lists[p] for p in pick])
        boot_arrays = [a[rows] for a in arrays]
        try:
            r = fn(*boot_arrays)
            if isinstance(r, tuple):
                r = r[0]
            vals.append(r)
        except Exception:
            continue
    vals = np.array(vals)
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5), vals


def report_regime(name, I, E, fu, fv, frame_id):
    n = len(I)
    print(f"\n=== {name}: N={n} ===")

    # (1) sanity check vs. Table tab:gap_split
    r_marg = np.corrcoef(I, E)[0, 1]
    r_part, n_eff, expl = pc1_partial_corr(I, E, fu, fv)
    screened = 1 - (r_part / r_marg) ** 2
    print(f"  [1] sanity check: r_marginal={r_marg:.3f}  r_partial(PC1)={r_part:.3f}  "
          f"screened={screened*100:.1f}%")

    # (2) R^2 of I, E ~ flux jointly
    r2_I_flux = r2_multivariate(I, np.column_stack([fu, fv]))
    r2_E_flux = r2_multivariate(E, np.column_stack([fu, fv]))
    lo, hi, _ = bootstrap_ci_by_frame(
        lambda I_, fu_, fv_: r2_multivariate(I_, np.column_stack([fu_, fv_])),
        frame_id, I, fu, fv)
    print(f"  [2] R^2(I ~ fu,fv)  = {r2_I_flux:.3f}  95% CI=[{lo:.3f},{hi:.3f}]")
    lo, hi, _ = bootstrap_ci_by_frame(
        lambda E_, fu_, fv_: r2_multivariate(E_, np.column_stack([fu_, fv_])),
        frame_id, E, fu, fv)
    print(f"      R^2(E ~ fu,fv)  = {r2_E_flux:.3f}  95% CI=[{lo:.3f},{hi:.3f}]")

    # (3) R^2 of flux ~ I, E jointly (the redundancy-relevant direction)
    r2_fu_IE = r2_multivariate(fu, np.column_stack([I, E]))
    r2_fv_IE = r2_multivariate(fv, np.column_stack([I, E]))
    lo, hi, _ = bootstrap_ci_by_frame(
        lambda fu_, I_, E_: r2_multivariate(fu_, np.column_stack([I_, E_])),
        frame_id, fu, I, E)
    print(f"  [3] R^2(fu ~ I,E)   = {r2_fu_IE:.3f}  95% CI=[{lo:.3f},{hi:.3f}]")
    lo, hi, _ = bootstrap_ci_by_frame(
        lambda fv_, I_, E_: r2_multivariate(fv_, np.column_stack([I_, E_])),
        frame_id, fv, I, E)
    print(f"      R^2(fv ~ I,E)   = {r2_fv_IE:.3f}  95% CI=[{lo:.3f},{hi:.3f}]")

    # (4) first canonical correlation between (fu,fv) and (I,E)
    cc1 = first_canonical_correlation(np.column_stack([fu, fv]), np.column_stack([I, E]))
    lo, hi, _ = bootstrap_ci_by_frame(
        lambda fu_, fv_, I_, E_: first_canonical_correlation(
            np.column_stack([fu_, fv_]), np.column_stack([I_, E_])),
        frame_id, fu, fv, I, E)
    print(f"  [4] first canonical correlation (fu,fv) vs (I,E) = {cc1:.3f}  "
          f"95% CI=[{lo:.3f},{hi:.3f}]")

    return dict(r_marginal=r_marg, r_partial=r_part, screened=screened,
                r2_I_flux=r2_I_flux, r2_E_flux=r2_E_flux,
                r2_fu_IE=r2_fu_IE, r2_fv_IE=r2_fv_IE, cc1=cc1, n=n)


def main():
    print(f"loading {DATA_FILE} ...")
    I, E, fu, fv, edge_gap, frame_id = load(DATA_FILE)
    print(f"N={len(I)} organism-instances")

    touching = edge_gap < 0
    isolated = edge_gap > 0
    print(f"touching (edge_gap<0): N={touching.sum()} ({touching.mean()*100:.1f}%)")
    print(f"isolated (edge_gap>0): N={isolated.sum()} ({isolated.mean()*100:.1f}%)")

    res_touch = report_regime("TOUCHING (edge_gap<0)",
                               I[touching], E[touching], fu[touching], fv[touching],
                               frame_id[touching])
    res_iso = report_regime("ISOLATED (edge_gap>0)",
                             I[isolated], E[isolated], fu[isolated], fv[isolated],
                             frame_id[isolated])

    print("\n=== summary: touching vs isolated ===")
    print(f"  screened fraction:              touching={res_touch['screened']*100:5.1f}%   "
          f"isolated={res_iso['screened']*100:5.1f}%")
    print(f"  R^2(flux ~ I,E) [max of fu,fv]: touching={max(res_touch['r2_fu_IE'],res_touch['r2_fv_IE']):5.3f}   "
          f"isolated={max(res_iso['r2_fu_IE'],res_iso['r2_fv_IE']):5.3f}")
    print(f"  first canonical correlation:    touching={res_touch['cc1']:5.3f}   "
          f"isolated={res_iso['cc1']:5.3f}")
    print("\nIf the canonical correlation / R^2(flux~I,E) is comparably modest in both "
          "regimes despite the screened fraction being far higher for touching, that "
          "argues AGAINST the geometric-redundancy/near-tautology explanation, and FOR "
          "flux genuinely capturing most of the real interaction when organisms touch. "
          "If it is much higher for touching specifically, that supports the redundancy "
          "reading instead: near-complete screening there would be partly mechanical.")

    np.savez("flux_redundancy_result.npz",
             **{f"touch_{k}": v for k, v in res_touch.items()},
             **{f"iso_{k}": v for k, v in res_iso.items()})
    print("\nsaved flux_redundancy_result.npz")


if __name__ == "__main__":
    main()
