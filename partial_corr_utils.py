"""
Shared utility for computing the I-E partial correlation, controlling
for blanket flux (Phi_u, Phi_v), in a way that is robust to the
near-total collinearity between those two variables.

BACKGROUND (found 2026-09-27, while checking referee Comment #3):
Phi_u and Phi_v are highly collinear in every dataset in this project
(r ~ -0.98 to -0.99, reflecting the reaction's stoichiometry -- v is
produced roughly in proportion to u consumed). The paper's headline
partial correlations condition on both jointly via ordinary least
squares. NOTE (updated 2026-10-09, pc1_details.py): on the current
merged-individuation datasets (single trajectory and both 1000-realization
ensembles) the joint result is NOT within the range spanned by Phi_u alone
and Phi_v alone (joint about -0.29 to -0.30; singles -0.18 to -0.21; PC1
-0.19 to -0.20), even though the standardized design matrix is well
conditioned (condition number ~11). So the joint estimate is a real, larger
number, not a rounding artifact; PC1 is used as the smaller, more
conservative estimate. The joint fit is additionally NOT reliably stable on smaller or
re-stratified subsamples (e.g. splitting by nearest-neighbor gap, or
using a different individuation method): there the near-singular design
matrix (condition number > 10^6) can fit a tiny, poorly-determined
residual direction that happens to correlate with I or E individually
without capturing any real shared structure, producing a partial
correlation OUTSIDE the range either flux variable alone would give --
in one case this produced an apparent "the blanket makes I and E MORE
dependent" result that turned out to be entirely this artifact.

pc1_partial_corr() avoids this by conditioning on a single combined
flux regressor (the leading principal component of standardized
(Phi_u, Phi_v), which captures ~99% of their joint variance) instead of
the two raw variables. Use this for any NEW partial-correlation check
on a subsample or alternative individuation method; the paper's
existing, already-verified full-dataset numbers do not need to be
recomputed with it, but any new one should at minimum be checked against
check_partial_corr_stability() before being trusted.
"""
import numpy as np


def _residualize(y, B):
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ coef


def pc1_partial_corr(I, E, fu, fv, return_direction=False):
    """Partial correlation of I and E, controlling for a single
    combined flux regressor (PC1 of standardized fu, fv)."""
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    Ig, Eg, fug, fvg = I[good], E[good], fu[good], fv[good]
    n = len(Ig)
    S = np.column_stack([fug, fvg])
    mean, std = S.mean(axis=0), S.std(axis=0)
    Sc = (S - mean) / std
    U, Sv, Vt = np.linalg.svd(Sc, full_matrices=False)
    explained = Sv[0] ** 2 / np.sum(Sv ** 2)
    p1 = U[:, 0] * Sv[0]
    B = np.column_stack([np.ones(n), p1])
    r = np.corrcoef(_residualize(Ig, B), _residualize(Eg, B))[0, 1]
    if return_direction:
        return r, n, explained, (mean, std, Vt[0])
    return r, n, explained


def check_partial_corr_stability(I, E, fu, fv):
    """Diagnostic: compares the standard two-variable joint partial
    correlation against conditioning on fu alone and fv alone. Returns
    (r_both, r_fu_only, r_fv_only, condition_number, is_suspect). If
    r_both falls outside [min(r_fu_only, r_fv_only), max(...)], that is
    the signature of the collinearity artifact described above, and
    pc1_partial_corr should be used instead for that dataset/subsample."""
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    Ig, Eg, fug, fvg = I[good], E[good], fu[good], fv[good]
    n = len(Ig)
    X = np.column_stack([np.ones(n), fug, fvg])
    cond = np.linalg.cond(X)
    r_fu = np.corrcoef(_residualize(Ig, np.column_stack([np.ones(n), fug])),
                        _residualize(Eg, np.column_stack([np.ones(n), fug])))[0, 1]
    r_fv = np.corrcoef(_residualize(Ig, np.column_stack([np.ones(n), fvg])),
                        _residualize(Eg, np.column_stack([np.ones(n), fvg])))[0, 1]
    r_both = np.corrcoef(_residualize(Ig, X), _residualize(Eg, X))[0, 1]
    lo, hi = sorted([r_fu, r_fv])
    is_suspect = r_both < lo - 1e-9 or r_both > hi + 1e-9
    return r_both, r_fu, r_fv, cond, is_suspect
