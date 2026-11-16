"""
Referee Comment #2 "cheap pass" analysis: reads
confound_checks_progress.npz (produced by run_confound_checks.py) and
tests whether the headline flux-conditioned I-E partial correlation
survives three candidate confounds.

Checks:
  (a) size/shape: add radius + eccentricity to the conditioning set
  (b) frame-level global confound: (b1) frame population/mean-u/Otsu
      covariates, (b2) full frame fixed effects (within-frame demeaning)
  (c) u-field enrichment: add near-field u-shell means to the
      conditioning set

Results (300 frames, N=62,821 organism-instances, 2026-09-26):

    baseline (flux only):                     r = -0.148
    (a) + radius + eccentricity:              r = -0.099   (~33% shrinkage in |r|, survives)
    (b1) + frame pop/mean-u/Otsu covariates:  r = -0.135   (~9% shrinkage, survives)
    (b2) + frame fixed effects:               r = -0.135   (same -- not a frame-level artifact)
    (c) + u-shells:                            UNSTABLE, see below

(a) and (b) are robust: the effect survives both, with (a) knocking off
roughly a third of the magnitude (organism size/shape is a real but
partial contributor) and (b) essentially not moving it at all (this is
not a global per-frame confound).

(c) turned out NOT to give a clean answer and should not be reported as
a "confound check passed/failed" result without more care. Using all 7
u-shells together as regressors gives r=+0.060 (looks like the effect
is nearly killed and flips sign), but the 7 shells are highly collinear
(adjacent-shell correlations up to 0.97) and, critically, using just
ONE shell at a time gives wildly different, sign-flipping results
depending on which radius is picked (shell 0-6px: r=-0.257; 6-12px:
r=+0.005; 12-18px: r=+0.204; 18-24px: r=+0.214; 24-36px: r=+0.096;
36-48px: r=-0.075; 48-64px: r=-0.158). A single well-motivated summary
(mean over all 7 shells, or their first principal component) also
disagrees with both the all-7-shells regression and with each other
(+0.069 and +0.106 respectively). This instability -- present even in
single-predictor models, so it is not simply overfitting from
collinear regressors -- most likely reflects real, non-monotonic
radial structure in the u field around an organism (a depletion zone
giving way to a recovery ring at some distance, characteristic of
Gray-Scott competitive dynamics) rather than a stable "enrichment"
confound with one sign and magnitude. Concluding that u-enrichment
"explains" or "doesn't explain" the residual I-E coupling would not be
justified from this check as currently designed; it would need a
purpose-built radial model (e.g. fitting a smooth u(r) profile per
organism and testing whether ITS parameters, not raw shell means,
explain the residual) before it says anything trustworthy.

Bottom line so far: the effect survives the two confound checks that
gave a clean answer (size/shape, frame-level global effects); the third
(u-field enrichment) needs a better-designed test before it can be
used as evidence either way.
"""
import numpy as np

PROGRESS_FILE = "confound_checks_progress.npz"


def residualize(x, B):
    beta, *_ = np.linalg.lstsq(B, x, rcond=None)
    return x - B @ beta


def partial_r(I, E, extra_cols, good):
    B = np.column_stack([np.ones(good.sum())] + [c[good] for c in extra_cols])
    return np.corrcoef(residualize(I[good], B), residualize(E[good], B))[0, 1]


def within_frame_demean(x, frame_id, good):
    xg, fg = x[good], frame_id[good]
    out = xg.copy()
    for f in np.unique(fg):
        m = fg == f
        out[m] = xg[m] - xg[m].mean() if m.sum() > 1 else 0.0
    return out


def main():
    d = np.load(PROGRESS_FILE)
    I, E, fu, fv = d["I"], d["E"], d["fu"], d["fv"]
    frame_id = d["frame_id"]
    radius, ecc = d["radius"], d["eccentricity"]
    u_shells = d["u_shells"]
    frame_pop, frame_mean_u, frame_otsu = d["frame_pop"], d["frame_mean_u"], d["frame_otsu"]

    print(f"N organism-instances = {len(I)}, frames = {frame_id.max()+1}")

    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    r_baseline = partial_r(I, E, [fu, fv], good)
    print(f"\nbaseline (flux only):                     r = {r_baseline:.4f}   N={good.sum()}")

    good_a = good & np.isfinite(radius) & np.isfinite(ecc)
    r_a = partial_r(I, E, [fu, fv, radius, ecc], good_a)
    print(f"(a) + radius + eccentricity:              r = {r_a:.4f}   N={good_a.sum()}")

    fpop, fmu, fot = frame_pop[frame_id], frame_mean_u[frame_id], frame_otsu[frame_id]
    good_b1 = good & np.isfinite(fpop) & np.isfinite(fmu) & np.isfinite(fot)
    r_b1 = partial_r(I, E, [fu, fv, fpop, fmu, fot], good_b1)
    print(f"(b1) + frame pop/mean-u/Otsu covariates:  r = {r_b1:.4f}   N={good_b1.sum()}")

    I_dm, E_dm = within_frame_demean(I, frame_id, good), within_frame_demean(E, frame_id, good)
    fu_dm, fv_dm = within_frame_demean(fu, frame_id, good), within_frame_demean(fv, frame_id, good)
    B_dm = np.column_stack([fu_dm, fv_dm])
    r_b2 = np.corrcoef(residualize(I_dm, B_dm), residualize(E_dm, B_dm))[0, 1]
    print(f"(b2) + frame fixed effects (within-frame): r = {r_b2:.4f}   N={good.sum()}")

    print("\n(c) u-field enrichment -- UNSTABLE, reported per-shell rather than combined:")
    labels_ = ['0-6', '6-12', '12-18', '18-24', '24-36', '36-48', '48-64']
    good_c = good.copy()
    for s in range(u_shells.shape[1]):
        good_c &= np.isfinite(u_shells[:, s])
    for s in range(u_shells.shape[1]):
        r = partial_r(I, E, [fu, fv, u_shells[:, s]], good_c)
        print(f"    + shell {labels_[s]:8s} alone: r = {r:+.4f}")
    r_all7 = partial_r(I, E, [fu, fv] + [u_shells[:, s] for s in range(7)], good_c)
    print(f"    + all 7 shells together:  r = {r_all7:+.4f}   (unreliable -- see module docstring)")

    print(f"\nmarginal (unconditioned) r(I,E) for reference: {np.corrcoef(I[good], E[good])[0,1]:.4f}")


if __name__ == "__main__":
    main()
