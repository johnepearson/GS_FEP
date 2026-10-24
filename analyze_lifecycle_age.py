"""
Does organism AGE confound the I-E partial-correlation-given-flux result
(main text §3.3.3)? Uses lifecycle_progress.pkl
(run_lifecycle.py --n-steps 2000 --fast), which records I, E, fu, fv
(via organism_interior_exterior_states_fast) for every
confirmed organism lineage, every SAMPLE_EVERY=5 time units, with I/E
computed every 4th sampled frame (IE_EVERY_N_FRAMES=4).

Builds one flat table of (lineage_id, t, age=t-birth, I, E, fu, fv)
from BOTH completed lifecycles (tr.examples: died or split) and
lineages still alive at the end of the run (tr.history, using
tr.birth_time) -- there is no reason to discard the latter, they are
equally valid measurements, just right-censored.

Tests age three ways:
  (a) age added as a THIRD covariate alongside the PC1(fu,fv) flux
      regressor -- does the partial correlation survive?
  (b) age-tercile stratification -- is the partial correlation stable
      across young / mid / old organisms, or does it only appear in
      one age band?
  (c) age vs I and age vs E marginally -- sanity check on whether
      organisms systematically change state as they age at all.

Bootstrap CIs (500 replicates, resampling lineages -- not
measurements -- to respect the within-lineage autocorrelation of
repeated measurements on the same organism over time).
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
import run_lifecycle  # noqa: E402 -- needed so pickle can resolve LifecycleTracker
# NB: only register the one name pickle needs to look up on `__main__`
# (LifecycleTracker was pickled with module='__main__' since
# run_lifecycle.py was run as a script). Do NOT bulk-update
# __main__'s __dict__ with vars(run_lifecycle) -- that clobbers this
# script's own __name__ and main(), silently turning `if __name__ ==
# "__main__": main()` below into a no-op.
sys.modules["__main__"].LifecycleTracker = run_lifecycle.LifecycleTracker

from partial_corr_utils import pc1_partial_corr, check_partial_corr_stability  # noqa: E402

PROGRESS_FILE = "lifecycle_progress.pkl"
N_BOOT = 500
RNG_SEED = 0


def load_tracker(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten(tracker):
    """One row per (lineage, sampled frame) with I, E both finite."""
    rows = []  # lineage_id, t, age, I, E, fu, fv

    for ex_idx, ex in enumerate(tracker.examples):
        birth = ex["birth"]
        lid_str = f"completed_{ex_idx}"
        for m in ex["history"]:
            rows.append((lid_str, m["t"], m["t"] - birth, m["I"], m["E"], m["fu"], m["fv"]))

    for lid, hist in tracker.history.items():
        birth = tracker.birth_time.get(lid, 0)
        lid_str = f"active_{lid}"
        for m in hist:
            rows.append((lid_str, m["t"], m["t"] - birth, m["I"], m["E"], m["fu"], m["fv"]))

    lineage_id = np.array([r[0] for r in rows], dtype=object)
    t = np.array([r[1] for r in rows], dtype=float)
    age = np.array([r[2] for r in rows], dtype=float)
    I = np.array([r[3] for r in rows], dtype=float)
    E = np.array([r[4] for r in rows], dtype=float)
    fu = np.array([r[5] for r in rows], dtype=float)
    fv = np.array([r[6] for r in rows], dtype=float)

    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv) & np.isfinite(age)
    return lineage_id[good], t[good], age[good], I[good], E[good], fu[good], fv[good]


def _residualize(y, B):
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ coef


def partial_corr_with_age(I, E, fu, fv, age):
    """Partial correlation of I, E controlling for PC1(fu, fv) AND age jointly."""
    n = len(I)
    S = np.column_stack([fu, fv])
    Sc = (S - S.mean(axis=0)) / S.std(axis=0)
    U, Sv, Vt = np.linalg.svd(Sc, full_matrices=False)
    p1 = U[:, 0] * Sv[0]
    B = np.column_stack([np.ones(n), p1, age])
    r = np.corrcoef(_residualize(I, B), _residualize(E, B))[0, 1]
    return r, n


def bootstrap_ci_by_lineage(fn, lineage_id, *arrays, n_boot=N_BOOT, seed=RNG_SEED):
    """Cluster bootstrap: resample whole lineages with replacement, not
    individual rows, since repeated measurements on the same organism
    are autocorrelated."""
    rng = np.random.default_rng(seed)
    unique_lids = np.unique(lineage_id)
    # map lineage -> row indices
    idx_of = {}
    for i, lid in enumerate(lineage_id):
        idx_of.setdefault(lid, []).append(i)
    idx_lists = [np.array(idx_of[lid]) for lid in unique_lids]

    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique_lids), size=len(unique_lids))
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


def main():
    print(f"loading {PROGRESS_FILE} ...")
    tr = load_tracker(PROGRESS_FILE)
    print(f"t={tr.t}  completed lifecycles={len(tr.examples)}  "
          f"active lineages={len(tr.history)}  pending episodes={len(tr.episodes)}")

    lineage_id, t, age, I, E, fu, fv = flatten(tr)
    n_lineages = len(np.unique(lineage_id))
    print(f"\nflattened: N={len(I)} measurements with finite I,E,fu,fv "
          f"across {n_lineages} lineages")
    print(f"age range: [{age.min():.0f}, {age.max():.0f}], "
          f"mean={age.mean():.1f}, median={np.median(age):.1f}")

    # --- (0) baseline: same-dataset partial corr, no age, for reference ---
    r0, n0, expl0 = pc1_partial_corr(I, E, fu, fv)
    r_both, r_fu, r_fv, cond, suspect = check_partial_corr_stability(I, E, fu, fv)
    print(f"\n[baseline] partial corr I,E | flux (PC1), THIS dataset: "
          f"r={r0:.4f}  N={n0}  (PC1 explains {expl0*100:.1f}% of flux variance)")
    print(f"  stability check: r_both(OLS)={r_both:.4f}  r_fu_only={r_fu:.4f}  "
          f"r_fv_only={r_fv:.4f}  cond={cond:.1e}  suspect={suspect}")
    lo0, hi0, _ = bootstrap_ci_by_lineage(
        lambda I_, E_, fu_, fv_: pc1_partial_corr(I_, E_, fu_, fv_)[0],
        lineage_id, I, E, fu, fv)
    print(f"  95% CI (lineage-cluster bootstrap, N_boot={N_BOOT}): [{lo0:.4f}, {hi0:.4f}]")

    # --- (a) age as a third covariate ---
    r_age, n_age = partial_corr_with_age(I, E, fu, fv, age)
    print(f"\n[a] partial corr I,E | flux(PC1) + age: r={r_age:.4f}  N={n_age}")
    lo_a, hi_a, _ = bootstrap_ci_by_lineage(
        lambda I_, E_, fu_, fv_, age_: partial_corr_with_age(I_, E_, fu_, fv_, age_)[0],
        lineage_id, I, E, fu, fv, age)
    print(f"  95% CI: [{lo_a:.4f}, {hi_a:.4f}]")
    shrink = (1 - r_age / r0) * 100
    print(f"  vs baseline r={r0:.4f}: {shrink:+.1f}% change in |r| from adding age")

    # --- (b) age-tercile stratification ---
    print(f"\n[b] age-tercile stratification (partial corr | flux(PC1), within each band):")
    edges = np.quantile(age, [0, 1/3, 2/3, 1.0])
    labels = ["young", "mid", "old"]
    for lo_e, hi_e, name in zip(edges[:-1], edges[1:], labels):
        sel = (age >= lo_e) & (age <= hi_e) if name == "old" else (age >= lo_e) & (age < hi_e)
        if sel.sum() < 30:
            print(f"  {name:5s} [{lo_e:.0f},{hi_e:.0f}]: N={sel.sum()} too small, skipped")
            continue
        r_s, n_s, _ = pc1_partial_corr(I[sel], E[sel], fu[sel], fv[sel])
        lo_s, hi_s, _ = bootstrap_ci_by_lineage(
            lambda I_, E_, fu_, fv_: pc1_partial_corr(I_, E_, fu_, fv_)[0],
            lineage_id[sel], I[sel], E[sel], fu[sel], fv[sel])
        print(f"  {name:5s} age[{lo_e:6.0f},{hi_e:6.0f}]: N={n_s:6d}  r={r_s:.4f}  "
              f"95% CI=[{lo_s:.4f}, {hi_s:.4f}]")

    # --- (c) marginal age effects, sanity check ---
    r_age_I = np.corrcoef(age, I)[0, 1]
    r_age_E = np.corrcoef(age, E)[0, 1]
    print(f"\n[c] marginal correlations with age (sanity check): "
          f"r(age,I)={r_age_I:.4f}  r(age,E)={r_age_E:.4f}")

    np.savez("lifecycle_age_analysis.npz",
             lineage_id_hash=np.array([hash(x) for x in lineage_id]),
             t=t, age=age, I=I, E=E, fu=fu, fv=fv,
             r_baseline=r0, r_with_age=r_age)
    print("\nsaved flat arrays to lifecycle_age_analysis.npz")


if __name__ == "__main__":
    main()
