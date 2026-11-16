"""
Direct response to a sharp, correctly-argued reviewer concern (2026-09-29
review): the unscaled Frenzel-Pompe/KSG CMI estimator (conditional_mi.py's
cmi_ksg) does its nearest-neighbor search with Chebyshev (max-norm)
distance over the raw, concatenated (X, Y, Z) coordinates. Blanket flux
(fu, fv) sits around 1e-4 in magnitude here, while I, E, and boundary
concentration (u_bdry, v_bdry) sit around 0.1 -- two to three orders of
magnitude larger. Reading cmi_ksg's own implementation confirms the
mechanism the reviewer describes: it finds the distance eps to the k-th
neighbor in the FULL (X,Y,Z) joint space (which, under max-norm with Z's
tiny range, is essentially always set by the X or Y coordinates, never
by Z), then counts neighbors within that SAME eps in the marginal (X,Z),
(Y,Z), (Z) subspaces. If eps is orders of magnitude larger than Z's
entire range, essentially every point already lies within eps of every
other point along the Z axis, so the Z-marginal counts stop depending on
Z at all -- the estimator is not really conditioning on B, and CMI(X;Y|Z)
collapses toward the unconditioned CMI(X;Y). The paper's own evidence is
consistent with this: for isolated organisms, "CMI given flux" reported
in Table tab:isolated_nonlinear matches the unconditioned I(I;E) to three
significant figures, while conditioning on boundary concentration (on
the same 0.1-ish scale as I, E) does change the estimate.

This script provides the two checks the review calls for, extended to
BOTH the touching and isolated regimes (the review's Table 5 only
covered isolated; the touching subset's nonlinear CMI under flux has
never been run at all, scaled or not -- relevant to the abstract's claim
about the contact regime):

  1. QUICK SANITY CHECK: run the unscaled estimator on (I, E, flux) and
     on (I, E, flux*1e4), same data, only the conditioning variable's
     units changed. If the CMI value moves substantially between the
     two, that confirms the estimator is scale-sensitive to the point of
     not really conditioning on flux at its natural scale.
  2. FULL RERUN: z-score I, E, and the flux pair jointly (matching the
     paper's existing sec:limitations scaling-sensitivity convention),
     recompute observed CMI and the 30-cluster conditional-permutation
     null, for BOTH touching and isolated regimes. This supersedes the
     population-level-only z-scoring check already in the paper and
     produces the actual numbers Table tab:isolated_nonlinear needs.

BOUNDARY CONCENTRATION (u_bdry, v_bdry): this script does NOT include
that candidate blanket variable, because it isn't in
positive_gap_progress_v2.npz (I, E, fu, fv, edge_gap only) -- the
original isolated-subset boundary-concentration check used a separate
data recording pass not available in this sandbox. If you still have
that file (or can point me to the script that produced it), tell me its
name/path and I'll extend this script to rerun the (u_bdry, v_bdry) rows
of Table 5 the same way; the flux checks below don't depend on it.

Usage:
    python run_cmi_scale_check_regimes.py
"""
import time

import numpy as np
from sklearn.cluster import KMeans

from conditional_mi import cmi_ksg

DATA_FILE = "positive_gap_progress_v2.npz"
N_SUB = 8000
N_CLUSTERS = 30
N_NULL = 300
SEED = 0
SCALE_FACTOR = 1e4  # for the quick sanity check


def load(path):
    d = np.load(path)
    I, E, fu, fv, edge_gap = d["I"], d["E"], d["fu"], d["fv"], d["edge_gap"]
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv) & np.isfinite(edge_gap)
    return I[good], E[good], fu[good], fv[good], edge_gap[good]


def subsample(rng, n_sub, *arrays):
    n = len(arrays[0])
    if n <= n_sub:
        return arrays
    idx = rng.choice(n, n_sub, replace=False)
    return tuple(a[idx] for a in arrays)


def conditional_null(I, E, B, n_clusters=N_CLUSTERS, n_null=N_NULL, seed=SEED, label=""):
    """Same 30-cluster conditional-permutation null construction as
    run_cmi_conditional_null.py / the paper's Section 3.3."""
    observed = cmi_ksg(I, E, B, k=5)

    B_std = (B - B.mean(axis=0)) / B.std(axis=0)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(B_std)
    cluster_labels = km.labels_

    null_vals = np.empty(n_null)
    t0 = time.time()
    for trial in range(n_null):
        rng2 = np.random.default_rng(1000 + trial)
        E_shuffled = E.copy()
        for c in range(n_clusters):
            members = np.where(cluster_labels == c)[0]
            if len(members) > 1:
                E_shuffled[members] = E[members[rng2.permutation(len(members))]]
        null_vals[trial] = cmi_ksg(I, E_shuffled, B, k=5)
        if (trial + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    [{label}] null trial {trial+1}/{n_null}  "
                  f"({elapsed:.0f}s elapsed, ~{elapsed/(trial+1):.2f}s/trial)")

    null_mean, null_std = null_vals.mean(), null_vals.std()
    excess = observed - null_mean
    n_sd = excess / null_std if null_std > 0 else np.nan
    return observed, null_mean, null_std, excess, n_sd


def report_regime(name, I, E, fu, fv):
    n = len(I)
    print(f"\n{'='*70}\n{name}: N={n} (subsample size {min(n, N_SUB)})\n{'='*70}")

    rng = np.random.default_rng(SEED)
    Is, Es, fus, fvs = subsample(rng, N_SUB, I, E, fu, fv)
    B = np.column_stack([fus, fvs])

    # unconditioned reference: condition on a constant column, which
    # collapses cmi_ksg to (approximately) the unconditioned MI, for
    # comparison against the paper's existing reported unconditioned
    # figure on a related subsample (Table tab:isolated_nonlinear:
    # 0.100 nats).
    unconditioned = cmi_ksg(Is, Es, np.zeros((len(Is), 1)), k=5)
    print(f"unconditioned I(I;E) on this subsample = {unconditioned:.4f} nats "
          f"(cf. paper's reported 0.100 nats for isolated organisms elsewhere)")

    # --- (1) quick sanity check: does scaling flux move the estimate? ---
    print(f"\n[1] quick sanity check: unscaled flux vs flux*{SCALE_FACTOR:.0e}")
    cmi_raw = cmi_ksg(Is, Es, B, k=5)
    cmi_scaled_input = cmi_ksg(Is, Es, B * SCALE_FACTOR, k=5)
    print(f"    CMI(I,E | flux)                  = {cmi_raw:.4f} nats")
    print(f"    CMI(I,E | flux*{SCALE_FACTOR:.0e})            = {cmi_scaled_input:.4f} nats")
    moved = abs(cmi_scaled_input - cmi_raw)
    print(f"    difference = {moved:.4f} nats "
          f"({'MOVES SUBSTANTIALLY -- scale artifact confirmed' if moved > 0.01 else 'small'})")

    # --- (2) full rerun: z-scored I, E, flux, with 30-cluster null ---
    print(f"\n[2] full rerun, z-scored inputs, 30-cluster conditional-permutation null "
          f"({N_NULL} trials)...")
    Is_z = (Is - Is.mean()) / Is.std()
    Es_z = (Es - Es.mean()) / Es.std()
    B_z = (B - B.mean(axis=0)) / B.std(axis=0)
    observed, null_mean, null_std, excess, n_sd = conditional_null(
        Is_z, Es_z, B_z, label=name)
    print(f"\n    observed CMI (z-scored)   = {observed:.4f} nats")
    print(f"    null mean                = {null_mean:.4f} nats")
    print(f"    null std                 = {null_std:.4f} nats")
    print(f"    null-subtracted excess   = {excess:.4f} nats")
    print(f"    null-standard-deviations = {n_sd:.1f}")

    return dict(observed=observed, null_mean=null_mean, null_std=null_std,
                excess=excess, n_sd=n_sd, cmi_raw=cmi_raw, cmi_scaled_input=cmi_scaled_input)


def main():
    print(f"loading {DATA_FILE} ...")
    I, E, fu, fv, edge_gap = load(DATA_FILE)
    touching = edge_gap < 0
    isolated = edge_gap > 0
    print(f"N={len(I)}  touching={touching.sum()}  isolated={isolated.sum()}")

    res_touch = report_regime("TOUCHING (edge_gap<0) -- flux blanket, z-scored",
                               I[touching], E[touching], fu[touching], fv[touching])
    res_iso = report_regime("ISOLATED (edge_gap>0) -- flux blanket, z-scored",
                             I[isolated], E[isolated], fu[isolated], fv[isolated])

    print(f"\n{'='*70}\nsummary\n{'='*70}")
    print(f"  touching: unscaled CMI(I,E|flux)={res_touch['cmi_raw']:.4f}  "
          f"z-scored excess={res_touch['excess']:.4f} nats ({res_touch['n_sd']:.1f} null-SDs)")
    print(f"  isolated: unscaled CMI(I,E|flux)={res_iso['cmi_raw']:.4f}  "
          f"z-scored excess={res_iso['excess']:.4f} nats ({res_iso['n_sd']:.1f} null-SDs)")
    print("\nNOTE: boundary-concentration (u_bdry, v_bdry) blanket rows are NOT included "
          "here -- see the script's docstring. Send the data file/script used for the "
          "original isolated-subset boundary-concentration check and this can be extended "
          "to cover it, for both regimes, the same way.")

    np.savez("cmi_scale_check_regimes_result.npz",
             **{f"touch_{k}": v for k, v in res_touch.items()},
             **{f"iso_{k}": v for k, v in res_iso.items()})
    print("\nsaved cmi_scale_check_regimes_result.npz")


if __name__ == "__main__":
    main()
