"""
Analyses computed from the master single-trajectory dataset (run_master_trajectory.py), no simulation.

    python analyze_master.py                       # master_trajectory_progress.npz
    python analyze_master.py --data other.npz --n-boot 2000

Reports (all on the merged, min_distance=24 segmentation):
  1. Headline partial correlation r(I,E | flux_u, flux_v): joint OLS and the single-PC1 regressor,
     Fisher 95% CI and a frame-level cluster bootstrap.
  2. Flux asymmetry: fraction of organisms with U flux inward / V flux outward.
  3. Same partial correlation with the whole-mask-erosion interior/exterior definition.
  4. Clean (merge_size == 1) vs crowded (merge_size > 1): marginal, PC1 and joint partial correlations.
  5. Correlation-length curve: partial corr of I with the mean v in each exterior shell, given flux.
  6. Boundary mechanism: corr(boundary_v, edge_gap) and boundary_v for clean and crowded organisms.
  7. Dropout diagnostic: visible spots left unlabelled per frame.

For the existing scripts that read the older file formats (Table 2 replacement, decay curve, NN-null CMI),
run master_to_legacy.py to write files they can read.
"""
import argparse
import numpy as np

from partial_corr_utils import pc1_partial_corr


def resid(y, B):
    return y - B @ np.linalg.lstsq(B, y, rcond=None)[0]


def pcorr(a, b, *controls):
    B = np.column_stack([np.ones(len(a))] + list(controls)) if controls else np.ones((len(a), 1))
    return np.corrcoef(resid(a, B), resid(b, B))[0, 1]


def fisher_ci(r, n, k=2):
    se = 1 / np.sqrt(max(n - 3 - k, 1))
    z = np.arctanh(r)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def cluster_boot(fn, frame, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(frame)
    groups = [np.where(frame == i)[0] for i in ids]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        out.append(fn(np.concatenate([groups[p] for p in pick])))
    return np.percentile(out, [2.5, 97.5])


def line(name, r, n, ci=None, boot=None):
    s = f"  {name:52s} r={r:+.4f}  N={n:,}"
    if ci is not None:
        s += f"  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
    if boot is not None:
        s += f"  frame-bootstrap [{boot[0]:+.4f}, {boot[1]:+.4f}]"
    print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    d = np.load(args.data)
    I, E, fu, fv = d["I"], d["E"], d["fu"], d["fv"]
    fr, gap, ms = d["frame_id"], d["edge_gap"], d["merge_size"]
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    print(f"{args.data}: {len(d['f_pop'])} frames, {len(I):,} organism-instances ({good.sum():,} with finite I,E,flux); "
          f"min_distance={int(d['min_peak_distance'])}\n")

    print("1. Headline partial correlation r(I,E | flux)")
    r = pcorr(I[good], E[good], fu[good], fv[good]); n = good.sum()
    line("joint OLS (flux_u, flux_v)", r, n, fisher_ci(r, n),
         cluster_boot(lambda ix: pcorr(I[good][ix], E[good][ix], fu[good][ix], fv[good][ix]), fr[good], args.n_boot))
    r1, n1, expl = pc1_partial_corr(I[good], E[good], fu[good], fv[good])
    line(f"PC1 regressor (explains {expl:.1%} of flux var.)", r1, n1, fisher_ci(r1, n1, 1))
    print(f"  marginal r(I,E) = {np.corrcoef(I[good], E[good])[0,1]:+.4f}")

    print("\n2. Flux asymmetry (criterion B)")
    print(f"  U flux inward: {np.mean(fu[good] < 0):.4f}   V flux outward: {np.mean(fv[good] > 0):.4f}")

    print("\n3. Interior/exterior by whole-mask erosion (alternative definition)")
    Ia, Ea = d["I_adapt"], d["E_adapt"]
    g2 = np.isfinite(Ia) & np.isfinite(Ea) & np.isfinite(fu) & np.isfinite(fv)
    r2 = pcorr(Ia[g2], Ea[g2], fu[g2], fv[g2])
    r2p = pc1_partial_corr(Ia[g2], Ea[g2], fu[g2], fv[g2])[0]
    line("erosion definition, PC1", r2p, g2.sum(), fisher_ci(r2p, g2.sum(), 1))
    line("erosion definition, joint OLS (can be unstable)", r2, g2.sum(), fisher_ci(r2, g2.sum()))

    print("\n4. Clean vs crowded (merge_size == 1 vs > 1)")
    for name, m in (("clean (merge_size == 1)", ms == 1), ("crowded (merge_size > 1)", ms > 1)):
        mm = good & m
        if mm.sum() > 50:
            ru, rv = pcorr(I[mm], E[mm], fu[mm]), pcorr(I[mm], E[mm], fv[mm])
            rj = pcorr(I[mm], E[mm], fu[mm], fv[mm])
            rp = pc1_partial_corr(I[mm], E[mm], fu[mm], fv[mm])[0]
            rm = np.corrcoef(I[mm], E[mm])[0, 1]
            stable = min(ru, rv) <= rj <= max(ru, rv)
            line(f"{name}  [marginal]", rm, mm.sum())
            line(f"{name}  [PC1, screened {100*(1-(rp/rm)**2):.1f}%]", rp, mm.sum())
            line(f"{name}  [joint, {'stable' if stable else 'UNSTABLE'}]", rj, mm.sum())
    print(f"  fraction crowded: {np.mean(ms > 1):.3f};  edge gap: median {np.nanmedian(gap):.1f} px, "
          f"min {np.nanmin(gap):.1f} px (essentially no negative gaps after the merge)")

    print("\n5. Correlation-length curve: partial corr of I with shell-mean v, given flux (PC1 regressor, as in the paper; joint OLS for reference)")
    radii = d["radii"]; vs = d["v_shells"]; R = np.nanmedian(d["radius_px"])
    for s in range(vs.shape[1]):
        m = good & np.isfinite(vs[:, s])
        if m.sum() > 100:
            rp = pc1_partial_corr(I[m], vs[m, s], fu[m], fv[m])[0]
            mid = 0.5 * (radii[s] + radii[s + 1]) / R
            print(f"  shell [{radii[s]:>3}-{radii[s+1]:>3}] px ({mid:4.2f} radii)  PC1 r={rp:+.4f}   joint r={pcorr(I[m], vs[m, s], fu[m], fv[m]):+.4f}   N={m.sum():,}")

    print("\n6. Boundary mechanism")
    bv = d["boundary_v"]; m = np.isfinite(bv) & np.isfinite(gap)
    print(f"  corr(boundary_v, edge_gap) = {np.corrcoef(bv[m], gap[m])[0,1]:+.4f}  N={m.sum():,}")
    for name, mk in (("clean", ms == 1), ("crowded", ms > 1)):
        mk = m & mk
        if mk.sum():
            print(f"  boundary_v, {name}: median {np.median(bv[mk]):.4f}  (N={mk.sum():,})")

    print("\n7. Dropout diagnostic")
    print(f"  visible spots left unlabelled per frame: mean {np.mean(d['f_unlabelled']):.3f}, max {int(np.max(d['f_unlabelled']))}")
    print(f"  organisms per frame: mean {np.mean(d['f_pop']):.1f}  (raw watershed regions {np.mean(d['f_raw_pop']):.1f})")


if __name__ == "__main__":
    main()
