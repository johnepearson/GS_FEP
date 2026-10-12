"""
Follow-up checks of the conditional-independence result, computed from the master single-trajectory run
(master_trajectory_progress.npz, merged segmentation, min_distance 24). No simulation.

    python3 analyze_master_followups.py                  # everything (about 10-15 min, dominated by the CMI nulls)
    python3 analyze_master_followups.py --skip-cmi       # linear checks only (seconds)

PC1 flux regressor throughout (paper convention). Sections
  A. Confound checks on the full population: + radius & eccentricity; + frame-level covariates; frame fixed effects;
     all together; + near-field u-shell means (all seven, their mean, and each alone); + local crowding density
     (neighbours within 80 px).
  B. By regime (full / clean = merge_size 1 / crowded = merge_size > 1):
     marginal and PC1 partial r; frame fixed effects; + radius and u-shell covariates; ridge joint (Phi_u, Phi_v);
     boundary-concentration conditioning (joint (u_b, v_b), each alone, stability criterion);
     nonlinear CMI (flux and boundary concentration, z-scored, nearest-neighbour null) on an N=8000 subsample.
  C. Richer-blanket linear check: flux alone vs flux + the organism's own v in 16 exterior shells (out to ~6 radii),
     against the far-field exterior (mean v in shells beyond 8.5 radii), plus the screening curve (adding shells one at a time).

Needs analyze_pooled_collider_cmi.py (CMI estimator) and partial_corr_utils.py in the same folder.
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree

from partial_corr_utils import pc1_partial_corr


def resid(y, B):
    return y - B @ np.linalg.lstsq(B, y, rcond=None)[0]


def pc1_of(fu, fv):
    Z = np.column_stack([(fu - fu.mean()) / fu.std(), (fv - fv.mean()) / fv.std()])
    w, V = np.linalg.eigh(np.cov(Z.T))
    return Z @ V[:, np.argmax(w)]


def pr(I, E, fu, fv, covs=None):
    """partial r(I,E) given the PC1 flux regressor (+ optional covariate columns)"""
    cols = [np.ones(len(I)), pc1_of(fu, fv)]
    if covs is not None:
        covs = np.asarray(covs, float)
        cols += [covs] if covs.ndim == 1 else [covs[:, j] for j in range(covs.shape[1])]
    B = np.column_stack(cols)
    return np.corrcoef(resid(I, B), resid(E, B))[0, 1]


def fe(x, frame):
    out = x.copy()
    for f in np.unique(frame):
        m = frame == f
        out[m] = x[m] - x[m].mean()
    return out


def fisher(r, n, k=1):
    se = 1 / np.sqrt(max(n - 3 - k, 1)); z = np.arctanh(r)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def show(name, r, n, base=None):
    s = f"  {name:58s} r={r:+.4f}  N={n:,}"
    if base is not None:
        s += f"   ({100 * (abs(r) / abs(base) - 1):+.0f}% in |r|)"
    print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--skip-cmi", action="store_true")
    args = ap.parse_args()

    d = np.load(args.data)
    I, E, fu, fv = d["I"], d["E"], d["fu"], d["fv"]
    ub, vb = d["boundary_u"], d["boundary_v"]
    frame, ms = d["frame_id"], d["merge_size"]
    rad, ecc = d["radius_px"], d["eccentricity"]
    ush = d["u_shells"]
    fids = np.unique(frame)
    assert len(fids) == len(d["f_pop"]), "frame arrays do not line up with frame ids"
    idx = np.searchsorted(fids, frame)
    fpop, fmu, fot = d["f_pop"][idx], d["f_mean_u"][idx], d["f_otsu"][idx]
    N = 1024
    dens = np.zeros(len(I))
    for f in fids:
        m = np.where(frame == f)[0]
        if len(m) > 1:
            tree = cKDTree(np.column_stack([d["cy"][m], d["cx"][m]]) % N, boxsize=N)
            dens[m] = tree.query_ball_point(np.column_stack([d["cy"][m], d["cx"][m]]) % N, r=80, return_length=True) - 1
    good = (np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv) & np.isfinite(ub) & np.isfinite(vb)
            & np.isfinite(ush).all(axis=1))
    print(f"{args.data}: {len(I):,} organism-instances, {good.sum():,} with all variables finite; "
          f"min_distance={int(d['min_peak_distance'])}\n")

    # ---------------- A ----------------
    print("A. Confound checks, full population (PC1 flux regressor)")
    g = good
    base = pr(I[g], E[g], fu[g], fv[g])
    show("baseline (flux only)", base, g.sum())
    show("+ radius, eccentricity", pr(I[g], E[g], fu[g], fv[g], np.column_stack([rad[g], ecc[g]])), g.sum(), base)
    fcov = np.column_stack([fpop[g], fmu[g], fot[g]])
    show("+ frame population, mean u, Otsu threshold", pr(I[g], E[g], fu[g], fv[g], fcov), g.sum(), base)
    show("frame fixed effects (within-frame demeaning)",
         pr(fe(I[g], frame[g]), fe(E[g], frame[g]), fe(fu[g], frame[g]), fe(fv[g], frame[g])), g.sum(), base)
    allc = np.column_stack([rad[g], ecc[g], fpop[g], fmu[g], fot[g]])
    show("+ radius, eccentricity, frame covariates (all together)", pr(I[g], E[g], fu[g], fv[g], allc), g.sum(), base)
    show("+ all seven near-field u-shell means", pr(I[g], E[g], fu[g], fv[g], ush[g]), g.sum(), base)
    show("+ mean of the seven u-shells", pr(I[g], E[g], fu[g], fv[g], ush[g].mean(axis=1)), g.sum(), base)
    ur = d["u_shell_radii"]
    for j in range(ush.shape[1]):
        show(f"+ u-shell {int(ur[j])}-{int(ur[j+1])} px alone", pr(I[g], E[g], fu[g], fv[g], ush[g][:, j]), g.sum(), base)
    show("+ local crowding density (neighbours within 80 px)", pr(I[g], E[g], fu[g], fv[g], dens[g]), g.sum(), base)
    print()

    # ---------------- B ----------------
    from numpy.linalg import solve
    regimes = [("full", good), ("clean (merge_size == 1)", good & (ms == 1)), ("crowded (merge_size > 1)", good & (ms > 1))]
    rng = np.random.default_rng(0)
    cmi_mod = None
    if not args.skip_cmi:
        import analyze_pooled_collider_cmi as cmi_mod
    for name, m in regimes:
        print(f"B. {name}: N={m.sum():,}")
        i, e, u_, v_ = I[m], E[m], fu[m], fv[m]
        rm = np.corrcoef(i, e)[0, 1]
        rp = pc1_partial_corr(i, e, u_, v_)[0]
        lo, hi = fisher(rp, m.sum())
        print(f"  marginal r = {rm:+.4f};  PC1 partial r = {rp:+.4f}  (naive 95% CI [{lo:+.4f}, {hi:+.4f}], "
              f"{100 * (1 - (rp / rm) ** 2):.1f}% of marginal variance screened)")
        show("frame fixed effects", pr(fe(i, frame[m]), fe(e, frame[m]), fe(u_, frame[m]), fe(v_, frame[m])), m.sum(), rp)
        show("+ radius and the seven u-shells", pr(i, e, u_, v_, np.column_stack([rad[m], ush[m]])), m.sum(), rp)
        Z = np.column_stack([(u_ - u_.mean()) / u_.std(), (v_ - v_.mean()) / v_.std()])
        for alpha in (0, 1, 10, 100):
            def rres(y):
                y = y - y.mean()
                return y - Z @ solve(Z.T @ Z + alpha * np.eye(2), Z.T @ y)
            print(f"  ridge joint (Phi_u, Phi_v), alpha={alpha:>3}:  r = {np.corrcoef(rres(i), rres(e))[0, 1]:+.4f}")
        ru = np.corrcoef(resid(i, np.column_stack([np.ones(len(i)), u_])), resid(e, np.column_stack([np.ones(len(i)), u_])))[0, 1]
        rv = np.corrcoef(resid(i, np.column_stack([np.ones(len(i)), v_])), resid(e, np.column_stack([np.ones(len(i)), v_])))[0, 1]
        Bj = np.column_stack([np.ones(len(i)), u_, v_])
        rj = np.corrcoef(resid(i, Bj), resid(e, Bj))[0, 1]
        print(f"  flux alone: Phi_u r={ru:+.4f}, Phi_v r={rv:+.4f}; joint r={rj:+.4f} "
              f"({'stable' if min(ru, rv) <= rj <= max(ru, rv) else 'UNSTABLE: outside the single-variable range'}; "
              f"{100 * (1 - (rj / rm) ** 2):+.1f}% of marginal variance 'screened')")
        bu_, bv_ = ub[m], vb[m]
        def p1(x):
            Bx = np.column_stack([np.ones(len(i)), x]); return np.corrcoef(resid(i, Bx), resid(e, Bx))[0, 1]
        Bb = np.column_stack([np.ones(len(i)), bu_, bv_])
        rb = np.corrcoef(resid(i, Bb), resid(e, Bb))[0, 1]
        rbu, rbv = p1(bu_), p1(bv_)
        print(f"  boundary concentration (u_b, v_b): joint r={rb:+.4f} (95% CI [{fisher(rb, m.sum(), 2)[0]:+.4f}, "
              f"{fisher(rb, m.sum(), 2)[1]:+.4f}]), u_b alone {rbu:+.4f}, v_b alone {rbv:+.4f}, "
              f"{'stable' if min(rbu, rbv) <= rb <= max(rbu, rbv) else 'UNSTABLE: outside the single-variable range'}; "
              f"{100 * (1 - (rb / rm) ** 2):.1f}% of marginal variance screened")
        if cmi_mod is not None and name != "full":
            k = np.arange(m.sum())
            if len(k) > args.n_sub:
                k = rng.choice(len(k), size=args.n_sub, replace=False)
            iz, ez = cmi_mod.zscore(i[k]), cmi_mod.zscore(e[k])
            mi0 = cmi_mod.mutual_information_ksg(iz, ez)
            print(f"  nonlinear (N={len(k):,} subsample): unconditioned MI(I;E) = {mi0:.4f} nats")
            for lab, Zc in (("flux (z-scored)", np.column_stack([cmi_mod.zscore(u_[k]), cmi_mod.zscore(v_[k])])),
                            ("boundary concentration (z-scored)", np.column_stack([cmi_mod.zscore(bu_[k]), cmi_mod.zscore(bv_[k])]))):
                c, nm, ns, nsd = cmi_mod.cmi_with_null(iz, ez, Zc, n_trials=args.n_trials)
                print(f"    CMI(I;E | {lab:34s}) = {c:.4f} nats; null {nm:.4f} (std {ns:.4f}); {nsd:.1f} null-SDs; "
                      f"excess over null {c - nm:.4f}; screened {100 * (1 - c / mi0):+.1f}%")
        print()

    # ---------------- C ----------------
    print("C. Richer-blanket linear check (full population)")
    edges = d["radii"]; vs = d["v_shells"]
    R = np.nanmedian(rad)
    mid = 0.5 * (edges[:-1] + edges[1:]) / R
    near = np.where(mid <= 6.3)[0][:16]
    far = np.where(mid >= 8.5)[0]
    mc = good & np.isfinite(vs).all(axis=1)
    ff = vs[mc][:, far].mean(axis=1)
    print(f"  near-field shells: {len(near)} shells, mid-radii {mid[near[0]]:.2f}-{mid[near[-1]]:.2f} organism radii (R = {R:.1f} px); "
          f"far-field: {len(far)} shells, {mid[far[0]]:.2f}-{mid[far[-1]]:.2f} radii")
    r0 = pr(I[mc], ff, fu[mc], fv[mc])
    show("flux alone vs far-field exterior", r0, mc.sum())
    rall = pr(I[mc], ff, fu[mc], fv[mc], vs[mc][:, near])
    show(f"flux + own v in {len(near)} near-field shells", rall, mc.sum(), r0)
    print("  screening curve (shells added one at a time, innermost first):")
    for n in range(1, len(near) + 1):
        print(f"    n={n:2d}  r = {pr(I[mc], ff, fu[mc], fv[mc], vs[mc][:, near[:n]]):+.4f}")


if __name__ == "__main__":
    main()
