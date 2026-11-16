"""Data-processing-inequality (bottleneck) check for candidate blankets B (supplement; added in the revision).

If B were a Markov blanket, the chain I -> B -> E would hold (I = interior state, E = exterior state), and the data
processing inequality would give
        MI(I;E) <= MI(I;B)      and      MI(I;E) <= MI(B;E).
A violation rejects B as a blanket without any conditioning, and the excess is a lower bound on the information that
bypasses B.  Passing the check proves nothing: the inequalities are necessary, not sufficient.  (In particular, a collider
such as the flux passes it while still failing  MI(I;E|B) = 0.)

For each candidate B (flux (fu, fv); boundary concentration (boundary_u, boundary_v)) and for each of S random subsamples of N
organism-instances, the script computes the KSG estimate (max-norm, k neighbours, z-scored variables, as in
analyze_pooled_collider_cmi.py) of MI(I;E), MI(I;B), MI(B;E), and the gaps
        gap_IB = MI(I;E) - MI(I;B)       gap_BE = MI(I;E) - MI(B;E)       (positive = inequality violated).
Reported: the mean over subsamples, the standard deviation over subsamples (estimator and subsample variation only: the
subsamples overlap, so this is NOT the sampling uncertainty of the population, which needs the cluster/block bootstrap
of the other scripts), and the number of subsamples with a positive gap.

Input: master_trajectory_progress.npz (fields I, E, fu, fv, boundary_u, boundary_v).
Usage (from the build folder or local_package):
    python3 analyze_dpi_bottleneck.py --self-test        # synthetic data with known answers, no real data needed
    python3 analyze_dpi_bottleneck.py                    # real data, 20 subsamples of 8000
    python3 analyze_dpi_bottleneck.py --macros           # also write results/dpi.tex
"""
import argparse
import numpy as np

import analyze_pooled_collider_cmi as P

mi = P.mutual_information_ksg
zs = P.zscore


def zcols(*cols):
    return np.column_stack([zs(c) for c in cols])


def triple(I, E, B, k):
    """MI(I;E), MI(I;B), MI(B;E) on already z-scored arrays."""
    return mi(I, E, k=k), mi(I, B, k=k), mi(B, E, k=k)


# ----------------------------------------------------------------------------------------------------------------------
def self_test(k):
    print("Self-test on synthetic data (no real data needed).")
    rng = np.random.default_rng(3)
    n = 8000
    # 1. Estimator bias at different dimensions: jointly Gaussian, exact MI known.
    #    X scalar, Y 2-dimensional with correlation rho between X and each Y component, Y components independent given X.
    rho = 0.6
    X = rng.normal(size=n)
    Y = np.column_stack([rho * X + np.sqrt(1 - rho ** 2) * rng.normal(size=n) for _ in range(2)])
    exact_1d = -0.5 * np.log(1 - rho ** 2)
    # exact MI(X; Y1,Y2): Y = rho*X*1 + noise; Var(Y|X)=(1-rho^2) I, Var(Y)=rho^2 11' + (1-rho^2) I
    cov_y = rho ** 2 * np.ones((2, 2)) + (1 - rho ** 2) * np.eye(2)
    exact_2d = 0.5 * np.log(np.linalg.det(cov_y) / (1 - rho ** 2) ** 2)
    est_1d = mi(zs(X), zs(Y[:, 0]), k=k)
    est_2d = mi(zs(X).reshape(-1, 1), zcols(Y[:, 0], Y[:, 1]), k=k)
    print(f"  scalar-scalar MI:  exact {exact_1d:.4f}   estimate {est_1d:.4f}")
    print(f"  scalar-2D MI:      exact {exact_2d:.4f}   estimate {est_2d:.4f}   (a bias of a few 0.01 would limit what gaps can be believed)")

    reps = 8
    for label in ["chain      I -> B -> E (DPI must hold)",
                  "bypass     E also depends directly on I (DPI violated)",
                  "collider   B = I + E + noise (DPI holds, conditional independence fails)"]:
        gaps_ib, gaps_be = [], []
        for r in range(reps):
            g = np.random.default_rng(100 + r)
            Iv = g.normal(size=n)
            if label.startswith("chain"):
                B1 = 0.8 * Iv + 0.6 * g.normal(size=n)
                B2 = 0.8 * Iv + 0.6 * g.normal(size=n)
                Ev = 0.5 * B1 + 0.5 * B2 + 0.7 * g.normal(size=n)
            elif label.startswith("bypass"):
                B1 = 0.5 * Iv + 0.9 * g.normal(size=n)
                B2 = 0.5 * Iv + 0.9 * g.normal(size=n)
                Ev = 0.9 * Iv + 0.3 * (B1 + B2) + 0.3 * g.normal(size=n)
            else:
                Ev = g.normal(size=n)
                B1 = Iv + Ev + 0.3 * g.normal(size=n)
                B2 = Iv - 0.5 * Ev + 0.3 * g.normal(size=n)
            Iz, Ez, Bz = zs(Iv), zs(Ev), zcols(B1, B2)
            m_ie, m_ib, m_be = triple(Iz, Ez, Bz, k)
            gaps_ib.append(m_ie - m_ib)
            gaps_be.append(m_ie - m_be)
        print(f"  {label}")
        print(f"      gap_IB = MI(I;E)-MI(I;B): mean {np.mean(gaps_ib):+.4f}  sd {np.std(gaps_ib, ddof=1):.4f}  positive in {sum(g > 0 for g in gaps_ib)}/{reps}")
        print(f"      gap_BE = MI(I;E)-MI(B;E): mean {np.mean(gaps_be):+.4f}  sd {np.std(gaps_be, ddof=1):.4f}  positive in {sum(g > 0 for g in gaps_be)}/{reps}")
    print("Expected: chain -> gaps clearly negative; bypass -> at least one gap clearly positive; collider -> gaps negative.")


# ----------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--n-subsamples", type=int, default=20)
    ap.add_argument("--k", type=int, default=P.K_MI)
    ap.add_argument("--macros", action="store_true", help="write results/dpi.tex")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(a.k)
        return

    d = np.load(a.data)
    need = ["I", "E", "fu", "fv", "boundary_u", "boundary_v"]
    for f in need:
        if f not in d:
            raise SystemExit(f"field {f!r} not in {a.data}; available: {list(d.keys())}")
    arr = {f: d[f].astype(float) for f in need}
    ok = np.all([np.isfinite(arr[f]) for f in need], axis=0)
    arr = {f: v[ok] for f, v in arr.items()}
    N = int(ok.sum())
    print(f"{a.data}: {N} organism-instances with all fields finite; {a.n_subsamples} subsamples of {min(a.n_sub, N)}; k={a.k}")

    names = [("flux", ("fu", "fv")), ("boundary concentration", ("boundary_u", "boundary_v"))]
    rows = {nm: [] for nm, _ in names}
    mi_ie = []
    for s in range(a.n_subsamples):
        rng = np.random.default_rng(s)
        idx = rng.choice(N, size=min(a.n_sub, N), replace=False)
        I, E = zs(arr["I"][idx]), zs(arr["E"][idx])
        m_ie = mi(I, E, k=a.k)
        mi_ie.append(m_ie)
        for nm, (c1, c2) in names:
            B = zcols(arr[c1][idx], arr[c2][idx])
            rows[nm].append((m_ie, mi(I, B, k=a.k), mi(B, E, k=a.k)))
        print(f"  subsample {s + 1}/{a.n_subsamples} done", flush=True)

    mi_ie = np.array(mi_ie)
    print(f"\nMI(I;E) = {mi_ie.mean():.4f} nats  (sd over subsamples {mi_ie.std(ddof=1):.4f})")
    items = [("DpiSubsamples", f"{a.n_subsamples}"), ("DpiNSub", f"{min(a.n_sub, N)}"), ("DpiMIIE", f"{mi_ie.mean():.5f}")]
    tag = {"flux": "Flux", "boundary concentration": "Bdry"}
    for nm, _ in names:
        r = np.array(rows[nm])
        m_ie, m_ib, m_be = r[:, 0], r[:, 1], r[:, 2]
        g_ib, g_be = m_ie - m_ib, m_ie - m_be
        print(f"\nB = {nm}")
        print(f"  MI(I;B) = {m_ib.mean():.4f} (sd {m_ib.std(ddof=1):.4f})    MI(B;E) = {m_be.mean():.4f} (sd {m_be.std(ddof=1):.4f})")
        print(f"  gap_IB = MI(I;E)-MI(I;B) = {g_ib.mean():+.4f} (sd {g_ib.std(ddof=1):.4f}); positive in {int((g_ib > 0).sum())}/{len(g_ib)} subsamples")
        print(f"  gap_BE = MI(I;E)-MI(B;E) = {g_be.mean():+.4f} (sd {g_be.std(ddof=1):.4f}); positive in {int((g_be > 0).sum())}/{len(g_be)} subsamples")
        verdict = "VIOLATED" if (g_ib.mean() > 2 * g_ib.std(ddof=1) or g_be.mean() > 2 * g_be.std(ddof=1)) else "not violated (no evidence either way)"
        print(f"  data processing inequality: {verdict}")
        t = tag[nm]
        items += [(f"Dpi{t}MIIB", f"{m_ib.mean():.5f}"), (f"Dpi{t}MIBE", f"{m_be.mean():.5f}"),
                  (f"Dpi{t}GapIB", f"{g_ib.mean():.5f}"), (f"Dpi{t}GapBE", f"{g_be.mean():.5f}"),
                  (f"Dpi{t}GapIBSD", f"{g_ib.std(ddof=1):.5f}"), (f"Dpi{t}GapBESD", f"{g_be.std(ddof=1):.5f}"),
                  (f"Dpi{t}GapIBPos", f"{int((g_ib > 0).sum())}"), (f"Dpi{t}GapBEPos", f"{int((g_be > 0).sum())}")]
    if a.macros:
        from texout import write_macros
        write_macros("results/dpi.tex", "analyze_dpi_bottleneck.py", items)
        print("\nwrote results/dpi.tex")


if __name__ == "__main__":
    main()
