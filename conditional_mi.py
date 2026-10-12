"""
Nonparametric conditional mutual information estimator (Frenzel-Pompe /
KSG-style, k-NN based), for testing I(X;Y|Z) without assuming linearity.

This is the rigorous complement to the linear partial-correlation test
used in step 3b: it can catch a nonlinear channel from exterior to
interior that a straight-line residualization would miss entirely.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def cmi_ksg(X, Y, Z, k=5):
    """
    Estimate I(X;Y|Z) via the Frenzel-Pompe (2007) k-NN estimator
    (the conditional generalization of Kraskov-Stogbauer-Grassberger).
    X, Y, Z: arrays of shape (N,) or (N, d). Returns a scalar, in nats.
    """
    X, Y, Z = _as2d(X), _as2d(Y), _as2d(Z)
    N = X.shape[0]
    XYZ = np.concatenate([X, Y, Z], axis=1)
    XZ = np.concatenate([X, Z], axis=1)
    YZ = np.concatenate([Y, Z], axis=1)

    tree_xyz = cKDTree(XYZ)
    dists, _ = tree_xyz.query(XYZ, k=k + 1, p=np.inf)
    eps = dists[:, -1]  # distance to k-th neighbor (excluding self at index 0)

    tree_xz = cKDTree(XZ)
    tree_yz = cKDTree(YZ)
    tree_z = cKDTree(Z)

    def count_within(tree, pts, radii):
        counts = np.empty(len(pts), dtype=int)
        shrink = 1 - 1e-10
        for i in range(len(pts)):
            r = radii[i] * shrink
            idx = tree.query_ball_point(pts[i], r=r, p=np.inf)
            counts[i] = len(idx) - 1  # exclude self
        return counts

    n_xz = count_within(tree_xz, XZ, eps)
    n_yz = count_within(tree_yz, YZ, eps)
    n_z = count_within(tree_z, Z, eps)

    cmi = digamma(k) - np.mean(digamma(n_xz + 1) + digamma(n_yz + 1) - digamma(n_z + 1))
    return cmi


def _self_test(n=4000, seed=0, k=5):
    rng = np.random.default_rng(seed)

    # --- Test 1: X and Y each depend on Z, but are conditionally
    # independent given Z (independent noise only). True CMI ~ 0.
    Z = rng.standard_normal((n, 2))
    X_null = Z @ np.array([0.7, -0.3]) + 0.5 * rng.standard_normal(n)
    Y_null = Z @ np.array([-0.4, 0.6]) + 0.5 * rng.standard_normal(n)
    cmi_null = cmi_ksg(X_null, Y_null, Z, k=k)

    # --- Test 2: same, but with an added SHARED noise term not routed
    # through Z -- a genuine direct dependency. True CMI clearly > 0.
    shared = rng.standard_normal(n)
    X_dep = Z @ np.array([0.7, -0.3]) + 0.8 * shared + 0.3 * rng.standard_normal(n)
    Y_dep = Z @ np.array([-0.4, 0.6]) + 0.8 * shared + 0.3 * rng.standard_normal(n)
    cmi_dep = cmi_ksg(X_dep, Y_dep, Z, k=k)

    print(f"self-test (n={n}, k={k}):")
    print(f"  null case   (true CMI ~ 0):   estimated CMI = {cmi_null:.4f} nats")
    print(f"  dependent case (true CMI>0):  estimated CMI = {cmi_dep:.4f} nats")
    print(f"  ratio dep/null magnitude: {abs(cmi_dep)/max(abs(cmi_null),1e-6):.1f}x")

    ok = (abs(cmi_null) < 0.05) and (cmi_dep > 0.1) and (cmi_dep > 5 * abs(cmi_null))
    print("  PASS" if ok else "  FAIL -- estimator not behaving as expected")
    return ok


if __name__ == "__main__":
    _self_test()
