"""Follow-up to diag_size_shape.py: is the sign flip when radius is added a collider / over-conditioning effect?

    python3 diag_size_shape2.py

Radius is nearly collinear with the flux (r(radius, Phi_u) = -0.87), so 'flux + radius' conditions on the part of the
flux that is NOT explained by size, which is mostly (I,E)-driven signal (a common effect of I and E). Within narrow
radius bins we print, WITHOUT flux and WITH flux: r(I,E), r(I,PC1), r(E,PC1), and the PC1-partial r(I,E).
If r(I,E) within a bin is about as negative as the marginal value while the flux-partial turns positive, the flip comes
from conditioning on the common effect (flux), not from size explaining the dependence.
"""
import numpy as np


def resid(y, B):
    return y - B @ np.linalg.lstsq(B, y, rcond=None)[0]


def pc1_of(fu, fv):
    Z = np.column_stack([(fu - fu.mean()) / fu.std(), (fv - fv.mean()) / fv.std()])
    w, V = np.linalg.eigh(np.cov(Z.T))
    p = Z @ V[:, np.argmax(w)]
    return p if np.corrcoef(p, fu)[0, 1] > 0 else -p


def cc(a, b):
    return np.corrcoef(a, b)[0, 1]


def block(i, e, u, v, r, label):
    p = pc1_of(u, v)
    B1 = np.column_stack([np.ones(len(i)), p])
    Br = np.column_stack([np.ones(len(i)), r])
    Bpr = np.column_stack([np.ones(len(i)), p, r])
    print(f"{label}: N={len(i):,}")
    print(f"   marginal r(I,E) {cc(i,e):+.3f} | r(I,PC1) {cc(i,p):+.3f} | r(E,PC1) {cc(e,p):+.3f} | r(radius,PC1) {cc(r,p):+.3f}")
    print(f"   partial r(I,E | radius only) {cc(resid(i,Br),resid(e,Br)):+.3f} | | PC1 only {cc(resid(i,B1),resid(e,B1)):+.3f}"
          f" | PC1 + radius {cc(resid(i,Bpr),resid(e,Bpr)):+.3f}")
    # PC1 residual after removing radius: how much of what is left is I,E signal?
    pr = resid(p, Br)
    print(f"   PC1 residual after radius: R2(PC1~radius) {1-pr.var()/p.var():.3f};  corr of that residual with I {cc(pr,i):+.3f}, with E {cc(pr,e):+.3f}")


d = np.load("master_trajectory_progress.npz")
I, E, fu, fv, ms, rad = d["I"], d["E"], d["fu"], d["fv"], d["merge_size"], d["radius_px"]
good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv) & np.isfinite(rad)
for name, m in (("full", good), ("clean", good & (ms == 1))):
    i, e, u, v, r = I[m], E[m], fu[m], fv[m], rad[m]
    block(i, e, u, v, r, name)
    qs = np.percentile(r, [0, 20, 40, 60, 80, 100])
    for k in range(5):
        s = (r >= qs[k]) & (r <= qs[k + 1])
        block(i[s], e[s], u[s], v[s], r[s], f"  {name} radius bin [{qs[k]:.1f}-{qs[k+1]:.1f}]px")
    print()
