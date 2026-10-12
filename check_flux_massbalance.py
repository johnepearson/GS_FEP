"""
Does the stored u-flux of each organism track its total reaction rate sum(u v^2)?

Mass balance for u over an organism's mask (steady-state limit):
    outward diffusive u-flux  =  -sum(u v^2)*dx^2  +  F * sum(1-u)*dx^2  -  d/dt sum(u)
so  -flux_u  ~  c1 * uv2_sum  -  c2 * area  with c1 ~ dx^2 and c2 ~ F*(mean 1-u)*dx^2.

Reads the existing lifecycle_progress.pkl (needs the updated tracker's uv2_sum/area). No simulation.
Usage:  python3 check_flux_massbalance.py
"""
import numpy as np
from run_lifecycle import LifecycleTracker  # noqa: F401  (pickle needs the class in __main__)
from rebuild_fe_comparison_1024 import tolerant_load


def r2(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    return b, 1 - res.var() / y.var(), res


def partial(a, b, c):
    ra = a - c * (a @ c) / (c @ c)
    rb = b - c * (b @ c) / (c @ c)
    return np.corrcoef(ra, rb)[0, 1]


def main():
    tr = tolerant_load("lifecycle_progress.pkl")
    rows = [(h['fu'], h['uv2_sum'], h['area'], h['t'])
            for e in tr.examples for h in e['history'] if 'uv2_sum' in h]
    a = np.array(rows, float)
    ok = np.all(np.isfinite(a[:, :3]), axis=1)
    fu, s, area = -a[ok, 0], a[ok, 1], a[ok, 2]
    print(f"{ok.sum()} organism-instances")
    print(f"sign check: fraction with u-flux inward (-fu > 0) = {(fu > 0).mean():.4f}")
    print(f"r(-flux_u, total uv2)                 = {np.corrcoef(fu, s)[0,1]:.4f}")
    print(f"r(-flux_u, area)                      = {np.corrcoef(fu, area)[0,1]:.4f}")
    print(f"r(total uv2, area)                    = {np.corrcoef(s, area)[0,1]:.4f}")
    print(f"partial r(-flux_u, total uv2 | area)  = {partial(fu, s, area):.4f}")
    b1, R1, _ = r2(fu, s[:, None])
    b2, R2, _ = r2(fu, np.column_stack([s, area]))
    print(f"fit  -flux_u = c1*uv2_sum:            c1={b1[0]:.5g}  R^2={R1:.4f}")
    print(f"fit  -flux_u = c1*uv2_sum + c2*area:  c1={b2[0]:.5g}  c2={b2[1]:.5g}  R^2={R2:.4f}")
    print(f"     c2/c1 = {b2[1]/b2[0]:.4g}   (mass balance predicts c1=dx^2 and c2/c1 = -F*mean(1-u), about -F)")
    if all('omu_sum' in h for e in tr.examples[:5] for h in e['history']):
        ck = np.load("checkpoint_1024_settled.npz")      # the run's own parameters, not assumed values
        L, n, F = float(ck["L"]), int(ck["n"]), float(ck["F"])
        print(f"parameters from checkpoint: L={L}, n={n}, F={F}")
        dx2 = (L / n) ** 2
        om = np.array([h['omu_sum'] for e in tr.examples for h in e['history'] if 'uv2_sum' in h], float)[ok]
        # exact quasi-steady balance, no free coefficients:  -flux_u = dx2*(uv2_sum - F*omu_sum)
        pred = dx2 * (s - F * om)
        print(f"NO-FIT balance  -flux_u vs dx^2*(sum uv2 - F*sum(1-u)):  r={np.corrcoef(fu, pred)[0,1]:.4f}, "
              f"median ratio measured/predicted = {np.median(fu/pred):.3f}")
        print(f"                -flux_u vs dx^2*sum uv2 alone:           r={np.corrcoef(fu, dx2*s)[0,1]:.4f}, "
              f"median ratio = {np.median(fu/(dx2*s)):.3f}")
        b3, R3, _ = r2(fu, np.column_stack([s, om]))
        print(f"fit  -flux_u = c1*uv2_sum + c2*omu_sum:  c1/dx2={b3[0]/dx2:.4f} (expect 1)  c2/c1={b3[1]/b3[0]:.4f} (expect {-F})  R^2={R3:.4f}")
    # the same within narrow area bins, so size cannot drive it
    edges = np.quantile(area, np.linspace(0, 1, 11))
    rs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (area >= lo) & (area <= hi)
        rs.append(np.corrcoef(fu[m], s[m])[0, 1])
    print("r(-flux_u, total uv2) inside area deciles: " + ", ".join(f"{r:.3f}" for r in rs))


if __name__ == "__main__":
    main()
