"""
Free-energy comparison with a proper SIZE control (reviewer request).

The paper's instance-level check (rebuild_instance_level_check.py) controlled for a radial-distance proxy of size
that does not track organism size. Here every organism-instance carries its measured area (pixels), and we ask whether
the F_template vs Friston-F correlation survives controlling for size properly:
  - how strongly each F tracks size (r with sqrt(area), the equivalent radius)
  - partial r controlling for a flexible function of equivalent radius (cubic polynomial) and age
  - r within narrow size strata (quintiles of equivalent radius), pooled with no size variation left to exploit
  - how well the old size proxy tracked the measured size

Needs (same as rebuild_instance_level_check.py): lifecycle_progress.pkl, fe_template_1024.npz, the seed-1000
ensemble used by fe_model.load_constants(). Run from local_package/:
    python3 run_fe_size_control.py
"""
import numpy as np
import rebuild_instance_level_check as R


def resid_poly(v, X):
    A = np.column_stack([np.ones(len(v))] + X)
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def pcorr(x, y, X):
    return np.corrcoef(resid_poly(x, X), resid_poly(y, X))[0, 1]


def main():
    tr = R.tolerant_load("lifecycle_progress.pkl")
    rows = R.collect_instances(tr)
    template = np.load("fe_template_1024.npz")["template"]
    keep = []
    for rec in rows:
        if "area" not in rec:
            raise SystemExit("records have no 'area' field; lifecycle_progress.pkl is older than run_lifecycle.py's area record")
        vals = [rec["I"], rec["E"], rec["fu"], rec["fv"], rec["area"]]
        if not all(np.isfinite(vals)) or rec["area"] <= 0:
            continue
        ft = R.f_template(rec["profile"], template)
        sz = R.size_proxy(rec["profile"])
        if not (np.isfinite(ft) and np.isfinite(sz)):
            continue
        keep.append((rec["I"], rec["E"], rec["fu"], rec["fv"], rec["area"], rec["age"], ft, sz))
    D = np.array(keep)
    I, E, fu, fv, area, age, ft, szp = D.T
    N = len(I)
    mu_q = R.SLOPE * I + R.INTERCEPT
    F = R.kl_term(mu_q, R.SIGMA_Q2, R.MU_P, R.SIGMA_P2) + R.accuracy_term(np.column_stack([fu, fv]), mu_q, R.SIGMA_Q2, R.A, R.B, R.SIGMA)
    req = np.sqrt(area / np.pi)
    print(f"N = {N} organism-instances with area; equivalent radius median {np.median(req):.1f} px, IQR [{np.percentile(req,25):.1f}, {np.percentile(req,75):.1f}]")

    print("\nHow well does the OLD size proxy track measured size?")
    print(f"  r(old proxy, equivalent radius) = {np.corrcoef(szp, req)[0,1]:+.3f}")

    print("\nHow strongly does each quantity track size (r with equivalent radius)?")
    for name, v in (("F_template", ft), ("Friston F", F), ("I", I), ("E", E), ("flux_u", fu), ("flux_v", fv), ("age", age)):
        print(f"  {name:11s} {np.corrcoef(v, req)[0,1]:+.3f}")

    r0 = np.corrcoef(ft, F)[0, 1]
    print(f"\nUncontrolled r(F_template, Friston F) = {r0:+.4f}")
    ctrl = {
        "age only": [age],
        "old proxy only": [szp],
        "equivalent radius (linear)": [req],
        "equivalent radius, cubic": [req, req**2, req**3],
        "equivalent radius cubic + age": [req, req**2, req**3, age],
        "equivalent radius cubic + log area + age": [req, req**2, req**3, np.log(area), age],
    }
    print("Partial r controlling for:")
    pc = {}
    for k, X in ctrl.items():
        pc[k] = pcorr(ft, F, X)
        print(f"  {k:42s} {pc[k]:+.4f}")
    from texout import num, write_macros
    cubic = pc["equivalent radius, cubic"]
    kl = R.kl_term(mu_q, R.SIGMA_Q2, R.MU_P, R.SIGMA_P2)
    acc = F - kl
    edges = np.percentile(req, [0, 20, 40, 60, 80, 100])
    srs, sws = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        mk = (req >= lo_) & (req <= hi_)
        srs.append(np.corrcoef(ft[mk], F[mk])[0, 1]); sws.append(mk.sum())
    cc = lambda a_, b_: np.corrcoef(a_, b_)[0, 1]
    SW = ["One", "Two", "Three", "Four", "Five"]
    items = [
        ("FeSizeN", num(N, 0)), ("FeSizeRZero", num(r0, 4)),
        ("FeSizeRTemplate", num(cc(ft, req), 3)), ("FeSizeRFriston", num(cc(F, req), 3)),
        ("FeSizeRLinear", num(pc["equivalent radius (linear)"], 4)), ("FeSizeRCubic", num(cubic, 4)),
        ("FeSizeShare", num(100 * (1 - (cubic / r0) ** 2), 1)),
        ("FeSizeMedian", num(np.median(req), 1)), ("FeSizeQLo", num(np.percentile(req, 25), 1)), ("FeSizeQHi", num(np.percentile(req, 75), 1)),
        ("FeSizeProxyR", num(cc(szp, req), 2)), ("FeSizeRI", num(cc(I, req), 2)), ("FeSizeRE", num(cc(E, req), 2)),
        ("FeSizeRFluxU", num(cc(fu, req), 2)), ("FeSizeRFluxV", num(cc(fv, req), 2)),
        ("FeSizeAge", num(pc["age only"], 4)), ("FeSizeProxy", num(pc["old proxy only"], 4)),
        ("FeSizeCubicAge", num(pc["equivalent radius cubic + age"], 4)),
        ("FeSizeCubicLogAge", num(pc["equivalent radius cubic + log area + age"], 4)),
        ("FeSizeStrataMean", num(np.average(srs, weights=sws), 2)), ("FeSizeStratumN", num(np.mean(sws), 0)),
        ("FeSizeKLF", num(cc(kl, ft), 2)), ("FeSizeAccF", num(cc(acc, ft), 2)),
        ("FeSizeKLReq", num(cc(kl, req), 2)), ("FeSizeAccReq", num(cc(acc, req), 2))]
    items += [(f"FeSizeStratum{w}", num(r_, 2)) for w, r_ in zip(SW, srs)]
    write_macros("results/fesize.tex", "run_fe_size_control.py", items)

    print("\nWithin size strata (quintiles of equivalent radius):")
    edges = np.percentile(req, [0, 20, 40, 60, 80, 100])
    rs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (req >= a) & (req <= b)
        r = np.corrcoef(ft[m], F[m])[0, 1]
        rs.append(r); ws.append(m.sum())
        print(f"  radius {a:5.1f}-{b:5.1f} px  N={m.sum():6d}  r = {r:+.4f}")
    print(f"  mean within-stratum r = {np.average(rs, weights=ws):+.4f}")

    print("\nF-components: does the flux part track size more than the state part?")
    kl = R.kl_term(mu_q, R.SIGMA_Q2, R.MU_P, R.SIGMA_P2)
    acc = F - kl
    for name, v in (("KL (state) term", kl), ("accuracy (flux) term", acc)):
        print(f"  {name:22s} r with F_template {np.corrcoef(v, ft)[0,1]:+.3f}   r with equivalent radius {np.corrcoef(v, req)[0,1]:+.3f}")
    print("\n(All confidence intervals omitted: instances within a lineage are not independent.)")


if __name__ == "__main__":
    main()
