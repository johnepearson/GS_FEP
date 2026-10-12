"""
Surprisal of the observed boundary flux under the paper's Friston generative model, followed through the
division cycle next to F_template and the Friston F.

Model (same constants as friston_F, refit by fe_model.py):  s = E ~ N(mu_p, sigma_p^2);  o = (fu, fv)
given s ~ N(A s + B, Sigma).  Everything below is exact (Gaussian):
  surprisal  = -ln p(o)   with   o ~ N(A mu_p + B,  Sigma + sigma_p^2 A A^T)
  true posterior p(s|o) = N(m_post, v_post)
  gap        = KL( q(s|I) || p(s|o) )  >= 0
and the identity  F = surprisal + gap  is checked on every organism-instance before anything is reported.

Usage:   python3 analyze_surprisal.py
Needs lifecycle_progress.pkl and fe_template_1024.npz, as analyze_uv2_lifecycle.py does.
"""
import numpy as np
from run_lifecycle import LifecycleTracker  # noqa: F401  (needed to unpickle)
import rebuild_fe_comparison_1024 as R
from rebuild_fe_comparison_1024 import tolerant_load, friston_F
from analyze_uv2_lifecycle import curve, z


def surprisal_and_gap(I, fu, fv, mu_p, sp2, A, B, Sigma, slope, icpt, sq2):
    o = np.array([fu, fv])
    S_o = Sigma + sp2 * np.outer(A, A)
    m_o = A * mu_p + B
    d = o - m_o
    _, ld = np.linalg.slogdet(S_o)
    surpr = np.log(2 * np.pi) + 0.5 * ld + 0.5 * d @ np.linalg.solve(S_o, d)
    Si = np.linalg.inv(Sigma)
    prec = 1.0 / sp2 + A @ Si @ A
    v_post = 1.0 / prec
    m_post = v_post * (mu_p / sp2 + A @ Si @ (o - B))
    mu_q = slope * I + icpt
    gap = np.log(np.sqrt(v_post) / np.sqrt(sq2)) + (sq2 + (mu_q - m_post) ** 2) / (2 * v_post) - 0.5
    return surpr, gap


def selftest():
    rng = np.random.default_rng(1)
    for _ in range(5):
        mu_p, sp2 = rng.normal(), rng.uniform(.5, 2)
        A, B = rng.normal(size=2), rng.normal(size=2)
        M = rng.normal(size=(2, 2)); Sigma = M @ M.T + 0.1 * np.eye(2)
        slope, icpt, sq2 = rng.normal(), rng.normal(), rng.uniform(.3, 1)
        I, fu, fv = rng.normal(), rng.normal(), rng.normal()
        mu_q = slope * I + icpt
        KL = np.log(np.sqrt(sp2 / sq2)) + (sq2 + (mu_q - mu_p) ** 2) / (2 * sp2) - .5
        Si = np.linalg.inv(Sigma); o = np.array([fu, fv]); res = o - (mu_q * A + B)
        _, ld = np.linalg.slogdet(Sigma)
        F = KL + 0.5 * (2 * np.log(2 * np.pi) + ld) + 0.5 * (res @ Si @ res + sq2 * A @ Si @ A)
        s, g = surprisal_and_gap(I, fu, fv, mu_p, sp2, A, B, Sigma, slope, icpt, sq2)
        assert abs(F - (s + g)) < 1e-9, (F, s, g)
    print("self-test passed: F = surprisal + KL(q || posterior) on random Gaussian models")


def main():
    selftest()
    tr = tolerant_load("lifecycle_progress.pkl")
    examples = [e for e in tr.examples if e['cause'] == 'split']
    C = (R.MU_P, R.SIGMA_P2, R.A, R.B, R.SIGMA, R.SLOPE, R.INTERCEPT, R.SIGMA_Q2)
    So = R.SIGMA + R.SIGMA_P2 * np.outer(R.A, R.A)
    print(f"condition number: Sigma {np.linalg.cond(R.SIGMA):.3g},  Sigma+sigma_p^2 A A^T {np.linalg.cond(So):.3g}")

    rows = []
    worst = 0.0
    for e in examples:
        for h in e['history']:
            vals = (h['I'], h['E'], h['fu'], h['fv'])
            if not all(np.isfinite(vals)):
                continue
            s, g = surprisal_and_gap(h['I'], h['fu'], h['fv'], *C)
            F = friston_F(h['I'], h['E'], h['fu'], h['fv'])
            # NOTE: friston_F uses the observed organism's own flux with q; identity holds exactly
            worst = max(worst, abs(F - (s + g)))
            h['surprisal'], h['gap'] = s, g
            rows.append((F, s, g, h.get('area', np.nan)))
    R_ = np.array(rows)
    print(f"{len(R_)} organism-instances; max |F - (surprisal + gap)| = {worst:.3g}")
    F, S, G, A_ = R_.T
    print(f"mean F {F.mean():.3f}   mean surprisal {S.mean():.3f}   mean gap {G.mean():.3f}   "
          f"(gap is {100*G.mean()/F.mean():.1f}% of F by mean)")
    print(f"gap: min {G.min():.4f}  median {np.median(G):.4f}  95th pct {np.percentile(G,95):.4f}  max {G.max():.4f}")
    ok = np.isfinite(A_)
    print(f"instance-level r(surprisal, F) = {np.corrcoef(S, F)[0,1]:.3f};  r(gap, F) = {np.corrcoef(G, F)[0,1]:.3f}")
    if ok.any():
        print(f"instance-level r with area: surprisal {np.corrcoef(S[ok], A_[ok])[0,1]:.3f}, "
              f"F {np.corrcoef(F[ok], A_[ok])[0,1]:.3f}, gap {np.corrcoef(G[ok], A_[ok])[0,1]:.3f}")

    template = np.load("fe_template_1024.npz")["template"]
    res = {k: curve(examples, template, k) for k in ("template", "friston", "surprisal", "gap", "area")}
    common = sorted(set.intersection(*[set(res[k][0].tolist()) for k in res]))
    M = {k: np.array([res[k][1][list(res[k][0]).index(c)] for c in common]) for k in res}
    Zs = {k: z(M[k]) for k in M}
    c = np.array(common)
    print(f"\ndivision-aligned curves, matched bins n={len(common)}")
    for a, b in (("surprisal", "friston"), ("surprisal", "template"), ("surprisal", "area"),
                 ("gap", "friston"), ("gap", "template"), ("gap", "surprisal"), ("gap", "area")):
        print(f"  r({a}, {b}) = {np.corrcoef(Zs[a], Zs[b])[0,1]:+.3f}")
    for k in ("template", "friston", "surprisal", "gap"):
        m = M[k]; pre, post = c < 0, c >= 0
        print(f"  {k:9s} pre-division peak t={c[pre][np.argmax(m[pre])]:.0f}, "
              f"post-division trough t={c[post][np.argmin(m[post])]:.0f}, "
              f"z at t=0 {Zs[k][int(np.argmin(np.abs(c)))]:+.2f}, z at end {Zs[k][-1]:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(c, Zs['template'], 'o-', ms=3, color="steelblue", label=r"$\mathcal{F}_{\mathrm{template}}$")
    ax[0].plot(c, Zs['friston'], 's-', ms=3, color="crimson", label=r"Friston $\mathcal{F}$")
    ax[0].plot(c, Zs['surprisal'], '^-', ms=3, color="darkgreen", label=r"surprisal $-\ln p(o)$")
    ax[0].axvline(0, color="gray", lw=0.8, ls="--")
    ax[0].set_xlabel("time relative to division"); ax[0].set_ylabel("$z$-scored value"); ax[0].legend(frameon=False)
    x, y, se = res['gap']
    ax[1].errorbar(x, y, yerr=2 * se, fmt='o-', ms=3, color="purple")
    ax[1].axvline(0, color="gray", lw=0.8, ls="--")
    ax[1].set_xlabel("time relative to division"); ax[1].set_ylabel(r"gap  $=\mathcal{F}-$surprisal (nats), $\pm2$ SE")
    fig.tight_layout(); fig.savefig("fe_surprisal_1024.png", dpi=150)
    print("saved fe_surprisal_1024.png (send this back, with the printed numbers)")


if __name__ == "__main__":
    main()
