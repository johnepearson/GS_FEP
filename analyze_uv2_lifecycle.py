"""
Entropy-production comparison through the division cycle at 1024^2.

Follows the reaction-rate prefactor mean(u v^2) (per-area mean over each
organism's own merged mask) through the life cycle, aligned to the moment
of division and averaged over the N confirmed divisions, next to the two
free-energy quantities (F_template and the Friston F) built exactly as in
rebuild_fe_comparison_1024.py. All three curves are z-scored so only their
shapes are compared. Reports the matched-bin correlations and the timing of
each curve's pre-division peak and post-division trough.

Needs lifecycle_progress.pkl written by the UPDATED run_lifecycle.py (it
records uv2 per organism per frame) and fe_template_1024.npz.

Usage:
    python3 analyze_uv2_lifecycle.py
"""
import numpy as np
# the pickle was written by run_lifecycle.py running as __main__, so the class must be visible here
from run_lifecycle import LifecycleTracker  # noqa: F401
from rebuild_fe_comparison_1024 import (tolerant_load, free_energy_template, friston_F,
                                        PRE_BINS, POST_BINS)


def curve(examples, template, kind):
    out = {}
    zero = {True: [], False: []}
    for is_pre in (True, False):
        bins = PRE_BINS if is_pre else POST_BINS
        keep = (lambda b: b <= 0) if is_pre else (lambda b: b >= 0)
        for b in bins:
            out.setdefault(float(b), [])
        for e in examples:
            if not e['history']:
                continue
            split_t = e['death'] if is_pre else e['birth']
            for h in e['history']:
                rel = h['t'] - split_t
                if rel < bins[0] - 3 or rel > bins[-1] + 3:
                    continue
                idx = np.argmin(np.abs(bins - rel))
                if abs(bins[idx] - rel) > 3 or not keep(bins[idx]):
                    continue
                if kind == "template":
                    val = free_energy_template(h['profile'], template)
                elif kind == "friston":
                    val = friston_F(h['I'], h['E'], h['fu'], h['fv'])
                else:
                    val = h.get(kind, np.nan)
                if np.isfinite(val):
                    if bins[idx] == 0:
                        zero[is_pre].append(val)
                    else:
                        out[float(bins[idx])].append(val)
    # bin 0: use the pre-division side if it has data, else the post-division side (never mixed)
    out[0.0] = zero[True] if len(zero[True]) > 3 else zero[False]
    xs = sorted(b for b in out if len(out[b]) > 3)
    return (np.array(xs), np.array([np.mean(out[b]) for b in xs]),
            np.array([np.std(out[b]) / np.sqrt(len(out[b])) for b in xs]))


def z(x):
    return (x - x.mean()) / x.std()


def main():
    tr = tolerant_load("lifecycle_progress.pkl")
    examples = [e for e in tr.examples if e['cause'] == 'split']
    have = sum('uv2' in h for e in examples for h in e['history'])
    tot = sum(len(e['history']) for e in examples)
    print(f"confirmed divisions N={len(examples)} (expect 695); uv2 recorded in {have}/{tot} history entries")
    have_sum = sum('uv2_sum' in h for e in examples for h in e['history'])
    print(f"uv2_sum/area recorded in {have_sum}/{tot} entries")
    if have == 0 or have_sum == 0:
        raise SystemExit("no uv2 in this pickle: rerun the UPDATED run_lifecycle.py first")
    template = np.load("fe_template_1024.npz")["template"]

    res = {k: curve(examples, template, k) for k in ("template", "friston", "uv2", "uv2_sum", "area")}
    common = sorted(set.intersection(*[set(res[k][0].tolist()) for k in res]))
    M = {k: np.array([res[k][1][list(res[k][0]).index(c)] for c in common]) for k in res}
    Z = {k: z(M[k]) for k in M}
    print(f"matched bins n={len(common)}")
    print(f"  r(F_template, Friston F) = {np.corrcoef(Z['template'], Z['friston'])[0,1]:.3f}")
    print(f"  r(uv2, F_template)       = {np.corrcoef(Z['uv2'], Z['template'])[0,1]:.3f}")
    print(f"  r(uv2, Friston F)        = {np.corrcoef(Z['uv2'], Z['friston'])[0,1]:.3f}")
    print(f"  r(uv2_sum, F_template)   = {np.corrcoef(Z['uv2_sum'], Z['template'])[0,1]:.3f}")
    print(f"  r(uv2_sum, Friston F)    = {np.corrcoef(Z['uv2_sum'], Z['friston'])[0,1]:.3f}")
    print(f"  r(area, F_template)      = {np.corrcoef(Z['area'], Z['template'])[0,1]:.3f}")
    print(f"  r(uv2_sum, area)         = {np.corrcoef(Z['uv2_sum'], Z['area'])[0,1]:.3f}")
    c = np.array(common)
    for k in ("template", "friston", "uv2", "uv2_sum", "area"):
        m = M[k]
        pre, post = c < 0, c >= 0
        print(f"  {k:9s} pre-division peak at t={c[pre][np.argmax(m[pre])]:.0f}, "
              f"post-division trough at t={c[post][np.argmin(m[post])]:.0f}, "
              f"first bin {m[0]:.5g}, at t=0 {m[int(np.argmin(np.abs(c)))]:.5g}, last bin {m[-1]:.5g}")
    # when does each curve first rise above its own post-division trough-to-plateau midpoint? keep simple:
    for k in ("template", "friston", "uv2", "uv2_sum", "area"):
        m = M[k]
        print(f"  {k:9s} z-score at t=0: {Z[k][int(np.argmin(np.abs(c)))]:+.2f}   at the end: {Z[k][-1]:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(c, Z['template'], 'o-', ms=3, color="steelblue", label=r"$\mathcal{F}_{\mathrm{template}}$")
    ax.plot(c, Z['friston'], 's-', ms=3, color="crimson", label=r"Friston $\mathcal{F}$")
    ax.plot(c, Z['uv2'], '^-', ms=3, color="darkgreen", label=r"mean$(uv^2)$")
    ax.plot(c, Z['uv2_sum'], 'v-', ms=3, color="darkorange", label=r"total $\sum uv^2$ (whole organism)")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("time relative to division")
    ax.set_ylabel("$z$-scored value")
    ax.set_title(f"$1024\\times1024$, $N={len(examples)}$ confirmed divisions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("fe_four_curves_1024.png", dpi=150)
    print("saved fe_four_curves_1024.png (send this back, with the printed numbers)")


if __name__ == "__main__":
    main()
