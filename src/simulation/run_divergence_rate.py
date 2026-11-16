"""
Divergence-rate check of the chaotic regime (supplement, "Lyapunov-exponent-proxy consistency", and its figure).

If the character of the chaos changed over a long trajectory, the rate at which two noise-perturbed copies of the same
state separate would differ between widely separated reference times.  This script advances the settled 1024x1024
checkpoint, saves the state at each reference time (default 10,000, 319,000 and 469,000 steps after the checkpoint),
and from each reference branches two copies perturbed by independent Gaussian noise (sigma = 1e-3, as in the ensemble
runs), runs both for 6000 steps and records mean |v_a - v_b| every 100 steps.  The growth rate is the slope of
ln(mean |v_a - v_b|) against time over 100-800 steps.  It also prints the value the difference approaches for two
INDEPENDENT draws from the frame's own v distribution (mean |v - v'| over random pixel pairs).

The long advance dominates the run time (about 470,000 steps; about 41 minutes with the C solver on 12 cores).
Reference states are saved to divergence_refs.npz, so an interrupted run resumes and the branching can be repeated
in a few minutes (with the saved states, only the branching runs).  The printed summary includes the step at which
each pair first reaches 90% of the independent-draw value.

Usage (from local_package; use the C solver):
    python3 run_divergence_rate.py --fast
    python3 run_divergence_rate.py --fast --ref-times 10000 100000 200000      # shorter
Output: divergence_rate_results.npz and lyapunov_comparison_fixed.png
"""
import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOISE_SIGMA = 1e-3


def pair_mean_abs_diff(v):
    """E|v - v'| for two independent draws from the pixel distribution of v (exact, via the sorted values)."""
    x = np.sort(v.ravel())
    n = len(x)
    return 2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2


def get_solver(fast):
    if fast:
        from gray_scott_spectral_c import PseudoSpectralGSFast as S
    else:
        from gray_scott_spectral import PseudoSpectralGS as S
    return S


def uv_of(sim):
    return sim.get_uv() if hasattr(sim, "get_uv") else (sim.u, sim.v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint_1024_settled.npz")
    ap.add_argument("--ref-times", type=int, nargs="+", default=[10000, 319000, 469000])
    ap.add_argument("--branch-steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--fit-max", type=int, default=800)
    ap.add_argument("--noise", type=float, default=NOISE_SIGMA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="use the C solver")
    ap.add_argument("--refs-file", default="divergence_refs.npz")
    ap.add_argument("--out", default="lyapunov_comparison_fixed.png")
    a = ap.parse_args()
    S = get_solver(a.fast)
    times = sorted(a.ref_times)

    # ---- reference states
    refs = {}
    if os.path.exists(a.refs_file):
        d = np.load(a.refs_file)
        refs = {int(t): (d[f"u{t}"], d[f"v{t}"]) for t in times if f"u{t}" in d.files}
        print(f"resuming: reference states already saved for {sorted(refs)}")
    todo = [t for t in times if t not in refs]
    if todo:
        sim = S.from_checkpoint(a.checkpoint)
        pos = 0
        t0 = time.time()
        for t in todo:
            while pos < t:
                chunk = min(1000, t - pos)
                sim.run(chunk)
                pos += chunk
                if pos % 10000 == 0:
                    print(f"  advanced to step {pos:,} of {todo[-1]:,}  ({time.time()-t0:.0f}s)", flush=True)
            u, v = uv_of(sim)
            refs[t] = (np.array(u), np.array(v))
            np.savez(a.refs_file, **{f"{k}{tt}": refs[tt][i] for tt in refs for i, k in enumerate("uv")})
            print(f"saved reference state at step {t:,}")
        n_grid = sim.n
    else:
        n_grid = refs[times[0]][0].shape[0]

    # ---- branch two perturbed copies from each reference
    rng = np.random.default_rng(a.seed)
    checkpoints = np.arange(a.every, a.branch_steps + 1, a.every)
    out = {}
    for t in times:
        u0, v0 = refs[t]
        sims = []
        for _ in range(2):
            s = S.from_checkpoint(a.checkpoint)
            s.from_checkpoint_fields(u0 + a.noise * rng.standard_normal(u0.shape),
                                     v0 + a.noise * rng.standard_normal(v0.shape))
            sims.append(s)
        diffs = []
        for _ in checkpoints:
            for s in sims:
                s.run(a.every)
            va, vb = uv_of(sims[0])[1], uv_of(sims[1])[1]
            diffs.append(np.mean(np.abs(va - vb)))
        diffs = np.array(diffs)
        sel = checkpoints <= a.fit_max
        slope, icpt = np.polyfit(checkpoints[sel], np.log(diffs[sel]), 1)
        bench = pair_mean_abs_diff(v0)
        hit = np.where(diffs >= 0.9 * bench)[0]
        t90 = int(checkpoints[hit[0]]) if len(hit) else None
        out[t] = dict(diffs=diffs, rate=slope, icpt=icpt, bench=bench, t90=t90)
        print(f"t_ref={t:>7,}: rate {slope:.5f} per step (fit {checkpoints[sel][0]}-{checkpoints[sel][-1]});  "
              f"|va-vb| at {checkpoints[-1]} steps = {diffs[-1]:.4f}  "
              f"(independent-draw benchmark {out[t]['bench']:.4f}, ratio {diffs[-1]/out[t]['bench']:.2f});  "
              f"90% of benchmark first reached at step {t90 if t90 else 'never'}", flush=True)
        print("    ratio to benchmark every 500 steps: " + ", ".join(
            f"{c}:{d/bench:.2f}" for c, d in zip(checkpoints, diffs) if c % 500 == 0), flush=True)
    rates = np.array([out[t]["rate"] for t in times])
    print(f"\nrates: {', '.join(f'{r:.5f}' for r in rates)};  spread (SD/mean) {rates.std()/rates.mean()*100:.1f}%, "
          f"range {(rates.max()-rates.min())/rates.mean()*100:.1f}%")
    np.savez("divergence_rate_results.npz", times=np.array(times), steps=checkpoints,
             diffs=np.array([out[t]["diffs"] for t in times]), rates=rates,
             bench=np.array([out[t]["bench"] for t in times]), noise=a.noise,
             t90=np.array([out[t]["t90"] or -1 for t in times]))

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    cols = ["steelblue", "darkorange", "green", "crimson", "purple"]
    for c, t in zip(cols, times):
        r = out[t]
        ax.semilogy(checkpoints, r["diffs"], "o-", color=c, label=f"t={t}: rate={r['rate']:.5f}")
        sel = checkpoints <= a.fit_max
        ax.semilogy(checkpoints[sel], np.exp(r["icpt"] + r["rate"] * checkpoints[sel]), "--", color=c, alpha=0.5)
        ax.axhline(r["bench"], color=c, lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("time since branching (relative)")
    ax.set_ylabel("mean |v_a - v_b|  (log scale)")
    ax.set_title("Divergence rate from widely separated reference points (dotted: independent-draw value)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
