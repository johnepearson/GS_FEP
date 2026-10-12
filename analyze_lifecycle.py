"""
Analyze the output of run_lifecycle.py: lifespan distribution,
crowding-vs-lifespan, and the template-deviation free energy
lifecycle curve. Run this after run_lifecycle.py has produced
lifecycle_progress.pkl (does not require the ensemble to be finished --
that's only needed for the separate Friston free energy analysis).

Usage:
    python3.12 analyze_lifecycle.py
"""
import numpy as np
from run_lifecycle import LifecycleTracker, MAX_R, N_BINS

EARLY_WINDOW = 100   # time units after birth, for the crowding measurement


def free_energy(profile, template):
    good = np.isfinite(profile) & np.isfinite(template)
    if not good.any():
        return np.nan
    return np.mean((profile[good] - template[good]) ** 2)


def main():
    tr = LifecycleTracker.load("lifecycle_progress.pkl")
    examples = tr.examples
    print(f"Total completed lifecycles: {len(examples)}")

    causes = [e['cause'] for e in examples]
    n_split = causes.count('split')
    n_death = causes.count('death')
    print(f"  ended by division: {n_split}")
    print(f"  ended by death: {n_death}")

    durations = np.array([e['death'] - e['birth'] for e in examples])
    print(f"\nLifespan (all): median={np.median(durations):.0f}  mean={durations.mean():.1f}")
    split_durations = np.array([e['death'] - e['birth'] for e in examples if e['cause'] == 'split'])
    death_durations = np.array([e['death'] - e['birth'] for e in examples if e['cause'] == 'death'])
    if len(split_durations):
        print(f"  ending in division: median={np.median(split_durations):.0f}")
    if len(death_durations):
        print(f"  ending in death: median={np.median(death_durations):.0f}")

    crowd_durations, crowd_nndist = [], []
    for e in examples:
        if e['cause'] != 'split' or not e['history']:
            continue
        birth = e['birth']
        early_vals = [h['nn_dist'] for h in e['history'] if (h['t'] - birth) <= EARLY_WINDOW
                      and np.isfinite(h['nn_dist'])]
        if len(early_vals) < 2:
            continue
        crowd_durations.append(e['death'] - e['birth'])
        crowd_nndist.append(np.mean(early_vals))
    if len(crowd_durations) > 10:
        crowd_durations = np.array(crowd_durations)
        crowd_nndist = np.array(crowd_nndist)
        r = np.corrcoef(crowd_durations, crowd_nndist)[0, 1]
        N = len(crowd_durations)
        se = 1 / np.sqrt(N - 3)
        z = 0.5 * np.log((1 + r) / (1 - r))
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        print(f"\nCrowding (early-window nn_dist) vs split-lifespan:")
        print(f"  N={N}  r={r:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    else:
        print(f"\nNot enough confirmed divisions with early-window data yet "
              f"({len(crowd_durations)}) for the crowding analysis.")

    try:
        d = np.load("fe_template_1024.npz")
        template = d["template"]
        pre_bins = np.arange(-150, 5, 5)
        post_bins = np.arange(0, 155, 5)

        def bin_examples(examples, is_pre):
            binned = {b: [] for b in (pre_bins if is_pre else post_bins)}
            for e in examples:
                if e['cause'] != 'split' or not e['history']:
                    continue
                split_t = e['death'] if is_pre else e['birth']
                for h in e['history']:
                    rel = h['t'] - split_t
                    bins = pre_bins if is_pre else post_bins
                    if rel < bins[0] - 3 or rel > bins[-1] + 3:
                        continue
                    idx = np.argmin(np.abs(bins - rel))
                    if abs(bins[idx] - rel) <= 3:
                        fe = free_energy(h['profile'], template)
                        if np.isfinite(fe):
                            binned[bins[idx]].append(fe)
            return binned

        pre_binned = bin_examples(examples, True)
        post_binned = bin_examples(examples, False)
        print(f"\nTemplate free energy lifecycle (pre-split / post-split):")
        for b in sorted(pre_binned):
            if len(pre_binned[b]) > 3:
                print(f"  t={b:>5} (pre):  F={np.mean(pre_binned[b]):.5f}  n={len(pre_binned[b])}")
        for b in sorted(post_binned):
            if len(post_binned[b]) > 3:
                print(f"  t={b:>5} (post): F={np.mean(post_binned[b]):.5f}  n={len(post_binned[b])}")
    except FileNotFoundError:
        print("\n(fe_template_1024.npz not found -- skipping free energy lifecycle curve)")


if __name__ == "__main__":
    main()
