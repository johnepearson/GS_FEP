"""Reads count_decorr.npz (from run_count_decorrelation.py). For each window of checkpoints prints the mean
count, the spread across realizations, and the mean pairwise Pearson correlation of the realizations' count
histories inside that window (the statistic behind the supplement's r = 0.349 for 3 checkpoints), with a
2-standard-error bootstrap over realizations. Also prints the count at every checkpoint (mean, SD)."""
import itertools
import numpy as np

d = np.load("count_decorr.npz")
t, C = d["times"], d["counts"].astype(float)
R, T = C.shape
print(f"{R} realizations, {T} checkpoints, spacing {t[1]-t[0]} steps")
W = 8
rng = np.random.default_rng(0)


def mean_pair_r(X):
    r = []
    for i, j in itertools.combinations(range(X.shape[0]), 2):
        if X[i].std() > 0 and X[j].std() > 0:
            r.append(np.corrcoef(X[i], X[j])[0, 1])
    return np.mean(r)


def boot(X, n=300):
    v = [mean_pair_r(X[rng.integers(0, X.shape[0], X.shape[0])]) for _ in range(n)]
    return np.nanstd(v)


print(f"\nwindows of {W} checkpoints (non-overlapping)")
print(f"{'steps':>14} {'mean count':>11} {'SD across runs':>15} {'mean pair r':>18}")
for s in range(0, T - W + 1, W):
    X = C[:, s:s + W]
    print(f"{t[s]:>6}-{t[s+W-1]:<7} {X.mean():>11.1f} {X.std(axis=0).mean():>15.2f} "
          f"{mean_pair_r(X):>9.3f} ± {2*boot(X):.3f}")
print("\ncount at each checkpoint (mean, SD across runs):")
for k in range(T):
    print(f"{t[k]:>7} {C[:, k].mean():>8.1f} {C[:, k].std():>6.2f}")
