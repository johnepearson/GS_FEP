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
wins = []
for s in range(0, T - W + 1, W):
    X = C[:, s:s + W]
    rr_, ee_ = mean_pair_r(X), 2 * boot(X)
    wins.append((s, rr_, ee_))
    print(f"{t[s]:>6}-{t[s+W-1]:<7} {X.mean():>11.1f} {X.std(axis=0).mean():>15.2f} "
          f"{rr_:>9.3f} ± {ee_:.3f}")
from params import load
from texout import num, write_macros
burn = load("initialize.tex")["BurninOne"]
nan = float("nan")
at = [w for w in wins if t[w[0]] >= burn]
first = at[0] if at else (wins[-1] if wins else (0, nan, nan))
zero = [w for w in wins if abs(w[1]) <= w[2]]
WORDS = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]
win_items = []
for k, w in enumerate(WORDS):
    if k < len(wins):
        st = wins[k][0]
        win_items += [(f"CountWin{w}R", num(wins[k][1], 3)), (f"CountWin{w}Err", num(wins[k][2], 3)),
                      (f"CountWin{w}Start", num(t[st], 0)), (f"CountWin{w}End", num(t[st + W - 1], 0))]
    else:
        win_items += [(f"CountWin{w}{q}", "nan") for q in ("R", "Err", "Start", "End")]
sd_all = C.std(axis=0)
win_items += [("CountCheckpoints", num(T, 0)), ("CountSpacing", num(t[1] - t[0], 0) if T > 1 else "nan"),
              ("CountSDStart", num(sd_all[0], 1)), ("CountSDLate", num(sd_all[-W:].mean(), 1))]
write_macros("results/countdecorran.tex", "analyze_count_decorrelation.py", win_items +
             [("CountRAtBurnin", num(first[1], 3)), ("CountRAtBurninErr", num(first[2], 3)),
              ("CountWindowStart", num(t[first[0]], 0)),
              ("CountZeroStep", num(t[zero[0][0]], 0) if zero else "nan")])
print("\ncount at each checkpoint (mean, SD across runs):")
for k in range(T):
    print(f"{t[k]:>7} {C[:, k].mean():>8.1f} {C[:, k].std():>6.2f}")
