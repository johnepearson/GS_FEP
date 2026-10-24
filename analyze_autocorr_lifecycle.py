"""Autocorrelation times (lag where the pooled, lineage-wise ACF first drops below 1/e) of I, E, Phi_u, Phi_v from the merged-organism
lifecycle tracker (lifecycle_progress.pkl). These are the autocorrelation times quoted in main text §3.1.

    python3 analyze_autocorr_lifecycle.py

History entries are every 5 time units (flux) and every 20 (I, E; the others are NaN). For each lineage the series is demeaned; the ACF at lag k is
sum over lineages of products of demeaned values over all finite pairs k steps apart, divided by the same pooled sum at lag 0 restricted to the same
pairs' variance (pooled variance of finite values). tau_1/e is found by linear interpolation at the first crossing.
"""
import numpy as np
from run_lifecycle import LifecycleTracker, SAMPLE_EVERY

tr = LifecycleTracker.load("lifecycle_progress.pkl")
series = [e["history"] for e in tr.examples if e.get("history")]
series += [h for h in getattr(tr, "history", {}).values() if h]
print(f"{len(series)} lineage histories; sampling every {SAMPLE_EVERY} time units")
MAXLAG = 60  # steps


def acf(key):
    num = np.zeros(MAXLAG + 1); den = np.zeros(MAXLAG + 1)
    for h in series:
        t = np.array([m["t"] for m in h], dtype=float)
        x = np.array([m[key] for m in h], dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 3:
            continue
        x = x - np.nanmean(x)
        step = np.round((t - t[0]) / SAMPLE_EVERY).astype(int)
        grid = np.full(step.max() + 1, np.nan)
        grid[step] = x
        n = len(grid)
        for k in range(0, min(MAXLAG, n - 1) + 1):
            a, b = grid[: n - k], grid[k:]
            g = np.isfinite(a) & np.isfinite(b)
            num[k] += np.sum(a[g] * b[g])
            den[k] += np.sum(a[g] ** 2 + b[g] ** 2) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


for key in ("I", "E", "fu", "fv"):
    r = acf(key)
    lags = np.where(np.isfinite(r))[0]          # I and E exist only every 4th lag (20 time units)
    lags = lags[np.insert(r[lags] < 1 / np.e, 0, False)[:-1] | True]  # keep all finite lags
    rr = r[lags]
    below = np.where(rr < 1 / np.e)[0]
    if len(below) == 0 or below[0] == 0:
        print(f"{key:3s}: ACF never drops below 1/e within {MAXLAG*SAMPLE_EVERY} time units")
        continue
    j = below[0]
    k0, k1 = lags[j - 1], lags[j]
    tau = (k0 + (rr[j - 1] - 1 / np.e) / (rr[j - 1] - rr[j]) * (k1 - k0)) * SAMPLE_EVERY
    print(f"{key:3s}: tau_1/e = {tau:.0f} time units   (ACF at lags " + ", ".join(f"{int(l)*SAMPLE_EVERY}: {x:.2f}" for l, x in zip(lags[:9], rr[:9])) + ")")
