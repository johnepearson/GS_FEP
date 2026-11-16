"""Autocorrelation function of the population count (and of the mean activator) along the single long trajectory.

Reads master_trajectory_progress.npz (run_master_trajectory.py): f_pop, the number of organisms in each frame, f_mean_v,
the mean activator in each frame, and f_time, the time of each frame.  Frames are SAMPLE steps apart.

For each series it prints the normalised autocorrelation function at a few lags and
  efold    the first lag (in time steps) at which the autocorrelation falls below 1/e,
  zero     the first lag at which it crosses zero,
  tauint   the integrated autocorrelation time in time steps, 1/2 + sum of rho(k) up to the first lag where
           k >= 5 tau_int(k) (Sokal's automatic window),
  neff     the effective number of independent frames, F / (2 tau_int) with tau_int in frames.

Writes results/population_acf.tex with PopEfold, PopZero, PopTauInt, PopNeff for the count.  The mean activator is printed
for comparison only: its autocorrelation oscillates, so the windowed correlation time is not meaningful."""
import numpy as np
from texout import num, write_macros

d = np.load("master_trajectory_progress.npz")
t = np.asarray(d["f_time"], float)
step = float(np.median(np.diff(t)))
F = len(t)


def acf(x, maxlag):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    c = np.fft.irfft(f * np.conj(f))[:maxlag + 1] / np.arange(n, n - maxlag - 1, -1)
    return c / c[0]


def summarise(name, x):
    maxlag = min(F // 2, 400)
    rho = acf(x, maxlag)
    below = np.nonzero(rho < 1 / np.e)[0]
    efold = below[0] * step if len(below) else np.nan
    zero_i = np.nonzero(rho < 0)[0]
    zero = zero_i[0] * step if len(zero_i) else np.nan
    tau = 0.5
    window = maxlag
    for k in range(1, maxlag + 1):
        tau += rho[k]
        if k >= 5 * tau:
            window = k
            break
    neff = F / (2 * tau)
    print(f"{name}: {F} frames, {step:.0f} steps apart; e-folding {efold:.0f} steps, first zero {zero:.0f} steps, "
          f"integrated autocorrelation time {tau * step:.0f} steps (window {window} frames), "
          f"effective independent frames {neff:.0f}")
    print("  autocorrelation at lag (steps): " + "  ".join(f"{k * step:.0f}: {rho[k]:+.3f}" for k in (1, 2, 5, 10, 20, 50) if k <= maxlag))
    return efold, zero, tau * step, neff


p = summarise("population count", d["f_pop"])
v = summarise("mean activator", d["f_mean_v"])
write_macros("results/population_acf.tex", "population_acf.py", [
    ("PopEfold", num(p[0], 1)), ("PopZero", num(p[1], 1)), ("PopTauInt", num(p[2], 1)), ("PopNeff", num(p[3], 1))])
