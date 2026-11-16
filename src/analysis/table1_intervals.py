"""Confidence intervals of Table 1, and its Trajectory row.

For each ensemble (seed bases SeedOne, SeedTwo, SeedThree of initialize.tex) and for the single trajectory
(master_trajectory_progress.npz) this computes the PC1 partial correlation of I and E and two 95% intervals:
  naive      Fisher interval with the number of organism-instances N (treats them as independent), one control variable;
  resampled  2000 bootstrap replicates that resample whole clusters (realisations for the ensembles, frames for the
             trajectory), with the PC1 direction of the flux fixed from the pooled data (random seed 0).

Run from build/.  Writes results/table1_intervals.tex with, for k in One, Two, Three:
   Ens<k>NaiveLo, Ens<k>NaiveHi, Ens<k>BootLo, Ens<k>BootHi   (partial r)
   Ens<k>MargLo, Ens<k>MargHi (marginal r) and Ens<k>ScrLo, Ens<k>ScrHi (screened %), from the same bootstrap resamples
and for the trajectory:
   TrajN, TrajMarginal, TrajPartial, TrajScreened, TrajNaiveLo, TrajNaiveHi, TrajBootLo, TrajBootHi
(marginal_screened_ensembles.py writes the N, marginal r, partial r and screened fraction of the ensembles.)"""
import numpy as np
from analyze_ensemble_pc1 import load, pcorr_moments
from partial_corr_utils import pc1_partial_corr
from params import load as load_params
from texout import num, write_macros

P = load_params("initialize.tex")
N_BOOT = 2000


def fisher(r, n, k=1):
    se = 1 / np.sqrt(max(n - 3 - k, 1))
    z = np.arctanh(r)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def analyse(I, E, fu, fv, cluster):
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    I, E, fu, fv, cluster = I[good], E[good], fu[good], fv[good], cluster[good]
    r, n, _, (mean, std, v1) = pc1_partial_corr(I, E, fu, fv, return_direction=True)
    marginal = np.corrcoef(I, E)[0, 1]
    pc = ((np.column_stack([fu, fv]) - mean) / std) @ v1
    X = np.column_stack([I, E, pc])
    ids = np.unique(cluster)
    cn = np.zeros(len(ids)); S = np.zeros((len(ids), 3)); Q = np.zeros((len(ids), 3, 3))
    for k, c in enumerate(ids):
        x = X[cluster == c]
        cn[k] = len(x); S[k] = x.sum(0); Q[k] = x.T @ x
    within = np.array([pcorr_moments(cn[k], S[k], Q[k]) for k in range(len(ids))])
    rng = np.random.default_rng(0)
    boot = np.empty(N_BOOT); bmarg = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(ids), len(ids))
        nb, Sb, Qb = cn[idx].sum(), S[idx].sum(0), Q[idx].sum(0)
        boot[b] = pcorr_moments(nb, Sb, Qb)
        C = Qb / nb - np.outer(Sb / nb, Sb / nb)
        bmarg[b] = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
    bscr = 100 * (1 - boot ** 2 / bmarg ** 2)
    return dict(within=within, cn=cn, margboot=tuple(np.percentile(bmarg, [2.5, 97.5])), scrboot=tuple(np.percentile(bscr, [2.5, 97.5])), n=n, marginal=marginal, partial=r, screened=100 * (1 - r ** 2 / marginal ** 2),
                naive=fisher(r, n), boot=tuple(np.percentile(boot, [2.5, 97.5])))


def show(name, a):
    print(f"{name}: N={a['n']:,}  marginal r={a['marginal']:+.4f}  PC1 partial r={a['partial']:+.4f}  "
          f"screened={a['screened']:.1f}%  naive CI [{a['naive'][0]:+.4f}, {a['naive'][1]:+.4f}]  "
          f"resampled CI [{a['boot'][0]:+.4f}, {a['boot'][1]:+.4f}]  "
          f"marginal CI [{a['margboot'][0]:+.4f}, {a['margboot'][1]:+.4f}]  screened CI [{a['scrboot'][0]:.1f}, {a['scrboot'][1]:.1f}]%")


macros = []
screened_all = []
partials = []
for word in ("One", "Two", "Three"):
    seed = P["Seed" + word]
    out = load(seed)
    if out is None:
        print(f"seed {seed}: no files found")
        continue
    I, E, fu, fv, rid = out[:5]
    a = analyse(I, E, fu, fv, rid)
    show(f"ensemble {word} (seed {seed})", a)
    screened_all.append(a["screened"])
    partials.append(a["partial"])
    w = a["within"]
    macros += [(f"Ens{word}RsqMarginal", num(a["marginal"] ** 2, 3)), (f"Ens{word}RsqPartial", num(a["partial"] ** 2, 3))]
    macros += [(f"Ens{word}SameSign", num(int(np.sum(np.sign(w) == np.sign(a["partial"]))), 0)),
               (f"Ens{word}Realisations", num(len(w), 0)), (f"Ens{word}RealMean", num(w.mean(), 4)),
               (f"Ens{word}RealMedian", num(np.median(w), 4)), (f"Ens{word}RealSD", num(w.std(), 3)),
               (f"Ens{word}RealQOne", num(np.percentile(w, 25), 4)), (f"Ens{word}RealQThree", num(np.percentile(w, 75), 4)),
               (f"Ens{word}PerRealN", num(a["cn"].mean(), 1)), (f"Ens{word}PerRealNSD", num(a["cn"].std(), 1))]
    macros += [(f"Ens{word}NaiveLo", num(a["naive"][0], 6)), (f"Ens{word}NaiveHi", num(a["naive"][1], 6)),
               (f"Ens{word}BootLo", num(a["boot"][0], 6)), (f"Ens{word}BootHi", num(a["boot"][1], 6)),
               (f"Ens{word}MargLo", num(a["margboot"][0], 6)), (f"Ens{word}MargHi", num(a["margboot"][1], 6)),
               (f"Ens{word}ScrLo", num(a["scrboot"][0], 3)), (f"Ens{word}ScrHi", num(a["scrboot"][1], 3))]

import os
if os.path.exists("master_trajectory_progress.npz"):
    d = np.load("master_trajectory_progress.npz")
    a = analyse(d["I"], d["E"], d["fu"], d["fv"], d["frame_id"])
    show("trajectory", a)
    screened_all.append(a["screened"])
    macros += [("TrajRsqMarginal", num(a["marginal"] ** 2, 3)), ("TrajRsqPartial", num(a["partial"] ** 2, 3))]
    macros += [("TrajN", num(a["n"], 0)), ("TrajMarginal", num(a["marginal"], 6)), ("TrajPartial", num(a["partial"], 6)),
               ("TrajScreened", num(a["screened"], 3)), ("TrajNaiveLo", num(a["naive"][0], 6)), ("TrajNaiveHi", num(a["naive"][1], 6)),
               ("TrajBootLo", num(a["boot"][0], 6)), ("TrajBootHi", num(a["boot"][1], 6)),
               ("TrajMargLo", num(a["margboot"][0], 6)), ("TrajMargHi", num(a["margboot"][1], 6)),
               ("TrajScrLo", num(a["scrboot"][0], 3)), ("TrajScrHi", num(a["scrboot"][1], 3))]
else:
    print("trajectory: master_trajectory_progress.npz not found")
macros += [("EnsSpread", num(max(partials) - min(partials), 3) if partials else "nan"), ("ScreenedMin", num(min(screened_all), 3) if screened_all else "nan"), ("ScreenedMax", num(max(screened_all), 3) if screened_all else "nan")]
have = {k for k, _ in macros}
for word in ("One", "Two", "Three"):
    for q in ("RsqMarginal", "RsqPartial", "SameSign", "Realisations", "RealMean", "RealMedian", "RealSD", "RealQOne", "RealQThree", "PerRealN", "PerRealNSD"):
        if f"Ens{word}{q}" not in have:
            macros.append((f"Ens{word}{q}", "nan"))
write_macros("results/table1_intervals.tex", "table1_intervals.py", macros)
