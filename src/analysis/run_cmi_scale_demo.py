"""
Scale sensitivity of the nearest-neighbour CMI estimator (ESM, "CMI Null Calibration and the Scale-Sensitivity Artefact").

On one N-point subsample of the ensemble and of the single trajectory, computes I(I;E | flux) with
  (a) the raw inputs (flux ~1e-4, I and E ~1e-1),
  (b) flux multiplied by 1e4 and nothing else changed,
  (c) all four variables z-scored (the convention used everywhere in the paper),
and the local nearest-neighbour null for (c).
Usage:  python3 run_cmi_scale_demo.py --ens ensemble:1000 [--n-sub 8000] [--n-null 100] [--quick]
"""
import argparse
import numpy as np
from conditional_mi import cmi_ksg
from run_u_enriched_cmi import zscore, local_null
from run_cmi_independent_frames import load
from texout import num, write_macros


def one(d, n_sub, n_null, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d["I"]), min(n_sub, len(d["I"])), replace=False)
    I, E = d["I"][idx], d["E"][idx]
    flux = np.column_stack([d["fu"][idx], d["fv"][idx]])
    raw = cmi_ksg(I, E, flux)
    scaled = cmi_ksg(I, E, flux * 1e4)
    Iz, Ez, Fz = zscore(I)[:, 0], zscore(E)[:, 0], zscore(flux)
    z = cmi_ksg(Iz, Ez, Fz)
    nl = local_null(Iz, Ez, Fz, n_null, 0)
    return raw, scaled, z, nl.mean(), nl.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--ens", default="ensemble:1000")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_sub, a.n_null = 2000, 10
    items = [("ScaleNSub", num(a.n_sub, 0)), ("ScaleNNull", num(a.n_null, 0))]
    for tag, spec in (("Ens", a.ens), ("Traj", "master")):
        r = one(load(spec), a.n_sub, a.n_null)
        print(f"{tag}: raw {r[0]:.4f}  flux*1e4 {r[1]:.4f}  z-scored {r[2]:.4f}  null {r[3]:.4f} (SD {r[4]:.4f})")
        items += [("Scale" + tag + "Raw", num(r[0], 3)), ("Scale" + tag + "Times", num(r[1], 3)),
                  ("Scale" + tag + "Z", num(r[2], 3)), ("Scale" + tag + "GainPercent", num(100 * (r[2] / r[0] - 1) if r[0] else float("nan"), 0)), ("Scale" + tag + "NullMean", num(r[3], 4)),
                  ("Scale" + tag + "NullSD", num(r[4], 4))]
    write_macros("results/cmiscaledemo.tex", "run_cmi_scale_demo.py", items)


if __name__ == "__main__":
    main()
