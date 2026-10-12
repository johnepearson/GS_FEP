"""
Nonlinear (CMI) check on the level-set individuation (reviewer request): the main text previously reported only the
linear partial correlation for the level-set method. This runs the paper's z-scored Frenzel-Pompe CMI with the local
nearest-neighbour null on the level-set organisms saved by level_set_from_master.py (levelset_states.npz).

Reports, for single-peak organisms, recovered multi-peak organisms, and the pooled set (N=8000 subsample each, or all
if fewer): MI(I;E), CMI(I;E | flux_u, flux_v), the local-null mean and SD (100 trials), and the null-SD distance.
If conditioning on flux RAISES the dependence above MI(I;E), the collider signature of the watershed analysis is present
under the level-set method too.

    python3 level_set_from_master.py     # once, to write levelset_states.npz (about as long as before)
    python3 run_levelset_cmi.py --jobs 6
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from conditional_mi import cmi_ksg
from run_u_enriched_cmi import zscore, local_null


def one(args):
    name, I, E, B, n_null, seed = args
    Ii, Ei = zscore(I)[:, 0], zscore(E)[:, 0]
    Z = zscore(B)
    mi = cmi_ksg(Ii, Ei, np.zeros((len(Ii), 1)) + np.random.default_rng(seed).normal(size=(len(Ii), 1)) * 1e-6)
    cmi = cmi_ksg(Ii, Ei, Z)
    nl = local_null(Ii, Ei, Z, n_null, seed)
    return name, dict(mi=mi, cmi=cmi, null_mean=nl.mean(), null_sd=nl.std(), n=len(Ii))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="levelset_states.npz")
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.data)
    sets = {}
    for name in ("single", "multi"):
        arr = {k: d[f"{name}_{k}"] for k in ("I", "E", "fu", "fv")}
        sets[name] = arr
    sets["pooled"] = {k: np.concatenate([sets["single"][k], sets["multi"][k]]) for k in ("I", "E", "fu", "fv")}
    rng = np.random.default_rng(a.seed)
    jobs = []
    for name, arr in sets.items():
        g = np.isfinite(arr["I"]) & np.isfinite(arr["E"]) & np.isfinite(arr["fu"]) & np.isfinite(arr["fv"])
        idx = np.where(g)[0]
        idx = rng.choice(idx, min(a.n_sub, len(idx)), replace=False)
        jobs.append((name, arr["I"][idx], arr["E"][idx], np.column_stack([arr["fu"][idx], arr["fv"][idx]]), a.n_null, a.seed))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(one, jobs))
    print("level-set organisms; z-scored CMI in nats, local nearest-neighbour null")
    for name, r in res.items():
        sd = (r["cmi"] - r["null_mean"]) / r["null_sd"]
        print(f"{name:7s} N={r['n']:6d}  MI(I;E)={r['mi']:.4f}  CMI(I;E|flux)={r['cmi']:.4f}  "
              f"ratio={r['cmi']/r['mi']:.2f}  null mean={r['null_mean']:.4f} (SD {r['null_sd']:.4f})  excess={sd:.1f} null SDs")
    np.savez("levelset_cmi.npz", **{f"{n}_{k}": v for n, r in res.items() for k, v in r.items()})


if __name__ == "__main__":
    main()
