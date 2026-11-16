"""
How long until organism counts in noise-perturbed copies stop tracking each other?

Each realization: load the settled checkpoint, add independent Gaussian noise (sigma 1e-3) to u and v
(exactly as the production ensemble does), then run --steps steps, recording the organism count
(same segmentation and merge as the ensemble, marker spacing 24) every --every steps. The count just
after the noise (step 0) is recorded too.

Resumable: results are saved after every finished realization to count_decorr.npz. Stopping with
Ctrl-C loses only the realization in progress; running the same command again continues.

Run:   GS_FFT_WORKERS=12 python3 run_count_decorrelation.py --n-real 12 --steps 15000
Then:  python3 analyze_count_decorrelation.py
"""
import argparse, os, time
import numpy as np
from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim
import run_ensemble_merged as M

OUT = "count_decorr.npz"


def count_now(sp):
    proxy = GrayScottSim(n=sp.n, L=sp.L, Du=sp.Du, Dv=sp.Dv, F=sp.F, k=sp.k, dt=sp.dt)
    proxy.u, proxy.v = sp.u, sp.v
    raw_labels, raw_n = proxy.segment_organisms(pad=M.PAD, min_peak_distance=M.MIN_PEAK_DISTANCE)
    if raw_n == 0:
        return 0
    labels, n, _ = M.merge_overlapping_organisms(raw_labels, raw_n, proxy)
    return int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=12)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--seed-base", type=int, default=5000)
    a = ap.parse_args()
    times = np.arange(0, a.steps + 1, a.every)
    if os.path.exists(OUT):
        d = np.load(OUT)
        if not np.array_equal(d["times"], times):
            raise SystemExit(f"{OUT} was made with different --steps/--every; move it aside first")
        counts, seeds = list(d["counts"]), list(d["seeds"])
        print(f"resuming: {len(seeds)} realizations already done")
    else:
        counts, seeds = [], []
    while len(seeds) < a.n_real:
        seed = a.seed_base + len(seeds)
        rng = np.random.default_rng(seed)
        sp = PseudoSpectralGS.from_checkpoint(M.CHECKPOINT_FILE)
        sp.from_checkpoint_fields(sp.u + M.NOISE_SIGMA * rng.standard_normal((sp.n, sp.n)),
                                  sp.v + M.NOISE_SIGMA * rng.standard_normal((sp.n, sp.n)))
        t0 = time.time()
        row = [count_now(sp)]
        for t in times[1:]:
            sp.run(a.every)
            row.append(count_now(sp))
            print(f"seed {seed}  step {t}/{a.steps}  count {row[-1]}  ({time.time()-t0:.0f}s)", flush=True)
        counts.append(row); seeds.append(seed)
        np.savez(OUT, times=times, counts=np.array(counts), seeds=np.array(seeds))
        print(f"saved {len(seeds)}/{a.n_real} realizations", flush=True)


if __name__ == "__main__":
    main()
