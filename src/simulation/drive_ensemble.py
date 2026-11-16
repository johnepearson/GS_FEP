"""Run one ensemble of realisations with several worker processes, then stop.

Usage: python3 drive_ensemble.py One|Two|Three        (run from build/; parameters come from initialize.tex)

Reads Seed<k> (seed base), Runs<k> (number of realisations) and Burnin<k> (burn-in steps), Spacing<k> (steps between the three samples) for k = One, Two or Three.
The number of worker processes is the environment variable JOBS (default 1). It does not change the results: each
realisation has its own seed, so any split of the seeds over workers gives the same data.
Each worker is run_ensemble_merged.py's main(); the output files are ensemble_merged_seed<base>_worker<i>.npz."""
import multiprocessing as mp
import os
import sys

os.environ.setdefault("GS_FFT_THREADS", "1")   # one FFT thread per worker: the workers already use the cores

import run_ensemble_merged as rem
from params import load

P = load("initialize.tex")
which = sys.argv[1]
seed, runs, burnin = P["Seed" + which], P["Runs" + which], P["Burnin" + which]
spacing = P["Spacing" + which]
# Read the settled checkpoint here, in the parent process: that records the dependency on it (the workers are
# separate processes), and it stops a run whose checkpoint was made with different parameters.
import numpy as np
_ck = np.load("checkpoint_1024_settled.npz")
for _key, _name in (("F", "Fval"), ("k", "kval"), ("Du", "Duval"), ("Dv", "Dvval"), ("L", "Lval"), ("n", "gridn"),
                    ("dt", "dtval")):
    if abs(float(_ck[_key]) - float(P[_name])) > 1e-12 * max(1.0, abs(float(P[_name]))):
        raise SystemExit(f"checkpoint_1024_settled.npz has {_key} = {float(_ck[_key])} but initialize.tex has "
                         f"{_name} = {P[_name]}")
jobs = max(1, int(os.environ.get("JOBS", "1")))
jobs = min(jobs, runs)


def worker(i):
    sys.argv = ["run_ensemble_merged.py", "--fast", "--n-realizations", str(runs), "--n-workers", str(jobs),
                "--worker-id", str(i), "--seed-base", str(seed), "--burn-in-steps", str(burnin),
                "--sample-spacing-steps", str(spacing)]
    rem.main()


if __name__ == "__main__":
    print(f"ensemble {which}: seed base {seed}, {runs} realisations, burn-in {burnin} steps, sample spacing {spacing} steps, {jobs} workers")
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker, args=(i,)) for i in range(jobs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        raise SystemExit("a worker failed")
