# Gray–Scott self-replicating spots and the Free Energy Principle: analysis code

Code for "Does a Flux-Defined Markov Blanket Screen Self-Replicating Gray–Scott Spots? A Computational Test of a Blanket Condition from the Free Energy Principle" (J. E. Pearson), submitted to J. R. Soc. Interface. https://github.com/johnepearson/GS_FEP

It contains the solver, the production scripts that generate the data, and every analysis script cited in the paper and its
electronic supplementary material (ESM). The ESM table "Reproducibility" lists, for each claim, the script and command.
`FIGURES.md` lists which script makes which figure. `MANIFEST.txt` gives the size and SHA-256 of every file.

## Requirements

Python 3 with `numpy scipy scikit-image scikit-learn matplotlib` (see `requirements.txt`). `tigramite` and `numba` are needed
only for the estimator cross-check (`run_cmi_crosscheck.py`). The fast solver needs FFTW and a C compiler:

    gcc -O3 -march=native -fPIC -shared -o libgs_spectral.so gs_spectral_c.c -lfftw3 -lm

The scripts that take `--fast` use this solver; without it they use the slower numpy solver. The number of FFT threads is set
by `GS_FFT_THREADS`. The results in the paper were produced on a 12-core machine.

## Model and numerics

Gray–Scott equations with F = 0.018, k = 0.055, Du = 2e-5, Dv = 1e-5, on a periodic 1024 x 1024 grid (L = 2.5), pseudo-spectral
with a first-order exponential time-differencing step, dt = 1. `checkpoint_1024_settled.npz` is the settled starting field used
by every simulation. `check_c_solver.py` compares the C solver with the numpy solver.

## Workflow

1. **Generate data** (long runs; skip if you already have the data files):
   - `run_master_trajectory.py --n-frames 500 --fast`: one long trajectory, sampled every 100 steps. Writes
     `master_trajectory_progress.npz` and the archived fields in `master_fields/` (about 4 GB).
   - `run_ensemble_merged.py` (12 worker processes, see its header): the 1000-realization ensembles (seed bases 1000 and 9000).
   - `run_lifecycle.py` (add `--raw` for unmerged regions): the life-cycle tracker, writes `lifecycle_progress.pkl`.
   - Other simulations: `run_resolution_2x2_merged.py`, `run_3var_shell_data.py`, `run_divergence_rate.py`,
     `run_random_disks.py`, `run_count_decorrelation.py`, `run_min_distance_sweep.py`.
2. **Analyse.** Everything in the paper's tables and figures is computed from the data files above by the `analyze_*`, `run_*cmi*`,
   `rebuild_*`, `plot_*` and `check_*` scripts. `run_batch.py` runs them in five batches and writes one log per batch to
   `rerun_logs/`, so the printed numbers can be compared with the paper:

       python3 run_batch.py 1 --list     # show what batch 1 would run
       python3 run_batch.py 1            # fast analyses
       python3 run_batch.py 2            # life-cycle and free-energy analyses
       python3 run_batch.py 3            # heavy nonlinear-CMI jobs (uses 12 cores)
       python3 run_batch.py 4            # band-sweep seeds 1 and 2, placebo-disk sweep, Tigramite cross-check
       python3 run_batch.py 5            # full-population collider CMI, split-half check, depth sensitivity, merge rounds

   The batches do not rerun the simulations. They move cached results aside (suffix `.before_rerun`) so the CMI jobs recompute.

## Notes on reproducing the numbers

- The nonlinear CMI uses N = 8000 subsamples. Values such as the unconditioned MI(I;E) vary by about +-0.01 nats between
  subsamples, so a script run with a different seed gives nearby, not identical, values. The paper states which seed or how many
  seeds each number uses.
- The ensembles and the life-cycle tracker are stochastic simulations; rerunning them gives statistically equivalent, not
  identical, data.
- Two figures have no script: `split_duration_histogram.png` came from an exploratory run on an earlier version of the tracker
  that was not kept, and `schematic_combined.png` is a snapshot with hand-drawn annotations. Both are described in the ESM.

- The three-variable model's finite-difference cross-check of the spectral solver is not archived; the paper's claim there is
  stated without a script.
- Only the settled starting field (`checkpoint_1024_settled.npz`, 16 MB) is included. The generated data files (the master
  trajectory, ensembles and archived fields, several GB) are not in the repository; the commands above regenerate them.
  Shell commands assume the working directory is the repository folder.

## Citation

J. E. Pearson, "Does a Flux-Defined Markov Blanket Screen Self-Replicating Gray–Scott Spots? A Computational Test of a Blanket
Condition from the Free Energy Principle", J. R. Soc. Interface (submitted). Repository: https://github.com/johnepearson/GS_FEP

## Licence

MIT; see `LICENSE`.
