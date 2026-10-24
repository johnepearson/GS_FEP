# GS_FEP: code for "Does a Flux-Defined Markov Blanket Screen Self-Replicating Gray–Scott Spots?"

Code for the solvers, the individuation and lineage-tracking pipeline, the conditional mutual information (CMI)
estimator, and the scripts for every figure and table in the paper and its Electronic Supplementary Material (ESM).

## Contents

- `FIGURES.md`: for each figure, and for each number quoted from a small script, the script, the command and the data it reads.
  This is the table of commands for each result. The ESM's reproduction section gives the same information per quantity.
- `MANIFEST.txt`: size and SHA-256 of every file in this folder.
- `requirements.txt`: Python packages.
- `checkpoint_1024_settled.npz`: the settled starting field (1024 × 1024, F = 0.018, k = 0.055) from which every simulation is run.
- `run_batch.py`: reruns the analyses in five batches and writes one log per batch to `rerun_logs/`, so printed numbers can be
  compared with the paper. It re-analyses data files you already have; it does not rerun the simulations.

## Setup

Python 3 (the scripts use only standard scientific packages).

    pip install -r requirements.txt

The 1024 × 1024 simulations use a C pseudo-spectral solver (it halves the number of FFTs per step). It needs FFTW3 and a C
compiler; build it once in this folder:

    gcc -O3 -march=native -fPIC -shared -o libgs_spectral.so gs_spectral_c.c -lfftw3 -lm

Scripts that simulate take `--fast` to use it. Without `libgs_spectral.so` the pure-Python solver (`gray_scott.py`) is used.
Set `GS_FFT_THREADS=1` when running several worker processes in parallel.

## Producing the data

All three production runs start from `checkpoint_1024_settled.npz`:

| Data | Script | Used for |
|---|---|---|
| Single trajectory (500 frames, one every 100 time steps, fields archived to `master_fields/`) | `python3 run_master_trajectory.py --n-frames 500 --fast` | most single-trajectory analyses |
| Ensembles 1–3 (independent realisations, perturbed once from the checkpoint) | `run_ensemble_merged.py`; commands for the three ensembles are in the header of the script | Table 1, Figure 3, the free-energy model |
| Lifecycle tracker (lineages, divisions, deaths) | `python3 run_lifecycle.py --fast --n-steps 10000` (10,000 time units, the length used for the paper; resumes with `--resume`) | free-energy comparison of life-cycle trajectories |

Ensemble runs are split over worker processes (`--n-workers`, `--worker-id`) and resume with `--resume`. Each
realisation is independent, so an interrupted worker loses no completed work.

## Analysing

Each analysis script's docstring says what it computes, which data it reads, and where the result appears in the paper.
`FIGURES.md` gives the command for every figure. Typical order: produce the data above, then
`python3 run_batch.py 1` onwards, and compare the logs with the paper.

A regenerated figure can differ in detail from the published one (a different frame, a different random subsample);
compare the printed numbers with the text.

## Conventions

- `Phi_u`, `Phi_v` are the net diffusive fluxes of `u` and `v` across an organism's boundary. "PC1" is the first principal
  component of the two (the paper's flux regressor); "joint" means both fluxes entered as separate regressors.
- Ensemble worker files are named `ensemble_merged_seed{seed_base}_worker{id}.npz`; seed bases 1000, 9000 and 20000 are
  Ensembles 1, 2 and 3.
- "Seed" means a random-number seed only. The starting state of every run is the settled checkpoint.
