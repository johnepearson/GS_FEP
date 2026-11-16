# GS_FEP: code for "Does a Flux-Defined Markov Blanket Screen Self-Replicating Gray–Scott Spots?"

Code for the solvers, the individuation and lineage-tracking pipeline, the conditional mutual information (CMI)
estimator, and the scripts for every figure and table in the paper and its Electronic Supplementary Material (ESM).

**Which branch is this?** The `main` branch is the code as submitted with the paper (flat layout, run by hand). The
`restructure` branch is the same analysis reorganised so that one Makefile rebuilds the numbers, tables and figures,
with a reproducible environment and automated tests. The scientific content is the same; the commands and the folder
layout are different.

## Try it (about 10 to 15 minutes in all, almost all of it step 3)

1. Install Miniforge (a small conda) and create the environment. Step-by-step instructions are in `setup/README.md`.

       git clone -b restructure https://github.com/johnepearson/GS_FEP.git
       cd GS_FEP
       conda env create -f setup/environment.yml
       conda activate gsfep
       bash setup/check_setup.sh        # every line should say OK

2. Run the tests (under a minute; they do not run a simulation):

       make test

3. Run the scaled-down pipeline end to end (about 10 minutes on a 12-core Mac plugged in, up to 15 on battery):

       make QUICK=1

   This runs every simulation and analysis stage at a tiny size, so it checks that the whole chain runs. **The numbers
   it prints are not the paper's numbers** (a few frames instead of hundreds, for example). Results and logs go to
   `build_quick/`, which is not kept in git; `make clean-quick` removes it.

The full-size run takes much longer and is what produced the paper's numbers. The Makefile builds the C solver
(`src/simulation/libgs_spectral.so`) automatically, using the FFTW library that the conda environment provides.
Set `GS_FFT_THREADS=1` when running several worker processes in parallel.

A regenerated figure can differ in detail from the published one (a different frame, a different random subsample);
compare the printed numbers with the text.

## The full run and the paper

The first 30 lines of `Makefile` list every option. The main ones:

    make -n FULL=1        show what would run, and run nothing
    make FULL=1           the full-size run: every simulation and analysis at the size used for the paper,
                          writing build/results.tex (about 30 hours, to within roughly 25%, on a 12-core Mac; make uses all cores, see JOBS in the Makefile)
    make FULL=1 pdfs      the same, then build build/paper.pdf and build/supplement.pdf
    make QUICK=1 pdfs     the scaled-down chain plus PDFs in build_quick/ (the numbers in them are not the paper's)
    make times            how long each stage took, longest first
    make check            compare the printed output with the validated output in tests/golden/ (full run only)

Without `FULL=1`, make stops before it would start a full-size simulation, so a full run is never started by accident.
A stage reruns only when its script, a library it imports, a data file it reads, or a parameter it uses has changed,
so an interrupted run can be restarted and finished stages are not repeated.

Building the PDFs needs `pdflatex` and `bibtex` (TeX Live or MacTeX). They are not part of the conda environment;
`setup/check_setup.sh` reports whether they are found.

## Conventions

- `Phi_u`, `Phi_v` are the net diffusive fluxes of `u` and `v` across an organism's boundary. "PC1" is the first principal
  component of the two (the paper's flux regressor); "joint" means both fluxes entered as separate regressors.
- Seed bases 1000, 9000 and 20000 are Ensembles 1, 2 and 3. "Seed" means a random-number seed only: the starting state of
  every run is the same settled field (1024 x 1024, F = 0.018, k = 0.055).
- Each analysis script's docstring says what it computes, which data it reads, and where the result appears in the paper.

## Layout

| Path | Contents |
|---|---|
| `Makefile`, `stages.mk`, `params.py` | the build: which stage makes which result, and the parameters |
| `src/simulation/` | the solvers (pure Python and C/FFTW) and the scripts that simulate |
| `src/analysis/` | individuation, partial correlation, CMI and the other analyses |
| `src/plotscripts/` | figure scripts |
| `src/tools/` | build helpers (stage timing, LaTeX macro output) |
| `src/TeX/` | `paper.tex`, `supplement.tex`, the bibliography, the figures and the settings for each size of run |
| `tests/` | unit tests, script checks, and `golden/` (validated output of the full-size run) |
| `setup/` | the conda environment and a script that checks it |
| `REVISION_TODO.md`, `CHANGELOG.md` | what changed since submission and what is still to do |

The comments at the end of the Makefile say which stage makes which figure and table.

## Not in git

Simulation output (`build*/`, `*.npz`), compiled libraries (`*.so`) and caches are generated on your machine.
