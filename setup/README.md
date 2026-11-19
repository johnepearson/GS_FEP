# Setting up the environment

These steps work on macOS and Linux (on Windows, use WSL and follow the Linux steps). Type each command in a terminal.

## Once per computer

1. Install a C compiler.
   macOS: `xcode-select --install`   Linux: `sudo apt install build-essential`
2. Install Miniforge (a small conda that uses the conda-forge channel), then close the terminal and open a new one. If you already have conda or mamba (`conda --version` prints a version), skip this step; if the installer says the directory already exists, you already have it, so do not delete it and go on to the next section:
   ```
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
   The installer asks you to press ENTER, then to type `yes` to accept the license, then to press ENTER to accept the install location (and to answer `yes` if it asks whether to initialize conda). The download is a file named `Miniforge3-...sh`; you can delete it afterwards.
3. Only if you want the PDFs: install LaTeX (macOS: MacTeX or BasicTeX; Linux: `sudo apt install texlive-latex-recommended`). `make` itself does not need it.

## Once per copy of the project

Get the code. This creates a folder named `GS_FEP` and goes into it:

```
git clone -b restructure https://github.com/johnepearson/GS_FEP.git
cd GS_FEP
```

Then, from inside that folder (the one that contains the `Makefile`):

```
conda env create -f setup/environment.yml
conda activate gsfep
bash setup/check_setup.sh
```

The last line should end with `No failures`.

## Every new terminal

`GS_FEP` is the folder that `git clone` created, the one that contains the `Makefile`. If you cloned it somewhere else or gave it another name, `cd` to that folder instead (`ls` should show `Makefile`).

```
cd GS_FEP
conda activate gsfep
```

The prompt then starts with `(gsfep)`.

## Check that everything works

From the project folder, with `gsfep` active:

```
make test
make QUICK=1
```

`make test` runs the unit tests (about 30 seconds; it does not run a simulation). `make QUICK=1` runs the whole pipeline at a tiny size (about 10 minutes on a 12-core Mac). Its numbers are not the paper's numbers; it only checks that everything runs. The full-size run is `make FULL=1` (about 30 hours); see the top-level README.

## When you are done

`conda deactivate` leaves the environment; the `(gsfep)` disappears from the prompt. Closing the terminal does the same.

## If something goes wrong

- `python3: command not found`, or `No module named numpy`: the environment is not active. Run `conda activate gsfep`.
- `fftw3.h: No such file`: run `conda install -n gsfep fftw` and delete `src/simulation/libgs_spectral.so` so that it is rebuilt.
- `check_setup.sh` prints `[FAIL]` or `[WARN]` lines: each one has a suggested fix. If you are stuck, email the full output of the script.
