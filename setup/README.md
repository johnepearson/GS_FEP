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
3. Only if you want the PDFs: install LaTeX (macOS: MacTeX or BasicTeX; Linux: `sudo apt install texlive-latex-recommended`). `make` itself does not need it.

## Once per copy of the project

From the project folder (the one that contains the Makefile):

```
conda env create -f setup/environment.yml
conda activate gsfep
bash setup/check_setup.sh
```

The last line should end with `No failures`.

## Every new terminal

```
cd <the project folder>
conda activate gsfep
```

The prompt then starts with `(gsfep)`. Then `make test` runs the unit tests (a few seconds) and `make` builds the results.

## If something goes wrong

- `python3: command not found`, or `No module named numpy`: the environment is not active. Run `conda activate gsfep`.
- `fftw3.h: No such file`: run `conda install -n gsfep fftw` and delete `src/simulation/libgs_spectral.so` so that it is rebuilt.
- `check_setup.sh` prints `[FAIL]` or `[WARN]` lines: each one has a suggested fix. If you are stuck, email the full output of the script.
