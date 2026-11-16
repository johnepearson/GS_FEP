#!/usr/bin/env bash
# Checks that the environment can build the project.  Changes nothing on your computer.
# Run from the project folder (the one with the Makefile), with the environment active:
#     conda activate gsfep
#     bash setup/check_setup.sh
# Each line is [ OK ], [WARN] or [FAIL], with a suggested fix.  WARN does not stop a build; FAIL does.

nok=0; nwarn=0; nfail=0
ok()   { printf '[ OK ] %s\n' "$1"; nok=$((nok+1)); }
warn() { printf '[WARN] %s\n       fix: %s\n' "$1" "$2"; nwarn=$((nwarn+1)); }
bad()  { printf '[FAIL] %s\n       fix: %s\n' "$1" "$2"; nfail=$((nfail+1)); }

if [ ! -f Makefile ] || [ ! -f requirements.txt ]; then
    bad "this folder has no Makefile or requirements.txt" "cd to the project folder, then run  bash setup/check_setup.sh"
fi

# 1. Python: the Makefile runs python3, so that is the one that must be the environment's
if command -v python3 >/dev/null 2>&1; then
    pyv=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
    pyp=$(python3 -c 'import sys; print(sys.executable)')
    case "$pyv" in
        3.11.*) ok "python3 is $pyv ($pyp)";;
        *)      warn "python3 is $pyv ($pyp); the full run used 3.11" "conda activate gsfep";;
    esac
    if [ -n "$CONDA_PREFIX" ]; then
        case "$pyp" in
            "$CONDA_PREFIX"*) ok "python3 comes from the active conda environment";;
            *) bad "python3 ($pyp) is not the active conda environment's ($CONDA_PREFIX)" "conda deactivate; conda activate gsfep   (the shell may find another Python first)";;
        esac
    else
        warn "no conda environment is active" "conda activate gsfep"
    fi
else
    bad "python3 not found" "conda activate gsfep"
fi

# 2. make
if command -v make >/dev/null 2>&1; then
    mv=$(make --version 2>/dev/null | head -n 1)
    ok "make: $mv"
else
    bad "make not found" "macOS: xcode-select --install   Linux: sudo apt install make"
fi

# 3. Python packages against the versions in requirements.txt
if command -v python3 >/dev/null 2>&1 && [ -f requirements.txt ]; then
    python3 - requirements.txt <<'PY' > /tmp/gsfep_check_$$.txt 2>&1
import re, sys
from importlib import metadata
for line in open(sys.argv[1]):
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
    if not m:
        continue
    name, want = m.groups()
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        print("FAIL", name, want, "missing"); continue
    print("OK" if have == want else "WARN", name, want, have)
PY
    while read -r tag name want have; do
        case "$tag" in
            OK)   ok "$name $have";;
            WARN) warn "$name is $have; requirements.txt has $want (results may differ in the last digits)" "python3 -m pip install -r requirements.txt";;
            FAIL) bad "$name is missing" "python3 -m pip install -r requirements.txt";;
        esac
    done < /tmp/gsfep_check_$$.txt
    rm -f /tmp/gsfep_check_$$.txt
fi

# 4. pytest and coverage (for make test and make coverage)
for p in pytest coverage; do
    if python3 -c "import $p" 2>/dev/null; then ok "$p is installed"; else warn "$p is not installed" "conda install pytest coverage"; fi
done

# 5. A C compiler and FFTW: compile and run a one-line FFTW program with the same flags the Makefile uses
tmp=$(mktemp -d)
cat > "$tmp/t.c" <<'C'
#include <fftw3.h>
#include <stdio.h>
int main(void) { fftw_init_threads(); fftw_plan_with_nthreads(1); double *a = fftw_malloc(8 * sizeof(double));
  printf("fftw ok\n"); fftw_free(a); return 0; }
C
flags=""
[ -n "$CONDA_PREFIX" ] && flags="-I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
if ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
    bad "no C compiler (gcc or cc)" "macOS: xcode-select --install   Linux: sudo apt install build-essential"
elif gcc $flags "$tmp/t.c" -o "$tmp/t" -lfftw3_threads -lfftw3 -lpthread -lm >"$tmp/log" 2>&1 && "$tmp/t" >/dev/null 2>&1; then
    ok "a C program using FFTW compiles and runs"
else
    bad "could not compile and run a C program that uses FFTW (the C solver needs it): $(head -n 1 "$tmp/log")" "conda install fftw   (and on macOS: xcode-select --install)"
fi
rm -rf "$tmp"

# 6. LaTeX, only needed for make pdfs
for t in pdflatex bibtex; do
    if command -v $t >/dev/null 2>&1; then ok "$t found"; else warn "$t not found (only needed for make pdfs)" "macOS: install MacTeX or BasicTeX   Linux: sudo apt install texlive-latex-recommended"; fi
done

printf '\nSummary: %d ok, %d warnings, %d failures\n' "$nok" "$nwarn" "$nfail"
if [ "$nfail" -eq 0 ]; then echo "No failures. You can run:  make test"; exit 0; else echo "Fix the [FAIL] lines first."; exit 1; fi
