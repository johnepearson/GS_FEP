#!/bin/sh
# Builds paper.pdf and supplement.pdf. Run from build/ with TEXINPUTS and BIBINPUTS pointing at src/TeX (the Makefile does this).
# Each document reads the other's labels from a plain file (paper_labels.tex, supplement_labels.tex) that this script
# regenerates from the .aux files, so the two documents compile on their own.
set -e
q() { "$@" >/dev/null 2>&1 || true; }
[ -f paper_labels.tex ] || : > paper_labels.tex
[ -f supplement_labels.tex ] || : > supplement_labels.tex
q pdflatex -interaction=nonstopmode paper.tex
q bibtex paper
q pdflatex -interaction=nonstopmode supplement.tex
q bibtex supplement
# Two rounds: the first fills the two label files (so the second round has no unresolved references to
# typeset as "??", which would change line and page breaks), the second makes the final layout.
for round in 1 2; do
  for i in 1 2; do
    q pdflatex -interaction=nonstopmode paper.tex
    q pdflatex -interaction=nonstopmode supplement.tex
  done
  grep "^.newlabel" paper.aux > paper_labels.tex
  grep "^.newlabel" supplement.aux > supplement_labels.tex
done
q pdflatex -interaction=nonstopmode paper.tex
q pdflatex -interaction=nonstopmode supplement.tex
np=$(grep -c '^!' paper.log || true); ns=$(grep -c '^!' supplement.log || true)
echo "paper: $(pdfinfo paper.pdf | awk '/Pages/{print $2}') pages, $np errors; supplement: $(pdfinfo supplement.pdf | awk '/Pages/{print $2}') pages, $ns errors"
[ "$np" = 0 ] && [ "$ns" = 0 ]
