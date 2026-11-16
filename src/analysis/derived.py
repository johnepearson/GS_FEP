"""Quantities derived from the model parameters alone.

Run from build/.  Writes results/derived.tex:
   Tau        the relaxation time 1/(F+k) of the featureless background state
   DvOverDu   the diffusivity ratio Dv/Du"""
from params import load
from texout import num, write_macros

P = load("initialize.tex")
tau = 1.0 / (P["Fval"] + P["kval"])
ratio = P["Dvval"] / P["Duval"]
print(f"tau = 1/(F+k) = {tau:.4f};  Dv/Du = {ratio:.4f}")
write_macros("results/derived.tex", "derived.py", [("Tau", num(tau, 6)), ("DvOverDu", num(ratio, 6))])
