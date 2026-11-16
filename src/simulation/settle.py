"""Settle the Gray-Scott field: start from the standard seeded square patch (random seed 0) and run
SettleSteps steps, then save the checkpoint every later simulation starts from.

Usage: python3 settle.py OUT.npz            (run from build/; parameters come from initialize.tex)
       python3 settle.py OUT.npz --fast     (C solver; needs libgs_spectral.so)

Parameters read: Fval, kval, Duval, Dvval, Lval, gridn, dtval, SettleSteps, PatchHalf (half-width in pixels of the initial square patch),
SettleSeed (random seed of the +-1% noise on the initial field)."""
import sys
import numpy as np
from params import load as load_params
from gray_scott_spectral import PseudoSpectralGS

P = load_params("initialize.tex")
out = sys.argv[1]
n, F, k = int(P["gridn"]), P["Fval"], P["kval"]
Solver = PseudoSpectralGS
if "--fast" in sys.argv:
    from gray_scott_spectral_c import PseudoSpectralGSFast as Solver


def make_initial_field(n, half, seed):
    c = n // 2
    u = np.ones((n, n), dtype=np.float64)
    v = np.zeros((n, n), dtype=np.float64)
    sl = slice(c - half, c + half)
    u[sl, sl] = 0.5
    v[sl, sl] = 0.25
    rng = np.random.default_rng(seed)
    u *= 1.0 + 0.01 * rng.uniform(-1, 1, size=(n, n))
    mask = v > 0
    v[mask] *= 1.0 + 0.01 * rng.uniform(-1, 1, size=(n, n))[mask]
    return u, v


sim = Solver(n=n, L=P["Lval"], Du=P["Duval"], Dv=P["Dvval"], F=F, k=k, dt=P["dtval"])
u0, v0 = make_initial_field(n, int(P["PatchHalf"]), int(P["SettleSeed"]))
sim.from_checkpoint_fields(u0, v0)
print(f"settling n={n} F={F} k={k} for {int(P['SettleSteps'])} steps")
sim.run(int(P["SettleSteps"]))
sim.save_checkpoint(out)
print(f"saved {out} at t={sim.time:.0f}")
