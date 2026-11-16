"""
Check that the C/FFTW solver (gs_spectral_c.c via gray_scott_spectral_c.py) reproduces the pure-Python pseudo-spectral solver
(gray_scott_spectral.py) from the same 1024 x 1024 checkpoint.  Reports the largest and rms differences in u and v after
increasing numbers of steps.  Rounding differences are amplified by the chaotic dynamics, so the difference grows slowly with
the number of steps; the check is that it starts at rounding level (about 1e-15) and grows smoothly.

    python3 check_c_solver.py
    python3 check_c_solver.py --targets 10 100 1000
"""
import argparse
import time
import numpy as np
from gray_scott_spectral import PseudoSpectralGS
from gray_scott_spectral_c import PseudoSpectralGSFast

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoint_1024_settled.npz")
ap.add_argument("--targets", type=int, nargs="+", default=[10, 100, 300, 1000, 2000, 3000, 4000])
a = ap.parse_args()
py = PseudoSpectralGS.from_checkpoint(a.checkpoint)
c = PseudoSpectralGSFast.from_checkpoint(a.checkpoint)
uc, vc = c.get_uv()
print(f"n={py.n}, dt={py.dt}; difference at start: max|du| {np.abs(py.u - uc).max():.2e}")
done = 0
for target in a.targets:
    t0 = time.time(); py.run(target - done); tp = time.time() - t0
    t0 = time.time(); c.run(target - done); tc = time.time() - t0
    done = target
    uc, vc = c.get_uv()
    print(f"steps {target:5d}: max|du| {np.abs(py.u - uc).max():.2e}  max|dv| {np.abs(py.v - vc).max():.2e}  "
          f"rms dv {np.sqrt(np.mean((py.v - vc) ** 2)):.2e}   (Python {tp:.0f}s, C {tc:.1f}s for this block)", flush=True)
