"""Compare two checkpoints: python3 compare_checkpoint.py A.npz B.npz
Prints the largest absolute and relative differences in u and v, and whether the scalar entries agree."""
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for key in ("n", "L", "Du", "Dv", "F", "k", "dt", "step_count", "time"):
    if key in a.files and key in b.files:
        print(f"{key:11s} {float(a[key]):<12g} {float(b[key]):<12g} {'same' if float(a[key]) == float(b[key]) else 'DIFFERENT'}")
for key in ("u", "v"):
    d = np.abs(a[key] - b[key])
    print(f"{key}: max |diff| = {d.max():.3e}   max relative = {(d / np.maximum(np.abs(b[key]), 1e-30)).max():.3e}   identical = {bool(d.max() == 0)}")
