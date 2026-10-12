"""
Lifespan distribution from the lifecycle tracker (lifecycle_progress.pkl, written by run_lifecycle.py),
by cause of termination (ordinary death vs. division). Prints N and medians so the supplement's numbers
can be checked against the same record, and saves lifespan_1024.png.

Organisms present at t=0 have unknown true age (their 'birth' is recorded as 0); the medians are printed
both with and without them.

Usage:  python3 plot_lifespan.py
"""
import numpy as np
from run_lifecycle import LifecycleTracker  # noqa: F401  (needed to unpickle)
from rebuild_fe_comparison_1024 import tolerant_load

tr = tolerant_load("lifecycle_progress.pkl")
ex = tr.examples
life = np.array([e['death'] - e['birth'] for e in ex], float)
cause = np.array([e['cause'] for e in ex])
born0 = np.array([e['birth'] == 0 for e in ex])
print(f"completed lifecycles N={len(ex)}  (divisions {np.sum(cause=='split')}, deaths {np.sum(cause=='death')})")
print(f"  median overall {np.median(life):.0f}   deaths {np.median(life[cause=='death']):.0f}   "
      f"divisions {np.median(life[cause=='split']):.0f}")
k = ~born0
print(f"excluding the {int(born0.sum())} organisms present at t=0: N={int(k.sum())}  median overall {np.median(life[k]):.0f}   "
      f"deaths {np.median(life[k & (cause=='death')]):.0f}   divisions {np.median(life[k & (cause=='split')]):.0f}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4.6))
bins = np.arange(0, np.percentile(life, 99.5) + 50, 50)
ax.hist([life[cause == 'death'], life[cause == 'split']], bins=bins, stacked=True,
        color=["#7f7f7f", "#1f4e8c"], label=["ordinary death", "division"])
ax.axvline(660, color="crimson", ls="--", lw=1.2, label="660 (Pearson 1993, isolated spot)")
ax.set_xlabel("lifespan (time units)"); ax.set_ylabel("number of lifecycles")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("lifespan_1024.png", dpi=200)
print("saved lifespan_1024.png (send this back, with the printed numbers)")
