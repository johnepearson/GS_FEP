"""
Field snapshot of the self-replicating-spot regime at 1024 x 1024 (paper: Figure 1, the spots snapshot).

Plots v (blue = 0, yellow, red = highest v in the frame) for one archived frame of the master trajectory.
The published panel is one particular frame; any frame of the same trajectory looks the same in character, so this
script takes --frame (default 250) from master_fields/frame_XXXX.npz, or, if that folder is absent, the final
state stored in master_trajectory_progress.npz.

Usage:  python3 plot_snapshot.py
        python3 plot_snapshot.py --frame 100 --out spots_snapshot.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields-dir", default="master_fields")
    ap.add_argument("--frame", type=int, default=250)
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--out", default="spots_snapshot.png")
    a = ap.parse_args()
    path = os.path.join(a.fields_dir, f"frame_{a.frame:04d}.npz")
    if os.path.exists(path):
        v = np.load(path)["v"].astype(float)
        src = path
    else:
        v = np.load(a.data)["v"].astype(float)
        src = f"{a.data} (final state)"
    cmap = LinearSegmentedColormap.from_list("byr", ["#0000ff", "#ffff00", "#ff0000"])
    fig = plt.figure(figsize=(7, 7), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(v, cmap=cmap, vmin=0.0, vmax=v.max(), interpolation="bilinear")
    ax.axis("off")
    fig.savefig(a.out, dpi=200)
    print(f"saved {a.out} from {src}; v range {v.min():.3f}-{v.max():.3f}")


if __name__ == "__main__":
    main()
