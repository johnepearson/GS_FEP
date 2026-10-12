"""
make_merge_figure.py -- two-panel illustration of the merge correction on a REAL
merged organism from the settled 1024 checkpoint (production segmentation and
production merge function; nothing is mocked).

Panel (a): raw watershed. Two regions (black outline + black dividing line), their
centroids (white x), and the equivalent-radius circle sqrt(area/pi) of each.
The circles overlap (edge-to-edge gap < 0), so the pair is merged.
Panel (b): after the merge. Same field; no circles and no dividing line: one organism
with a single outline and a single (merged) centroid.

Run from local_package:  python3 make_merge_figure.py [--frame F] [--out DIR]
Writes merge_before.png, merge_after.png.
"""
import argparse, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
import run_ensemble_merged as prod
from gray_scott import GrayScottSim

HALF = 40  # half-width of the crop in pixels (zoomed; keeps part of the upper-left neighbor)
DEPTHS = (0.5, 0.7, 0.9)  # depth fractions d drawn in panel (b)

def load():
    d = np.load(prod.CHECKPOINT_FILE, allow_pickle=True)
    sim = GrayScottSim(n=int(d["n"]), L=float(d["L"]), Du=float(d["Du"]), Dv=float(d["Dv"]),
                       F=float(d["F"]), k=float(d["k"]), dt=float(d["dt"]))
    sim.u, sim.v = d["u"].copy(), d["v"].copy()
    return sim

def find_pairs(sim):
    raw, n = sim.segment_organisms(pad=prod.PAD, min_peak_distance=prod.MIN_PEAK_DISTANCE)
    lab, nn, ms = prod.merge_overlapping_organisms(raw, n, sim)
    _, radii = sim.organism_geometry(raw, n)
    radii_px = radii / sim.dx
    cen = prod.periodic_centroids(raw, n, sim.n)
    out = []
    for i, j in prod.find_adjacent_pairs(raw):
        d = prod.periodic_dist(cen[i-1], cen[j-1], sim.n)
        gap = d - radii_px[i-1] - radii_px[j-1]
        if gap < 0 and lab[raw == i][0] == lab[raw == j][0]:
            out.append((i, j, d, radii_px[i-1], radii_px[j-1], gap))
    return raw, n, lab, nn, ms, cen, out

def crop(arr, cy, cx, half, N):
    ys = (np.arange(int(round(cy))-half, int(round(cy))+half) % N)
    xs = (np.arange(int(round(cx))-half, int(round(cx))+half) % N)
    return arr[np.ix_(ys, xs)], int(round(cy))-half, int(round(cx))-half

def draw(ax, v, lab_crop, rawc, cen_c, radii, i, j, show_split, extent, merged_c=None, mask=None):
    ax.imshow(v, cmap="jet", origin="upper", interpolation="nearest")
    both = np.isin(rawc, [i, j])
    ax.contour(both.astype(float), levels=[0.5], colors="k", linewidths=3)
    if show_split:
        # dividing line = pixels where the two regions touch
        a = rawc == i; b = rawc == j
        edge = (ndimage.binary_dilation(a) & b) | (ndimage.binary_dilation(b) & a)
        yy, xx = np.nonzero(edge)
        ax.contour(((rawc == i)).astype(float), levels=[0.5], colors="k", linewidths=3)
        ax.contour(((rawc == j)).astype(float), levels=[0.5], colors="k", linewidths=3)
    th = np.linspace(0, 2*np.pi, 200)
    for (cy, cx), r in zip(cen_c, radii):
        if show_split:
            ax.plot(cx + r*np.cos(th), cy + r*np.sin(th), "--", color="magenta", lw=2.5)
            ax.plot(cx, cy, "x", color="white", ms=14, mew=3)
    if merged_c is not None:
        ax.plot(merged_c[1], merged_c[0], 'x', color='white', ms=5, mew=1.8)
    if mask is not None:
        dist = ndimage.distance_transform_edt(mask)
        R = np.sqrt(mask.sum() / np.pi)
        for d, ls, lw in zip(DEPTHS, ("-", "--", ":"), (2.5, 2.5, 3.8)):
            ax.contour(dist, levels=[d * R], colors="white", linewidths=lw, linestyles=ls)
    ax.set_xlim(-0.5, v.shape[1]-0.5); ax.set_ylim(v.shape[0]-0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); ap.add_argument("--pick", type=int, default=1)
    ap.add_argument("--advance", type=int, default=0, help="unused; reserved")
    a = ap.parse_args()
    sim = load(); N = sim.n
    raw, n, lab, nn, ms, cen, pairs = find_pairs(sim)
    print(f"raw regions {n}, merged organisms {nn}, merged pairs (overlapping+adjacent): {len(pairs)}")
    if not pairs:
        sys.exit("no merged pair in the checkpoint frame; step the simulation forward and retry")
    pairs.sort(key=lambda t: t[5])
    for k, t in enumerate(pairs[:10]):
        print(f"  [{k}] i={t[0]} j={t[1]} dist={t[2]:.1f} r_i+r_j={t[3]+t[4]:.1f} gap={t[5]:.1f}")
    i, j, d, ri, rj, gap = pairs[min(a.pick, len(pairs)-1)]
    mid_y = (cen[i-1][0] + cen[j-1][0]) / 2; mid_x = (cen[i-1][1] + cen[j-1][1]) / 2
    # periodic-safe midpoint
    dy = prod.periodic_delta(cen[j-1][0], cen[i-1][0], N); dx = prod.periodic_delta(cen[j-1][1], cen[i-1][1], N)
    my = (cen[i-1][0] + dy/2) % N; mx = (cen[i-1][1] + dx/2) % N
    v, oy, ox = crop(sim.v, my, mx, HALF, N)
    rawc, _, _ = crop(raw, my, mx, HALF, N)
    labc, _, _ = crop(lab, my, mx, HALF, N)
    cc = []
    for q in (i, j):
        cyq = (cen[q-1][0] - oy + N/2) % N - N/2 if False else prod.periodic_delta(cen[q-1][0], oy, N)
        cxq = prod.periodic_delta(cen[q-1][1], ox, N)
        cc.append((cyq, cxq))
    radii = [ri, rj]
    mcen = prod.periodic_centroids(lab, nn, N)[int(lab[raw == i][0]) - 1]
    merged_c = (prod.periodic_delta(mcen[0], oy, N), prod.periodic_delta(mcen[1], ox, N))
    for fname, split, title in [
        ("merge_before.png", True,
         f"Edge-to-edge gap = {d:.1f} $-$ {ri+rj:.1f} = {gap:.1f} px $<0$"),
        ("merge_after.png", False, "After the merge correction: one organism")]:
        fig = plt.figure(figsize=(6.2, 7.6))
        ax = fig.add_axes([0.05, 0.20, 0.90, 0.734])
        draw(ax, v, rawc, rawc, cc, radii, i, j, split, None, None if split else merged_c, None if split else (labc == labc[int(round(merged_c[0])), int(round(merged_c[1]))]))
        ax.set_title(title, fontsize=12, pad=8)
        from matplotlib.lines import Line2D
        if split:
            h = [Line2D([0],[0],color="k",lw=3,label="Watershed regions (masks disjoint)"),
                 Line2D([0],[0],color="magenta",lw=2.5,ls="--",label=r"Equivalent-radius circle ($\sqrt{\mathrm{area}/\pi}$)"),
                 Line2D([0],[0],color="white",marker="x",ms=10,mew=3,lw=0,label="Centroid")]
            ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure, fontsize=10, frameon=True, facecolor="#6b6b6b", edgecolor="k", framealpha=1.0, labelcolor="white", borderpad=0.8)
        else:
            h = [Line2D([0],[0],color="k",lw=3,label="Merged organism (single outline)"),
                 Line2D([0],[0],color="white",marker="x",ms=10,mew=3,lw=0,label="Centroid of the merged organism"),
                 Line2D([0],[0],color="white",lw=2.5,ls="-",label="Interior core boundary, $d=0.5$ (used throughout)"),
                 Line2D([0],[0],color="white",lw=2.5,ls="--",label="$d=0.7$"),
                 Line2D([0],[0],color="white",lw=3.8,ls=":",label="$d=0.9$")]
            ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure, fontsize=10, frameon=True, facecolor="#6b6b6b", edgecolor="k", framealpha=1.0, labelcolor="white", borderpad=0.8)
        fig.savefig(f"{a.out}/{fname}", dpi=200); plt.close(fig)
    print("wrote merge_before.png, merge_after.png")

if __name__ == "__main__":
    main()
