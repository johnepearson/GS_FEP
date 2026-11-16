"""
Sweep the watershed marker spacing (peak_local_max min_distance) and measure
whether it removes the segmentation dropout (healthy spots with no mask) without
over-splitting spots.

One simulation is run from the settled checkpoint; at every sampled frame the SAME
field is segmented with each min_distance (production pipeline: raw watershed +
overlap merge), so the values are compared on identical data.

Per min_distance it reports
  masks/frame           mean number of final (merged) organisms
  raw -> merged         mean watershed regions before / after the overlap merge
                        (a large gap means the smaller spacing is cutting spots in pieces
                        and the merge step is repairing it)
  unlabelled/frame      connected components of (v > Otsu) that received NO label,
                        with area >= MIN_AREA px and peak v at least PEAK_EXCESS above threshold
                        (= a visible spot that the segmentation lost)
     with own local max   how many of those have a true local maximum (+-LM_WIN px). Only those can
                        ever be rescued by a smaller min_distance.
  vanish / appear       masks in a frame with NO overlap at all in the next frame (and the reverse),
                        per frame. A dropout shows up as one vanish followed by one appear.
  small masks           fraction of masks with equivalent radius < SMALL_R px (fragments)
  radius                median / p10 equivalent radius of masks (px)

Usage:
    python run_min_distance_sweep.py --n-frames 500
    python run_min_distance_sweep.py --n-frames 20 --values 16 32     # quick test
"""
import argparse
import time
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim
import run_ensemble_merged as prod

PAD = 80
SAMPLE_EVERY_STEPS = 10
CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
MIN_AREA = 50          # px
PEAK_EXCESS = 0.05     # peak v above the Otsu threshold for a "visible spot"
LM_WIN = 5             # px half-window for "has its own local maximum"
SMALL_R = 10.0         # px


def segment(proxy, md):
    raw, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=md)
    if raw_n == 0:
        return raw, 0, 0
    labels, n, _ = prod.merge_overlapping_organisms(raw, raw_n, proxy)
    return labels, n, raw_n


def unlabelled_spots(v, labels):
    """Visible spots with no label. Returns list of (area, peak_excess, has_local_max)."""
    thr = threshold_otsu(v)
    comp, nc = ndi.label((v > thr) & (labels == 0))   # no periodic join; fine for a count
    out = []
    if nc == 0:
        return out
    idx = np.arange(1, nc + 1)
    areas = ndi.sum(np.ones_like(v), comp, idx)
    peaks = ndi.maximum(v, comp, idx)
    pos = ndi.maximum_position(v, comp, idx)
    N = v.shape[0]
    for a, p, (y, x) in zip(areas, peaks, pos):
        if a < MIN_AREA or p - thr < PEAK_EXCESS:
            continue
        yy = (np.arange(y - LM_WIN, y + LM_WIN + 1)) % N
        xx = (np.arange(x - LM_WIN, x + LM_WIN + 1)) % N
        win = v[np.ix_(yy, xx)]
        out.append((a, p - thr, bool(v[y, x] >= win.max() - 1e-12)))
    return out


def overlap_sets(prev, curr, n_prev, n_curr):
    """Return (#prev labels with no overlap in curr, #curr labels with no overlap in prev)."""
    both = (prev > 0) & (curr > 0)
    pp = np.unique(prev[both])
    cc = np.unique(curr[both])
    return n_prev - len(pp), n_curr - len(cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=500)
    ap.add_argument("--values", type=int, nargs="+", default=[8, 16, 24, 32])
    ap.add_argument("--out", default="min_distance_sweep")
    args = ap.parse_args()
    vals = args.values

    sim = PseudoSpectralGS.from_checkpoint(CHECKPOINT_FILE)
    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)
    proxy.u, proxy.v = sim.u, sim.v

    S = {md: dict(n=[], raw=[], unl=[], unl_lm=[], vanish=[], appear=[], radii=[]) for md in vals}
    prev = {}
    t0 = time.time()
    for fr in range(1, args.n_frames + 1):
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.u, sim.v
        for md in vals:
            labels, n, raw_n = segment(proxy, md)
            s = S[md]
            s["n"].append(n); s["raw"].append(raw_n)
            u = unlabelled_spots(proxy.v, labels)
            s["unl"].append(len(u)); s["unl_lm"].append(sum(1 for q in u if q[2]))
            if n:
                _, rad = proxy.organism_geometry(labels, n)
                s["radii"].extend((rad / proxy.dx).tolist())
            if md in prev:
                pl, pn = prev[md]
                v_, a_ = overlap_sets(pl, labels, pn, n)
                s["vanish"].append(v_); s["appear"].append(a_)
            prev[md] = (labels, n)
        if fr % 20 == 0:
            print(f"frame {fr}/{args.n_frames}  {(time.time()-t0)/fr:.1f}s/frame", flush=True)

    print(f"\n{args.n_frames} frames, {SAMPLE_EVERY_STEPS} steps apart; visible-spot test: "
          f"area>={MIN_AREA}px, peak >= {PEAK_EXCESS} above Otsu; local max window +-{LM_WIN}px\n")
    hdr = (f"{'min_dist':>8} {'masks':>7} {'raw':>7} {'raw-merged':>10} {'unlabelled':>10} "
           f"{'w/ own max':>10} {'vanish':>7} {'appear':>7} {'small%':>7} {'R med':>6} {'R p10':>6}")
    print(hdr)
    for md in vals:
        s = S[md]
        r = np.array(s["radii"])
        print(f"{md:>8d} {np.mean(s['n']):7.1f} {np.mean(s['raw']):7.1f} "
              f"{np.mean(s['raw']) - np.mean(s['n']):10.2f} {np.mean(s['unl']):10.2f} "
              f"{np.mean(s['unl_lm']):10.2f} {np.mean(s['vanish']):7.2f} {np.mean(s['appear']):7.2f} "
              f"{100*np.mean(r < SMALL_R):7.2f} {np.median(r):6.1f} {np.percentile(r,10):6.1f}")
    print("\ncolumns (all are per-frame means): masks = final organisms; raw = watershed regions before merge;"
          "\n  unlabelled = visible spots with no mask; w/ own max = those with a true local maximum"
          "\n  (the only ones a smaller min_distance can rescue); vanish/appear = masks with no overlap"
          "\n  with the next/previous frame (dropout = one of each); small% = masks with R < "
          f"{SMALL_R:.0f} px.")
    np.savez(args.out + ".npz", values=np.array(vals),
             **{f"{k}_{md}": np.array(S[md][k]) for md in vals for k in ("n", "raw", "unl", "unl_lm", "vanish", "appear", "radii")})


if __name__ == "__main__":
    main()
