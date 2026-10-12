"""
What produces the low-shape-correlation tail in Section 3.1?

Re-runs the rotation-aligned shape-correlation check (run_shape_stability.py)
on production (watershed + overlap-merge) organisms, and tags every matched
transition with
  div_now   this transition is a division: the earlier organism is overlapped
            by >= 2 later organisms, each covering >= DIV_FRAC of its area
  ttd       frames until the lineage next divides (inf if none within the run)
  age       frames since the lineage's last division (nan if before the run)
  gap       edge-to-edge distance (pixels) to the nearest other organism
and then reports how much of the tail each explains.

Usage:
    python run_shape_tail_test.py --n-frames 400
"""
import argparse, time
import numpy as np
from scipy import ndimage

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, match_organisms_by_overlap
from run_shape_stability import get_all_shape_vectors, best_rotation_correlation
import run_ensemble_merged as prod

MIN_PEAK_DISTANCE = 24
PAD = 80
SAMPLE_EVERY_STEPS = 10
CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
DIV_FRAC = 0.2
NEAR = 5          # frames: "close to a division" window


def segment(proxy):
    raw, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
    if raw_n == 0:
        return raw, 0
    lab, n, _ = prod.merge_overlapping_organisms(raw, raw_n, proxy)
    return lab, n


def geometry(proxy, labels, n):
    _, radii = proxy.organism_geometry(labels, n)
    return prod.periodic_centroids(labels, n, proxy.n), radii


def best_rotation_rms(r1, r2):
    """Smallest rms difference of r(theta) over circular shifts, as a fraction of mean r1."""
    n = len(r1)
    best = min(np.sqrt(np.mean((r1 - np.roll(r2, k)) ** 2)) for k in range(n))
    return best / r1.mean()


def nearest_gap(cen, rad_px, N):
    n = len(cen)
    gap = np.full(n, np.nan)
    if n < 2:
        return gap
    dy = cen[:, None, 0] - cen[None, :, 0]; dx = cen[:, None, 1] - cen[None, :, 1]
    dy -= N * np.round(dy / N); dx -= N * np.round(dx / N)
    d = np.hypot(dy, dx) - rad_px[:, None] - rad_px[None, :]
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def overlaps(prev_labels, curr_labels, prev_n, curr_n):
    both = (prev_labels > 0) & (curr_labels > 0)
    joint = prev_labels[both].astype(np.int64) * (curr_n + 1) + curr_labels[both]
    vals, cnt = np.unique(joint, return_counts=True)
    return vals // (curr_n + 1), vals % (curr_n + 1), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--out", default="shape_tail_test")
    args = ap.parse_args()

    sim = PseudoSpectralGS.from_checkpoint(CHECKPOINT_FILE)
    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)
    proxy.u, proxy.v = sim.u, sim.v
    N = sim.n

    labels, n = segment(proxy)
    cen, rad = geometry(proxy, labels, n)
    vecs = get_all_shape_vectors(proxy, labels, n, cen, rad)
    gap = nearest_gap(cen, rad / proxy.dx, N)
    frames = []   # per frame: dict of arrays indexed by organism id-1
    t0 = time.time()
    cur = dict(n=n, gap=gap, rad=rad / proxy.dx, nxt=np.zeros(n, int), corr=np.full(n, np.nan), rms=np.full(n, np.nan), anis=np.full(n, np.nan), ev=np.zeros(n, int),
               div=np.zeros(n, bool), daughters=[[] for _ in range(n)])
    for fr in range(1, args.n_frames + 1):
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.u, sim.v
        nl, nn = segment(proxy)
        if nn == 0:
            break
        ncen, nrad = geometry(proxy, nl, nn)
        nvecs = get_all_shape_vectors(proxy, nl, nn, ncen, nrad)
        fwd, bwd = match_organisms_by_overlap(labels, nl, n, nn)
        p_ids, c_ids, cnt = overlaps(labels, nl, n, nn)
        area_prev = np.bincount(labels.ravel(), minlength=n + 1)
        for p, c, k in zip(p_ids, c_ids, cnt):
            if k >= DIV_FRAC * area_prev[p]:
                cur["daughters"][p - 1].append(int(c))
        for p in range(1, n + 1):
            if len(cur["daughters"][p - 1]) >= 2:
                cur["div"][p - 1] = True
        for p, (c, frac) in fwd.items():
            if c is None or frac <= 0.5 or len(bwd.get(c, [])) != 1:
                continue
            if p not in vecs or c not in nvecs:
                continue
            cur["nxt"][p - 1] = c
            cur["corr"][p - 1] = best_rotation_correlation(vecs[p], nvecs[c])
            cur["rms"][p - 1] = best_rotation_rms(vecs[p], nvecs[c])
            cur["anis"][p - 1] = vecs[p].std() / vecs[p].mean()
        for p in range(1, n + 1):
            if cur["div"][p - 1]:
                cur["ev"][p - 1] = 1                       # division
                continue
            c, frac = fwd.get(p, (None, 0.0))
            if c is None or frac < 0.2:
                cur["ev"][p - 1] = 2                       # death: (almost) nothing overlaps it next frame
            elif len(bwd.get(c, [])) > 1:
                cur["ev"][p - 1] = 3                       # merge / fusion: its successor also claims another organism
            elif cur["nxt"][p - 1] == 0:
                cur["ev"][p - 1] = 4                       # ambiguous (<50% overlap, or no usable shape vector)
        frames.append(cur)
        labels, n, cen, rad, vecs = nl, nn, ncen, nrad, nvecs
        cur = dict(n=n, gap=nearest_gap(cen, rad / proxy.dx, N), rad=rad / proxy.dx,
                   nxt=np.zeros(n, int), corr=np.full(n, np.nan),
                   rms=np.full(n, np.nan), anis=np.full(n, np.nan), ev=np.zeros(n, int),
                   div=np.zeros(n, bool), daughters=[[] for _ in range(n)])
        if fr % 20 == 0:
            print(f"frame {fr}/{args.n_frames} organisms={n} {(time.time()-t0)/fr:.2f}s/frame", flush=True)

    F = len(frames)
    # time to next division: backward pass along 1:1 chains
    ttd = [np.full(f["n"], np.inf) for f in frames]
    for t in range(F - 1, -1, -1):
        for i in range(frames[t]["n"]):
            if frames[t]["div"][i]:
                ttd[t][i] = 0
            else:
                j = frames[t]["nxt"][i]
                if j and t + 1 < F:
                    ttd[t][i] = 1 + ttd[t + 1][j - 1]
    # time to the next non-continuation event of ANY type, and its type
    tte = [np.full(f["n"], np.inf) for f in frames]
    typ = [np.full(f["n"], 5) for f in frames]            # 5 = run ended first
    for t in range(F - 1, -1, -1):
        for i in range(frames[t]["n"]):
            e = frames[t]["ev"][i]
            if e != 0:
                tte[t][i], typ[t][i] = 0, e
            else:
                j = frames[t]["nxt"][i]
                if j and t + 1 < F:
                    tte[t][i] = 1 + tte[t + 1][j - 1]
                    typ[t][i] = typ[t + 1][j - 1]
    # age since last division: forward pass; daughters born at t+1
    age = [np.full(f["n"], np.nan) for f in frames]
    for t in range(F - 1):
        for i in range(frames[t]["n"]):
            if frames[t]["div"][i]:
                for c in frames[t]["daughters"][i]:
                    age[t + 1][c - 1] = 0
        for i in range(frames[t]["n"]):
            j = frames[t]["nxt"][i]
            if j and not frames[t]["div"][i] and np.isfinite(age[t][i]):
                age[t + 1][j - 1] = age[t][i] + 1

    rows = []
    for t in range(F):
        f = frames[t]
        for i in range(f["n"]):
            if np.isfinite(f["corr"][i]):
                rows.append((f["corr"][i], f["div"][i], ttd[t][i], age[t][i], f["gap"][i], f["rad"][i], f["rms"][i], f["anis"][i], tte[t][i], typ[t][i]))
    R = np.array(rows, float)
    corr, div, ttdv, agev, gapv, radv, rmsv, anisv, ttev, typv = R.T
    div = div.astype(bool)
    np.savez(args.out + ".npz", corr=corr, div_now=div, ttd=ttdv, age=agev, gap=gapv, rad=radv, rms=rmsv, anis=anisv, tte=ttev, typ=typv)

    print(f"\nN matched transitions = {len(corr)};  division transitions among them = {div.sum()} "
          f"({100*div.mean():.2f}%)")
    print(f"all:   median={np.median(corr):.3f} mean={corr.mean():.3f} p10={np.percentile(corr,10):.3f} p1={np.percentile(corr,1):.3f}")
    pre = (~div) & (ttdv >= 1) & (ttdv <= NEAR)
    post = (~div) & np.isfinite(agev) & (agev <= NEAR) & ~pre
    other = ~(div | pre | post)
    for name, m in (("division transition", div), (f"pre-division (1-{NEAR} frames before)", pre),
                    (f"post-division (<= {NEAR} frames after)", post), ("everything else", other)):
        c = corr[m]
        if len(c):
            print(f"  {name:38s} n={m.sum():6d}  median={np.median(c):.3f}  p10={np.percentile(c,10):.3f}  p1={np.percentile(c,1):.3f}")
    for thr in (0.667, 0.435):
        tail = corr < thr
        print(f"\ntail corr < {thr}: n={tail.sum()} ({100*tail.mean():.2f}% of transitions)")
        for name, m in (("division transition", div), ("pre-division", pre), ("post-division", post), ("everything else", other)):
            print(f"   {name:22s} {100*(tail & m).sum()/max(tail.sum(),1):5.1f}% of tail   "
                  f"(tail rate within group {100*(tail & m).sum()/max(m.sum(),1):5.1f}%)")
        un = tail & other
        if un.sum() > 20:
            g_all = np.nanmedian(gapv[other]); g_un = np.nanmedian(gapv[un])
            print(f"   unexplained tail: median nearest-neighbor gap {g_un:.1f} px vs {g_all:.1f} px for other non-tail "
                  f"({100*np.mean(gapv[un] < 0):.1f}% vs {100*np.mean(gapv[other & ~tail] < 0):.1f}% overlapping circles)")
    print("\nAmplitude-based shape change (rms of r(theta) difference / radius, best rotation):")
    print(f"  all: median={np.median(rmsv):.3f} p90={np.percentile(rmsv,90):.3f} p99={np.percentile(rmsv,99):.3f}")
    tail = corr < 0.667
    print(f"  in the correlation tail (<0.667): median={np.median(rmsv[tail]):.3f} p90={np.percentile(rmsv[tail],90):.3f}; "
          f"{100*np.mean(rmsv[tail] < 0.05):.0f}% of tail transitions change shape by < 5% of the radius")
    print(f"  rms by group: " + "; ".join(f"{nm} median={np.median(rmsv[m]):.3f} p99={np.percentile(rmsv[m],99):.3f} (n={m.sum()})"
          for nm, m in (("division", div), ("pre-division", pre), ("post-division", post), ("other", other)) if m.sum() > 20))
    for q in (99, 95):
        big = rmsv > np.percentile(rmsv, q)
        print(f"  top {100-q}% of shape changes (rms > {np.percentile(rmsv, q):.3f}): "
              + ", ".join(f"{nm} {100*np.mean(m[big]):.1f}%" for nm, m in (("division", div), ("pre-division", pre), ("post-division", post), ("other", other))))
    print(f"  fraction of transitions with rms < 5% of radius: {100*np.mean(rmsv < 0.05):.1f}%;  < 10%: {100*np.mean(rmsv < 0.10):.1f}%")
    for nm, x in (("anisotropy std(r)/mean(r)", anisv), ("equivalent radius (px)", radv)):
        e = np.percentile(x, [0, 20, 40, 60, 80, 100]); e[-1] += 1e-9
        print(f"  by {nm}:")
        for lo, hi in zip(e[:-1], e[1:]):
            m = (x >= lo) & (x < hi)
            print(f"     {lo:7.3f}-{hi:7.3f} n={m.sum():6d}  median corr={np.median(corr[m]):.3f}  tail={100*np.mean(corr[m] < 0.667):5.1f}%  median rms={np.median(rmsv[m]):.3f}")
    names = {1: "division", 2: "death", 3: "merge", 4: "ambiguous", 5: "run ended"}
    n_ev = {k: int(sum((f["ev"] == k).sum() for f in frames)) for k in (1, 2, 3, 4)}
    print("\nOutcome of each organism's transition (all organism-instances):", {names[k]: v for k, v in n_ev.items()})
    big = rmsv > np.percentile(rmsv, 99)
    print(f"Large shape changes = rms > {np.percentile(rmsv, 99):.3f} of radius (top 1%, n={big.sum()}); "
          f"share of those, and rate within group, by how the track next ends and how many frames away:")
    bins = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 1e9)]
    print(f"  {'next event':10s} " + " ".join(f"{('%d' % lo if lo == hi else ('%d-%d' % (lo, hi) if hi < 1e8 else '%d+' % lo)):>16s}" for lo, hi in bins))
    for k in (1, 2, 3, 4, 5):
        cells = []
        for lo, hi in bins:
            m = (typv == k) & (ttev >= lo) & (ttev <= hi)
            cells.append(f"{100*(big & m).sum()/max(big.sum(),1):5.1f}% /{100*(big & m).sum()/max(m.sum(),1):5.2f}%" if m.sum() else "              -")
        print(f"  {names[k]:10s} " + " ".join(f"{c:>16s}" for c in cells))
    print("  (each cell: share of all large changes / rate within that cell; n per cell below)")
    for k in (1, 2, 3, 4, 5):
        print(f"  n {names[k]:10s} " + " ".join(f"{int(((typv == k) & (ttev >= lo) & (ttev <= hi)).sum()):16d}" for lo, hi in bins))
    for k in (2, 3, 4):
        m = (typv == k) & (ttev >= 1) & (ttev <= 5)
        if m.sum() > 20:
            print(f"  median rms, 1-5 frames before {names[k]}: {np.median(rmsv[m]):.3f}  p90={np.percentile(rmsv[m],90):.3f}  (n={m.sum()})")
    nd = ~div
    print(f"\nexcluding division transitions: median={np.median(corr[nd]):.3f} p10={np.percentile(corr[nd],10):.3f} p1={np.percentile(corr[nd],1):.3f}")
    nd2 = other
    print(f"excluding division, pre- and post-division: median={np.median(corr[nd2]):.3f} p10={np.percentile(corr[nd2],10):.3f} p1={np.percentile(corr[nd2],1):.3f}")


if __name__ == "__main__":
    main()
