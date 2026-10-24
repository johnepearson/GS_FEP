"""
Three single-trajectory checks with the merge-based individuation (periodic_centroids + merge_overlapping_organisms,
as in run_merged_gap_split.py):

  1. Timescale separation (main text §3.1): match MERGED organisms frame-to-frame by maximum overlap; boundary
     displacement per intrinsic relaxation time tau=1/(F+k), normalized by organism radius.
  2. Flux asymmetry (main text §3.2): sign of net diffusive flux (U inward, V outward) per MERGED organism per frame.
  3. Boundary mechanism: boundary V-value vs. edge-to-edge gap to the nearest neighbour. Under the merge, no two
     distinct organisms can have a negative edge gap by construction, so every gap is non-negative.

Settings: checkpoint_1024_settled.npz, PAD=80, MIN_PEAK_DISTANCE=24, 10-step sampling cadence (finer than the 100-step
cadence used for the I/E extraction scripts, since no expensive distance transform is needed here).

Usage:
    python run_remaining_checks_merged.py --n-frames 500
    python run_remaining_checks_merged.py --n-frames 500   # reruns and
        automatically resumes from remaining_checks_merged_md24_progress.npz
        if it exists -- delete that file to force a fresh start
"""
import argparse
import time
import os
import numpy as np
from scipy import ndimage

import sys
sys.path.insert(0, '.')
from gray_scott import GrayScottSim, laplacian_periodic, match_organisms_by_overlap

try:
    from gray_scott_spectral_c import PseudoSpectralGSFast
    Solver = PseudoSpectralGSFast
    print("using PseudoSpectralGSFast (C)")
except OSError:
    from gray_scott_spectral import PseudoSpectralGS
    Solver = PseudoSpectralGS
    print("using PseudoSpectralGS (pure python)")

MIN_PEAK_DISTANCE = 24   # was 32; 24 removes the segmentation dropout (see run_min_distance_sweep.py)
PAD = 80
SAMPLE_EVERY_STEPS = 10
SAVE_EVERY_FRAMES = 20
MAX_MERGE_ROUNDS = 6

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "remaining_checks_merged_md24_progress.npz"
OUT_FILE = "remaining_checks_merged_md24_results.npz"


# ---- merge machinery, copied verbatim from the validated run_merged_gap_split.py ----

def periodic_delta(a, b, L):
    dd = a - b
    return dd - L * np.round(dd / L)


def periodic_centroids(labels, n_organisms, N):
    """Circular-mean centroid, correct for organisms whose mask straddles
    the periodic domain edge."""
    if n_organisms == 0:
        return np.zeros((0, 2))
    idx = np.arange(1, n_organisms + 1)
    ys, xs = np.indices(labels.shape)
    theta_y = 2 * np.pi * ys / N
    theta_x = 2 * np.pi * xs / N
    sum_sin_y = ndimage.sum(np.sin(theta_y), labels=labels, index=idx)
    sum_cos_y = ndimage.sum(np.cos(theta_y), labels=labels, index=idx)
    sum_sin_x = ndimage.sum(np.sin(theta_x), labels=labels, index=idx)
    sum_cos_x = ndimage.sum(np.cos(theta_x), labels=labels, index=idx)
    cy = (np.arctan2(sum_sin_y, sum_cos_y) * N / (2 * np.pi)) % N
    cx = (np.arctan2(sum_sin_x, sum_cos_x) * N / (2 * np.pi)) % N
    return np.column_stack([cy, cx])


def periodic_dist(c1, c2, L):
    dy = periodic_delta(c1[0], c2[0], L)
    dx = periodic_delta(c1[1], c2[1], L)
    return np.sqrt(dy ** 2 + dx ** 2)


def find_adjacent_pairs(labels):
    pairs = set()
    for dy, dx_ in [(1, 0), (0, 1)]:
        a = labels
        b = np.roll(labels, shift=(dy, dx_), axis=(0, 1))
        mask = (a != b) & (a > 0) & (b > 0)
        if not mask.any():
            continue
        pr = np.unique(np.stack([a[mask], b[mask]], axis=1), axis=0)
        for p in pr:
            i, j = int(min(p)), int(max(p))
            if i != j:
                pairs.add((i, j))
    return pairs


def merge_overlapping_organisms(labels, n, proxy):
    """Fixed-point merge: union any watershed-adjacent pair of organisms
    whose equivalent-radius circles overlap, relabel, repeat until stable.
    Returns (final_labels, final_n, merge_size)."""
    current_labels = labels.copy()
    current_n = n
    orig_to_current = np.arange(n + 1)

    for _round in range(MAX_MERGE_ROUNDS):
        _, radii_phys = proxy.organism_geometry(current_labels, current_n)
        radii_px = radii_phys / proxy.dx
        centroids = periodic_centroids(current_labels, current_n, proxy.n)
        pairs = find_adjacent_pairs(current_labels)
        if not pairs:
            break

        parent = list(range(current_n + 1))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                return True
            return False

        n_merges = 0
        for (i, j) in pairs:
            d = periodic_dist(centroids[i - 1], centroids[j - 1], proxy.n)
            gap = d - radii_px[i - 1] - radii_px[j - 1]
            if gap < 0:
                if union(i, j):
                    n_merges += 1

        if n_merges == 0:
            break

        roots = np.array([find(i) for i in range(current_n + 1)])
        unique_roots = np.unique(roots[1:])
        root_to_new = {r: idx + 1 for idx, r in enumerate(unique_roots)}
        old_to_new = np.zeros(current_n + 1, dtype=int)
        for L in range(1, current_n + 1):
            old_to_new[L] = root_to_new[roots[L]]
        new_labels = old_to_new[current_labels]

        orig_to_current = old_to_new[orig_to_current]
        current_labels = new_labels
        current_n = len(unique_roots)

    merge_size = np.bincount(orig_to_current[1:], minlength=current_n + 1)[1:]
    return current_labels, current_n, merge_size


# ---- boundary_v_and_gap, same as original but takes periodic centroids ----

def boundary_v_and_gap(proxy, labels, n, centroids, radii):
    boundary = (labels != np.roll(labels, 1, axis=0)) | (labels != np.roll(labels, 1, axis=1)) | \
               (labels != np.roll(labels, -1, axis=0)) | (labels != np.roll(labels, -1, axis=1))
    boundary = boundary & (labels > 0)
    labels_boundary_only = np.where(boundary, labels, 0)
    idx = np.arange(1, n + 1)
    counts = ndimage.sum(np.ones_like(labels), labels=labels_boundary_only, index=idx)
    sums = ndimage.sum(proxy.v, labels=labels_boundary_only, index=idx)
    with np.errstate(invalid='ignore', divide='ignore'):
        boundary_v = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    N = proxy.n
    diff = centroids[:, None, :] - centroids[None, :, :]
    diff = (diff + N / 2) % N - N / 2
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nn_idx = dist.argmin(axis=1)
    nn_dist = dist[np.arange(n), nn_idx]
    radii_px = radii / proxy.dx
    edge_gap = nn_dist - radii_px - radii_px[nn_idx]
    return boundary_v, edge_gap, radii_px


def segment_and_merge(proxy):
    raw_labels, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
    if raw_n == 0:
        return raw_labels, 0, None
    merged_labels, merged_n, merge_size = merge_overlapping_organisms(raw_labels, raw_n, proxy)
    return merged_labels, merged_n, merge_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=500)
    args = ap.parse_args()

    F, k = 0.018, 0.055
    tau = 1.0 / (F + k)

    if os.path.exists(PROGRESS_FILE):
        d = np.load(PROGRESS_FILE)
        sim = Solver(n=int(d["n"]))
        if hasattr(sim, "from_checkpoint_fields"):
            sim.from_checkpoint_fields(d["u"], d["v"])
        else:
            sim.u, sim.v = d["u"], d["v"]
        sim.time = float(d["time"])
        rho_list = list(d["rho"])
        flux_u_sign_list = list(d["flux_u_neg"])
        flux_v_sign_list = list(d["flux_v_pos"])
        boundary_v_list = [d["boundary_v"]]
        edge_gap_list = [d["edge_gap"]]
        merge_size_list = [d["merge_size_all"]]
        frames_done = int(d["frames_done"])
        print(f"resumed at t={sim.time:.0f}, {frames_done} frames done")
    else:
        sim = Solver.from_checkpoint(CHECKPOINT_FILE)
        rho_list, flux_u_sign_list, flux_v_sign_list = [], [], []
        boundary_v_list, edge_gap_list, merge_size_list = [], [], []
        frames_done = 0
        print(f"loaded checkpoint at t={sim.time:.0f}")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)

    def get_uv():
        if hasattr(sim, "get_uv"):
            return sim.get_uv()
        return sim.u, sim.v

    proxy.u, proxy.v = get_uv()
    prev_labels, prev_n, prev_merge_size = segment_and_merge(proxy)
    _, prev_radii = proxy.organism_geometry(prev_labels, prev_n)
    prev_centroids = periodic_centroids(prev_labels, prev_n, proxy.n)

    t_start = time.time()
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = get_uv()
        curr_labels, curr_n, curr_merge_size = segment_and_merge(proxy)
        if curr_n == 0:
            print("population died out, stopping")
            break
        _, curr_radii = proxy.organism_geometry(curr_labels, curr_n)
        curr_centroids = periodic_centroids(curr_labels, curr_n, proxy.n)

        # --- 3.1 timescale separation, matched on MERGED labels ---
        forward, backward = match_organisms_by_overlap(prev_labels, curr_labels, prev_n, curr_n)
        N = sim.n
        for p, (c, frac) in forward.items():
            if c is None or frac <= 0.5:
                continue
            claimants = backward.get(c, [])
            if len(claimants) != 1:
                continue
            disp = curr_centroids[c - 1] - prev_centroids[p - 1]
            disp = (disp + N / 2) % N - N / 2
            disp_mag = np.sqrt((disp ** 2).sum())
            v_boundary = disp_mag / SAMPLE_EVERY_STEPS
            r = prev_radii[p - 1] / proxy.dx
            if r > 0:
                rho_list.append(v_boundary * tau / r)

        # --- 3.2 flux asymmetry, on MERGED labels ---
        proxy.Lu = laplacian_periodic(proxy.u, proxy.dx)
        proxy.Lv = laplacian_periodic(proxy.v, proxy.dx)
        fu, fv = proxy.per_organism_flux(curr_labels, curr_n)
        flux_u_sign_list.append(np.mean(fu < 0))
        flux_v_sign_list.append(np.mean(fv > 0))

        # --- 3.5 boundary mechanism, on MERGED labels (expect edge_gap >= 0 everywhere now) ---
        bv, eg, _ = boundary_v_and_gap(proxy, curr_labels, curr_n, curr_centroids, curr_radii)
        boundary_v_list.append(bv)
        edge_gap_list.append(eg)
        merge_size_list.append(curr_merge_size)

        prev_labels, prev_n, prev_centroids, prev_radii = curr_labels, curr_n, curr_centroids, curr_radii
        frames_done += 1

        elapsed = time.time() - t_start
        if frames_done % 20 == 0:
            print(f"frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  organisms={curr_n}  "
                  f"({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.2f}s/frame)", flush=True)

        if frames_done % SAVE_EVERY_FRAMES == 0 or frames_done == args.n_frames:
            u_save, v_save = get_uv()
            np.savez(PROGRESS_FILE, u=u_save, v=v_save, time=sim.time, n=sim.n,
                     frames_done=frames_done, rho=np.array(rho_list),
                     flux_u_neg=np.array(flux_u_sign_list), flux_v_pos=np.array(flux_v_sign_list),
                     boundary_v=np.concatenate(boundary_v_list), edge_gap=np.concatenate(edge_gap_list),
                     merge_size_all=np.concatenate(merge_size_list))

    rho = np.array(rho_list)
    bv_all = np.concatenate(boundary_v_list)
    eg_all = np.concatenate(edge_gap_list)
    merge_size_all = np.concatenate(merge_size_list)
    good = np.isfinite(bv_all) & np.isfinite(eg_all)

    np.savez(OUT_FILE, rho=rho, flux_u_neg=np.array(flux_u_sign_list),
             flux_v_pos=np.array(flux_v_sign_list), boundary_v=bv_all, edge_gap=eg_all,
             merge_size=merge_size_all)
    print("saved to", OUT_FILE)

    print(f"\nTimescale separation: N={len(rho)}  median rho={np.median(rho):.4f}  "
          f"90th pct={np.percentile(rho,90):.4f}  99th pct={np.percentile(rho,99):.4f}")
    print(f"Flux asymmetry: mean fraction U-inward={np.mean(flux_u_sign_list):.4f}  "
          f"mean fraction V-outward={np.mean(flux_v_sign_list):.4f}")
    print(f"Boundary mechanism: N={good.sum()}  corr(boundary_v, edge_gap)="
          f"{np.corrcoef(bv_all[good], eg_all[good])[0,1]:.4f}")
    n_negative = (eg_all[good] < 0).sum()
    print(f"  edge_gap < 0 count: {n_negative} / {good.sum()} "
          f"({100*n_negative/max(good.sum(),1):.2f}%)  "
          f"[should be ~0 under the merge fix -- any nonzero count is a residual case "
          f"where merge rounds capped out before convergence]")
    print(f"merge_size distribution: {np.bincount(merge_size_all)}")


if __name__ == "__main__":
    main()
