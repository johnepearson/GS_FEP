"""
Merge-aware redo of run_resolution_2x2.py (the main text \\S3.7 /
ESM \\S~S6 resolution x interior-definition factorial), which never got
the "spots don't overlap" merge-based segmentation fix used everywhere
else in this revision -- it still runs raw watershed with no step
fusing overlapping organisms, and only saves the final aggregated
per-organism I/E/flux arrays (not per-frame labels or fields), so the
existing resolution_2x2_{256,1024}_progress.npz files cannot be
retrofitted after the fact. This reruns the full 2x2 design from
scratch with the merge step inserted, at the same production scale
(400 frames) and the same two resolutions {256, 1024} crossed with the
same two interior definitions {whole-mask adaptive erosion,
local-distance-transform}, computed on the SAME merged organisms within
each resolution so the two definitions remain exactly comparable.

Merge machinery (merge_overlapping_organisms, periodic_centroids, etc.)
is copied verbatim from the validated run_merged_gap_split.py /
run_decay_curve_merged.py, inserted right after segment_organisms and
before flux/I/E extraction, exactly as in those scripts.

Same settle/sample/solver conventions as the original script
(SETTLE_STEPS=3800, SAMPLE_EVERY_STEPS=100, same initial condition,
same --resume-checkpoint support for the 1024 arm).

Usage:
    python run_resolution_2x2_merged.py --resolution 256 --n-frames 400
    python run_resolution_2x2_merged.py --resolution 1024 --n-frames 400 --resume-checkpoint checkpoint_1024_settled.npz
    python run_resolution_2x2_merged.py --resolution 256 --resume
"""
import argparse
import time
import numpy as np
from scipy import ndimage

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
from run_confound_checks import (organism_interior_exterior_states_fast,
                                  organism_interior_exterior_states_adaptive_fast)

try:
    from gray_scott_spectral_c import PseudoSpectralGSFast
    _HAVE_FAST = True
except OSError:
    _HAVE_FAST = False

SETTLE_STEPS = 3800
SAMPLE_EVERY_STEPS = 100
SAVE_EVERY_FRAMES = 10
DEPTH_FRACTION = 0.5
KEEP_FRACTION = 0.5
MAX_MERGE_ROUNDS = 6


def params_for_resolution(n, min_distance=None):
    """min_distance: marker spacing in pixels. Default 6 * n/256 (= 24 at 1024, the spacing used for all
    1024 results after the dropout fix; the earlier default of 8 * n/256 was the old 32 at 1024)."""
    scale = n / 256
    return dict(
        pad=int(round(20 * scale)),
        min_peak_distance=int(min_distance) if min_distance else int(round(6 * scale)),
        exterior_ring_px=int(round(6 * scale)),
        crop_ie=int(round(22.5 * scale)),
    )


def make_initial_field(n, F, k):
    c = n // 2
    half = max(1, int(round(10 * n / 256)))
    u = np.ones((n, n), dtype=np.float64)
    v = np.zeros((n, n), dtype=np.float64)
    sl = slice(c - half, c + half)
    u[sl, sl] = 0.5
    v[sl, sl] = 0.25
    rng = np.random.default_rng(0)
    u *= 1.0 + 0.01 * rng.uniform(-1, 1, size=(n, n))
    mask = v > 0
    v[mask] *= 1.0 + 0.01 * rng.uniform(-1, 1, size=(n, n))[mask]
    return u, v


# ---- merge machinery, copied verbatim from the validated run_merged_gap_split.py ----

def periodic_delta(a, b, L):
    dd = a - b
    return dd - L * np.round(dd / L)


def periodic_centroids(labels, n_organisms, N):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[256, 1024])
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-checkpoint", type=str, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--min-distance", type=int, default=None,
                    help="marker spacing in px (default 6 at 256, 24 at 1024)")
    args = ap.parse_args()

    if args.fast and not _HAVE_FAST:
        raise SystemExit("--fast requested but libgs_spectral.so isn't built for this machine.")
    Solver = PseudoSpectralGSFast if args.fast else PseudoSpectralGS

    n = args.resolution
    P = params_for_resolution(n, args.min_distance)
    md = P["min_peak_distance"]
    progress_file = f"resolution_2x2_merged_{n}_md{md}_progress.npz"
    print(f"n={n}  marker spacing min_distance={md} px  -> {progress_file}")
    keys = ["I_adapt", "E_adapt", "I_dist", "E_dist", "fu", "fv", "merge_size"]

    if args.resume:
        d = np.load(progress_file)
        sim = Solver(n=n, F=0.018, k=0.055, dt=1.0)
        sim.from_checkpoint_fields(d["u"], d["v"])
        sim.time = float(d["time"])
        lists = {k: [d[k]] for k in keys}
        frames_done = int(d["frames_done"])
        print(f"resumed n={n} at t={sim.time:.0f}, {frames_done} frames done")
    elif args.resume_checkpoint:
        sim = Solver.from_checkpoint(args.resume_checkpoint)
        lists = {k: [] for k in keys}
        frames_done = 0
        print(f"loaded checkpoint {args.resume_checkpoint} at n={sim.n}, t={sim.time:.0f}")
    else:
        u0, v0 = make_initial_field(n, 0.018, 0.055)
        sim = Solver(n=n, F=0.018, k=0.055, dt=1.0)
        sim.from_checkpoint_fields(u0, v0)
        print(f"settling n={n} field for {SETTLE_STEPS} steps...")
        t0 = time.time()
        sim.run(SETTLE_STEPS)
        print(f"  settled at t={sim.time:.0f} ({time.time()-t0:.0f}s)")
        lists = {k: [] for k in keys}
        frames_done = 0

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)

    t_start = time.time()
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        if hasattr(sim, "get_uv"):
            proxy.u, proxy.v = sim.get_uv()
        else:
            proxy.u, proxy.v = sim.u, sim.v
        raw_labels, raw_n = proxy.segment_organisms(pad=P["pad"], min_peak_distance=P["min_peak_distance"])

        if raw_n > 0:
            labels, n_org, merge_size = merge_overlapping_organisms(raw_labels, raw_n, proxy)

            proxy.Lu = laplacian_periodic(sim.u, sim.dx)
            proxy.Lv = laplacian_periodic(sim.v, sim.dx)
            flux_u, flux_v = proxy.per_organism_flux(labels, n_org)
            centroids = periodic_centroids(labels, n_org, proxy.n)

            I_a, E_a = organism_interior_exterior_states_adaptive_fast(
                proxy, labels, n_org, centroids, keep_fraction=KEEP_FRACTION,
                exterior_ring_px=P["exterior_ring_px"], crop=P["crop_ie"])
            I_d, E_d = organism_interior_exterior_states_fast(
                proxy, labels, n_org, centroids, depth_fraction=DEPTH_FRACTION,
                exterior_ring_px=P["exterior_ring_px"], crop=P["crop_ie"])

            for k, v_ in zip(keys, [I_a, E_a, I_d, E_d, flux_u, flux_v, merge_size]):
                lists[k].append(v_)
        else:
            n_org = 0

        frames_done += 1
        elapsed = time.time() - t_start
        if frames_done % 10 == 0:
            n_so_far = sum(len(x) for x in lists["I_adapt"])
            print(f"n={n} frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  "
                  f"raw_n={raw_n}  merged_n={n_org}  N so far={n_so_far}  "
                  f"({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.2f}s/frame)")

        if frames_done % SAVE_EVERY_FRAMES == 0 or frames_done == args.n_frames:
            save_kwargs = {k: np.concatenate(v_) for k, v_ in lists.items()}
            np.savez(progress_file, u=sim.u, v=sim.v, time=sim.time, n=sim.n,
                     frames_done=frames_done, **save_kwargs)

    N_total = sum(len(x) for x in lists["I_adapt"])
    print(f"\ndone: n={n}, {frames_done} frames, N={N_total} organism-instances, "
          f"saved to {progress_file}")
    ms_all = np.concatenate(lists["merge_size"]) if lists["merge_size"] else np.array([], dtype=int)
    if len(ms_all):
        print(f"fraction that are merges of 2+ original sub-regions: {(ms_all > 1).mean():.4f}")


if __name__ == "__main__":
    main()
