"""
Merge-aware redo of run_ensemble_clustered.py -- the script behind
sec:conditional_independence's 3.3.2 ("Linear partial correlation,
independent ensemble") headline numbers (pooled r=-0.151/-0.153 on
~560-610k organism-instances from 1000 independent noise-perturbed
realizations, plus the per-realization distribution and realization-
cluster bootstrap in Figure realization_distribution).

Same experimental design as run_ensemble_clustered.py, same seed range
(9000+i) for direct trajectory-level comparability -- the underlying
u/v dynamics are deterministic and unaffected by the merge fix, so
reusing these seeds means any difference between this dataset and the
original is attributable ONLY to the "spots don't overlap" merge-based
segmentation, not to a different random draw of realizations.

The one substantive change: after segmenting each sample, organisms
whose equivalent-radius circles overlap (negative edge-to-edge gap)
are merged into one before flux/I/E are computed, via the same
fixed-point union-find merge (merge_overlapping_organisms) and
periodic-correct centroid (periodic_centroids) already validated in
run_merged_gap_split.py and run_remaining_checks_merged.py. Also
additionally records merge_size per organism-instance (diagnostic
only, not used in the partial correlation).

NOTE: this script now also covers the PRIMARY ensemble (seeds 1000+i,
run_ensemble_parallel.py's dataset, N=559,147 pre-merge-fix -- sec
3.3.2's FIRST paragraph, and the source dataset for sec 3.3.3's CMI
N=8000 subsample), via --seed-base 1000, in addition to the second/
robustness ensemble (seeds 9000+i, --seed-base 9000, the default --
already run and reported separately). Same script, same merge fix,
independent seed ranges -- just run it twice, once per seed base.
Progress files are named ensemble_merged_seed{seed_base}_worker{id}.npz,
so the two runs' outputs never collide and can proceed independently
(or concurrently, resources permitting, though 24 processes at once on
a 12-core machine is not recommended -- finish one seed base's 12
workers before starting the other).

Usage (12 parallel worker processes -- matches the 12 FFTW threads this
machine reported; GS_FFT_THREADS=1 per worker avoids threads-within-
processes contention, since each worker gets its own core this way):

    for i in $(seq 0 11); do
        GS_FFT_THREADS=1 python3 run_ensemble_merged.py \\
            --n-realizations 1000 --n-workers 12 --worker-id $i --fast \\
            --seed-base 1000 \\
            > ensemble_merged_seed1000_worker_$i.log 2>&1 &
    done
    wait

(Omit --seed-base, or pass --seed-base 9000 explicitly, to rerun the
already-completed second ensemble instead -- not needed again unless
a worker's output was lost.)

Resume an interrupted worker with --resume (per-worker, exactly as in
run_ensemble_clustered.py -- each realization is independent, so
resuming never redoes completed work). If a worker dies, just rerun
its own command above with --resume added; the others are unaffected.

Rough cost estimate (from the smoke test and run_remaining_checks_merged.py's
measured ~0.069s/integration-step on this machine): 3300 steps/realization
(3000 burn-in + 3x100 sampling) is about 3.8 minutes/realization, so 1000
realizations is roughly 63 single-core-hours total. Split 12 ways that's
roughly 5-6 hours wall-clock (1000/12 ~= 84 realizations/worker), though
real throughput depends on how the 12 FFTW threads were actually
distributed across physical cores when idle vs. all 12 processes busy at
once -- running all 12 workers simultaneously will contend for cores in
a way a single worker's benchmark doesn't capture, so treat 5-6h as
optimistic and expect it could run longer. It's fully resumable per
worker, so it's safe to background with nohup and check back later.
"""
import argparse
import sys
import time
import numpy as np
from scipy import ndimage

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
from run_confound_checks import organism_interior_exterior_states_fast

try:
    from gray_scott_spectral_c import PseudoSpectralGSFast
    _HAVE_FAST = True
except OSError:
    _HAVE_FAST = False

DEPTH_FRACTION = 0.5
EXTERIOR_RING_PX = 24
MIN_PEAK_DISTANCE = 24     # was 32; 24 removes the segmentation dropout (see run_min_distance_sweep.py)
PAD = 80
CROP_IE = 90
MAX_MERGE_ROUNDS = 6

BURN_IN_STEPS = 3000
N_SAMPLES_PER_REALIZATION = 3
SAMPLE_SPACING_STEPS = 100
NOISE_SIGMA = 1e-3

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"

HEARTBEAT_CHUNK_STEPS = 300

ARCHIVE_EVERY = 10          # save u, v and the final labels of every 10th realization (0 = never)
ARCHIVE_DIR = "ensemble_fields"


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


EXTRA_KEYS = ["sample_idx", "cy", "cx", "radius_px", "area_px", "nn_gap_px", "n_org_frame", "raw_n_frame"]


def nearest_edge_gap(centroids, radius_px, N):
    """Edge-to-edge gap (px) to the nearest other organism, periodic, using
    equivalent-radius circles (the same gap definition the merge step uses)."""
    n = len(centroids)
    if n < 2:
        return np.full(n, np.nan)
    dy = centroids[:, None, 0] - centroids[None, :, 0]
    dx = centroids[:, None, 1] - centroids[None, :, 1]
    dy -= N * np.round(dy / N); dx -= N * np.round(dx / N)
    d = np.hypot(dy, dx) - radius_px[:, None] - radius_px[None, :]
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def run_one_realization(seed, burn_in_steps, Solver, worker_id=None, log=print, archive=False):
    tag = f"[worker {worker_id}] " if worker_id is not None else ""
    rng = np.random.default_rng(seed)
    sp = Solver.from_checkpoint(CHECKPOINT_FILE)
    u_noisy = sp.u + NOISE_SIGMA * rng.standard_normal((sp.n, sp.n))
    v_noisy = sp.v + NOISE_SIGMA * rng.standard_normal((sp.n, sp.n))
    sp.from_checkpoint_fields(u_noisy, v_noisy)

    t0 = time.time()
    done = 0
    while done < burn_in_steps:
        chunk = min(HEARTBEAT_CHUNK_STEPS, burn_in_steps - done)
        sp.run(chunk)
        done += chunk
        elapsed = time.time() - t0
        log(f"{tag}  seed={seed}  burn-in {done}/{burn_in_steps} steps "
            f"({elapsed:.0f}s elapsed, ~{elapsed/done*1000:.0f}ms/step)")

    I_list, E_list, fu_list, fv_list, ms_list = [], [], [], [], []
    extra = {k: [] for k in EXTRA_KEYS}
    arch = {}
    for s_idx in range(N_SAMPLES_PER_REALIZATION):
        sp.run(SAMPLE_SPACING_STEPS)
        log(f"{tag}  seed={seed}  sample {s_idx+1}/{N_SAMPLES_PER_REALIZATION} stepped "
            f"({time.time()-t0:.0f}s elapsed total)")
        proxy = GrayScottSim(n=sp.n, L=sp.L, Du=sp.Du, Dv=sp.Dv, F=sp.F, k=sp.k, dt=sp.dt)
        if hasattr(sp, "get_uv"):
            proxy.u, proxy.v = sp.get_uv()
        else:
            proxy.u, proxy.v = sp.u, sp.v
        raw_labels, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
        if raw_n == 0:
            continue

        labels, n, merge_size = merge_overlapping_organisms(raw_labels, raw_n, proxy)

        proxy.Lu = laplacian_periodic(proxy.u, proxy.dx)
        proxy.Lv = laplacian_periodic(proxy.v, proxy.dx)
        flux_u, flux_v = proxy.per_organism_flux(labels, n)
        # periodic-correct centroid -- plain organism_geometry centroids are wrong for any
        # merged organism whose mask straddles the domain edge
        centroids = periodic_centroids(labels, n, proxy.n)
        I, E = organism_interior_exterior_states_fast(
            proxy, labels, n, centroids, depth_fraction=DEPTH_FRACTION,
            exterior_ring_px=EXTERIOR_RING_PX, crop=CROP_IE)
        I_list.append(I); E_list.append(E); fu_list.append(flux_u); fv_list.append(flux_v)
        ms_list.append(merge_size)

        # per-instance covariates, so later confound / stratification checks need no new simulation
        _, radii_phys = proxy.organism_geometry(labels, n)
        radius_px = radii_phys / proxy.dx
        area_px = np.pi * radius_px ** 2
        gap = nearest_edge_gap(centroids, radius_px, proxy.n)
        extra["sample_idx"].append(np.full(n, s_idx, dtype=np.int8))
        extra["cy"].append(centroids[:, 0]); extra["cx"].append(centroids[:, 1])
        extra["radius_px"].append(radius_px); extra["area_px"].append(area_px)
        extra["nn_gap_px"].append(gap)
        extra["n_org_frame"].append(np.full(n, n, dtype=np.int16))
        extra["raw_n_frame"].append(np.full(n, raw_n, dtype=np.int16))
        if archive:
            arch[f"u{s_idx}"] = proxy.u.astype(np.float32)
            arch[f"v{s_idx}"] = proxy.v.astype(np.float32)
            arch[f"labels{s_idx}"] = labels.astype(np.int16)
            arch[f"raw_labels{s_idx}"] = raw_labels.astype(np.int16)

    if archive and arch:
        import os
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(ARCHIVE_DIR, f"seed{seed}.npz"), min_peak_distance=MIN_PEAK_DISTANCE, **arch)

    if not I_list:
        return ((np.array([]), np.array([]), np.array([]), np.array([]), np.array([], dtype=int)),
                {k: np.array([]) for k in EXTRA_KEYS})
    return ((np.concatenate(I_list), np.concatenate(E_list),
             np.concatenate(fu_list), np.concatenate(fv_list), np.concatenate(ms_list)),
            {k: np.concatenate(v) for k, v in extra.items()})


def save_progress(path, seeds_done, I_all, E_all, fu_all, fv_all, realization_id_all, merge_size_all, extra_all=None):
    extra_all = extra_all or {}
    np.savez(path, seeds_done=np.array(seeds_done),
             I=I_all, E=E_all, fu=fu_all, fv=fv_all,
             realization_id=realization_id_all, merge_size=merge_size_all,
             min_peak_distance=MIN_PEAK_DISTANCE, pad=PAD, depth_fraction=DEPTH_FRACTION,
             exterior_ring_px=EXTERIOR_RING_PX, crop_ie=CROP_IE, burn_in_steps=BURN_IN_STEPS,
             sample_spacing_steps=SAMPLE_SPACING_STEPS, noise_sigma=NOISE_SIGMA,
             **extra_all)


def load_progress(path):
    d = np.load(path)
    missing = [k for k in EXTRA_KEYS if k not in d.files]
    if missing:
        raise SystemExit(f"{path} was written by an older version of this script (no {missing}); "
                         "start fresh without --resume")
    if "sample_spacing_steps" in d.files and int(d["sample_spacing_steps"]) != SAMPLE_SPACING_STEPS:
        raise SystemExit(f"{path} used sample_spacing_steps={int(d['sample_spacing_steps'])}, now {SAMPLE_SPACING_STEPS}; "
                         "do not mix settings in one resumed run")
    if "burn_in_steps" in d.files and int(d["burn_in_steps"]) != BURN_IN_STEPS:
        raise SystemExit(f"{path} used burn_in_steps={int(d['burn_in_steps'])}, now {BURN_IN_STEPS}; "
                         "do not mix settings in one resumed run")
    if int(d["min_peak_distance"]) != MIN_PEAK_DISTANCE:
        raise SystemExit(f"{path} used min_peak_distance={int(d['min_peak_distance'])}, now {MIN_PEAK_DISTANCE}")
    return (list(d["seeds_done"]), [d["I"]], [d["E"]], [d["fu"]], [d["fv"]],
            [d["realization_id"]], [d["merge_size"]], {k: [d[k]] for k in EXTRA_KEYS})


def main():
    global MIN_PEAK_DISTANCE, SAMPLE_SPACING_STEPS, BURN_IN_STEPS
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-realizations", type=int, required=True,
                     help="TOTAL realizations across all workers combined")
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--worker-id", type=int, required=True, help="0-indexed")
    ap.add_argument("--burn-in-steps", type=int, default=BURN_IN_STEPS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fast", action="store_true",
                     help="use the C solver (libgs_spectral.so). Strongly recommended "
                          "at this scale (1000 realizations) -- see README.md's C "
                          "solver build instructions if libgs_spectral.so isn't built.")
    ap.add_argument("--seed-base", type=int, default=9000,
                     help="Seed range base: seeds are {seed_base}+i. Default 9000 matches "
                          "run_ensemble_clustered.py's second/robustness ensemble (sec "
                          "3.3.2's realization-distribution and cluster-bootstrap check). "
                          "Pass --seed-base 1000 instead to target the PRIMARY ensemble "
                          "(run_ensemble_parallel.py's seeds, sec 3.3.2's first paragraph "
                          "and the sec 3.3.3 CMI subsample's source dataset) -- same script, "
                          "same merge fix, just a different, independent set of 1000 "
                          "physical realizations, reanalyzed/regenerated the same way.")
    ap.add_argument("--sample-spacing-steps", type=int, default=SAMPLE_SPACING_STEPS,
                     help="integration steps between the 3 samples of one realization (default 100; "
                          "use 1000 for the long-spacing rerun)")
    ap.add_argument("--min-distance", type=int, default=MIN_PEAK_DISTANCE,
                     help="watershed marker spacing (default 24)")
    ap.add_argument("--archive-every", type=int, default=ARCHIVE_EVERY,
                     help="save u, v and labels for every N-th realization (0 = never); about 10 MB "
                          "per sample before compression")
    args = ap.parse_args()
    MIN_PEAK_DISTANCE = args.min_distance
    SAMPLE_SPACING_STEPS = args.sample_spacing_steps
    BURN_IN_STEPS = args.burn_in_steps

    if args.fast and not _HAVE_FAST:
        raise SystemExit("--fast requested but libgs_spectral.so isn't built for this "
                          "machine. Build it first (see README.md), or drop --fast "
                          "(NOT recommended at 1000 realizations -- will be very slow).")
    Solver = PseudoSpectralGSFast if args.fast else PseudoSpectralGS

    progress_file = f"ensemble_merged_seed{args.seed_base}_worker{args.worker_id}.npz"

    # Seed range is {seed_base}+i. With the default seed_base=9000, this is the
    # SAME 1000 physical realizations as run_ensemble_clustered.py (deliberately
    # reused for direct trajectory-level comparability -- any difference from
    # that dataset's numbers is then attributable only to the segmentation
    # change). With --seed-base 1000, this instead targets the PRIMARY
    # ensemble's seed range (run_ensemble_parallel.py), for the same reason.
    my_seeds = [args.seed_base + i for i in range(args.n_realizations) if i % args.n_workers == args.worker_id]

    if args.resume:
        seeds_done, I_list, E_list, fu_list, fv_list, rid_list, ms_list, extra_lists = load_progress(progress_file)
        print(f"[worker {args.worker_id}] resumed: {len(seeds_done)}/{len(my_seeds)} realizations done")
    else:
        seeds_done = []
        I_list, E_list, fu_list, fv_list, rid_list, ms_list = [], [], [], [], [], []
        extra_lists = {k: [] for k in EXTRA_KEYS}
        print(f"[worker {args.worker_id}] starting: responsible for {len(my_seeds)} realizations "
              f"({'C/fast' if args.fast else 'pure-Python'} solver)")

    remaining_seeds = [s for s in my_seeds if s not in seeds_done]
    t_start = time.time()
    for idx, seed in enumerate(remaining_seeds):
        archive = args.archive_every > 0 and (seed - args.seed_base) % args.archive_every == 0
        (I, E, fu, fv, ms), ex = run_one_realization(
            seed, args.burn_in_steps, Solver, worker_id=args.worker_id,
            log=lambda msg: print(msg, flush=True), archive=archive)
        if len(I) > 0:
            for k in EXTRA_KEYS:
                extra_lists[k].append(ex[k])
            I_list.append(I); E_list.append(E); fu_list.append(fu); fv_list.append(fv)
            rid_list.append(np.full(len(I), seed, dtype=np.int64))
            ms_list.append(ms)
        seeds_done.append(seed)

        elapsed = time.time() - t_start
        done_this_session = idx + 1
        n_so_far = sum(len(x) for x in I_list)
        eta_h = (len(remaining_seeds) - done_this_session) * elapsed / done_this_session / 3600
        print(f"[worker {args.worker_id}] {len(seeds_done)}/{len(my_seeds)}  "
              f"({elapsed:.0f}s elapsed this session, ~{elapsed/done_this_session:.0f}s/realization, "
              f"~{eta_h:.1f}h remaining)  organism-instances so far: {n_so_far}")

        I_all = np.concatenate(I_list) if I_list else np.array([])
        E_all = np.concatenate(E_list) if E_list else np.array([])
        fu_all = np.concatenate(fu_list) if fu_list else np.array([])
        fv_all = np.concatenate(fv_list) if fv_list else np.array([])
        rid_all = np.concatenate(rid_list) if rid_list else np.array([], dtype=np.int64)
        ms_all = np.concatenate(ms_list) if ms_list else np.array([], dtype=int)
        extra_all = {k: (np.concatenate(v) if v else np.array([])) for k, v in extra_lists.items()}
        save_progress(progress_file, seeds_done, I_all, E_all, fu_all, fv_all, rid_all, ms_all, extra_all)

    print(f"[worker {args.worker_id}] done: {len(seeds_done)}/{len(my_seeds)} realizations")


if __name__ == "__main__":
    main()
