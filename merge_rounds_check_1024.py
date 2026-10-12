"""
merge_rounds_check_1024.py

How many rounds does the merge-correction loop actually need on real frames?

Background. The paper's merge-based segmentation (run_ensemble_merged.py,
merge_overlapping_organisms) fuses touching organisms whose equivalent-radius
circles overlap. It does this in rounds: fuse, recompute every radius and
centroid, repeat. The production loop is capped at MAX_MERGE_ROUNDS = 6
iterations. Two questions were left open when we read the code:

  1. How many merge rounds do real frames need? Does any frame use more
     than 2, and does any frame hit the cap of 6 before converging (which
     would leave overlapping touching pairs unmerged)?
  2. The loop only tests pairs of organisms that TOUCH. Are there pairs that
     do NOT touch but whose equivalent-radius circles overlap anyway, which
     the loop never looks at?

This script answers both on frames from the settled 1024 checkpoint, using
the production segmentation (segment_organisms, pad=80, min_peak_distance=24)
exactly as run_ensemble_merged.py does.

Definitions used in the output:
  merge rounds   = rounds in which at least one fusion happened. (The loop
                   also spends one extra, final iteration confirming that
                   nothing is left to fuse; that iteration is not counted.)
  hit the cap    = the production loop's 6th iteration was still fusing, so
                   it exited without confirming convergence.

Fidelity. trace_merge() below is a line-for-line copy of the production
loop with counters added. On every real frame the script also runs the
production function itself and checks that both give identical labels, n and
merge_size, so any divergence between this copy and production is caught.

Run (from the local_package directory, like the other checks):
    python3 merge_rounds_check_1024.py --self-test
    python3 merge_rounds_check_1024.py
Optional: --frames 50 --stride 100   (defaults shown; stride matches the
production sampling spacing of 100 steps)
"""
import argparse
import sys
import time

import numpy as np

import run_ensemble_merged as prod
from gray_scott import GrayScottSim

CHECKPOINT = prod.CHECKPOINT_FILE
PAD = prod.PAD
MIN_PEAK_DISTANCE = prod.MIN_PEAK_DISTANCE
PROD_CAP = prod.MAX_MERGE_ROUNDS
BIG_CAP = 50          # effectively "run to convergence"


def trace_merge(labels, n, proxy, max_rounds):
    """Line-for-line copy of prod.merge_overlapping_organisms with tracing.
    Returns (final_labels, final_n, merge_size, trace) where trace has:
      merge_rounds : number of rounds in which >=1 fusion happened
      hit_cap      : True if the loop used all max_rounds iterations
                     and the last one was still fusing
      merges       : list, number of fusions in each merge round
    """
    current_labels = labels.copy()
    current_n = n
    orig_to_current = np.arange(n + 1)
    merges = []
    last_round_merged = False
    iterations = 0

    for _round in range(max_rounds):
        iterations += 1
        last_round_merged = False
        _, radii_phys = proxy.organism_geometry(current_labels, current_n)
        radii_px = radii_phys / proxy.dx
        centroids = prod.periodic_centroids(current_labels, current_n, proxy.n)
        pairs = prod.find_adjacent_pairs(current_labels)
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
            d = prod.periodic_dist(centroids[i - 1], centroids[j - 1], proxy.n)
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
        merges.append(n_merges)
        last_round_merged = True

    merge_size = np.bincount(orig_to_current[1:], minlength=current_n + 1)[1:]
    trace = dict(merge_rounds=len(merges),
                 hit_cap=bool(iterations == max_rounds and last_round_merged),
                 merges=merges)
    return current_labels, current_n, merge_size, trace


def overlapping_pairs(labels, n, proxy):
    """All pairs of organisms whose equivalent-radius circles overlap
    (periodic centroid distance minus both radii < 0), split into pairs that
    touch (4-neighbour adjacency, the same adjacency the merge loop uses)
    and pairs that do not touch. Returns (touching, not_touching), each a
    list of (i, j, gap_px)."""
    if n < 2:
        return [], []
    _, radii_phys = proxy.organism_geometry(labels, n)
    radii_px = radii_phys / proxy.dx
    cent = prod.periodic_centroids(labels, n, proxy.n)
    adj = prod.find_adjacent_pairs(labels)
    touching, not_touching = [], []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            d = prod.periodic_dist(cent[i - 1], cent[j - 1], proxy.n)
            gap = d - radii_px[i - 1] - radii_px[j - 1]
            if gap < 0:
                (touching if (i, j) in adj else not_touching).append((i, j, gap))
    return touching, not_touching


def analyze_frame(raw_labels, raw_n, proxy):
    """Everything this script measures for one segmented frame."""
    # production result (cap 6) and traced copy at the same cap must agree
    p_labels, p_n, p_ms = prod.merge_overlapping_organisms(raw_labels, raw_n, proxy)
    t_labels, t_n, t_ms, t_trace = trace_merge(raw_labels, raw_n, proxy, PROD_CAP)
    identical = (p_n == t_n and np.array_equal(p_labels, t_labels)
                 and np.array_equal(p_ms, t_ms))
    # run to convergence for the number of rounds actually needed
    c_labels, c_n, c_ms, c_trace = trace_merge(raw_labels, raw_n, proxy, BIG_CAP)
    touching, not_touching = overlapping_pairs(p_labels, p_n, proxy)
    return dict(raw_n=raw_n, final_n=p_n,
                identical_to_production=identical,
                rounds_needed=c_trace["merge_rounds"],
                merges_per_round=c_trace["merges"],
                hit_cap=t_trace["hit_cap"],
                n_differs_at_convergence=(c_n != p_n),
                touching_overlap=touching,
                nontouching_overlap=not_touching)


def summarize(results):
    nf = len(results)
    rounds = np.array([r["rounds_needed"] for r in results])
    print(f"\n=== merge rounds needed, over {nf} frames ===")
    vals, counts = np.unique(rounds, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  {v} merge round(s): {c} frame(s)")
    print(f"  frames needing MORE THAN 2 merge rounds: {(rounds > 2).sum()} of {nf}")
    print(f"  most rounds any frame needed: {rounds.max()}")
    n_cap = sum(r["hit_cap"] for r in results)
    n_diff = sum(r["n_differs_at_convergence"] for r in results)
    print(f"  frames that hit the production cap ({PROD_CAP}) while still fusing: {n_cap}")
    print(f"  frames where running to full convergence changes the organism count: {n_diff}")
    n_bad = sum(not r["identical_to_production"] for r in results)
    print(f"  traced copy vs production function: "
          f"{'identical on all frames' if n_bad == 0 else f'MISMATCH on {n_bad} frames (do not trust this run)'}")

    tot_org = sum(r["final_n"] for r in results)
    tt = sum(len(r["touching_overlap"]) for r in results)
    nt_pairs = [p for r in results for p in r["nontouching_overlap"]]
    print(f"\n=== overlapping equivalent-radius circles AFTER the production merge ===")
    print(f"  organisms in total over all frames: {tot_org}")
    print(f"  touching pairs still overlapping: {tt}"
          f"  (expected 0 unless the cap was hit)")
    print(f"  NON-touching pairs overlapping:   {len(nt_pairs)}"
          f"  in {sum(1 for r in results if r['nontouching_overlap'])} of {nf} frames")
    if nt_pairs:
        gaps = np.array([g for (_, _, g) in nt_pairs])
        print(f"    overlap depth (px): median {np.median(gaps):.2f}, "
              f"most overlapping {gaps.min():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--stride", type=int, default=100)
    args = ap.parse_args()

    Solver = prod.PseudoSpectralGSFast if prod._HAVE_FAST else prod.PseudoSpectralGS
    print(f"Loading real settled checkpoint: {CHECKPOINT}")
    sim = Solver.from_checkpoint(CHECKPOINT)
    print(f"Grid {sim.n}x{sim.n}, t={sim.time}, step_count={sim.step_count}")
    print(f"{args.frames} frames, {args.stride} steps apart; production segmentation "
          f"(pad={PAD}, min_peak_distance={MIN_PEAK_DISTANCE}), production cap = {PROD_CAP}\n")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F,
                         k=sim.k, dt=sim.dt)
    results = []
    t0 = time.time()
    for f in range(args.frames):
        sim.run(args.stride)
        proxy.u, proxy.v = sim.get_uv() if hasattr(sim, "get_uv") else (sim.u, sim.v)
        raw_labels, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
        if raw_n == 0:
            continue
        r = analyze_frame(raw_labels, raw_n, proxy)
        results.append(r)
        print(f"  frame {f:2d} (t={sim.time:.0f}): raw {r['raw_n']} -> {r['final_n']} organisms, "
              f"merge rounds {r['rounds_needed']} {r['merges_per_round']}"
              f"{'  HIT CAP' if r['hit_cap'] else ''}"
              f"  non-touching overlaps {len(r['nontouching_overlap'])}")
    print(f"\n({time.time() - t0:.0f}s)")
    summarize(results)


# ---------------------------------------------------------------- self-test

def _make_field(n, n_seeds, rmin, rmax, seed):
    """Synthetic label image: random periodic discs, each pixel given to the
    nearest disc that covers it, so overlapping discs become touching labels.
    Returns (labels, n_labels)."""
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, n, (n_seeds, 2))
    rad = rng.uniform(rmin, rmax, n_seeds)
    yy, xx = np.mgrid[0:n, 0:n]
    best = np.full((n, n), np.inf)
    lab = np.zeros((n, n), dtype=int)
    for i, (sy, sx) in enumerate(seeds):
        dy = yy - sy
        dy -= n * np.round(dy / n)
        dx = xx - sx
        dx -= n * np.round(dx / n)
        d = np.hypot(dy, dx)
        m = (d <= rad[i]) & (d < best)
        best[m] = d[m]
        lab[m] = i + 1
    used = np.unique(lab[lab > 0])
    remap = np.zeros(n_seeds + 1, dtype=int)
    remap[used] = np.arange(1, len(used) + 1)
    return remap[lab], len(used)


def self_test():
    """Synthetic label images with known structure; no solver, no checkpoint.
      A) well-separated discs: 0 merge rounds, nothing fused.
      B) one overlapping adjacent pair: exactly 1 merge round, fused to 1.
      C) many random dense configurations: the traced copy must equal the
         production function on every one, the converged result must leave
         NO touching overlapping pair (checked independently), merge_size
         must sum to the raw count, and at least one configuration must need
         2 or more merge rounds so that the multi-round path is exercised.
      D) cap detection: take a configuration that needs >=2 merge rounds,
         force max_rounds=1, and confirm hit_cap is reported and that the
         independent check then finds a touching overlapping pair left over.
      E) a non-touching overlapping pair (an elongated organism whose circle
         reaches a separated neighbour) is reported by overlapping_pairs and
         not fused."""
    n = 200
    proxy = GrayScottSim(n=n, L=2.5 * n / 1024)
    yy, xx = np.mgrid[0:n, 0:n]

    def two_discs(c1, c2, r1, r2):
        d1 = np.hypot(yy - c1[0], xx - c1[1])
        d2 = np.hypot(yy - c2[0], xx - c2[1])
        lab = np.zeros((n, n), dtype=int)
        lab[(d1 <= r1)] = 1
        lab[(d2 <= r2) & ((d2 < d1) | (d1 > r1))] = 2
        return lab

    # A
    lab = two_discs((100, 40), (100, 150), 10, 10)
    l, nn, ms, tr = trace_merge(lab, 2, proxy, PROD_CAP)
    assert nn == 2 and tr["merge_rounds"] == 0 and not tr["hit_cap"], (nn, tr)
    print("self_test A: OK (separated discs: 0 merge rounds)")

    # B
    lab = two_discs((100, 90), (100, 109), 10, 10)
    l, nn, ms, tr = trace_merge(lab, 2, proxy, PROD_CAP)
    assert nn == 1 and tr["merge_rounds"] == 1 and list(ms) == [2], (nn, tr, ms)
    print("self_test B: OK (overlapping pair: 1 merge round, fused into 1)")

    # C
    multi = None
    n_cfg = 0
    for seed in range(60):
        lab, nl = _make_field(n, 60, 8, 16, seed)
        p_l, p_n, p_ms = prod.merge_overlapping_organisms(lab, nl, proxy)
        t_l, t_n, t_ms, t_tr = trace_merge(lab, nl, proxy, PROD_CAP)
        assert p_n == t_n and np.array_equal(p_l, t_l) and np.array_equal(p_ms, t_ms), \
            f"traced copy differs from production at seed {seed}"
        c_l, c_n, c_ms, c_tr = trace_merge(lab, nl, proxy, BIG_CAP)
        assert int(c_ms.sum()) == nl, "merge_size must sum to the raw organism count"
        touching, _ = overlapping_pairs(c_l, c_n, proxy)
        assert not touching, f"converged result still has touching overlaps at seed {seed}"
        n_cfg += 1
        if c_tr["merge_rounds"] >= 2 and multi is None:
            multi = (lab, nl, c_tr["merge_rounds"])
    assert multi is not None, "no synthetic configuration needed 2+ merge rounds; adjust test field"
    print(f"self_test C: OK ({n_cfg} random configurations: traced copy == production, "
          f"converged result has no touching overlaps; a {multi[2]}-round case was exercised)")

    # D
    lab, nl, need = multi
    l1, n1, ms1, tr1 = trace_merge(lab, nl, proxy, 1)
    assert tr1["hit_cap"], "forcing a 1-round cap on a multi-round case must report hit_cap"
    touching, _ = overlapping_pairs(l1, n1, proxy)
    assert touching, "a capped run on a multi-round case must leave a touching overlap"
    print(f"self_test D: OK (cap forced to 1 on a {need}-round case: hit_cap reported, "
          f"{len(touching)} touching overlap(s) left, found independently)")

    # E: an elongated organism A (a bar, 6 rows x 40 columns, area 240, so its
    # equivalent-radius circle has radius ~8.7 px) and a small disc B sitting
    # off the bar's SHORT side. The bar is only 3 px from its own centroid to
    # its top edge, so B can be well outside the bar yet inside the bar's
    # circle: centroid distance ~11.5 px < 8.7 + 6.0 px, with two rows of
    # background between them so the two do not touch.
    lab = np.zeros((n, n), dtype=int)
    lab[97:103, 60:100] = 1                    # bar: rows 97-102, cols 60-99
    d = np.hypot(yy - 88, xx - 80)
    lab[d <= 6] = 2                            # disc, centre (88, 80), radius 6
    assert not prod.find_adjacent_pairs(lab), "E construction should not touch"
    touching, not_touching = overlapping_pairs(lab, 2, proxy)
    l, nn, ms, tr = trace_merge(lab, 2, proxy, PROD_CAP)
    assert not_touching and not touching, (touching, not_touching)
    assert nn == 2 and tr["merge_rounds"] == 0, "non-touching pairs must not be fused"
    print(f"self_test E: OK (non-touching overlapping pair reported, gap "
          f"{not_touching[0][2]:.1f}px, and not fused)")
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
