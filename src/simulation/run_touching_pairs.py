"""
Data collection for the "cancellation between pairs" hypothesis: for
touching organisms, is some of the apparent I-E coupling actually
routed through the specific neighbor sharing that boundary, rather
than (or in addition to) flux? Two testable versions of this:
  (a) competition: are the two members of a touching pair's own
      interior states anticorrelated (I_A vs I_B), consistent with
      drawing on the same locally limited resource?
  (b) contamination: is a touching organism's measured "exterior" E_A
      better explained by its specific neighbor's interior state I_B
      than by its own I_A -- i.e. does E_A partly just measure the
      neighbor, and if that neighbor's state runs opposite to A's own
      (per (a)), does that produce a structural cancellation in what
      the marginal I_A-E_A correlation is actually picking up?

Both need pair-LINKED data (which two organism-instances are the same
contact), which none of the existing datasets have --
positive_gap_progress_v2.npz records edge_gap per organism-instance
but not which other row is its partner.

This reproduces run_positive_gap_check_v2.py's exact production
methodology (same checkpoint, same solver settings, same watershed
segmentation parameters PAD=80/MIN_PEAK_DISTANCE=32, same I/E
definition depth_fraction=0.5/exterior_ring_px=24, same 100-step
sampling cadence, same nearest-neighbor/edge-gap computation) so this
new dataset is directly comparable to the paper's existing touching-
subset numbers, and adds: for organisms whose edge_gap < 0 AND whose
nearest-neighbor relationship is MUTUAL (i is j's nearest neighbor and
j is i's), record both members' (I, E, fu, fv) linked by one pair_id
per row -- a clean, unambiguous pairing rather than trying to
reconstruct one after the fact from unlinked per-organism data.

Usage:
    python run_touching_pairs.py --n-frames 400
    python run_touching_pairs.py --resume
"""
import argparse
import time
import numpy as np

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
from run_confound_checks import organism_interior_exterior_states_fast

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "touching_pairs_progress.npz"

PAD = 80
MIN_PEAK_DISTANCE = 32
DEPTH_FRACTION = 0.5
EXTERIOR_RING_PX = 24
SAMPLE_EVERY_STEPS = 100
SAVE_EVERY_FRAMES = 10


def nearest_neighbor_dist_and_idx(centroids, n):
    if len(centroids) < 2:
        return np.full(len(centroids), np.nan), np.full(len(centroids), -1)
    diff = centroids[:, None, :] - centroids[None, :, :]
    diff = (diff + n / 2) % n - n / 2
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nn_idx = dist.argmin(axis=1)
    nn_dist = dist[np.arange(len(centroids)), nn_idx]
    return nn_dist, nn_idx


def find_mutual_touching_pairs(nn_idx, edge_gap):
    """Returns a list of (i, j) index pairs, i<j, where i and j are
    each other's nearest neighbor AND edge_gap < 0 for both -- a
    clean, unambiguous touching-pair identification (no organism
    appears in more than one pair, since mutual-nearest-neighbor is a
    partial matching)."""
    n = len(nn_idx)
    pairs = []
    seen = set()
    for i in range(n):
        j = nn_idx[i]
        if j < 0 or j in seen or i in seen:
            continue
        if nn_idx[j] != i:
            continue  # not mutual
        if not (edge_gap[i] < 0 and edge_gap[j] < 0):
            continue
        lo, hi = min(i, j), max(i, j)
        pairs.append((lo, hi))
        seen.add(i)
        seen.add(j)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    keys = ["I_A", "E_A", "fu_A", "fv_A", "I_B", "E_B", "fu_B", "fv_B",
            "edge_gap", "frame_id", "pair_id"]
    if args.resume:
        d = np.load(PROGRESS_FILE)
        sim = PseudoSpectralGS(n=int(d["n"]))
        sim.u, sim.v, sim.time = d["u"], d["v"], float(d["time"])
        lists = {k: list(d[k]) for k in keys}
        frames_done = int(d["frames_done"])
        next_pair_id = int(d["next_pair_id"])
        print(f"resumed at t={sim.time:.0f}, {frames_done} frames done")
    else:
        sim = PseudoSpectralGS.from_checkpoint(CHECKPOINT_FILE)
        lists = {k: [] for k in keys}
        frames_done = 0
        next_pair_id = 0
        print(f"loaded checkpoint at t={sim.time:.0f}")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)

    t_start = time.time()
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.u, sim.v
        labels, n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)

        n_pairs_this_frame = 0
        if n > 1:
            proxy.Lu = laplacian_periodic(sim.u, sim.dx)
            proxy.Lv = laplacian_periodic(sim.v, sim.dx)
            flux_u, flux_v = proxy.per_organism_flux(labels, n)
            centroids, radii = proxy.organism_geometry(labels, n)
            radii_px = radii / proxy.dx
            I, E = organism_interior_exterior_states_fast(
                sim, labels, n, centroids, depth_fraction=DEPTH_FRACTION,
                exterior_ring_px=EXTERIOR_RING_PX)

            nn_dist, nn_idx = nearest_neighbor_dist_and_idx(centroids, sim.n)
            valid_nn = nn_idx >= 0
            edge_gap = np.full(n, np.nan)
            edge_gap[valid_nn] = (nn_dist[valid_nn] - radii_px[valid_nn]
                                   - radii_px[nn_idx[valid_nn]])

            pairs = find_mutual_touching_pairs(nn_idx, edge_gap)
            n_pairs_this_frame = len(pairs)
            for (i, j) in pairs:
                row = dict(I_A=I[i], E_A=E[i], fu_A=flux_u[i], fv_A=flux_v[i],
                           I_B=I[j], E_B=E[j], fu_B=flux_u[j], fv_B=flux_v[j],
                           edge_gap=edge_gap[i], frame_id=frames_done,
                           pair_id=next_pair_id)
                for k in keys:
                    lists[k].append(row[k])
                next_pair_id += 1

        frames_done += 1
        elapsed = time.time() - t_start
        if frames_done % 10 == 0:
            n_so_far = len(lists["I_A"])
            print(f"frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  "
                  f"n_organisms={n}  pairs this frame={n_pairs_this_frame}  "
                  f"N pairs so far={n_so_far}  "
                  f"({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.2f}s/frame)")

        if frames_done % SAVE_EVERY_FRAMES == 0 or frames_done == args.n_frames:
            save_kwargs = {k: np.array(lists[k]) for k in keys}
            np.savez(PROGRESS_FILE, u=sim.u, v=sim.v, time=sim.time, n=sim.n,
                     frames_done=frames_done, next_pair_id=next_pair_id, **save_kwargs)

    print(f"\ndone: {frames_done} frames, N={len(lists['I_A'])} touching pairs, "
          f"saved to {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
