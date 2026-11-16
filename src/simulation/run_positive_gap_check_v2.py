"""
Referee Comment #3 (this referee report, 2026-09-27): "The proposed
mechanism may not explain the effect it is invoked for." Figure 8/the
boundary-v-vs-gap check shows the relational-boundary correlation is
carried almost entirely by touching/overlapping pairs (negative edge
gap, ~18% of instances); positive-gap pairs sit on a flat plateau. The
referee's decisive test: repeat the MAIN conditional-independence
analysis (Section 3.3.1, the single-trajectory partial correlation
between I and E given blanket flux) restricted to organisms with a
clearly positive edge-to-edge gap. If the violation persists among
isolated (non-touching) organisms, the relational-boundary mechanism
cannot be the effect's main source. If it vanishes, the headline claim
needs to be narrowed to crowded/contact configurations.

This reproduces run_full_replacement.py's exact production
methodology (same solver settings, same watershed segmentation
parameters PAD=80/MIN_PEAK_DISTANCE=32, same interior/exterior
definition depth_fraction=0.5/exterior_ring_px=24, same 100-step
sampling cadence) with two additions:
  1. per-organism equivalent radius and periodic nearest-neighbor
     distance are recorded alongside I, E and flux, so each
     organism-instance can be classified by edge-to-edge gap
     (= nn_dist - own_radius - neighbor_radius) after the fact.
  2. interior/exterior states are computed with the crop-based,
     periodic-correct organism_interior_exterior_states_fast (see
     run_confound_checks.py) rather than the original whole-image
     organism_interior_exterior_states_distance, which has a
     previously-diagnosed wrap-padding bug for organisms within
     ~70-85px of a periodic domain edge (diagnose_residual.py). Using
     the corrected version here also gives a bug-fixed full-population
     number to compare against the paper's currently-reported
     (bug-affected) headline r=-0.111 for the same construction.

Usage:
    python run_positive_gap_check.py --n-frames 400
    python run_positive_gap_check.py --resume
"""
import argparse
import time
import numpy as np
from params import load as _load_params
_PARAMS = _load_params("initialize.tex")   # F, k, dt of the model
from scipy import ndimage

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
from run_confound_checks import organism_interior_exterior_states_fast, organism_shell_profile_field_fast

try:
    from gray_scott_spectral_c import PseudoSpectralGSFast
    _HAVE_FAST = True
except OSError:
    _HAVE_FAST = False

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "positive_gap_progress_v2.npz"

PAD = 80
MIN_PEAK_DISTANCE = 32
DEPTH_FRACTION = 0.5
EXTERIOR_RING_PX = 24
SAMPLE_EVERY_STEPS = 100  # matches run_full_replacement.py's production cadence
SAVE_EVERY_FRAMES = 10

# u-field shell schedule for the additional confound covariates requested by
# the reviewer (2026-09-27 report, "New problems" #2): a handful of coarse
# near-field u-shells, cheap enough not to meaningfully change the per-frame
# cost, to use as extra regressors on the isolated-organism subset alongside
# frame fixed effects / radius.
USHELL_RADII_PX = np.array([0, 20, 40, 80, 160])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fast", action="store_true",
                     help="use the C solver (several times faster; see README.md's C "
                          "solver section to build libgs_spectral.so first). This script "
                          "originally had no --fast option and would be prohibitively slow "
                          "on the pure-Python solver at 400 frames x 100 steps.")
    args = ap.parse_args()

    if args.fast and not _HAVE_FAST:
        raise SystemExit("--fast requested but libgs_spectral.so isn't built for this "
                          "machine. Build it first (see README.md), or drop --fast.")
    Solver = PseudoSpectralGSFast if args.fast else PseudoSpectralGS

    n_ushells = len(USHELL_RADII_PX) - 1
    keys = ["I", "E", "fu", "fv", "radius", "edge_gap", "frame_id"] + [f"ushell{s}" for s in range(n_ushells)]
    if args.resume:
        d = np.load(PROGRESS_FILE)
        n = int(d["n"])
        sim = Solver(n=n, F=_PARAMS["Fval"], k=_PARAMS["kval"], dt=_PARAMS["dtval"])
        sim.from_checkpoint_fields(d["u"], d["v"])
        sim.time = float(d["time"])
        lists = {k: [d[k]] for k in keys}
        frames_done = int(d["frames_done"])
        print(f"resumed at t={sim.time:.0f}, {frames_done} frames done")
    else:
        sim = Solver.from_checkpoint(CHECKPOINT_FILE)
        lists = {k: [] for k in keys}
        frames_done = 0
        print(f"loaded checkpoint at t={sim.time:.0f}")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)

    t_start = time.time()
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        if hasattr(sim, "get_uv"):
            proxy.u, proxy.v = sim.get_uv()
        else:
            proxy.u, proxy.v = sim.u, sim.v
        labels, n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)

        if n > 0:
            proxy.Lu = laplacian_periodic(proxy.u, proxy.dx)
            proxy.Lv = laplacian_periodic(proxy.v, proxy.dx)
            flux_u, flux_v = proxy.per_organism_flux(labels, n)
            centroids, radii = proxy.organism_geometry(labels, n)  # radii in physical units
            radii_px = radii / proxy.dx
            I, E = organism_interior_exterior_states_fast(
                proxy, labels, n, centroids, depth_fraction=DEPTH_FRACTION,
                exterior_ring_px=EXTERIOR_RING_PX)

            nn_dist, nn_idx = nearest_neighbor_dist_and_idx(centroids, sim.n)
            valid_nn = nn_idx >= 0
            edge_gap = np.full(n, np.nan)
            edge_gap[valid_nn] = (nn_dist[valid_nn] - radii_px[valid_nn]
                                   - radii_px[nn_idx[valid_nn]])

            frame_id = np.full(n, frames_done)
            u_shells = organism_shell_profile_field_fast(
                proxy.u, labels, n, centroids, USHELL_RADII_PX)

            vals = [I, E, flux_u, flux_v, radii_px, edge_gap, frame_id] + \
                   [u_shells[:, s] for s in range(n_ushells)]
            for k, v_ in zip(keys, vals):
                lists[k].append(v_)

        frames_done += 1
        elapsed = time.time() - t_start
        if frames_done % 10 == 0:
            n_so_far = sum(len(x) for x in lists["I"])
            print(f"frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  n_organisms={n}  "
                  f"N so far={n_so_far}  ({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.2f}s/frame)")

        if frames_done % SAVE_EVERY_FRAMES == 0 or frames_done == args.n_frames:
            save_kwargs = {k: np.concatenate(v_) for k, v_ in lists.items()}
            np.savez(PROGRESS_FILE, u=proxy.u, v=proxy.v, time=sim.time, n=sim.n,
                     frames_done=frames_done, **save_kwargs)

    print(f"\ndone: {frames_done} frames, N={sum(len(x) for x in lists['I'])} "
          f"organism-instances, saved to {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
