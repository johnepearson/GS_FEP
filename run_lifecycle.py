"""
Combined lifecycle tracker at n=1024, replacing the separate n=256
trackers (lineage/lifespan, crowding, template free energy, Friston
free energy) with ONE pass that records everything each of those
needs, so the (expensive) simulation is only stepped once.

Reuses the same split-confirmation logic validated earlier (50-time-
unit confirmation window; a split is confirmed real either by
surviving the window or by one child dying independently, and treated
as a false start if the children re-merge).

Per confirmed organism, per sampled frame, records:
  - t (time)
  - radial V profile (for template-deviation free energy, post-hoc)
  - I, E via the local-distance-transform definition (for Friston free
    energy, post-hoc, once ensemble-fitted Gaussian parameters are
    available)
  - blanket flux (fu, fv)
  - nearest-neighbor distance (for the crowding-vs-lifespan analysis;
    only the early window after birth is actually needed downstream,
    but recording every frame costs little extra and keeps this
    general-purpose)

Usage:
    python run_lifecycle.py                  # default: 2000 time units
    python run_lifecycle.py --n-steps 8000
    python run_lifecycle.py --resume

Progress saves periodically to lifecycle_progress.pkl (the name every downstream analysis script reads;
move any older file with that name aside before starting a fresh run).
Marker spacing is MIN_PEAK_DISTANCE = 24 px, and organisms are the MERGED regions (overlapping watershed regions fused),
so a 'division' is a merged organism that becomes two non-overlapping organisms.
"""
import argparse
import pickle
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
from run_confound_checks import organism_interior_exterior_states_fast
import run_ensemble_merged as prod   # merge_overlapping_organisms, periodic_centroids
# NOTE (2026-09-27): switched from organism_interior_exterior_states_distance
# (gray_scott_distance_transform.py) to the crop-based fast version --
# the old one has a wrap-duplication bug for organisms near a periodic
# edge (~20-26% of organism-instances in testing), which affected every
# I/E value this tracker recorded (used for the Friston free-energy
# work and, now, the life-cycle-stratified confound check for referee
# Comment #2).

try:
    from gray_scott_spectral_c import PseudoSpectralGSFast
    _HAVE_FAST = True
except OSError:
    _HAVE_FAST = False

FRAC_THRESHOLD = 0.25
CONFIRM_WINDOW = 50
SAMPLE_EVERY = 5            # fine temporal sampling, matching the n=256 lifecycle work
SAVE_EVERY_SECONDS = 300    # wall-clock-based progress saving

EXTERIOR_RING_PX = 24
MIN_PEAK_DISTANCE = 24   # was 32; 24 removes the segmentation dropout (spots losing their label for 10-80 time units)
PAD = 80
MAX_R, N_BINS = 80, 20      # radial profile range, scaled 4x from n=256's 0-20px
IE_EVERY_N_FRAMES = 4       # compute the (expensive) interior/exterior distance
                            # transform only every Nth sampled frame -- it dominates
                            # per-frame cost (~12s vs <0.4s for everything else
                            # combined) and is only needed for Friston free energy,
                            # not for split detection or the template/crowding data,
                            # which use every frame regardless

CROP_IE = 90

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "lifecycle_progress.pkl"
TRACK_RAW = False          # set by --raw: track raw watershed regions instead of merged organisms (comparison only)


def organism_radial_profile(v_field, N, cy, cx):
    bins = np.linspace(0, MAX_R, N_BINS + 1)
    pad = int(MAX_R) + 2
    r0, c0 = int(round(cy)), int(round(cx))
    rows = (np.arange(r0 - pad, r0 + pad + 1)) % N
    cols = (np.arange(c0 - pad, c0 + pad + 1)) % N
    sub_v = v_field[np.ix_(rows, cols)]
    yy, xx = np.meshgrid(np.arange(-pad, pad + 1), np.arange(-pad, pad + 1), indexing='ij')
    dy = yy - (cy - r0)
    dx = xx - (cx - c0)
    dist = np.sqrt(dy ** 2 + dx ** 2)
    profile = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = (dist >= bins[b]) & (dist < bins[b + 1])
        if mask.any():
            profile[b] = sub_v[mask].mean()
    return profile


def nn_distances(centroids, N):
    if len(centroids) < 2:
        return np.full(len(centroids), np.nan)
    tree = cKDTree(centroids, boxsize=N)
    dist, _ = tree.query(centroids, k=2)
    return dist[:, 1]


def overlaps(prev_labels, curr_labels, prev_n, curr_n):
    joint = prev_labels.astype(np.int64) * (curr_n + 1) + curr_labels.astype(np.int64)
    vals, counts = np.unique(joint, return_counts=True)
    p_ids = vals // (curr_n + 1)
    c_ids = vals % (curr_n + 1)
    prev_areas = np.bincount(prev_labels.ravel(), minlength=prev_n + 1)
    fwd = {}
    for p, c, cnt in zip(p_ids, c_ids, counts):
        if p == 0 or c == 0:
            continue
        fwd.setdefault(p, []).append((c, cnt / prev_areas[p]))
    return fwd


class LifecycleTracker:
    def __init__(self, checkpoint_path, fast=False):
        Solver = PseudoSpectralGSFast if fast else PseudoSpectralGS
        self.sim = Solver.from_checkpoint(checkpoint_path)
        self.prev_labels, self.prev_n = self._segment()
        self._next_id = 1
        self.confirmed_of_label = {i: self._new_id() for i in range(1, self.prev_n + 1)}
        self.episodes = {}
        self.birth_time = {lid: 0 for lid in self.confirmed_of_label.values()}
        self.history = {lid: [] for lid in self.confirmed_of_label.values()}
        self.pending_history = {}
        self.examples = []
        self.t = 0
        self._frame_counter = 0

    def _new_id(self):
        v = self._next_id
        self._next_id += 1
        return v

    def _segment(self):
        proxy = GrayScottSim(n=self.sim.n, L=self.sim.L, Du=self.sim.Du, Dv=self.sim.Dv,
                              F=self.sim.F, k=self.sim.k, dt=self.sim.dt)
        if hasattr(self.sim, "get_uv"):
            proxy.u, proxy.v = self.sim.get_uv()
        else:
            proxy.u, proxy.v = self.sim.u, self.sim.v
        raw_labels, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
        # Track MERGED organisms (watershed regions whose equivalent-radius circles overlap are fused), the same
        # individuation as every other analysis in the paper. At marker spacing 24 a single spot is often cut
        # into 2 raw regions; tracking raw regions would count those fragments as separate organisms and
        # persistent fragmentations as 'divisions'. A division is now a merged organism that becomes two
        # non-overlapping organisms.
        if raw_n == 0 or TRACK_RAW:
            labels, n = raw_labels, raw_n
        else:
            labels, n, _ = prod.merge_overlapping_organisms(raw_labels, raw_n, proxy)
        self._proxy = proxy
        return labels, n

    def _measure_all(self, labels, n):
        if n == 0:
            return {}
        proxy = self._proxy
        centroids = prod.periodic_centroids(labels, n, self.sim.n)   # periodic-correct (merged masks can straddle the edge)
        proxy.Lu = laplacian_periodic(proxy.u, proxy.dx)
        proxy.Lv = laplacian_periodic(proxy.v, proxy.dx)
        fu, fv = proxy.per_organism_flux(labels, n)

        self._frame_counter += 1
        if self._frame_counter % IE_EVERY_N_FRAMES == 0:
            I, E = organism_interior_exterior_states_fast(
                proxy, labels, n, centroids, depth_fraction=0.5,
                exterior_ring_px=EXTERIOR_RING_PX, crop=CROP_IE)
        else:
            I = np.full(n, np.nan)
            E = np.full(n, np.nan)

        # mean of u*v^2 over each (merged) organism's own mask: the reaction-rate prefactor that the
        # entropy-production comparison follows (a per-area mean, not an integral, so not driven by size)
        uv2 = ndimage.mean(proxy.u * proxy.v ** 2, labels=labels, index=np.arange(1, n + 1))
        uv2_sum = ndimage.sum(proxy.u * proxy.v ** 2, labels=labels, index=np.arange(1, n + 1))
        area = ndimage.sum(np.ones_like(proxy.u), labels=labels, index=np.arange(1, n + 1))
        omu_sum = ndimage.sum(1.0 - proxy.u, labels=labels, index=np.arange(1, n + 1))   # feed term: F * sum(1-u)
        u_sum = ndimage.sum(proxy.u, labels=labels, index=np.arange(1, n + 1))            # for d/dt sum(u) between frames
        nnd = nn_distances(centroids, self.sim.n)
        out = {}
        for i in range(n):
            profile = organism_radial_profile(proxy.v, self.sim.n, centroids[i, 0], centroids[i, 1])
            out[i + 1] = dict(t=self.t, profile=profile, I=I[i], E=E[i],
                               fu=fu[i], fv=fv[i], nn_dist=nnd[i], uv2=uv2[i],
                               uv2_sum=uv2_sum[i], area=area[i],
                               omu_sum=omu_sum[i], u_sum=u_sum[i])
        return out

    def advance(self, n_steps):
        t_end = self.t + n_steps
        while self.t < t_end:
            self.sim.run(SAMPLE_EVERY)
            self.t += SAMPLE_EVERY
            curr_labels, curr_n = self._segment()
            if curr_n == 0:
                break
            fwd = overlaps(self.prev_labels, curr_labels, self.prev_n, curr_n)
            measurements = self._measure_all(curr_labels, curr_n)

            new_confirmed_of_label = {}
            handled = set()

            for ep_id, ep in list(self.episodes.items()):
                lblA, lblB = ep['sibling_labels']
                bestA = sorted(fwd.get(lblA, []), key=lambda x: -x[1])
                bestB = sorted(fwd.get(lblB, []), key=lambda x: -x[1])
                cA = bestA[0][0] if bestA else None
                cB = bestB[0][0] if bestB else None

                if cA is not None and cA == cB:
                    new_confirmed_of_label[cA] = ep['parent_lineage_id']
                    handled.add(cA)
                    del self.episodes[ep_id]
                    continue

                died_A, died_B = cA is None, cB is None
                if died_A or died_B or (self.t - ep['split_time'] >= CONFIRM_WINDOW):
                    self.examples.append(dict(
                        birth=ep['parent_birth_time'], death=ep['split_time'], cause='split',
                        history=self.history.get(ep['parent_lineage_id'], [])))
                    self.history.pop(ep['parent_lineage_id'], None)
                    if not died_A:
                        lidA = self._new_id()
                        self.history[lidA] = self.pending_history.pop((ep_id, 'A'), [])
                        self.birth_time[lidA] = ep['split_time']
                        new_confirmed_of_label[cA] = lidA
                        handled.add(cA)
                    else:
                        self.examples.append(dict(birth=ep['split_time'], death=self.t,
                                                   cause='death', history=[]))
                    if not died_B:
                        lidB = self._new_id()
                        self.history[lidB] = self.pending_history.pop((ep_id, 'B'), [])
                        self.birth_time[lidB] = ep['split_time']
                        new_confirmed_of_label[cB] = lidB
                        handled.add(cB)
                    else:
                        self.examples.append(dict(birth=ep['split_time'], death=self.t,
                                                   cause='death', history=[]))
                    self.pending_history.pop((ep_id, 'A'), None)
                    self.pending_history.pop((ep_id, 'B'), None)
                    del self.episodes[ep_id]
                    continue

                if cA in measurements:
                    self.pending_history.setdefault((ep_id, 'A'), []).append(measurements[cA])
                if cB in measurements:
                    self.pending_history.setdefault((ep_id, 'B'), []).append(measurements[cB])
                ep['sibling_labels'] = [cA, cB]
                handled.add(cA); handled.add(cB)

            for p, lid in self.confirmed_of_label.items():
                lst = sorted(fwd.get(p, []), key=lambda x: -x[1])
                if len(lst) >= 2 and lst[0][1] > FRAC_THRESHOLD and lst[1][1] > FRAC_THRESHOLD:
                    c1, c2 = lst[0][0], lst[1][0]
                    if c1 in handled or c2 in handled:
                        continue
                    ep_id = self._new_id()
                    self.episodes[ep_id] = dict(split_time=self.t, parent_lineage_id=lid,
                                                 parent_birth_time=self.birth_time.get(lid, 0),
                                                 sibling_labels=[c1, c2])
                    if c1 in measurements:
                        self.pending_history[(ep_id, 'A')] = [measurements[c1]]
                    if c2 in measurements:
                        self.pending_history[(ep_id, 'B')] = [measurements[c2]]
                    handled.add(c1); handled.add(c2)
                elif lst:
                    c_best = lst[0][0]
                    if c_best not in handled:
                        new_confirmed_of_label[c_best] = lid
                        if c_best in measurements:
                            self.history.setdefault(lid, []).append(measurements[c_best])
                        handled.add(c_best)
                else:
                    self.examples.append(dict(birth=self.birth_time.get(lid, 0), death=self.t,
                                               cause='death', history=self.history.get(lid, [])))
                    self.history.pop(lid, None)

            for c in range(1, curr_n + 1):
                if c not in handled:
                    new_lid = self._new_id()
                    new_confirmed_of_label[c] = new_lid
                    self.birth_time[new_lid] = self.t
                    self.history[new_lid] = [measurements[c]] if c in measurements else []

            self.confirmed_of_label = new_confirmed_of_label
            self.prev_labels, self.prev_n = curr_labels, curr_n

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=2000,
                     help="total additional simulation time units to advance")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fast", action="store_true",
                     help="use the C solver (see README.md's C solver section to build "
                          "libgs_spectral.so first)")
    ap.add_argument("--raw", action="store_true",
                     help="comparison only: track raw watershed regions instead of merged organisms; "
                          "writes lifecycle_raw_progress.pkl and leaves lifecycle_progress.pkl alone")
    args = ap.parse_args()
    global TRACK_RAW, PROGRESS_FILE
    if args.raw:
        TRACK_RAW = True
        PROGRESS_FILE = "lifecycle_raw_progress.pkl"

    if args.fast and not _HAVE_FAST:
        raise SystemExit("--fast requested but libgs_spectral.so isn't built for this "
                          "machine. Build it first (see README.md), or drop --fast.")

    if args.resume:
        tr = LifecycleTracker.load(PROGRESS_FILE)
        print(f"resumed at t={tr.t}, {len(tr.examples)} completed lifecycles so far")
    else:
        tr = LifecycleTracker(CHECKPOINT_FILE, fast=args.fast)
        print(f"starting fresh at t={tr.t}")

    t_start_wall = time.time()
    t_target = tr.t + args.n_steps
    last_save = time.time()
    while tr.t < t_target:
        tr.advance(min(50, t_target - tr.t))
        elapsed = time.time() - t_start_wall
        print(f"t={tr.t}  ({elapsed:.0f}s elapsed)  completed lifecycles: {len(tr.examples)}  "
              f"pending episodes: {len(tr.episodes)}")
        if time.time() - last_save > SAVE_EVERY_SECONDS:
            tr.save(PROGRESS_FILE)
            last_save = time.time()
            print(f"  [progress saved to {PROGRESS_FILE}]")

    tr.save(PROGRESS_FILE)
    print(f"\nDone. t={tr.t}, {len(tr.examples)} completed lifecycles recorded.")


if __name__ == "__main__":
    main()
