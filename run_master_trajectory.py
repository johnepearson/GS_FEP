"""
Master single-trajectory run: ONE long trajectory from the settled checkpoint, sampled every
100 steps, production segmentation (watershed + overlap merge, min_distance=24), recording
everything the single-trajectory analyses in the paper need, so they can all be computed from
this one dataset (and from the archived fields) without simulating again.

Per organism-instance (one row each; frame_id + org_idx identify it):
  I, E, fu, fv                 distance-transform interior / exterior states and blanket fluxes (production)
  I_adapt, E_adapt             whole-mask-erosion interior / exterior (the alternative definition)
  cy, cx, radius_px, area_px   periodic-correct centroid, equivalent radius, area of the merged mask
  eccentricity                 regionprops eccentricity
  merge_size                   number of watershed regions merged into this organism
  nn_idx, nn_dist, edge_gap    nearest other organism (org_idx in the same frame), centroid distance,
                               edge-to-edge gap (centroid distance - both equivalent radii)
  boundary_u, boundary_v       mean u, v over the organism's own boundary pixels (its black outline)
  v_shells                     mean v in concentric exterior shells (RADII, 8 px steps to 160 px, 24 px to 400 px)
  u_shells                     mean u in shells (U_SHELL_RADII_PX)
Per frame: time, population (merged), raw watershed count, mean u, mean v, Otsu threshold of v,
           number of visible spots left unlabelled (dropout diagnostic).

Archive (--archive-every, default every frame): u, v (float32) and the final merged labels (int16) in
master_fields/frame_XXXX.npz, so any new analysis (alternative individuation, new blanket variables,
level sets, ...) can be run offline.  About 8 MB per frame; 500 frames is about 4 GB.

Usage (use the C solver):
    python run_master_trajectory.py --n-frames 500 --fast
    python run_master_trajectory.py --n-frames 500 --fast --resume       # continue an interrupted run
Progress: master_trajectory_progress.npz (data + solver state), saved every 10 frames.
"""
import argparse
import os
import time
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
import run_ensemble_merged as prod
from run_confound_checks import (organism_interior_exterior_states_fast,
                                 organism_interior_exterior_states_adaptive_fast,
                                 organism_shell_profile_field_fast, eccentricities)
from run_min_distance_sweep import unlabelled_spots

try:
    from gray_scott_spectral_c import PseudoSpectralGSFast
    _HAVE_FAST = True
except OSError:
    _HAVE_FAST = False

MIN_PEAK_DISTANCE = 24
PAD = 80
DEPTH_FRACTION = 0.5
EXTERIOR_RING_PX = 24
CROP_IE = 90
CROP_USHELL = 140
CROP_VSHELL = 460
SAMPLE_EVERY_STEPS = 100
SAVE_EVERY_FRAMES = 10
RADII = list(range(0, 161, 8)) + list(range(160, 401, 24))[1:]
U_SHELL_RADII_PX = [0, 6, 12, 18, 24, 36, 48, 64]
CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "master_trajectory_progress.npz"
ARCHIVE_DIR = "master_fields"

ROW_KEYS = ["frame_id", "org_idx", "I", "E", "fu", "fv", "I_adapt", "E_adapt", "cy", "cx", "radius_px",
            "area_px", "eccentricity", "merge_size", "nn_idx", "nn_dist", "edge_gap", "boundary_u",
            "boundary_v"]
MAT_KEYS = ["v_shells", "u_shells"]
FRAME_KEYS = ["f_time", "f_pop", "f_raw_pop", "f_mean_u", "f_mean_v", "f_otsu", "f_unlabelled"]


def boundary_means(labels, n, u, v):
    b = ((labels != np.roll(labels, 1, 0)) | (labels != np.roll(labels, 1, 1)) |
         (labels != np.roll(labels, -1, 0)) | (labels != np.roll(labels, -1, 1))) & (labels > 0)
    lb = np.where(b, labels, 0)
    idx = np.arange(1, n + 1)
    cnt = ndimage.sum(np.ones_like(labels), labels=lb, index=idx)
    with np.errstate(invalid="ignore", divide="ignore"):
        bu = np.where(cnt > 0, ndimage.sum(u, labels=lb, index=idx) / np.maximum(cnt, 1), np.nan)
        bv = np.where(cnt > 0, ndimage.sum(v, labels=lb, index=idx) / np.maximum(cnt, 1), np.nan)
    return bu, bv


def nearest_neighbour(centroids, radius_px, N):
    n = len(centroids)
    if n < 2:
        return np.full(n, -1), np.full(n, np.nan), np.full(n, np.nan)
    dy = centroids[:, None, 0] - centroids[None, :, 0]
    dx = centroids[:, None, 1] - centroids[None, :, 1]
    dy -= N * np.round(dy / N); dx -= N * np.round(dx / N)
    d = np.hypot(dy, dx)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(axis=1)
    nn_dist = d[np.arange(n), nn]
    gap = nn_dist - radius_px - radius_px[nn]
    return nn, nn_dist, gap


def process_frame(proxy, frame_id, sim_time, archive_dir, archive):
    raw_labels, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
    if raw_n == 0:
        return None
    labels, n, merge_size = prod.merge_overlapping_organisms(raw_labels, raw_n, proxy)
    proxy.Lu = laplacian_periodic(proxy.u, proxy.dx)
    proxy.Lv = laplacian_periodic(proxy.v, proxy.dx)
    fu, fv = proxy.per_organism_flux(labels, n)
    centroids = prod.periodic_centroids(labels, n, proxy.n)
    _, radii = proxy.organism_geometry(labels, n)
    radius_px = radii / proxy.dx
    I, E = organism_interior_exterior_states_fast(
        proxy, labels, n, centroids, depth_fraction=DEPTH_FRACTION,
        exterior_ring_px=EXTERIOR_RING_PX, crop=CROP_IE)
    Ia, Ea = organism_interior_exterior_states_adaptive_fast(
        proxy, labels, n, centroids, keep_fraction=DEPTH_FRACTION,
        exterior_ring_px=EXTERIOR_RING_PX, crop=CROP_IE)
    ecc = eccentricities(labels, n)
    nn, nn_dist, gap = nearest_neighbour(centroids, radius_px, proxy.n)
    bu, bv = boundary_means(labels, n, proxy.u, proxy.v)
    vsh = organism_shell_profile_field_fast(proxy.v, labels, n, centroids, RADII, crop=CROP_VSHELL)
    ush = organism_shell_profile_field_fast(proxy.u, labels, n, centroids, U_SHELL_RADII_PX, crop=CROP_USHELL)
    rows = dict(frame_id=np.full(n, frame_id, dtype=np.int32), org_idx=np.arange(n, dtype=np.int32),
                I=I, E=E, fu=fu, fv=fv, I_adapt=Ia, E_adapt=Ea, cy=centroids[:, 0], cx=centroids[:, 1],
                radius_px=radius_px, area_px=np.pi * radius_px ** 2, eccentricity=ecc,
                merge_size=merge_size.astype(np.int16), nn_idx=nn.astype(np.int32), nn_dist=nn_dist,
                edge_gap=gap, boundary_u=bu, boundary_v=bv)
    frame = dict(f_time=sim_time, f_pop=n, f_raw_pop=raw_n, f_mean_u=proxy.u.mean(), f_mean_v=proxy.v.mean(),
                 f_otsu=threshold_otsu(proxy.v), f_unlabelled=len(unlabelled_spots(proxy.v, labels)))
    if archive:
        os.makedirs(archive_dir, exist_ok=True)
        np.savez_compressed(os.path.join(archive_dir, f"frame_{frame_id:04d}.npz"),
                            u=proxy.u.astype(np.float32), v=proxy.v.astype(np.float32),
                            labels=labels.astype(np.int16), raw_labels=raw_labels.astype(np.int16),
                            time=sim_time, min_peak_distance=MIN_PEAK_DISTANCE)
    return rows, dict(v_shells=vsh, u_shells=ush), frame


def save(sim, frames_done, R, M, Fr):
    out = dict(u=sim.u, v=sim.v, time=sim.time, n=sim.n, frames_done=frames_done, radii=np.array(RADII),
               u_shell_radii=np.array(U_SHELL_RADII_PX), min_peak_distance=MIN_PEAK_DISTANCE, pad=PAD,
               depth_fraction=DEPTH_FRACTION, exterior_ring_px=EXTERIOR_RING_PX, crop_ie=CROP_IE,
               sample_every_steps=SAMPLE_EVERY_STEPS)
    for k in ROW_KEYS:
        out[k] = np.concatenate(R[k]) if R[k] else np.array([])
    for k in MAT_KEYS:
        out[k] = np.concatenate(M[k], axis=0) if M[k] else np.array([])
    for k in FRAME_KEYS:
        out[k] = np.array(Fr[k])
    np.savez(PROGRESS_FILE, **out)


def main():
    global MIN_PEAK_DISTANCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fast", action="store_true", help="use the C solver (libgs_spectral.so)")
    ap.add_argument("--min-distance", type=int, default=MIN_PEAK_DISTANCE)
    ap.add_argument("--archive-every", type=int, default=1,
                    help="save u, v and labels every N-th frame (0 = never)")
    args = ap.parse_args()
    MIN_PEAK_DISTANCE = args.min_distance
    if args.fast and not _HAVE_FAST:
        raise SystemExit("--fast requested but libgs_spectral.so isn't built for this machine")
    Solver = PseudoSpectralGSFast if args.fast else PseudoSpectralGS

    R = {k: [] for k in ROW_KEYS}; M = {k: [] for k in MAT_KEYS}; Fr = {k: [] for k in FRAME_KEYS}
    if args.resume:
        d = np.load(PROGRESS_FILE)
        if int(d["min_peak_distance"]) != MIN_PEAK_DISTANCE:
            raise SystemExit(f"progress file used min_peak_distance={int(d['min_peak_distance'])}")
        sim = Solver(n=int(d["n"]))
        if hasattr(sim, "from_checkpoint_fields"):
            sim.from_checkpoint_fields(d["u"], d["v"])
        else:
            sim.u, sim.v = d["u"], d["v"]
        sim.time = float(d["time"])
        frames_done = int(d["frames_done"])
        for k in ROW_KEYS: R[k] = [d[k]]
        for k in MAT_KEYS: M[k] = [d[k]]
        for k in FRAME_KEYS: Fr[k] = list(d[k])
        print(f"resumed at t={sim.time:.0f}, {frames_done} frames done")
    else:
        sim = Solver.from_checkpoint(CHECKPOINT_FILE)
        frames_done = 0
        print(f"loaded checkpoint at t={sim.time:.0f}; min_distance={MIN_PEAK_DISTANCE}")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)
    t0 = time.time(); done0 = frames_done
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        if hasattr(sim, "get_uv"):
            proxy.u, proxy.v = sim.get_uv()
        else:
            proxy.u, proxy.v = sim.u, sim.v
        archive = args.archive_every > 0 and frames_done % args.archive_every == 0
        res = process_frame(proxy, frames_done, sim.time, ARCHIVE_DIR, archive)
        if res is None:
            print(f"t={sim.time:.0f}: population died out, stopping"); break
        rows, mats, frame = res
        for k in ROW_KEYS: R[k].append(rows[k])
        for k in MAT_KEYS: M[k].append(mats[k])
        for k in FRAME_KEYS: Fr[k].append(frame[k])
        frames_done += 1
        if frames_done % 10 == 0 or frames_done == args.n_frames:
            el = time.time() - t0
            per = el / (frames_done - done0)
            n_so_far = sum(len(x) for x in R["I"])
            print(f"frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  organisms={frame['f_pop']}  "
                  f"unlabelled={frame['f_unlabelled']}  N so far={n_so_far}  "
                  f"({el:.0f}s, ~{per:.1f}s/frame, ~{(args.n_frames-frames_done)*per/3600:.1f}h left)", flush=True)
        if frames_done % SAVE_EVERY_FRAMES == 0:
            save(sim, frames_done, R, M, Fr)
    save(sim, frames_done, R, M, Fr)

    I = np.concatenate(R["I"]); E = np.concatenate(R["E"])
    fu = np.concatenate(R["fu"]); fv = np.concatenate(R["fv"])
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    B = np.column_stack([np.ones(good.sum()), fu[good], fv[good]])
    res = lambda y: y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
    r = np.corrcoef(res(I[good]), res(E[good]))[0, 1]
    print(f"\ndone: {frames_done} frames, N={good.sum()} organism-instances; "
          f"partial r(I,E | flux) = {r:.4f}; unlabelled spots/frame = {np.mean(Fr['f_unlabelled']):.3f}")
    print(f"saved to {PROGRESS_FILE}" + (f"; fields in {ARCHIVE_DIR}/" if args.archive_every else ""))


if __name__ == "__main__":
    main()
