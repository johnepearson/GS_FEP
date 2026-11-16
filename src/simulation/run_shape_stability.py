"""
Direct test of shape (not just position) stability, addressing the
gap between rho (centroid displacement) and Friston's actual
criterion (stability of the blanket's coupling structure). An
organism could have perfectly stable position while its boundary
shape churns rapidly, or vice versa -- rho alone cannot distinguish
these cases.

For each organism at each frame, extracts a rotation-invariant shape
descriptor: the boundary contour resampled onto a fixed 64-point
angular grid (r(theta) relative to the organism's own centroid).
Matches organisms frame-to-frame using the same overlap-based
matching rho itself uses (match_organisms_by_overlap) -- this
naturally excludes ambiguous cases (splits, deaths) from the
comparison, exactly as it does for rho, since a split organism's mask
does not claim a single unambiguous future counterpart.

For each matched pair, finds the best-aligning rotation (brute-force
search over all 64 circular shifts) and reports the resulting
correlation between the two shape vectors. A value near 1 means the
shape is stable up to rotation; lower values indicate real shape
change (elongation, protrusions, deformation) independent of any
translation or rotation.

Usage:
    python run_shape_stability.py --n-frames 200
    python run_shape_stability.py --resume
"""
import argparse
import time
import numpy as np
from skimage import measure

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, match_organisms_by_overlap
import run_ensemble_merged as prod

MIN_PEAK_DISTANCE = 24
PAD = 80
SAMPLE_EVERY_STEPS = 10
N_ANGLES = 64
PAD_FACTOR = 2.5

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "shape_stability_progress_md24_merged.npz"


def organism_shape_vector(mask, cy, cx, n_angles=N_ANGLES):
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    dy = contour[:, 0] - cy
    dx = contour[:, 1] - cx
    theta = np.arctan2(dy, dx)
    r = np.sqrt(dy ** 2 + dx ** 2)
    order = np.argsort(theta)
    theta_s, r_s = theta[order], r[order]
    theta_ext = np.concatenate([theta_s - 2 * np.pi, theta_s, theta_s + 2 * np.pi])
    r_ext = np.concatenate([r_s, r_s, r_s])
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    return np.interp(angles, theta_ext, r_ext)


def crop_organism_mask(labels, N, cy_px, cx_px, radius_px, org_id):
    half = int(radius_px * PAD_FACTOR) + 5
    r0, c0 = int(round(cy_px)), int(round(cx_px))
    rows = (np.arange(r0 - half, r0 + half)) % N
    cols = (np.arange(c0 - half, c0 + half)) % N
    sub_labels = labels[np.ix_(rows, cols)]
    mask = (sub_labels == org_id)
    return mask, half, half


def best_rotation_correlation(r1, r2):
    n = len(r1)
    r1n = (r1 - r1.mean()) / (r1.std() + 1e-12)
    best = -1.0
    for shift in range(n):
        r2s = np.roll(r2, shift)
        r2n = (r2s - r2s.mean()) / (r2s.std() + 1e-12)
        c = np.mean(r1n * r2n)
        if c > best:
            best = c
    return best


def get_all_shape_vectors(proxy, labels, n, centroids, radii):
    N = proxy.n
    vecs = {}
    for i in range(1, n + 1):
        cy_px, cx_px = centroids[i - 1]
        r_px = radii[i - 1] / proxy.dx
        mask, cy_local, cx_local = crop_organism_mask(labels, N, cy_px, cx_px, r_px, i)
        if mask.sum() < 8:
            continue
        sv = organism_shape_vector(mask, cy_local, cx_local)
        if sv is not None:
            vecs[i] = sv
    return vecs


def segment_merged(proxy):
    """The paper's individuation: Otsu mask, markers at spacing MIN_PEAK_DISTANCE, marker-controlled watershed,
    then merge of touching regions whose equivalent-radius circles overlap. Returns labels, n, centroids
    (periodic-correct), radii (physical units)."""
    raw_labels, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
    if raw_n == 0:
        return raw_labels, 0, np.zeros((0, 2)), np.zeros(0)
    labels, n, _merge_size = prod.merge_overlapping_organisms(raw_labels, raw_n, proxy)
    _, radii = proxy.organism_geometry(labels, n)
    centroids = prod.periodic_centroids(labels, n, proxy.n)
    return labels, n, centroids, radii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.resume:
        d = np.load(PROGRESS_FILE)
        sim = PseudoSpectralGS(n=int(d["n"]))
        sim.u, sim.v, sim.time = d["u"], d["v"], float(d["time"])
        shape_corrs = list(d["shape_corrs"])
        frames_done = int(d["frames_done"])
        print(f"resumed at t={sim.time:.0f}, {frames_done} frames done, "
              f"{len(shape_corrs)} matched pairs so far")
    else:
        sim = PseudoSpectralGS.from_checkpoint(CHECKPOINT_FILE)
        shape_corrs = []
        frames_done = 0
        print(f"loaded checkpoint at t={sim.time:.0f}")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)
    proxy.u, proxy.v = sim.u, sim.v
    prev_labels, prev_n, prev_centroids, prev_radii = segment_merged(proxy)
    prev_vecs = get_all_shape_vectors(proxy, prev_labels, prev_n, prev_centroids, prev_radii)

    t_start = time.time()
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.u, sim.v
        curr_labels, curr_n, curr_centroids, curr_radii = segment_merged(proxy)
        if curr_n == 0:
            print("population died out, stopping")
            break
        curr_vecs = get_all_shape_vectors(proxy, curr_labels, curr_n, curr_centroids, curr_radii)

        forward, backward = match_organisms_by_overlap(prev_labels, curr_labels, prev_n, curr_n)
        for p, (c, frac) in forward.items():
            if c is None or frac <= 0.5:
                continue
            claimants = backward.get(c, [])
            if len(claimants) != 1:
                continue
            if p not in prev_vecs or c not in curr_vecs:
                continue
            corr = best_rotation_correlation(prev_vecs[p], curr_vecs[c])
            shape_corrs.append(corr)

        prev_labels, prev_n, prev_vecs = curr_labels, curr_n, curr_vecs
        frames_done += 1

        elapsed = time.time() - t_start
        if frames_done % 10 == 0:
            print(f"frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  organisms={curr_n}  "
                  f"({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.2f}s/frame)  "
                  f"matched pairs so far: {len(shape_corrs)}")

        np.savez(PROGRESS_FILE, u=sim.u, v=sim.v, time=sim.time, n=sim.n,
                 frames_done=frames_done, shape_corrs=np.array(shape_corrs))

    sc = np.array(shape_corrs)
    print(f"\nN matched organism-transitions: {len(sc)}")
    print(f"Shape correlation (rotation-aligned): median={np.median(sc):.4f}  "
          f"mean={sc.mean():.4f}")
    print(f"  10th pct={np.percentile(sc,10):.4f}  1st pct={np.percentile(sc,1):.4f}")
    print(f"Shape instability (1 - correlation): median={np.median(1-sc):.4f}  "
          f"90th pct={np.percentile(1-sc,90):.4f}  99th pct={np.percentile(1-sc,99):.4f}")


if __name__ == "__main__":
    main()
