"""
Section 3.3.5 check: level-set individuation, with no watershed and no
neighbor-dependent boundary construction at all.

Organisms are found as connected components of a FIXED threshold on v
(0.1356, the empirical boundary plateau value for well-separated
organisms established in Section~\\ref{sec:boundary_mechanism}), with no
watershed step and no basin competition -- component extent is
determined purely by where v crosses a single fixed value, so two
organisms only get separate components if v actually dips below
threshold between them. Components containing more than one local
maximum of v (touching/merged organisms this simpler method cannot
separate) or zero local maxima are excluded, exactly as described in
the paper (about 11% of the population at any sampled instant).

This is a full reconstruction, not the quick, small-N exploratory run
the paper's headline number (N=3,141) apparently came from -- that
script no longer exists (see run_boundary_fixed_radius.py's docstring
for the same situation with the Section 3.6 check). This version:
  - uses the SAME sampling cadence as the other single-trajectory
    production checks (SAMPLE_EVERY_STEPS=10, run_remaining_checks.py)
  - is run long enough to reach a sample size comparable to the other
    single-trajectory headline checks (N=77,595 in Section 3.3.1;
    N=104,473 in Section 3.5), not a quick one-off
  - uses the verified crop-based interior/exterior computation
    (organism_interior_exterior_states_fast, cross-checked exactly
    against the paper's production function away from periodic edges;
    see diagnose_residual.py) rather than the buggy whole-image
    np.pad(mode="wrap") version
  - additionally recovers per-organism states for the EXCLUDED
    multi-maximum components via a secondary, LOCAL watershed split
    (applied only within each multi-maximum component, using its own
    local maxima as markers), so the referee's selection-effect
    question -- whether discarding exactly the closest/touching pairs
    biases the reported correlation -- can be checked directly by
    comparing the single-peak-only result against the full
    (single-peak + recovered multi-peak) result.

Usage:
    python run_level_set_individuation.py --n-frames 450
    python run_level_set_individuation.py --resume
"""
import argparse
import time
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic, _merge_periodic_labels
from run_confound_checks import organism_interior_exterior_states_fast

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "level_set_progress.npz"

V_THRESHOLD = 0.1356
PAD = 80
MIN_PEAK_DISTANCE = 32
DEPTH_FRACTION = 0.5
EXTERIOR_RING_PX = 24
SAMPLE_EVERY_STEPS = 10  # matches run_remaining_checks.py / Section 3.5's cadence
SAVE_EVERY_FRAMES = 20


def find_all_peaks(v, pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE):
    """All local maxima of v, periodic-safe (pad, detect, crop, keep
    only peaks that land back in the original tile)."""
    n = v.shape[0]
    vp = np.pad(v, pad, mode="wrap")
    thresh = threshold_otsu(v)
    mask = vp > thresh
    coords = peak_local_max(vp, min_distance=min_peak_distance, labels=mask.astype(int))
    keep = (coords[:, 0] >= pad) & (coords[:, 0] < pad + n) & \
           (coords[:, 1] >= pad) & (coords[:, 1] < pad + n)
    return coords[keep] - pad


def level_set_components(v, threshold=V_THRESHOLD):
    """Connected components of {v > threshold}, periodic-safe via
    seam-merging (no wrap-padding, so none of the area-inflation risk
    that np.pad(mode='wrap') carries elsewhere in this codebase)."""
    mask = v > threshold
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n_features = ndimage.label(mask, structure=structure)
    labeled, n_merged = _merge_periodic_labels(labeled, n_features)
    return labeled, n_merged


def assign_peaks_to_components(labeled, peaks):
    """For each connected component id (1..n), count how many of the
    detected local maxima fall inside it."""
    if len(peaks) == 0:
        return np.zeros(labeled.max(), dtype=int)
    rows = np.clip(np.round(peaks[:, 0]).astype(int), 0, labeled.shape[0] - 1)
    cols = np.clip(np.round(peaks[:, 1]).astype(int), 0, labeled.shape[1] - 1)
    comp_ids = labeled[rows, cols]
    counts = np.zeros(labeled.max(), dtype=int)
    for cid in comp_ids:
        if cid > 0:
            counts[cid - 1] += 1
    return counts


def relabel_subset(labeled, keep_ids):
    """Build a fresh contiguous 1..n_keep label array containing only
    the components in keep_ids (in the order given)."""
    n = labeled.shape[0]
    out = np.zeros_like(labeled)
    remap = np.zeros(labeled.max() + 1, dtype=labeled.dtype)
    for new_id, old_id in enumerate(keep_ids, start=1):
        remap[old_id] = new_id
    out = remap[labeled]
    return out, len(keep_ids)


def split_multi_peak_components(labeled, counts, peaks, v):
    """Secondary, LOCAL watershed split of the excluded multi-maximum
    components only, using each component's own local maxima as
    markers -- recovers per-organism sub-labels for the 'ambiguous'
    11% so their I/E states can be computed too, for the selection-
    effect sensitivity check. Zero-maximum components are left out
    entirely (nothing to split on; these are not organisms this
    method can locate at all, matching the paper's description)."""
    multi_ids = np.where(counts >= 2)[0] + 1
    if len(multi_ids) == 0:
        return np.zeros_like(labeled), 0

    multi_mask = np.isin(labeled, multi_ids)
    if not multi_mask.any():
        return np.zeros_like(labeled), 0

    rows = np.clip(np.round(peaks[:, 0]).astype(int), 0, labeled.shape[0] - 1)
    cols = np.clip(np.round(peaks[:, 1]).astype(int), 0, labeled.shape[1] - 1)
    comp_of_peak = labeled[rows, cols]
    in_multi = np.isin(comp_of_peak, multi_ids)
    multi_peaks = peaks[in_multi]

    markers = np.zeros_like(labeled)
    for i, (r, c) in enumerate(multi_peaks, start=1):
        markers[int(round(r)), int(round(c))] = i

    split_labels = watershed(-v, markers=markers, mask=multi_mask)
    return split_labels, len(multi_peaks)


def compute_states(sim, proxy, labels, n, depth_fraction=DEPTH_FRACTION,
                    exterior_ring_px=EXTERIOR_RING_PX):
    if n == 0:
        return (np.array([]),) * 4
    centroids = np.array(ndimage.center_of_mass(np.ones_like(labels), labels=labels,
                                                  index=np.arange(1, n + 1)))
    I, E = organism_interior_exterior_states_fast(sim, labels, n, centroids,
                                                   depth_fraction=depth_fraction,
                                                   exterior_ring_px=exterior_ring_px)
    flux_u, flux_v = proxy.per_organism_flux(labels, n)
    return I, E, flux_u, flux_v


def partial_corr(I, E, flux_u, flux_v):
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(flux_u) & np.isfinite(flux_v)
    B = np.column_stack([np.ones(good.sum()), flux_u[good], flux_v[good]])
    def resid(x):
        beta, *_ = np.linalg.lstsq(B, x[good], rcond=None)
        return x[good] - B @ beta
    return np.corrcoef(resid(I), resid(E))[0, 1], good.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=450)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.resume:
        d = np.load(PROGRESS_FILE)
        sim = PseudoSpectralGS(n=int(d["n"]))
        sim.u, sim.v, sim.time = d["u"], d["v"], float(d["time"])
        keys = ["I_single", "E_single", "fu_single", "fv_single",
                "I_multi", "E_multi", "fu_multi", "fv_multi"]
        lists = {k: [d[k]] for k in keys}
        n_total_components = int(d["n_total_components"])
        n_single = int(d["n_single"])
        n_multi_excluded_peaks = int(d["n_multi_excluded_peaks"])
        n_zero = int(d["n_zero"])
        frames_done = int(d["frames_done"])
        print(f"resumed at t={sim.time:.0f}, {frames_done} frames done")
    else:
        sim = PseudoSpectralGS.from_checkpoint(CHECKPOINT_FILE)
        keys = ["I_single", "E_single", "fu_single", "fv_single",
                "I_multi", "E_multi", "fu_multi", "fv_multi"]
        lists = {k: [] for k in keys}
        n_total_components = n_single = n_multi_excluded_peaks = n_zero = 0
        frames_done = 0
        print(f"loaded checkpoint at t={sim.time:.0f}")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)

    t_start = time.time()
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.u, sim.v
        proxy.Lu = laplacian_periodic(sim.u, sim.dx)
        proxy.Lv = laplacian_periodic(sim.v, sim.dx)

        labeled, n_comp = level_set_components(sim.v)
        peaks = find_all_peaks(sim.v)
        n_total_components += n_comp

        if n_comp > 0:
            counts = assign_peaks_to_components(labeled, peaks)
            single_ids = np.where(counts == 1)[0] + 1
            n_zero += int((counts == 0).sum())

            labels_single, n_s = relabel_subset(labeled, single_ids)
            I_s, E_s, fu_s, fv_s = compute_states(sim, proxy, labels_single, n_s)
            for k, v_ in zip(["I_single", "E_single", "fu_single", "fv_single"],
                              [I_s, E_s, fu_s, fv_s]):
                lists[k].append(v_)
            n_single += n_s

            split_labels, n_m_peaks = split_multi_peak_components(labeled, counts, peaks, sim.v)
            n_multi_excluded_peaks += n_m_peaks
            if n_m_peaks > 0:
                I_m, E_m, fu_m, fv_m = compute_states(sim, proxy, split_labels, n_m_peaks)
                for k, v_ in zip(["I_multi", "E_multi", "fu_multi", "fv_multi"],
                                  [I_m, E_m, fu_m, fv_m]):
                    lists[k].append(v_)

        frames_done += 1
        elapsed = time.time() - t_start
        if frames_done % 20 == 0:
            print(f"frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  "
                  f"N_single so far={n_single}  N_multi-recovered so far={n_multi_excluded_peaks}  "
                  f"({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.2f}s/frame)")

        if frames_done % SAVE_EVERY_FRAMES == 0 or frames_done == args.n_frames:
            save_kwargs = {k: np.concatenate(v_) if len(v_) else np.array([]) for k, v_ in lists.items()}
            np.savez(PROGRESS_FILE, u=sim.u, v=sim.v, time=sim.time, n=sim.n,
                     frames_done=frames_done, n_total_components=n_total_components,
                     n_single=n_single, n_multi_excluded_peaks=n_multi_excluded_peaks,
                     n_zero=n_zero, **save_kwargs)

    I_single = np.concatenate(lists["I_single"])
    E_single = np.concatenate(lists["E_single"])
    fu_single = np.concatenate(lists["fu_single"])
    fv_single = np.concatenate(lists["fv_single"])
    I_multi = np.concatenate(lists["I_multi"]) if any(len(x) for x in lists["I_multi"]) else np.array([])
    E_multi = np.concatenate(lists["E_multi"]) if any(len(x) for x in lists["E_multi"]) else np.array([])
    fu_multi = np.concatenate(lists["fu_multi"]) if any(len(x) for x in lists["fu_multi"]) else np.array([])
    fv_multi = np.concatenate(lists["fv_multi"]) if any(len(x) for x in lists["fv_multi"]) else np.array([])

    print(f"\ndone: {frames_done} frames")
    print(f"total level-set components seen: {n_total_components}")
    print(f"  single-maximum (retained, paper's construction): {n_single} "
          f"({100*n_single/n_total_components:.1f}%)")
    print(f"  zero-maximum (excluded, no peak found): {n_zero} "
          f"({100*n_zero/n_total_components:.1f}%)")
    print(f"  multi-maximum components (excluded by paper; recovered here via local "
          f"watershed): peaks recovered = {n_multi_excluded_peaks}")

    r_single, N_single_good = partial_corr(I_single, E_single, fu_single, fv_single)
    print(f"\nSingle-peak-only (matches paper's Section 3.3.5 construction):")
    print(f"  N={N_single_good}  partial r = {r_single:.4f}")

    if len(I_multi) > 0:
        r_multi_only, N_multi_good = partial_corr(I_multi, E_multi, fu_multi, fv_multi)
        print(f"\nRecovered multi-maximum organisms only (the excluded 11%, watershed-split):")
        print(f"  N={N_multi_good}  partial r = {r_multi_only:.4f}")

        I_full = np.concatenate([I_single, I_multi])
        E_full = np.concatenate([E_single, E_multi])
        fu_full = np.concatenate([fu_single, fu_multi])
        fv_full = np.concatenate([fv_single, fv_multi])
        r_full, N_full_good = partial_corr(I_full, E_full, fu_full, fv_full)
        print(f"\nFull population (single-peak + recovered multi-peak organisms):")
        print(f"  N={N_full_good}  partial r = {r_full:.4f}")
        print(f"\n=> excluding the closest/touching pairs changes r from {r_full:.4f} "
              f"(full) to {r_single:.4f} (single-peak only)")


if __name__ == "__main__":
    main()
