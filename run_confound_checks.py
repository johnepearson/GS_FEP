"""
Referee Comment #2 (cheap half): a single production data-collection
pass recording, for every organism-instance, everything needed to test
three candidate confounds against the headline conditional-independence
result -- without touching the original ensemble production code.

For each sampled organism-instance, records:
  - I, E, flux_u, flux_v            (exactly as in Section 3.3, using
                                      the distance-transform interior/
                                      exterior definition actually used
                                      in the paper's pipeline)
  - frame_id                         (for the frame-level global-
                                      confound / frame-fixed-effects
                                      check)
  - radius (mm), eccentricity        (size/shape covariates)
  - u_shells                         (mean u in a handful of near-field
                                      annular shells beyond the
                                      organism's boundary, exactly the
                                      same construction as the existing
                                      v-shell enrichment test, generalized
                                      to an arbitrary field)
Also records, once per frame: population count, frame-mean u, and the
frame's Otsu(v) threshold -- for the frame-level confound check.

NOT included here: real lineage-tracked age / time-to-division (the
referee's stronger ask). That needs continuous frame-to-frame organism
tracking (see run_lifecycle.py / run_shape_stability.py for the
matching machinery) and is a separate, more expensive script -- this
one is deliberately scoped to what a single independent-snapshot pass
can answer, to check whether the effect survives these cheaper
controls before committing to that.

Usage:
    python run_confound_checks.py --n-frames 300
    python run_confound_checks.py --resume
"""
import argparse
import time
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import threshold_otsu

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic

try:
    from gray_scott_spectral_c import PseudoSpectralGSFast
    _HAVE_FAST = True
except OSError:
    _HAVE_FAST = False

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "confound_checks_progress.npz"

PAD = 80
MIN_PEAK_DISTANCE = 32
DEPTH_FRACTION = 0.5
EXTERIOR_RING_PX = 24
SAMPLE_EVERY_STEPS = 100  # match the single-trajectory §3.3.1 spacing, not the 10-step §3.5-style spacing
SAVE_EVERY_FRAMES = 20

U_SHELL_RADII_PX = [0, 6, 12, 18, 24, 36, 48, 64]  # 7 shells, out to ~3 organism radii
N_U_SHELLS = len(U_SHELL_RADII_PX) - 1

# Crop half-widths for the local-window implementations below. Each must
# exceed (measured max organism extent-from-centroid, ~49px) plus
# whatever the ring/dilation step reaches beyond the organism's own
# boundary. Verified empirically (diagnose_residual.py, 2026-09-26):
# comparing this crop-based method against the ORIGINAL production
# function (organism_interior_exterior_states_distance in
# gray_scott_distance_transform.py) on non-edge-affected organisms gives
# EXACT agreement (0.00000 max abs diff) once the crop covers the
# organism + ring with margin to spare. NOTE: the original function has
# an independent, previously-unknown bug for organisms within ~70-85px
# of a periodic domain edge (its np.pad(..., mode="wrap") duplicates
# such an organism's own pixels into the wrap margin, inflating its
# computed area/radius/threshold -- confirmed via a corner-case organism
# whose padded-array area was ~4x its true area). The crop-based
# functions below use direct periodic (modulo) indexing instead, so they
# do NOT have that bug and are used uniformly for ALL organisms here,
# not just non-edge ones.
CROP_IE = 90       # interior/exterior: organism extent (~49) + ring (24) + margin
CROP_USHELL = 140  # u-shells: organism extent (~49) + max shell radius (64) + margin


def _crop_mask_and_field(labels, field, i, cy, cx, n, crop):
    rows = (np.arange(cy - crop, cy + crop) % n).astype(int)
    cols = (np.arange(cx - crop, cx + crop) % n).astype(int)
    sub_labels = labels[np.ix_(rows, cols)]
    sub_field = field[np.ix_(rows, cols)]
    return sub_labels == i, sub_field


def organism_interior_exterior_states_fast(sim, labels, n_organisms, centroids,
                                            depth_fraction=DEPTH_FRACTION,
                                            exterior_ring_px=EXTERIOR_RING_PX,
                                            crop=CROP_IE):
    """Crop-based (periodic-correct) replacement for
    organism_interior_exterior_states_distance: same depth-fraction/
    exterior-ring definition, but each organism's distance transform and
    dilation run on a small local window (2*crop x 2*crop) built with
    direct modulo indexing, instead of a whole-image np.pad(mode="wrap")
    -- avoiding the wrap-duplication bug in the original function and
    running orders of magnitude faster. See diagnose_residual.py for the
    exactness check against the original."""
    n = sim.n
    structure = ndimage.generate_binary_structure(2, 2)
    I = np.full(n_organisms, np.nan)
    E = np.full(n_organisms, np.nan)
    for i in range(1, n_organisms + 1):
        cy, cx = centroids[i - 1]
        if not np.isfinite(cy):
            continue
        mask, v_c = _crop_mask_and_field(labels, sim.v, i, int(round(cy)), int(round(cx)), n, crop)
        if not mask.any():
            continue
        area = mask.sum()
        radius = np.sqrt(area / np.pi)
        threshold = depth_fraction * radius
        dist = ndimage.distance_transform_edt(mask)
        core = dist >= threshold
        if not core.any():
            max_depth = dist.max()
            core = dist >= (max_depth * 0.99) if max_depth > 0 else mask
        I[i - 1] = v_c[core].mean()
        dil = ndimage.binary_dilation(mask, structure=structure, iterations=exterior_ring_px)
        ring = dil & (~mask)
        if ring.any():
            E[i - 1] = v_c[ring].mean()
    return I, E


def organism_interior_exterior_states_adaptive_fast(sim, labels, n_organisms, centroids,
                                                      keep_fraction=0.5,
                                                      exterior_ring_px=EXTERIOR_RING_PX,
                                                      crop=CROP_IE):
    """Crop-based (periodic-correct) replacement for
    organism_interior_exterior_states_adaptive (gray_scott.py): same
    whole-mask-erosion definition (erode inward by a single depth
    scaled to the organism's own radius, rather than the local
    distance transform used by the _fast function above), but on a
    small local window per organism instead of a whole-image
    np.pad(mode="wrap") -- same speed benefit, added so the
    resolution/interior-definition 2x2 (run_resolution_2x2.py) doesn't
    have to pay the whole-image cost for either interior definition."""
    n = sim.n
    structure = ndimage.generate_binary_structure(2, 2)
    I = np.full(n_organisms, np.nan)
    E = np.full(n_organisms, np.nan)
    for i in range(1, n_organisms + 1):
        cy, cx = centroids[i - 1]
        if not np.isfinite(cy):
            continue
        mask, v_c = _crop_mask_and_field(labels, sim.v, i, int(round(cy)), int(round(cx)), n, crop)
        if not mask.any():
            continue
        area = mask.sum()
        radius = np.sqrt(area / np.pi)
        erode_px = max(1, int(round((1 - keep_fraction) * radius)))
        core = ndimage.binary_erosion(mask, structure=structure, iterations=erode_px)
        while not core.any() and erode_px > 0:
            erode_px -= 1
            core = ndimage.binary_erosion(mask, structure=structure, iterations=erode_px) \
                   if erode_px > 0 else mask
        I[i - 1] = v_c[core].mean()
        dil = ndimage.binary_dilation(mask, structure=structure, iterations=exterior_ring_px)
        ring = dil & (~mask)
        if ring.any():
            E[i - 1] = v_c[ring].mean()
    return I, E


def organism_shell_profile_field_fast(field, labels, n_organisms, centroids, radii, crop=CROP_USHELL):
    """Crop-based (periodic-correct) replacement for the whole-image
    organism_shell_profile_field: same annular-shell definition, but run
    on a small local window per organism instead of a whole-image
    np.pad(mode="wrap")."""
    n = labels.shape[0]
    n_shells = len(radii) - 1
    out = np.full((n_organisms, n_shells), np.nan)
    for i in range(1, n_organisms + 1):
        cy, cx = centroids[i - 1]
        if not np.isfinite(cy):
            continue
        mask, field_c = _crop_mask_and_field(labels, field, i, int(round(cy)), int(round(cx)), n, crop)
        if not mask.any():
            continue
        dist = ndimage.distance_transform_edt(~mask)
        for s in range(n_shells):
            shell = (dist > radii[s]) & (dist <= radii[s + 1])
            if shell.any():
                out[i - 1, s] = field_c[shell].mean()
    return out


def eccentricities(labels, n_organisms):
    ecc = np.full(n_organisms, np.nan)
    for region in regionprops(labels):
        if 1 <= region.label <= n_organisms:
            ecc[region.label - 1] = region.eccentricity
    return ecc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fast", action="store_true",
                     help="use the C solver (see README.md's C solver section to build "
                          "libgs_spectral.so first)")
    args = ap.parse_args()

    if args.fast and not _HAVE_FAST:
        raise SystemExit("--fast requested but libgs_spectral.so isn't built for this "
                          "machine. Build it first (see README.md), or drop --fast.")
    Solver = PseudoSpectralGSFast if args.fast else PseudoSpectralGS

    if args.resume:
        d = np.load(PROGRESS_FILE)
        sim = Solver(n=int(d["n"]))
        if hasattr(sim, "from_checkpoint_fields"):
            sim.from_checkpoint_fields(d["u"], d["v"])
        else:
            sim.u, sim.v = d["u"], d["v"]
        sim.time = float(d["time"])
        I_list = [d["I"]]; E_list = [d["E"]]; fu_list = [d["fu"]]; fv_list = [d["fv"]]
        frame_id_list = [d["frame_id"]]; radius_list = [d["radius"]]; ecc_list = [d["eccentricity"]]
        ushell_list = [d["u_shells"]]
        frame_pop_list = list(d["frame_pop"]); frame_mean_u_list = list(d["frame_mean_u"])
        frame_otsu_list = list(d["frame_otsu"])
        frames_done = int(d["frames_done"])
        print(f"resumed at t={sim.time:.0f}, {frames_done} frames done")
    else:
        sim = Solver.from_checkpoint(CHECKPOINT_FILE)
        I_list, E_list, fu_list, fv_list = [], [], [], []
        frame_id_list, radius_list, ecc_list, ushell_list = [], [], [], []
        frame_pop_list, frame_mean_u_list, frame_otsu_list = [], [], []
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

        frame_pop_list.append(n)
        frame_mean_u_list.append(proxy.u.mean())
        frame_otsu_list.append(threshold_otsu(proxy.v))

        if n > 0:
            proxy.Lu = laplacian_periodic(proxy.u, proxy.dx)
            proxy.Lv = laplacian_periodic(proxy.v, proxy.dx)
            flux_u, flux_v = proxy.per_organism_flux(labels, n)
            centroids, radii = proxy.organism_geometry(labels, n)
            I, E = organism_interior_exterior_states_fast(
                proxy, labels, n, centroids, depth_fraction=DEPTH_FRACTION, exterior_ring_px=EXTERIOR_RING_PX)
            ecc = eccentricities(labels, n)
            u_shells = organism_shell_profile_field_fast(proxy.u, labels, n, centroids, U_SHELL_RADII_PX)

            I_list.append(I); E_list.append(E); fu_list.append(flux_u); fv_list.append(flux_v)
            frame_id_list.append(np.full(n, frames_done))
            radius_list.append(radii); ecc_list.append(ecc); ushell_list.append(u_shells)

        frames_done += 1
        elapsed = time.time() - t_start
        if frames_done % 10 == 0:
            n_so_far = sum(len(x) for x in I_list)
            print(f"frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  n_organisms={n}  "
                  f"N so far={n_so_far}  ({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.2f}s/frame)")

        if frames_done % SAVE_EVERY_FRAMES == 0:
            np.savez(PROGRESS_FILE, u=sim.u, v=sim.v, time=sim.time, n=sim.n, frames_done=frames_done,
                     I=np.concatenate(I_list), E=np.concatenate(E_list),
                     fu=np.concatenate(fu_list), fv=np.concatenate(fv_list),
                     frame_id=np.concatenate(frame_id_list),
                     radius=np.concatenate(radius_list), eccentricity=np.concatenate(ecc_list),
                     u_shells=np.concatenate(ushell_list, axis=0),
                     frame_pop=np.array(frame_pop_list), frame_mean_u=np.array(frame_mean_u_list),
                     frame_otsu=np.array(frame_otsu_list))

    np.savez(PROGRESS_FILE, u=sim.u, v=sim.v, time=sim.time, n=sim.n, frames_done=frames_done,
             I=np.concatenate(I_list), E=np.concatenate(E_list),
             fu=np.concatenate(fu_list), fv=np.concatenate(fv_list),
             frame_id=np.concatenate(frame_id_list),
             radius=np.concatenate(radius_list), eccentricity=np.concatenate(ecc_list),
             u_shells=np.concatenate(ushell_list, axis=0),
             frame_pop=np.array(frame_pop_list), frame_mean_u=np.array(frame_mean_u_list),
             frame_otsu=np.array(frame_otsu_list))

    print(f"\ndone: {frames_done} frames, N={sum(len(x) for x in I_list)} organism-instances, "
          f"saved to {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
