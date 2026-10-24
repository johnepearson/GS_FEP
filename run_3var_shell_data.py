"""
Correlation-length data collection for the three-variable chemical-memory
Gray-Scott model, at n=1024 with the pseudo-spectral solver -- the proper,
full-scale version of the Dw=0 vs Dw=Dv comparison of ESM §S13.

Set GS_FFT_WORKERS to your CPU count for a large speedup, e.g.:
    GS_FFT_WORKERS=8 python3 run_3var_shell_data.py --dw 0.0 --target-n 100000
    GS_FFT_WORKERS=8 python3 run_3var_shell_data.py --dw 1e-5 --target-n 100000

Resumable: re-running the same command with the same --dw continues from
its own checkpoint and accumulated data file rather than starting over.
Prints per-frame timing so you can sanity-check progress early.
"""
import argparse
import os
import time
import numpy as np

from gray_scott_3var_spectral import ThreeVarPseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
from run_confound_checks import (organism_interior_exterior_states_fast,
                                  organism_shell_profile_field_fast)
import run_ensemble_merged as prod   # merge_overlapping_organisms, periodic_centroids
# Uses the crop-based, modulo-indexed fast functions (periodic edges handled
# by direct indexing), matching every other production script in this package.

GS_CHECKPOINT = "checkpoint_1024_settled.npz"  # existing 2-variable n=1024 checkpoint
EPS = 0.25
BURN_IN_STEPS = 3000
SAVE_EVERY = 300
RADII = list(range(0, 161, 8)) + list(range(160, 401, 24))[1:]  # matches main n=1024 shell schedule
MIN_PEAK_DISTANCE = 24   # current production individuation (was 32, unmerged)
PAD = 80
DEPTH_FRACTION = 0.5
EXTERIOR_RING_PX = 24
CROP_IE = 90        # organism extent (~49) + ring (24 or 6) + margin
CROP_SHELL = 460    # covers the full 400px shell schedule + organism extent + margin


def tag_for(dw):
    return "Dw0" if dw == 0.0 else "DwDv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dw", type=float, required=True, help="Dw value, e.g. 0.0 or 1e-5")
    ap.add_argument("--target-n", type=int, default=100000)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="if given, add Gaussian noise (sigma 1e-3) to u and v after loading the settled "
                         "checkpoint, and tag all output files with this seed; without it the run is "
                         "deterministic and identical every time")
    args = ap.parse_args()

    tag = tag_for(args.dw)
    if args.noise_seed is not None:
        tag = f"{tag}_s{args.noise_seed}"
    running_ckpt = f"checkpoint_3var_1024_{tag}_running.npz"   # dynamics checkpoint: reused (segmentation does not affect the dynamics)
    accum_file = f"shell_data_3var_1024_{tag}_md24.npz"       # NEW file: spacing-24 merged individuation

    if os.path.exists(running_ckpt):
        sim = ThreeVarPseudoSpectralGS.from_checkpoint(running_ckpt)
        print(f"resuming {tag} run from t={sim.time:.0f}")
    else:
        sim = ThreeVarPseudoSpectralGS.from_gs_checkpoint(GS_CHECKPOINT, eps=args.eps, dt=1.0, Dw=args.dw)
        if args.noise_seed is not None:
            rng = np.random.default_rng(args.noise_seed)
            sim.u = sim.u + 1e-3 * rng.standard_normal(sim.u.shape)
            sim.v = sim.v + 1e-3 * rng.standard_normal(sim.v.shape)
        print(f"starting fresh {tag} run (Dw={args.dw}, eps={args.eps}); burn-in {BURN_IN_STEPS} steps...")
        t0 = time.time()
        sim.run(BURN_IN_STEPS)
        print(f"burn-in done, t={sim.time:.0f}, {time.time()-t0:.0f}s")

    if os.path.exists(accum_file):
        d = np.load(accum_file)
        I, fu, fv, profiles, rad = d["I"], d["fu"], d["fv"], d["profiles"], d["radius_px"]
        print(f"resuming accumulation: {len(I)} organism-instances so far")
    else:
        I = np.array([]); fu = np.array([]); fv = np.array([]); rad = np.array([])
        profiles = np.zeros((0, len(RADII) - 1))

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)

    t_start = time.time()
    frame = 0
    while len(I) < args.target_n:
        t0 = time.time()
        sim.run(SAVE_EVERY)
        proxy.u, proxy.v = sim.u, sim.v
        proxy.Lu = laplacian_periodic(sim.u, sim.dx)
        proxy.Lv = laplacian_periodic(sim.v, sim.dx)

        raw_labels, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
        if raw_n == 0:
            print("population died out, stopping")
            break
        labels, n, _merge = prod.merge_overlapping_organisms(raw_labels, raw_n, proxy)
        flux_u, flux_v = proxy.per_organism_flux(labels, n)
        centroids = prod.periodic_centroids(labels, n, proxy.n)
        _c, radii_phys = proxy.organism_geometry(labels, n)
        I_frame, _E = organism_interior_exterior_states_fast(
            proxy, labels, n, centroids, depth_fraction=DEPTH_FRACTION,
            exterior_ring_px=EXTERIOR_RING_PX, crop=CROP_IE)
        profile_frame = organism_shell_profile_field_fast(
            proxy.v, labels, n, centroids, RADII, crop=CROP_SHELL)

        I = np.concatenate([I, I_frame])
        fu = np.concatenate([fu, flux_u])
        fv = np.concatenate([fv, flux_v])
        rad = np.concatenate([rad, radii_phys / proxy.dx])
        profiles = np.concatenate([profiles, profile_frame], axis=0)

        frame += 1
        dt_frame = time.time() - t0
        elapsed = time.time() - t_start
        if frame % 5 == 0:
            rate = len(I) / elapsed if elapsed > 0 else 0
            remaining = (args.target_n - len(I)) / rate if rate > 0 else float("inf")
            print(f"[{tag}] frame {frame}  t={sim.time:.0f}  n_organisms={n}  "
                  f"N={len(I)}/{args.target_n}  ({dt_frame:.1f}s/frame, "
                  f"~{remaining/60:.0f}min remaining)")

        # checkpoint every frame so a long run can be safely interrupted/resumed
        sim.save_checkpoint(running_ckpt)
        np.savez(accum_file, I=I, fu=fu, fv=fv, profiles=profiles, radius_px=rad, radii=np.array(RADII),
                 min_peak_distance=MIN_PEAK_DISTANCE)

    print(f"\n[{tag}] done: N={len(I)} organism-instances, saved to {accum_file}")


if __name__ == "__main__":
    main()
