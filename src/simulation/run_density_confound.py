"""
Check whether local organism density is a confound for the main
interior-vs-exterior partial correlation (Section 3.3): does adding
local density (neighbor count within a fixed radius) as an additional
regressor change the residual correlation meaningfully?

Uses a modest sample (this only needs to show the coefficient doesn't
move much, not establish a precise estimate in its own right).

Usage:
    python run_density_confound.py --n-frames 30
    python run_density_confound.py --resume
"""
import argparse
import time
import numpy as np
from scipy.spatial import cKDTree

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, laplacian_periodic
from gray_scott_distance_transform import organism_interior_exterior_states_distance

MIN_PEAK_DISTANCE = 32
PAD = 80
EXTERIOR_RING_PX = 24
DENSITY_RADIUS_PX = 80   # 20 * 4, matching the original density-radius scaling
SAMPLE_EVERY_STEPS = 100

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"
PROGRESS_FILE = "density_confound_progress.npz"


def local_density(centroids, N, radius):
    if len(centroids) < 2:
        return np.zeros(len(centroids))
    tree = cKDTree(centroids, boxsize=N)
    counts = tree.query_ball_point(centroids, r=radius, return_length=True)
    return counts - 1


def residualize(y, B):
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.resume:
        d = np.load(PROGRESS_FILE)
        sim = PseudoSpectralGS(n=int(d["n"]))
        sim.u, sim.v, sim.time = d["u"], d["v"], float(d["time"])
        I_list, E_list, fu_list, fv_list, dens_list = [d["I"]], [d["E"]], [d["fu"]], [d["fv"]], [d["density"]]
        frames_done = int(d["frames_done"])
        print(f"resumed at t={sim.time:.0f}, {frames_done} frames done")
    else:
        sim = PseudoSpectralGS.from_checkpoint(CHECKPOINT_FILE)
        I_list, E_list, fu_list, fv_list, dens_list = [], [], [], [], []
        frames_done = 0
        print(f"loaded checkpoint at t={sim.time:.0f}")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)

    t_start = time.time()
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.u, sim.v
        labels, n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
        if n == 0:
            print("population died out, stopping")
            break
        centroids, radii = proxy.organism_geometry(labels, n)
        proxy.Lu = laplacian_periodic(sim.u, sim.dx)
        proxy.Lv = laplacian_periodic(sim.v, sim.dx)
        fu, fv = proxy.per_organism_flux(labels, n)
        I, E = organism_interior_exterior_states_distance(
            proxy, labels, n, depth_fraction=0.5, exterior_ring_px=EXTERIOR_RING_PX)
        dens = local_density(centroids, sim.n, DENSITY_RADIUS_PX)

        I_list.append(I); E_list.append(E); fu_list.append(fu); fv_list.append(fv); dens_list.append(dens)
        frames_done += 1
        elapsed = time.time() - t_start
        print(f"frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  organisms={n}  "
              f"({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.1f}s/frame)")

        np.savez(PROGRESS_FILE, u=sim.u, v=sim.v, time=sim.time, n=sim.n, frames_done=frames_done,
                 I=np.concatenate(I_list), E=np.concatenate(E_list),
                 fu=np.concatenate(fu_list), fv=np.concatenate(fv_list),
                 density=np.concatenate(dens_list))

    I_all = np.concatenate(I_list); E_all = np.concatenate(E_list)
    fu_all = np.concatenate(fu_list); fv_all = np.concatenate(fv_list)
    dens_all = np.concatenate(dens_list)
    good = np.isfinite(I_all) & np.isfinite(E_all) & np.isfinite(fu_all) & np.isfinite(fv_all)
    I, E, fu, fv, dens = I_all[good], E_all[good], fu_all[good], fv_all[good], dens_all[good]
    N = len(I)

    B_without = np.column_stack([np.ones(N), fu, fv])
    r_without = np.corrcoef(residualize(I, B_without), residualize(E, B_without))[0, 1]

    B_with = np.column_stack([np.ones(N), fu, fv, dens])
    r_with = np.corrcoef(residualize(I, B_with), residualize(E, B_with))[0, 1]

    print(f"\nN={N}")
    print(f"Partial correlation WITHOUT density: r={r_without:.4f}")
    print(f"Partial correlation WITH density:    r={r_with:.4f}")
    print(f"Change: {abs(r_with - r_without):.4f}")


if __name__ == "__main__":
    main()
