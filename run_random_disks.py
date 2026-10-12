"""
Placebo-organism control for the blanket-thickness sweep.

Question: is the screening by a thick field band specific to organisms, or does an arbitrary disk of the same size, placed
anywhere in the pattern, show the same behaviour?  Offline, no new simulation: uses the archived fields of the master
trajectory (master_fields/frame_XXXX.npz).

For every STRIDE-th archived frame, each real organism is matched by one placebo disk of the same equivalent radius, centred at a
random position chosen so that the disk does not touch any organism and does not overlap another placebo disk.  For each
placebo disk the SAME measurements are made as for real organisms (same functions): interior state I (mean v in the core,
depth fraction 0.5), outward diffusive fluxes fu, fv across the disk edge, mean-v shells to 400 px and mean-u shells to 64 px
around the disk.  Output has the same layout as master_trajectory_progress.npz so run_band_sweep_cmi.py can read it.

    python3 run_random_disks.py --stride 3 --jobs 12
    python3 run_band_sweep_cmi.py --data random_disks.npz --tag _disks --only-all --jobs 12
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

from gray_scott import laplacian_periodic
from run_confound_checks import (organism_interior_exterior_states_fast, organism_shell_profile_field_fast)

N = 1024
CROP_IE, CROP_USHELL, CROP_VSHELL = 90, 140, 460
RADII = list(range(0, 161, 8)) + list(range(160, 401, 24))[1:]
U_SHELL_RADII_PX = [0, 6, 12, 18, 24, 36, 48, 64]
DEPTH_FRACTION, EXTERIOR_RING_PX = 0.5, 24


def dist_to_organisms(lab):
    """Periodic distance (px) from every pixel to the nearest organism pixel."""
    occ = lab > 0
    big = np.tile(occ, (3, 3))
    d = ndimage.distance_transform_edt(~big)
    return d[N:2 * N, N:2 * N]


def place_disks(lab, radii, rng, gap):
    d = dist_to_organisms(lab)
    centres, rads = [], []
    for r in radii:
        ok = np.argwhere(d >= r + gap)
        if len(ok) == 0:
            continue
        for _ in range(200):
            cy, cx = ok[rng.integers(len(ok))]
            good = True
            for (py, px), pr in zip(centres, rads):
                dy = abs(cy - py); dy = min(dy, N - dy)
                dx = abs(cx - px); dx = min(dx, N - dx)
                if np.hypot(dy, dx) < r + pr + 1:
                    good = False
                    break
            if good:
                centres.append((cy, cx)); rads.append(r)
                break
    return np.array(centres, dtype=float), np.array(rads)


def disk_labels(centres, rads):
    yy, xx = np.mgrid[0:N, 0:N]
    lab = np.zeros((N, N), dtype=np.int32)
    for i, ((cy, cx), r) in enumerate(zip(centres, rads), start=1):
        y0, y1 = int(cy - r - 2), int(cy + r + 3)
        x0, x1 = int(cx - r - 2), int(cx + r + 3)
        rows = np.arange(y0, y1) % N
        cols = np.arange(x0, x1) % N
        Y = (np.arange(y0, y1)[:, None] - cy) ** 2
        X = (np.arange(x0, x1)[None, :] - cx) ** 2
        m = (Y + X) <= r * r
        sub = lab[np.ix_(rows, cols)]
        sub[m] = i
        lab[np.ix_(rows, cols)] = sub
    return lab


def one_frame(args):
    path, seed, gap = args
    a = np.load(path)
    u, v, lab = a["u"].astype(np.float64), a["v"].astype(np.float64), a["labels"].astype(np.int32)
    n_org = int(lab.max())
    if n_org == 0:
        return None
    areas = ndimage.sum(np.ones_like(lab), labels=lab, index=np.arange(1, n_org + 1))
    radii = np.sqrt(areas / np.pi)
    rng = np.random.default_rng(seed)
    centres, rads = place_disks(lab, radii, rng, gap)
    n = len(rads)
    if n == 0:
        return None
    dl = disk_labels(centres, rads)
    return u, v, dl, centres, rads, n


def process(args):
    """Heavy part in the worker; needs solver constants, passed in."""
    path, seed, gap, Du, Dv, dx = args
    res = one_frame((path, seed, gap))
    if res is None:
        return None
    u, v, dl, centres, rads, n = res
    Lu, Lv = laplacian_periodic(u, dx), laplacian_periodic(v, dx)
    cell = dx * dx
    idx = np.arange(1, n + 1)
    fu = -Du * ndimage.sum(Lu, labels=dl, index=idx) * cell
    fv = -Dv * ndimage.sum(Lv, labels=dl, index=idx) * cell

    class P:  # minimal stand-in with the attributes the measurement functions use
        pass
    p = P(); p.n = N; p.u = u; p.v = v
    I, E = organism_interior_exterior_states_fast(p, dl, n, centres, depth_fraction=DEPTH_FRACTION,
                                                  exterior_ring_px=EXTERIOR_RING_PX, crop=CROP_IE)
    vsh = organism_shell_profile_field_fast(v, dl, n, centres, RADII, crop=CROP_VSHELL)
    ush = organism_shell_profile_field_fast(u, dl, n, centres, U_SHELL_RADII_PX, crop=CROP_USHELL)
    return dict(I=I, E=E, fu=fu, fv=fv, radius_px=rads, v_shells=vsh, u_shells=ush)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields-dir", default="master_fields")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gap", type=float, default=0.0, help="extra clearance (px) between a disk and any organism")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all selected frames (for quick tests)")
    ap.add_argument("--out", default="random_disks.npz")
    a = ap.parse_args()

    from gray_scott_spectral import PseudoSpectralGS
    sim = PseudoSpectralGS.from_checkpoint("checkpoint_1024_settled.npz")
    Du, Dv, dx = sim.Du, sim.Dv, sim.dx
    files = sorted(f for f in os.listdir(a.fields_dir) if f.startswith("frame_"))[::a.stride]
    if a.max_frames:
        files = files[:a.max_frames]
    jobs = [(os.path.join(a.fields_dir, f), a.seed * 100003 + i, a.gap, Du, Dv, dx) for i, f in enumerate(files)]
    print(f"{len(jobs)} frames, {a.jobs} workers")
    t0 = time.time()
    parts = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, r in enumerate(ex.map(process, jobs)):
            if r is not None:
                parts.append(r)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(jobs)} frames  ({time.time()-t0:.0f}s)", flush=True)
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    out["merge_size"] = np.ones(len(out["I"]), dtype=np.int16)
    np.savez(a.out, **out)
    good = np.isfinite(out["I"]) & np.isfinite(out["E"]) & np.isfinite(out["fu"]) & np.isfinite(out["fv"])
    print(f"wrote {a.out}: {len(out['I']):,} placebo disks ({good.sum():,} finite); "
          f"mean radius {out['radius_px'].mean():.1f} px; mean I {np.nanmean(out['I']):.4f}, SD I {np.nanstd(out['I']):.4f}")


if __name__ == "__main__":
    main()
