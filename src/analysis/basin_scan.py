#!/usr/bin/env python3
"""Basin scan: does a square seed in an n x n periodic box grow into a lasting spot population?

For each (n, side, seed): start from the trivial state (u=1, v=0) with a centred square of
side `side` pixels set to (u, v) = (0.5, 0.25) and 1% noise (Pearson 1993, as GrayScottSim),
run the production solver map (F=0.018, k=0.055, dt=1, dx = 2.5/1024) and record spot count
and mean v.  A run is called extinct when max(v) stays below --dead-vmax at two consecutive
records.  Results are appended to a CSV as each run finishes, so the scan can be killed.
Alive runs can be saved (--save-final DIR) and fed to lyapunov_spectrum.py with
    --from-frame DIR/final_n192_s20_seed0.npz --n 192 --corner 0 0
to decide steady versus chaotic.

Run from build/ (one FFT thread per run; parallel over runs):
  pilot : python3 -B ../src/analysis/basin_scan.py --n 256 --side 20 --seeds 1 --steps 100000 --every 1000 --verbose
  scan  : python3 -B ../src/analysis/basin_scan.py --n 128 160 192 224 256 --side 8 12 16 20 28 40 --seeds 3 --steps 60000 --workers 8 --out basin.csv
"""
import os
os.environ.setdefault('GS_FFT_WORKERS', '1')      # one FFT thread per run; we parallelise over runs
os.environ.setdefault('GS_FFT_THREADS', '1')
import argparse, csv, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation'))
MIN_PEAK_DISTANCE = 24    # as MIN_PEAK_DISTANCE in run_master_trajectory.py

def make_solver(n, fast):
    L = n * 2.5 / 1024
    if fast:
        from gray_scott_spectral_c import PseudoSpectralGSFast
        return PseudoSpectralGSFast(n=n, L=L)
    from gray_scott_spectral import PseudoSpectralGS
    return PseudoSpectralGS(n=n, L=L)

def seed_mask(n, side, shape):
    """Boolean mask of the seeded region, centred in the box.
    square  : side x side pixels.
    triangle: equilateral, side `side` pixels, apex up, centroid at the box centre."""
    m = np.zeros((n, n), dtype=bool); c = n // 2
    if shape == 'square':
        lo = c - side // 2; m[lo:lo + side, lo:lo + side] = True
    elif shape == 'triangle':
        h = side * np.sqrt(3) / 2.0
        top = c - h * 2.0 / 3.0                       # centroid is 2/3 of the height below the apex
        rows = np.arange(n)[:, None]; cols = np.arange(n)[None, :]
        t = rows - top                                 # 0 at the apex, h at the base
        m = (t >= 0) & (t <= h) & (np.abs(cols - c + 0.5) <= (side / 2.0) * (t / h))
    else:
        raise ValueError(shape)
    return m

def initial_state(n, side, seed, shape='square'):
    rng = np.random.default_rng(seed)
    u = np.ones((n, n)); v = np.zeros((n, n))
    m0 = seed_mask(n, side, shape)
    u[m0] = 0.5; v[m0] = 0.25
    u *= 1.0 + 0.01 * rng.uniform(-1, 1, size=(n, n))
    m = v > 0
    v[m] *= 1.0 + 0.01 * rng.uniform(-1, 1, size=(n, n))[m]
    return u, v

def tag(shape, n, side, seed):
    return f'final_n{n}_s{side}_seed{seed}.npz' if shape == 'square' else f'final_{shape}_n{n}_s{side}_seed{seed}.npz'

def get_uv(sim):
    return sim.get_uv() if hasattr(sim, 'get_uv') else (sim.u, sim.v)

def count_spots(v, proxy):
    proxy.v = v
    return proxy.segment_organisms(min_peak_distance=MIN_PEAK_DISTANCE)[1]

def one_run(job):
    n, side, seed, a = job
    from gray_scott import GrayScottSim
    t0 = time.time()
    sim = make_solver(n, a['fast'])
    t_base = 0
    if a['from_dir']:                     # continue a saved alive run (fields were saved as float32)
        d = np.load(os.path.join(a['from_dir'], tag(a['shape'], n, side, seed)))
        u0, v0 = d['u'].astype(np.float64), d['v'].astype(np.float64)
        t_base = int(d['time'])
    else:
        u0, v0 = initial_state(n, side, seed, a['shape'])
    sim.from_checkpoint_fields(u0, v0)
    proxy = GrayScottSim(n=n, L=n * 2.5 / 1024)
    rec = []; dead = 0; t_ext = None; step = 0
    while step < a['steps']:
        sim.run(a['every']); step += a['every']
        u, v = get_uv(sim)
        vmax = float(v.max())
        dead = dead + 1 if vmax < a['dead_vmax'] else 0
        sp = count_spots(v, proxy) if vmax >= a['dead_vmax'] else 0
        rec.append((step, sp, float(v.mean()), vmax))
        if a['verbose']:
            print(f'  n={n} side={side} seed={seed} t={t_base + step}: spots {sp}  mean v {v.mean():.4f}  max v {vmax:.3f}  [{time.time()-t0:.0f}s]', flush=True)
        if dead >= 2:
            t_ext = t_base + step - a['every']; break
    alive = t_ext is None
    sp_all = np.array([r[1] for r in rec]); q = max(1, len(sp_all) // 4)
    if alive and a['save_final']:
        os.makedirs(a['save_final'], exist_ok=True)
        np.savez(os.path.join(a['save_final'], tag(a['shape'], n, side, seed)),
                 u=get_uv(sim)[0].astype(np.float32), v=get_uv(sim)[1].astype(np.float32), time=t_base + step)
    return dict(shape=a['shape'], n=n, side=side, seed=seed, outcome='alive' if alive else 'extinct',
                t_extinct='' if alive else t_ext, steps_run=t_base + step, continued_from=t_base,
                spots_final=int(sp_all[-1]), spots_mean_last_quarter=round(float(sp_all[-q:].mean()), 2),
                spots_sd_last_quarter=round(float(sp_all[-q:].std()), 2), spots_peak=int(sp_all.max()),
                seconds=round(time.time() - t0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', default=[256])
    ap.add_argument('--side', type=int, nargs='+', default=[20], help='side of the seed in pixels (square side, or equilateral-triangle side)')
    ap.add_argument('--shape', choices=['square', 'triangle'], default='square')
    ap.add_argument('--seeds', type=int, default=3, help='number of seeds')
    ap.add_argument('--seed-start', dest='seed_start', type=int, default=0, help='first seed; seeds are seed_start .. seed_start+seeds-1')
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--every', type=int, default=500, help='record interval (steps)')
    ap.add_argument('--dead-vmax', dest='dead_vmax', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--fast', action='store_true', help='C solver (libgs_spectral.so), as production')
    ap.add_argument('--save-final', dest='save_final', default=None)
    ap.add_argument('--from-dir', dest='from_dir', default=None,
                    help='continue the alive runs saved in this directory (from --save-final); --steps is then the ADDITIONAL steps. '
                         'Use a different --save-final directory. Saved fields are float32, so the continuation is statistically, not bitwise, the same trajectory.')
    ap.add_argument('--out', default=None, help='CSV, appended to as runs finish')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args(); A = vars(a)
    jobs = [(n, s, sd, A) for n in a.n for s in a.side for sd in range(a.seed_start, a.seed_start + a.seeds) if s < n]
    if a.from_dir:
        have = lambda j: os.path.exists(os.path.join(a.from_dir, tag(a.shape, j[0], j[1], j[2])))
        jobs = [j for j in jobs if have(j)]
        if a.save_final and os.path.abspath(a.save_final) == os.path.abspath(a.from_dir):
            sys.exit('--save-final must differ from --from-dir')
    print(f'{len(jobs)} runs, {a.steps} steps each, {a.workers} worker(s)', flush=True)
    writer = fh = None
    if a.out:
        new = not os.path.exists(a.out); fh = open(a.out, 'a', newline='')
    it = Pool(a.workers).imap_unordered(one_run, jobs) if a.workers > 1 else map(one_run, jobs)
    for r in it:
        print(f"{r['shape'][:3]} n={r['n']:4d} side={r['side']:3d} seed={r['seed']}  {r['outcome']:7s}"
              f"{'  died at ' + str(r['t_extinct']) if r['outcome']=='extinct' else '  spots %d (mean %.1f sd %.1f, peak %d)' % (r['spots_final'], r['spots_mean_last_quarter'], r['spots_sd_last_quarter'], r['spots_peak'])}"
              f"  [{r['seconds']}s]", flush=True)
        if fh:
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(r)); 
                if new: writer.writeheader()
            writer.writerow(r); fh.flush()

if __name__ == '__main__':
    main()
