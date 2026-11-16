#!/usr/bin/env python3
"""Lyapunov spectrum and Kaplan-Yorke dimension of the (deterministic) Gray-Scott map.

Benettin/QR method with the EXACT tangent map of the exponential-Euler step used by
PseudoSpectralGS (production solver, F=0.018, k=0.055, dt=1), so no finite-difference perturbation size enters.  Small periodic box at
the production grid spacing (dx = L/n = 2.5/1024); the dimension is extensive, so the
result is reported per unit area and scaled to L = 2.5.

Run from build/ (writes nothing unless --out is given):
    python3 -B ../src/analysis/lyapunov_spectrum.py --from-frame master_fields/frame_0400.npz --m 60 --transient 3000 --steps 20000
    python3 -B ../src/analysis/lyapunov_spectrum.py --self-test
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation'))
from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim

from scipy.fft import rfft2, irfft2

def step_state(u, v, s):
    """One production step (exponential Euler, as in PseudoSpectralGS.step)."""
    Ru = -u * v * v + s.F * (1.0 - u)
    Rv = u * v * v - (s.F + s.k) * v
    un = irfft2(s.Eu * rfft2(u) + s.dt * s.phi1_u * rfft2(Ru), s=u.shape[-2:])
    vn = irfft2(s.Ev * rfft2(v) + s.dt * s.phi1_v * rfft2(Rv), s=u.shape[-2:])
    return un, vn

def step_tangent(u, v, du, dv, s):
    """Exact Jacobian of step_state applied to (du, dv); u, v (n,n); du, dv (m,n,n)."""
    a = v * v * du + 2 * u * v * dv          # d(u v^2)
    dRu = -a - s.F * du
    dRv = a - (s.F + s.k) * dv
    sh = u.shape[-2:]
    return (irfft2(s.Eu * rfft2(du) + s.dt * s.phi1_u * rfft2(dRu), s=sh),
            irfft2(s.Ev * rfft2(dv) + s.dt * s.phi1_v * rfft2(dRv), s=sh))

def ky_dimension(lam):
    lam = np.sort(lam)[::-1]; c = np.cumsum(lam)
    if lam[0] <= 0: return 0.0
    if c[-1] > 0: return float('nan')      # not enough exponents: dimension >= m
    j = int(np.nonzero(c < 0)[0][0])       # first index where partial sum < 0
    return j + c[j - 1] / abs(lam[j])

MIN_PEAK_DISTANCE = 24    # same as MIN_PEAK_DISTANCE in run_master_trajectory.py

def count_spots(v, n, L, _proxy=[]):
    """Number of organisms, using the same segmentation as the production analysis."""
    if not _proxy: _proxy.append(GrayScottSim(n=n, L=L))
    p = _proxy[0]; p.v = v
    return p.segment_organisms(min_peak_distance=MIN_PEAK_DISTANCE)[1]

def fmt_dky(lam):
    D = ky_dimension(lam)
    return f'>= {len(lam)} (raise --m)' if np.isnan(D) else f'{D:.1f}'

def orthonormalize(du, dv):
    m = du.shape[0]
    X = np.concatenate([du.reshape(m, -1), dv.reshape(m, -1)], axis=1).T   # (2n^2, m)
    Q, R = np.linalg.qr(X)
    sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1
    Q = Q * sgn; d = np.abs(np.diag(R))
    h = du[0].size
    return Q[:h].T.reshape(du.shape), Q[h:].T.reshape(dv.shape), d

def run(a):
    s = PseudoSpectralGS(n=a.n, L=a.n * 2.5 / 1024)
    t0 = time.time()
    if a.from_frame:                      # crop an n x n patch of an archived 1024^2 frame
        d = np.load(a.from_frame)
        r, c = a.corner
        u = d['u'][r:r + a.n, c:c + a.n].astype(np.float64).copy()
        v = d['v'][r:r + a.n, c:c + a.n].astype(np.float64).copy()
        assert u.shape == (a.n, a.n), 'patch runs off the frame'
        print(f'start from {a.from_frame} patch at {a.corner}; the cut edges relax during the transient')
    else:                                 # Pearson-style seed square + 1% noise; needs a long transient
        rng0 = np.random.default_rng(a.seed)
        u = np.ones((a.n, a.n)); v = np.zeros((a.n, a.n)); h = a.n // 2
        u[h-10:h+10, h-10:h+10] = 0.5; v[h-10:h+10, h-10:h+10] = 0.25
        u *= 1 + 0.01 * rng0.uniform(-1, 1, u.shape)
    for i in range(a.transient):
        u, v = step_state(u, v, s)
        if (i + 1) % 5000 == 0:
            print(f'  transient {i+1}/{a.transient}  spots {count_spots(v, a.n, a.n*2.5/1024)}  mean v {v.mean():.4f}  [{time.time()-t0:.0f}s]', flush=True)
    rng = np.random.default_rng(a.seed + 1)
    du = rng.standard_normal((a.m,) + u.shape); dv = rng.standard_normal((a.m,) + u.shape)
    du, dv, _ = orthonormalize(du, dv)
    logs = np.zeros(a.m); T = 0
    hist = []
    for i in range(a.steps):
        du, dv = step_tangent(u, v, du, dv, s)
        u, v = step_state(u, v, s)
        if (i + 1) % a.q == 0:
            du, dv, d = orthonormalize(du, dv)
            logs += np.log(d); T += a.q
            if (T % a.report) < a.q:
                lam = logs / (T * s.dt)
                hist.append((T, lam.copy(), v.mean()))
                nsp = count_spots(v, a.n, a.n*2.5/1024)
                print(f'  t={T}: lam1 {lam[0]:+.3e}  n_pos {(lam>0).sum()}  sum {lam.sum():+.3e}  '
                      f'D_KY {fmt_dky(lam)}  spots {nsp}  mean v {v.mean():.4f}  [{time.time()-t0:.0f}s]', flush=True)
    lam = logs / (T * s.dt)
    return lam, s

def self_test():
    s = PseudoSpectralGS(n=64, L=64 * 2.5 / 1024)
    rng = np.random.default_rng(0)
    u = 0.6 + 0.3 * rng.random(s.u.shape); v = 0.1 + 0.2 * rng.random(s.v.shape)
    for _ in range(50): u, v = step_state(u, v, s)
    du = rng.standard_normal((1,) + u.shape); dv = rng.standard_normal((1,) + u.shape)
    e = 1e-6
    tu, tv = step_tangent(u, v, du, dv, s)
    pu, pv = step_state(u + e * du[0], v + e * dv[0], s)
    qu, qv = step_state(u - e * du[0], v - e * dv[0], s)
    err = max(np.abs((pu - qu) / (2 * e) - tu[0]).max(), np.abs((pv - qv) / (2 * e) - tv[0]).max())
    print(f'tangent map vs central finite difference: max abs err {err:.2e}'); assert err < 1e-6
    # KY check on a known spectrum
    print('KY test (1, -0.5, -1 -> 2.0 + 0.5/1... ):', ky_dimension(np.array([1.0, 0.0, -0.5, -1.0])))
    print('self-test ok')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--m', type=int, default=60, help='number of tangent vectors')
    ap.add_argument('--transient', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--q', type=int, default=20, help='QR every q steps')
    ap.add_argument('--report', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--from-frame', default=None, help='e.g. master_fields/frame_0400.npz (recommended)')
    ap.add_argument('--corner', type=int, nargs=2, default=[0, 0], help='row col of the patch')
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test: return self_test()
    t_start = time.time()
    lam, s = run(a)
    L = a.n * 2.5 / 1024; D = ky_dimension(lam)
    print(f'\nbox L = {L:.4f} (n = {a.n}), m = {a.m} exponents, run length {a.steps} steps')
    print('exponents (first 15):', np.array2string(lam[:15], formatter={'float_kind': lambda x: f'{x:+.2e}'}))
    print(f'positive exponents: {(lam>0).sum()}   sum of all {a.m}: {lam.sum():+.3e}')
    if np.isnan(D): print('partial sum still positive after m exponents: D_KY > m. Increase --m.')
    else: print(f'Kaplan-Yorke dimension in this box: {D:.1f}   per unit area {D/L**2:.0f}   scaled to L=2.5: {D*(2.5/L)**2:.0f}')
    print(f'total elapsed {time.time()-t_start:.0f} s')
    if a.out: np.savez(a.out, lam=lam, n=a.n, m=a.m, steps=a.steps); print('wrote', a.out)

if __name__ == '__main__':
    main()
