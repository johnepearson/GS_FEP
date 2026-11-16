#!/usr/bin/env python3
"""Power spectrum of the archived u,v fields: how many Fourier modes carry the variance?

Run from build/:   python3 -B ../src/analysis/analyze_power_spectrum.py
Reads master_fields/frame_*.npz one at a time (u, v float32), removes the spatial mean,
averages |FFT|^2 over frames.  Because the dynamics are translation invariant, the PCA
modes of the fields are the Fourier modes, so this power spectrum IS the PCA spectrum.
Reports, for u and v: number of Fourier coefficients (real degrees of freedom, out of n^2)
needed to hold 90/99/99.9/99.99% of the fluctuation variance, and the matching wavenumber.
This is an upper-bound style count, not the attractor dimension.
Writes nothing unless --out is given (a .npz with the radial spectra).
"""
import argparse, glob, os
import numpy as np

def spectra(files, stride):
    acc = {'u': 0.0, 'v': 0.0}; nfr = 0; L = None; n = None
    for f in files[::stride]:
        d = np.load(f)
        for key in ('u', 'v'):
            a = d[key].astype(np.float64)
            n = a.shape[0]
            a = a - a.mean()
            acc[key] = acc[key] + np.abs(np.fft.fft2(a)) ** 2 / a.size ** 2
        nfr += 1
    return {k: v / nfr for k, v in acc.items()}, nfr, n

def summarize(P, n, L):
    kx = np.fft.fftfreq(n, d=1.0 / n)            # integer wavenumber index
    kr = np.hypot(*np.meshgrid(kx, kx, indexing='ij'))
    order = np.argsort(P.ravel())[::-1]
    c = np.cumsum(P.ravel()[order]); tot = c[-1]
    res = {}
    for frac in (0.90, 0.99, 0.999, 0.9999):
        m = int(np.searchsorted(c, frac * tot)) + 1
        kmax = kr.ravel()[order[:m]].max()
        res[frac] = (m, kmax, 2 * np.pi * kmax / L)
    return res, kr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='master_fields')
    ap.add_argument('--stride', type=int, default=1, help='use every stride-th frame')
    ap.add_argument('--L', type=float, default=2.5, help='box side, physical units')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, 'frame_*.npz')))
    if not files: raise SystemExit(f'no frames in {a.dir}')
    P, nfr, n = spectra(files, a.stride)
    print(f'{nfr} frames, grid {n}x{n} = {n*n} real degrees of freedom per field, L = {a.L}')
    out = {}
    for key in ('u', 'v'):
        res, kr = summarize(P[key], n, a.L)
        print(f'\nfield {key}: Fourier coefficients needed (real dof) for a fraction of the variance')
        print(f'  {"fraction":>9} {"modes":>9} {"% of n^2":>9} {"k_index":>8} {"k physical":>16}')
        for frac, (m, kmax, kphys) in res.items():
            print(f'  {frac:9.4f} {m:9d} {100*m/n**2:9.3f} {kmax:8.1f} {kphys:16.2f}')
        kb = np.rint(kr).astype(int).ravel()
        rad = np.bincount(kb, P[key].ravel()) / np.maximum(np.bincount(kb), 1)
        out[key] = rad
        j = np.nonzero(rad > 0)[0]
        print(f'  radial power: k=1 {rad[1]:.3e}, k=10 {rad[10]:.3e}, k=100 {rad[100]:.3e}, k={n//2} {rad[n//2]:.3e}')
    if a.out: np.savez(a.out, **out); print('wrote', a.out)

if __name__ == '__main__':
    main()
