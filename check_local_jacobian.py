"""
Local kinetics at organism boundaries and cores (paper: relaxation time tau; ESM, "Linear Stability of the Local Kinetics").

Reaction Jacobian of the Gray-Scott kinetics,   J(u,v) = [[-F - v^2, -2uv], [v^2, -F - k + 2uv]],   F = 0.018, k = 0.055.
Reads one archived frame (master_fields/frame_XXXX.npz: u, v, merged labels), takes for every organism
  boundary = the organism's own pixels that touch another label or the background (a one-pixel ring just inside its contour),
  core     = its pixels at least half the equivalent radius from its edge,
and prints, for the organism-mean (u, v) of each region and also pixel by pixel:
  mean u, v; the fraction satisfying the diagonal criterion 2uv > F + k; the fraction with a positive dominant eigenvalue
  (largest real part); the median dominant eigenvalue; the fraction of cores whose eigenvalues are a complex pair with
  negative real part.  Also prints 1/(F+k).
Supplement values to compare: 242 organisms; mean boundary u = 0.51, v = 0.16; mean core u = 0.31, v = 0.28; every boundary
has a positive dominant eigenvalue (median 0.036); about 51% of cores do.

Usage:  python3 check_local_jacobian.py --find-pop 242          # lists archived frames with that organism count
        python3 check_local_jacobian.py --frame 250
"""
import argparse
import glob
import os
import numpy as np
from scipy import ndimage

F, K = 0.018, 0.055


def dominant(u, v):
    u, v = np.atleast_1d(u), np.atleast_1d(v)
    out_re = np.empty(len(u)); cplx = np.zeros(len(u), bool)
    for i, (uu, vv) in enumerate(zip(u, v)):
        J = np.array([[-F - vv * vv, -2 * uu * vv], [vv * vv, -F - K + 2 * uu * vv]])
        w = np.linalg.eigvals(J)
        out_re[i] = w.real.max(); cplx[i] = np.abs(w.imag).max() > 1e-12
    return out_re, cplx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields-dir", default="master_fields")
    ap.add_argument("--frame", type=int, default=250)
    ap.add_argument("--find-pop", type=int, default=None)
    ap.add_argument("--labels-key", default="labels", choices=["labels", "raw_labels"],
                    help="merged organisms (default) or the raw watershed regions")
    a = ap.parse_args()
    print(f"1/(F+k) = {1/(F+K):.2f}")
    if a.find_pop is not None:
        paths = sorted(glob.glob(os.path.join(a.fields_dir, "frame_*.npz")))
        print(f"scanning {len(paths)} archived frames in {a.fields_dir}/")
        if not paths:
            raise SystemExit("no frames found: run this from the folder that holds master_fields/ (or pass --fields-dir)")
        counts = []
        for p in paths:
            lab = np.load(p)[a.labels_key]
            counts.append((len(np.unique(lab[lab > 0])), p))
        c = np.array([x[0] for x in counts])
        print(f"organisms per frame: min {c.min()}, median {int(np.median(c))}, max {c.max()}")
        near = sorted(counts, key=lambda x: (abs(x[0] - a.find_pop), x[1]))[:10]
        print(f"the 10 frames closest to {a.find_pop} organisms:")
        for n, p in near:
            print(f"  {n:4d} organisms  {p}")
        return
    d = np.load(os.path.join(a.fields_dir, f"frame_{a.frame:04d}.npz"))
    u, v, lab = d["u"].astype(float), d["v"].astype(float), d[a.labels_key].astype(int)
    N = lab.shape[0]
    ids = [k for k in np.unique(lab) if k > 0]
    print(f"frame {a.frame}: {len(ids)} organisms")
    nb = ((lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, -1, 0)) | (lab != np.roll(lab, 1, 1)) | (lab != np.roll(lab, -1, 1)))
    B_uv, C_uv, Bpix, Cpix = [], [], [], []
    for k in ids:
        m = lab == k
        ys, xs = np.nonzero(m)
        ang = lambda c: np.angle(np.exp(2j * np.pi * c / N).mean()) * N / (2 * np.pi)
        cy, cx = ang(ys), ang(xs)
        sy, sx = int(round(N // 2 - cy)), int(round(N // 2 - cx))
        mc = np.roll(np.roll(m, sy, 0), sx, 1)[N // 2 - 160:N // 2 + 160, N // 2 - 160:N // 2 + 160]
        edt = ndimage.distance_transform_edt(mc)
        rad = np.sqrt(mc.sum() / np.pi)
        core_c = mc & (edt >= 0.5 * rad)
        shift = lambda x: np.roll(np.roll(x, sy, 0), sx, 1)[N // 2 - 160:N // 2 + 160, N // 2 - 160:N // 2 + 160]
        uc, vc, bc = shift(u), shift(v), shift(nb) & mc
        if bc.sum() == 0 or core_c.sum() == 0:
            continue
        B_uv.append((uc[bc].mean(), vc[bc].mean())); C_uv.append((uc[core_c].mean(), vc[core_c].mean()))
        Bpix.append((uc[bc], vc[bc])); Cpix.append((uc[core_c], vc[core_c]))
    B_uv, C_uv = np.array(B_uv), np.array(C_uv)
    print(f"used {len(B_uv)} organisms with a non-empty boundary and core")
    print(f"mean boundary u = {B_uv[:,0].mean():.2f}, v = {B_uv[:,1].mean():.2f};  mean core u = {C_uv[:,0].mean():.2f}, v = {C_uv[:,1].mean():.2f}")
    for name, uv in (("boundary", B_uv), ("core", C_uv)):
        diag = (2 * uv[:, 0] * uv[:, 1] > F + K).mean()
        re, cx_ = dominant(uv[:, 0], uv[:, 1])
        print(f"  {name:8s} (organism means): diagonal criterion met {100*diag:.1f}%;  positive dominant eigenvalue {100*(re>0).mean():.1f}%;  "
              f"median dominant eigenvalue {np.median(re):+.4f};  complex pair with negative real part {100*((re<0)&cx_).mean():.1f}%")
    for name, pix in (("boundary", Bpix), ("core", Cpix)):
        uu = np.concatenate([p[0] for p in pix]); vv = np.concatenate([p[1] for p in pix])
        sel = np.random.default_rng(0).choice(len(uu), min(20000, len(uu)), replace=False)
        re, _ = dominant(uu[sel], vv[sel])
        print(f"  {name:8s} (pixel by pixel, 20,000 sampled): diagonal criterion {100*(2*uu[sel]*vv[sel]>F+K).mean():.1f}%;  "
              f"positive dominant eigenvalue {100*(re>0).mean():.1f}%")


if __name__ == "__main__":
    main()
