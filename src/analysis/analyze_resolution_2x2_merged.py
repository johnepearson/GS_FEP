"""
Resolution x interior-definition 2x2 table on the merged, min_distance-24-equivalent segmentation.

    python3 analyze_resolution_2x2_merged.py
    python3 analyze_resolution_2x2_merged.py --file256 resolution_2x2_merged_256_md6_progress.npz --file1024 master_trajectory_progress.npz

256 column : output of run_resolution_2x2_merged.py --resolution 256 (keys I_adapt,E_adapt,I_dist,E_dist,fu,fv).
1024 column: the master single-trajectory run (I_adapt,E_adapt = whole-mask erosion; I,E = distance-transform
             interior), which already contains both definitions on the same merged organisms.
PC1 flux regressor (paper convention) is the headline; joint OLS shown for reference.
"""
import argparse
import numpy as np
from partial_corr_utils import pc1_partial_corr


def joint(I, E, fu, fv):
    B = np.column_stack([np.ones(len(I)), fu, fv])
    res = lambda y: y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
    return np.corrcoef(res(I), res(E))[0, 1]


def report(title, d, pairs):
    fu, fv = d["fu"], d["fv"]
    print(f"=== {title} ===")
    res = {}
    for label, ik, ek in pairs:
        I, E = d[ik], d[ek]
        g = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
        r, n, expl = pc1_partial_corr(I[g], E[g], fu[g], fv[g])
        print(f"  {label:34s} N={n:,}  PC1 r={r:+.4f}   joint r={joint(I[g], E[g], fu[g], fv[g]):+.4f}   "
              f"(marginal {np.corrcoef(I[g], E[g])[0,1]:+.4f}; PC1 explains {expl:.1%})")
        res[ik] = (r, n)
    print()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file256", default="resolution_2x2_merged_256_md6_progress.npz")
    ap.add_argument("--file1024", default="master_trajectory_progress.npz")
    a = ap.parse_args()
    from texout import num, write_macros
    d = np.load(a.file256)
    r256 = report(f"256 x 256 ({a.file256}, {int(d['frames_done'])} frames)", d,
           [("whole-mask erosion (adaptive)", "I_adapt", "E_adapt"), ("distance-transform", "I_dist", "E_dist")])
    f256 = int(d["frames_done"])
    d = np.load(a.file1024)
    r1024 = report(f"1024 x 1024 ({a.file1024})", d,
           [("whole-mask erosion (adaptive)", "I_adapt", "E_adapt"), ("distance-transform", "I", "E")])
    A256, D256 = r256["I_adapt"], r256["I_dist"]
    A1024, D1024 = r1024["I_adapt"], r1024["I"]
    pct = lambda x, y: 100 * abs(abs(x) - abs(y)) / max(abs(x), abs(y))
    write_macros("results/res2x2.tex", "analyze_resolution_2x2_merged.py",
                 [("ResAdaptLow", num(A256[0], 4)), ("ResDistLow", num(D256[0], 4)),
                  ("ResAdaptHigh", num(A1024[0], 4)), ("ResDistHigh", num(D1024[0], 4)),
                  ("ResNLow", num(D256[1], 0)), ("ResNHigh", num(D1024[1], 0)), ("ResFramesLow", num(f256, 0)),
                  ("ResShiftAdapt", num(pct(A256[0], A1024[0]), 1)), ("ResShiftDist", num(pct(D256[0], D1024[0]), 1)),
                  ("ResShiftLow", num(pct(A256[0], D256[0]), 1)), ("ResShiftHigh", num(pct(A1024[0], D1024[0]), 1))])
