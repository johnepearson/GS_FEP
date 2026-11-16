"""Is the single trajectory stationary? Compares thirds (and the first 50 frames) of master_trajectory_progress.npz:
population per frame, marginal r(I,E), PC1 partial r, screened fraction.  Takes seconds.
Usage (from the directory holding the file):  python3 check_trajectory_stationarity.py"""
import numpy as np
from partial_corr_utils import pc1_partial_corr

d = np.load("master_trajectory_progress.npz")
fid, I, E, fu, fv = d["frame_id"], d["I"], d["E"], d["fu"], d["fv"]
ft, fp = d["f_time"], d["f_pop"]
nfr = len(ft)
print(f"{nfr} frames, t = {ft[0]:.0f} .. {ft[-1]:.0f}, {len(I):,} organism-instances")

def line(name, frames):
    m = np.isin(fid, frames)
    rm = np.corrcoef(I[m], E[m])[0, 1]
    rp = pc1_partial_corr(I[m], E[m], fu[m], fv[m]); rp = rp[0] if isinstance(rp, tuple) else rp
    pop = fp[np.isin(np.arange(len(fp)), frames - fid.min())] if False else None
    print(f"{name:<18} N={m.sum():>7,}  marginal r={rm:.4f}  PC1 partial r={rp:.4f}  screened={100*(1-rp**2/rm**2):.1f}%")

frames = np.arange(fid.min(), fid.max() + 1)
idx = np.arange(len(frames))
for name, sel in [("first 50 frames", idx[:50]), ("first third", idx[:nfr//3]), ("middle third", idx[nfr//3:2*nfr//3]),
                  ("last third", idx[2*nfr//3:]), ("all", idx)]:
    line(name, frames[sel])
    print(f"{'':<18} organisms per frame: mean {fp[sel].mean():.1f}, SD {fp[sel].std():.1f}")
