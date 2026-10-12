"""Radial v-profile template T(r) built from the MASTER trajectory's merged organisms (marker spacing 24).

Replaces the old fe_template_1024.npz (built from raw watershed regions, which at spacing 24 include fragments) and
the old ensemble_profiles_worker*.npz disjoint dataset (spacing 32, unmerged). Profiles are taken at the stored merged
centroids (master_trajectory_progress.npz cy/cx) from the archived v fields (master_fields/frame_XXXX.npz).
The master trajectory is a separate simulation from run_lifecycle.py's, so a template built here is independent of the
lifecycle trajectories it is evaluated on.

Writes:  master_profiles.npz   (frame_id, profiles [n_org x N_BINS])   -- the 'disjoint dataset' for the circularity check
         fe_template_1024.npz  (template, template_std, N, max_r, n_bins) -- what rebuild_fe_comparison_1024 / instance check load
Usage:   python3 build_template_from_master.py
"""
import os
import numpy as np
from run_lifecycle import organism_radial_profile, MAX_R, N_BINS

d = np.load("master_trajectory_progress.npz")
fid = d["frame_id"]
frames = np.unique(fid)
rows, profs = [], []
for k, f in enumerate(frames):
    path = os.path.join("master_fields", f"frame_{int(f):04d}.npz")
    if not os.path.exists(path):
        continue
    v = np.load(path)["v"].astype(np.float64)
    sel = np.where(fid == f)[0]
    for i in sel:
        profs.append(organism_radial_profile(v, v.shape[0], d["cy"][i], d["cx"][i]))
        rows.append(f)
    if (k + 1) % 50 == 0:
        print(f"{k+1}/{len(frames)} frames, {len(profs)} profiles", flush=True)
profs = np.array(profs)
frame_id = np.array(rows)
np.savez("master_profiles.npz", frame_id=frame_id, profiles=profs)
if os.path.exists("fe_template_1024.npz") and not os.path.exists("fe_template_1024.npz.before_master"):
    os.rename("fe_template_1024.npz", "fe_template_1024.npz.before_master")
template = np.nanmean(profs, axis=0)
np.savez("fe_template_1024.npz", template=template, template_std=np.nanstd(profs, axis=0),
         N=len(profs), max_r=MAX_R, n_bins=N_BINS)
print(f"done: N={len(profs)} merged-organism profiles from {len(frames)} frames; template saved to fe_template_1024.npz")
print("template T(r):", np.round(template, 4))
