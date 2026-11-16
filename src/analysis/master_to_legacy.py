"""
Convert the master single-trajectory dataset into the file formats the existing analysis scripts read,
so they run unchanged on the new data (pass the new file with --data / the script's file argument).

    python master_to_legacy.py                           # master_trajectory_progress.npz

Writes
  master_gap_split_results.npz   keys I, E, fu, fv, radius, merge_size, frame_id
        -> analyze_merged_gap_split.py (edit its DATA), replace_table2_merge_split.py --data ...,
           run_nn_null_scale_check_merged.py / run_nn_null_cmi_pc1.py --single-traj-file ...
  master_decay_curve.npz         keys I, fu, fv, profiles (mean v in exterior shells), radii
        -> plot_decay_curve_merged.py / plot_decay_curve_pc1.py --data ...
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="master_trajectory_progress.npz")
args = ap.parse_args()
d = np.load(args.data)
np.savez("master_gap_split_results.npz", I=d["I"], E=d["E"], fu=d["fu"], fv=d["fv"],
         radius=d["radius_px"], merge_size=d["merge_size"].astype(int), frame_id=d["frame_id"])
np.savez("master_decay_curve.npz", I=d["I"], E=d["E"], fu=d["fu"], fv=d["fv"], profiles=d["v_shells"],
         radii=d["radii"], frame_id=d["frame_id"])
print(f"wrote master_gap_split_results.npz and master_decay_curve.npz  ({len(d['I']):,} organism-instances, "
      f"min_peak_distance={int(d['min_peak_distance'])})")
