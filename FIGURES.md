# Figures: which script makes which

Run from the folder that holds the scripts and the data they read.  Data files come from the production scripts
(`run_master_trajectory.py`, `run_ensemble_merged.py`, `run_lifecycle.py`).

| Figure file | Script and command | Reads |
|---|---|---|
| spots_snapshot.png | `python3 plot_snapshot.py` | `master_fields/` (or `master_trajectory_progress.npz`) |
| schematic_combined.png | none kept: field snapshot with the interior, boundary, exterior-ring and far-exterior outlines overlaid, plus a hand-drawn schematic | |
| ensemble_1024.png | `python3 plot_ensemble_summary.py` | `ensemble_merged_seed1000_worker*.npz` |
| decay_curve_1024.png | `python3 master_to_legacy.py` then `python3 plot_decay_curve_pc1.py --data master_decay_curve.npz` | `master_trajectory_progress.npz` |
| screening_curve.png | `python3 plot_screening_curve.py` | `master_trajectory_progress.npz` |
| fe_four_curves_1024.png | `python3 analyze_uv2_lifecycle.py` | `lifecycle_progress.pkl` (`run_lifecycle.py`), `fe_template_1024.npz` (`build_template_from_master.py`) |
| lyapunov_comparison_fixed.png | `python3 run_divergence_rate.py --fast` (about 45 minutes, most of it advancing to step 469,000) | `checkpoint_1024_settled.npz` |
| split_duration_histogram.png | exploratory pass on an earlier 256 x 256 version of the lifecycle tracker with no confirmation window; the code was not kept (see the supplement) | |
| band_sweep_cmi.png, u_enriched_cmi.png | `python3 plot_blanket_figures.py` | `run_band_sweep_seed{0,1,2}.npz` (`run_band_sweep_cmi.py --seed 0/1/2`), `u_enriched_cmi_ring.npz` (`run_u_enriched_cmi.py`) |
| depth_sensitivity.png | `sweep_from_master.py`, `plot_depth_sensitivity_pc1.py` | `master_trajectory_progress.npz` |
| fe_surprisal_1024.png | `analyze_surprisal.py` | |
| merge_before.png, merge_after.png | `make_merge_figure.py` | |
| lifespan_1024.png | `plot_lifespan.py` | `lifecycle_progress.pkl` |
| memory_comparison_1024.png | `analyze_3var_shells.py` | `shell_data_3var_*.npz` (`run_3var_shell_data.py`) |

A regenerated figure can differ in detail from the published one (a different frame, a different random subsample);
compare the printed numbers with the text.
