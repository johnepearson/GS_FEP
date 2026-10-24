# Figures: which script makes which

Run from the folder that holds the scripts and the data they read.  Data files come from the production scripts
(`run_master_trajectory.py`, `run_ensemble_merged.py`, `run_lifecycle.py`).

| Figure file | Script and command | Reads |
|---|---|---|
| v_movie.mp4 (optional movie of the v field) | `python3 make_v_movie.py --start-field master_fields/frame_0200.npz --every 5 --crop 128 --scale 8 --n-frames 1200 --out v_movie_1200.mp4` | `master_fields/frame_0200.npz` |
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

Numbers quoted in the text that come from small scripts rather than figures:

| Number | Script | Reads |
|---|---|---|
| marginal r, partial r and screened fraction for each ensemble (Table 1) | `python3 marginal_screened_ensembles.py` | `ensemble_merged_seed{1000,9000,20000}_worker*.npz` |
| single trajectory: population and partial r in thirds of the run (stationarity) | `python3 check_trajectory_stationarity.py` | `master_trajectory_progress.npz` |
| radius shares about 70% of its variance with the flux PC1 | `python3 radius_flux_share.py` | `master_trajectory_progress.npz` |
| CMI on one organism per frame, per realisation, or thinned subsamples | `python3 run_cmi_independent_frames.py --data ensemble:1000 --jobs 12` (and `--data master`) | ensemble or master data |

A regenerated figure can differ in detail from the published one (a different frame, a different random subsample);
compare the printed numbers with the text.
