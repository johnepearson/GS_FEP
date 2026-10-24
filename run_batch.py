"""
Rerun the analysis scripts in batches and write ONE log file per batch, so the printed numbers can be compared with the paper.

  python3 run_batch.py 1            # fast analyses (about an hour; plot_ensemble_summary.py alone takes about 10 minutes)
  python3 run_batch.py 2            # life-cycle and free-energy analyses (under an hour)
  python3 run_batch.py 3            # heavy nonlinear-CMI jobs, including the independent-frame CMI (1-2 hours or more; uses 12 cores)
  python3 run_batch.py 4            # band-sweep seeds 1 and 2, placebo-disk sweep from scratch, Tigramite cross-check, then the band-sweep and u-enriched figures (about 1 hour)
  python3 run_batch.py 5            # full-population collider CMI (3 seeds), split-half check, depth sensitivity, merge rounds (about 45 minutes)
  python3 run_batch.py 1 --list     # show what would run, run nothing
  python3 run_batch.py 1 --only 7 8 # run only steps 7 and 8 of batch 1 (numbers as shown by --list); log is batch1_partial.txt

Each command's full output, exit status and run time go to rerun_logs/batch<N>.txt.  A failing command does not stop the
batch.  Send that one file back.  The batch does NOT rerun the simulations themselves (master trajectory, ensembles,
life-cycle tracker, resolution runs); it re-analyses the data files you already have.
Cached results are moved aside (name + .before_rerun) so the CMI jobs recompute; outputs that a script overwrites are copied
first (name + .before_rerun).
"""
import os
import shutil
import subprocess
import sys
import time

PY = "python3"
# (command, files to MOVE aside first (caches), files to COPY first (outputs that get overwritten))
B1 = [
    ("analyze_master.py", [], []),
    ("marginal_screened_ensembles.py", [], []),
    ("check_trajectory_stationarity.py", [], []),
    ("radius_flux_share.py", [], []),
    ("diag_size_shape.py", [], []),
    ("diag_size_shape2.py", [], []),
    ("pc1_details.py", [], []),
    ("analyze_ensemble_pc1.py", [], []),
    ("analyze_ensemble_merged.py --n-workers 12 --seed-base 1000", [], []),
    ("analyze_ensemble_merged.py --n-workers 12 --seed-base 9000", [], []),
    ("run_block_bootstrap_master.py", [], []),
    ("cluster_bootstrap_analysis.py master_trajectory_progress.npz --cluster-col frame_id --n-boot 2000", [], []),
    ("plot_screening_curve.py", [], []),
    ("check_local_jacobian.py --frame 454 --labels-key raw_labels", [], []),
    ("check_local_jacobian.py --frame 454", [], []),
    ("analyze_3var_shells.py", [], []),
    ("master_to_legacy.py", [], ["master_decay_curve.npz"]),
    ("plot_decay_curve_pc1.py --data master_decay_curve.npz", [], []),
    ("plot_ensemble_summary.py", [], []),
    ("analyze_resolution_2x2_merged.py", [], []),
    ("plot_timescale_flux_merged.py --data remaining_checks_merged_md24_results.npz", [], []),
    ("analyze_count_decorrelation.py", [], []),
    ("level_set_from_master.py", [], ["levelset_states.npz"]),
]
B2 = [
    ("count_lifecycle_events.py", [], []),
    ("analyze_lifecycle.py", [], []),
    ("plot_lifespan.py", [], []),
    ("analyze_autocorr_lifecycle.py", [], []),
    ("analyze_lifecycle_age.py", [], ["lifecycle_age_analysis.npz"]),
    ("run_lineage_mixture_decomposition.py", [], ["lineage_mixture_decomposition_result.npz"]),
    ("build_template_from_master.py", [], ["fe_template_1024.npz", "master_profiles.npz"]),
    ("rebuild_fe_comparison_1024.py", [], []),
    ("rebuild_instance_level_check.py", [], []),
    ("rebuild_circularity_check_v2.py", [], []),
    ("analyze_surprisal.py", [], []),
    ("analyze_uv2_lifecycle.py", [], []),
    ("check_flux_massbalance.py", [], []),
    ("run_fe_size_control.py", [], []),
    ("rebuild_fig5_and_fe_model.py", [], []),
    ("run_shape_tail_test.py --n-frames 400", [], []),
    ("check_c_solver.py", [], []),
]
B3 = [
    ("analyze_master_followups.py", [], []),
    ("run_cmi_crosscheck.py", [], []),
    ("run_null_calibration.py --n-rep 48 --jobs 12", ["null_calibration_cache.pkl"], ["null_calibration.npz"]),
    ("run_u_enriched_cmi.py --jobs 12", ["u_enriched_cmi_cache.pkl"], []),
    ("run_band_sweep_cmi.py --jobs 12", ["run_band_sweep_cache.pkl"], []),
    ("run_random_disks.py --stride 3 --jobs 12", [], ["random_disks.npz"]),
    ("run_band_sweep_cmi.py --data random_disks.npz --tag _disks --only-all --jobs 12", [], []),
    ("run_levelset_cmi.py --jobs 12", [], ["levelset_cmi.npz"]),
    ("run_cmi_independent_frames.py --data ensemble:1000 --jobs 12", [], []),
    ("run_cmi_independent_frames.py --data master --jobs 12", [], []),
]
B4 = [
    ("run_band_sweep_cmi.py --jobs 12 --seed 1", [], []),
    ("run_band_sweep_cmi.py --jobs 12 --seed 2", [], []),
    ("run_band_sweep_cmi.py --data random_disks.npz --tag _disks --only-all --jobs 12", ["run_band_sweep_cache_disks.pkl"], []),
    ("run_cmi_crosscheck.py", [], []),
    ("plot_blanket_figures.py", [], ["band_sweep_cmi.png", "u_enriched_cmi.png"]),
]
B5 = [
    ("analyze_pooled_collider_cmi.py --seeds 0 1 2 --data master_trajectory_progress.npz", [], []),
    ("rebuild_splithalf_check.py", [], []),
    ("plot_depth_sensitivity_pc1.py --data sweep_merged_progress.npz", [], ["depth_sensitivity.png"]),
    ("merge_rounds_check_1024.py", [], []),
]
BATCHES = {"1": B1, "2": B2, "3": B3, "4": B4, "5": B5}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BATCHES:
        raise SystemExit(__doc__)
    n = sys.argv[1]
    steps = BATCHES[n]
    only = None
    if "--only" in sys.argv:
        only = [int(x) for x in sys.argv[sys.argv.index("--only") + 1:] if x.isdigit()]
        steps = [st for i, st in enumerate(steps, 1) if i in only]
    if "--list" in sys.argv:
        for i, (c, mv, cp) in enumerate(steps, 1):
            print(f"{i:2d}. {PY} {c}" + (f"   [moves aside: {', '.join(mv)}]" if mv else "") + (f"   [backs up: {', '.join(cp)}]" if cp else ""))
        return
    os.makedirs("rerun_logs", exist_ok=True)
    out_path = f"rerun_logs/batch{n}" + ("_partial" if only else "") + ".txt"
    summary = []
    t_all = time.time()
    with open(out_path, "w") as log:
        for i, (c, mv, cp) in enumerate(steps, 1):
            script = c.split()[0]
            log.write(f"\n{'=' * 100}\n[{i}/{len(steps)}] {PY} {c}\n{'=' * 100}\n")
            if not os.path.exists(script):
                log.write(f"SCRIPT NOT FOUND: {script}\n")
                summary.append((i, c, "NOT FOUND", 0)); print(f"[{i}/{len(steps)}] {c}: NOT FOUND", flush=True)
                continue
            for f in mv:
                if os.path.exists(f):
                    shutil.move(f, f + ".before_rerun"); log.write(f"(moved {f} aside)\n")
            for f in cp:
                if os.path.exists(f) and not os.path.exists(f + ".before_rerun"):
                    shutil.copy2(f, f + ".before_rerun"); log.write(f"(backed up {f})\n")
            log.flush()
            t0 = time.time()
            r = subprocess.run([PY] + c.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            dt = time.time() - t0
            txt = r.stdout
            if len(txt) > 60000:
                txt = txt[:30000] + "\n...[output shortened]...\n" + txt[-30000:]
            log.write(txt + f"\n[exit status {r.returncode}, {dt:.0f} s]\n")
            log.flush()
            summary.append((i, c, "ok" if r.returncode == 0 else f"EXIT {r.returncode}", dt))
            print(f"[{i}/{len(steps)}] {c}: {summary[-1][2]} ({dt:.0f} s)", flush=True)
        log.write(f"\n{'=' * 100}\nSUMMARY (total {time.time() - t_all:.0f} s)\n")
        for i, c, st, dt in summary:
            log.write(f"{i:2d}  {st:10s} {dt:6.0f} s   {c}\n")
    print(f"\nwrote {out_path}  ({time.time() - t_all:.0f} s in all).  Send me that file.")


if __name__ == "__main__":
    main()
