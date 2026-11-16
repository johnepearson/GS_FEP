# =====================================================================================================================
# stages.mk -- WHAT TO RUN.  Included by the Makefile; the rules that use this data are in the Makefile.
#   Section 2:  the lists of stage names (the order here is the order they run in)
#   Sections 3-4:  for each stage, the script (SIMSCRIPT.NAME) and its arguments (SIMARGS.NAME)
#   Section 6:  the analyses that write result macros (ANALYSES, SCRIPT.NAME, ARGS.NAME)
# To add a stage: put its name in a list, and give it a SIMSCRIPT and a SIMARGS line.
# =====================================================================================================================


# ==============================================================================================================
# 2. THE LIST OF STAGES (the order here is the order they run in)
# ==============================================================================================================


# SIMS  = the simulations (they make the data).      POSTS = analyses of that data, run after all of the SIMS.
# Each stage NAME is described by two variables further down:
#     SIMSCRIPT.NAME  the Python script to run         SIMARGS.NAME  the arguments given to it
# SIMS and POSTS only hold names.  The link to SIMSCRIPT.NAME is made by writing out the name (see section 5).
# In an argument, @Name stands for the parameter Name in initialize.tex (its value is put in by trace_run.py).
# A stage is finished when build/sim/NAME.done exists.

SIMS := \
    settle ensOne ensTwo ensThree master lifecycle \
    remaining shapetail shapestab mindist confound liferaw \
    res256 res1024 countdecorr threevar0 threevar1 centroid \
    vrate posgap touching density dealias0 dealias1 \
    divergence

# POSTS: first the analyses of the simulation data, then the batch of plotting and macro-writing stages (BATCH).

POSTS := \
    paircancel confanalysis fluxredundancy regimeboot cmiscale dealiascmp

BATCH := \
    spotsnapshot mergefig anmaster anens1 anens2 anens3 \
    stationarity radiusflux diagsize diagsize2 pc1details anensp1 \
    blockboot clusterboot plotscreen jacraw jacmerged shells3var \
    legacy plotdecay plotens res2x2 plottimescale countdecorran \
    levelset lifeevents anlife plotlife lifeautocorr lifeage \
    lineagemix fetemplate fecomp instancelevel circularity surprisal \
    uv2life massbalance fesize fig5fe csolver followups \
    cmicross nullcal uenriched uenrichedfar bandsweep randomdisks \
    banddisks levelsetcmi indepens indepmaster cmiscaledemo bandsweep1 \
    bandsweep2 blanketfigs pooledcmi splithalf depthsweep plotdepth \
    mergerounds plotschematic

POSTS += $(BATCH)


# ==============================================================================================================
# 3. WHAT EACH STAGE RUNS
# ==============================================================================================================


# ---- 3a. The starting field ----

SIMSCRIPT.settle := settle.py
SIMARGS.settle   := checkpoint_1024_settled.npz --fast

# ---- 3b. The three ensembles (drive_ensemble.py reads Seed/Runs/Burnin/Spacing One, Two, Three from initialize.tex) ----

SIMSCRIPT.ensOne   := drive_ensemble.py
SIMARGS.ensOne     := One

SIMSCRIPT.ensTwo   := drive_ensemble.py
SIMARGS.ensTwo     := Two

SIMSCRIPT.ensThree := drive_ensemble.py
SIMARGS.ensThree   := Three

# ---- 3c. The long single run and the lifecycle tracker ----

SIMSCRIPT.master    := run_master_trajectory.py
SIMARGS.master      := --fast

SIMSCRIPT.lifecycle := run_lifecycle.py
SIMARGS.lifecycle   := --fast

# ---- 3d. Further simulations (frame counts etc. come from initialize.tex) ----

SIMSCRIPT.remaining   := run_remaining_checks_merged.py
SIMARGS.remaining     := --n-frames @FramesRemaining

SIMSCRIPT.shapetail   := run_shape_tail_test.py
SIMARGS.shapetail     := --n-frames @FramesShapeTail

SIMSCRIPT.shapestab   := run_shape_stability.py
SIMARGS.shapestab     := --n-frames @FramesShapeStab

SIMSCRIPT.mindist     := run_min_distance_sweep.py
SIMARGS.mindist       := --n-frames @FramesMinDist

SIMSCRIPT.confound    := run_confound_checks.py
SIMARGS.confound      := --n-frames @FramesConfound --fast

SIMSCRIPT.liferaw     := run_lifecycle.py
SIMARGS.liferaw       := --n-steps @LifeSteps --fast --raw

SIMSCRIPT.res256      := run_resolution_2x2_merged.py
SIMARGS.res256        := --resolution 256 --n-frames @FramesResLow --fast

SIMSCRIPT.res1024     := run_resolution_2x2_merged.py
SIMARGS.res1024       := --resolution 1024 --n-frames @FramesResHigh --fast --resume-checkpoint checkpoint_1024_settled.npz

SIMSCRIPT.countdecorr := run_count_decorrelation.py
SIMARGS.countdecorr   := --n-real @CountReal --steps @CountSteps

SIMSCRIPT.threevar0   := run_3var_shell_data.py
SIMARGS.threevar0     := --dw 0.0 --target-n @ThreeVarTarget --burn-in @ThreeVarBurnin

SIMSCRIPT.threevar1   := run_3var_shell_data.py
SIMARGS.threevar1     := --dw @Dvval --target-n @ThreeVarTarget --burn-in @ThreeVarBurnin

SIMSCRIPT.centroid    := run_centroid_radial_dist.py
SIMARGS.centroid      := --n-frames @FramesCentroid

SIMSCRIPT.vrate       := run_v_rate_dist.py
SIMARGS.vrate         := --n-frames @FramesVRate

SIMSCRIPT.posgap      := run_positive_gap_check_v2.py
SIMARGS.posgap        := --n-frames @FramesPosGap

SIMSCRIPT.touching    := run_touching_pairs.py
SIMARGS.touching      := --n-frames @FramesTouching

SIMSCRIPT.density     := run_density_confound.py
SIMARGS.density       := --n-frames @FramesDensity

SIMSCRIPT.dealias0    := run_dealiasing_check.py
SIMARGS.dealias0      := --dealias 0 --n-frames @FramesDealias

SIMSCRIPT.dealias1    := run_dealiasing_check.py
SIMARGS.dealias1      := --dealias 1 --n-frames @FramesDealias

SIMSCRIPT.divergence  := run_divergence_rate.py
SIMARGS.divergence    := --fast --ref-times @DivRefA @DivRefB @DivRefC --branch-steps @DivBranch

# ---- 3e. Analyses of that simulation data ----

SIMSCRIPT.paircancel     := analyze_pair_cancellation.py
SIMARGS.paircancel       := touching_pairs_progress.npz

SIMSCRIPT.confanalysis   := run_confound_analysis.py
SIMARGS.confanalysis     :=

SIMSCRIPT.fluxredundancy := run_flux_redundancy_check.py
SIMARGS.fluxredundancy   :=

SIMSCRIPT.regimeboot     := run_regime_difference_bootstrap.py
SIMARGS.regimeboot       := --data positive_gap_progress_v2.npz

SIMSCRIPT.cmiscale       := run_cmi_scale_check_regimes.py
SIMARGS.cmiscale         :=

SIMSCRIPT.dealiascmp     := compare_dealiasing_check.py
SIMARGS.dealiascmp       := --nodealias dealiasing_check_nodealias_progress.npz --dealias dealiasing_check_dealias_progress.npz


# ==============================================================================================================
# 4. THE BATCH: plotting and macro-writing stages (they read the data above)
# ==============================================================================================================


# ---- 4. figures ----

SIMSCRIPT.spotsnapshot := plot_snapshot.py
SIMARGS.spotsnapshot   := --frame @SnapshotFrame

SIMSCRIPT.mergefig     := make_merge_figure.py
SIMARGS.mergefig       := --out .


# ---- 4. master trajectory and ensembles ----

SIMSCRIPT.anmaster     := analyze_master.py
SIMARGS.anmaster       := --n-boot @BootReps

SIMSCRIPT.anens1       := analyze_ensemble_merged.py
SIMARGS.anens1         := --n-workers 0 --seed-base @SeedOne --n-boot @BootReps

SIMSCRIPT.anens2       := analyze_ensemble_merged.py
SIMARGS.anens2         := --n-workers 0 --seed-base @SeedTwo --n-boot @BootReps

SIMSCRIPT.anens3       := analyze_ensemble_merged.py
SIMARGS.anens3         := --n-workers 0 --seed-base @SeedThree --n-boot @BootReps

SIMSCRIPT.anensp1      := analyze_ensemble_pc1.py
SIMARGS.anensp1        :=

SIMSCRIPT.stationarity := check_trajectory_stationarity.py
SIMARGS.stationarity   :=

SIMSCRIPT.radiusflux   := radius_flux_share.py
SIMARGS.radiusflux     :=

SIMSCRIPT.diagsize     := diag_size_shape.py
SIMARGS.diagsize       :=

SIMSCRIPT.diagsize2    := diag_size_shape2.py
SIMARGS.diagsize2      :=

SIMSCRIPT.pc1details   := pc1_details.py
SIMARGS.pc1details     :=

SIMSCRIPT.blockboot    := run_block_bootstrap_master.py
SIMARGS.blockboot      :=

SIMSCRIPT.clusterboot  := cluster_bootstrap_analysis.py
SIMARGS.clusterboot    := master_trajectory_progress.npz --cluster-col frame_id --n-boot @BootReps

SIMSCRIPT.plotscreen   := plot_screening_curve.py
SIMARGS.plotscreen     :=


# ---- 4. timescale, flux, resolution ----

SIMSCRIPT.jacraw        := check_local_jacobian.py
SIMARGS.jacraw          := --frame @JacobianFrame --labels-key raw_labels

SIMSCRIPT.jacmerged     := check_local_jacobian.py
SIMARGS.jacmerged       := --frame @JacobianFrame

SIMSCRIPT.shells3var    := analyze_3var_shells.py
SIMARGS.shells3var      :=

SIMSCRIPT.legacy        := master_to_legacy.py
SIMARGS.legacy          :=

SIMSCRIPT.plotdecay     := plot_decay_curve_pc1.py
SIMARGS.plotdecay       := --data master_decay_curve.npz

SIMSCRIPT.plotens       := plot_ensemble_summary.py
SIMARGS.plotens         := --seed-base @SeedOne --n-cmi @CmiNSub --n-null @CmiTrials --pop-draws @CmiSubsamples --n-subsamples @CmiNullSubsamples

SIMSCRIPT.res2x2        := analyze_resolution_2x2_merged.py
SIMARGS.res2x2          :=

SIMSCRIPT.plottimescale := plot_timescale_flux_merged.py
SIMARGS.plottimescale   := --data remaining_checks_merged_md24_results.npz

SIMSCRIPT.countdecorran := analyze_count_decorrelation.py
SIMARGS.countdecorran   :=


# ---- 4. lifecycle and lineage ----

SIMSCRIPT.levelset     := level_set_from_master.py
SIMARGS.levelset       :=

SIMSCRIPT.lifeevents   := count_lifecycle_events.py
SIMARGS.lifeevents     :=

SIMSCRIPT.anlife       := analyze_lifecycle.py
SIMARGS.anlife         :=

SIMSCRIPT.plotlife     := plot_lifespan.py
SIMARGS.plotlife       :=

# Figure 2 of the paper (left: field snapshot from src/TeX/figures/schematic_left.png; right: the blanket diagram with
# arrows in both directions, drawn by the script). Writes figure2_blanket.png in build/. For plain lines use --lines.
SIMSCRIPT.plotschematic := make_figure2.py
SIMARGS.plotschematic   :=

SIMSCRIPT.lifeautocorr := analyze_autocorr_lifecycle.py
SIMARGS.lifeautocorr   :=

SIMSCRIPT.lifeage      := analyze_lifecycle_age.py
SIMARGS.lifeage        :=

SIMSCRIPT.lineagemix   := run_lineage_mixture_decomposition.py
SIMARGS.lineagemix     :=


# ---- 4. free-energy comparison ----

SIMSCRIPT.fetemplate    := build_template_from_master.py
SIMARGS.fetemplate      :=

SIMSCRIPT.fecomp        := rebuild_fe_comparison_1024.py
SIMARGS.fecomp          :=

SIMSCRIPT.instancelevel := rebuild_instance_level_check.py
SIMARGS.instancelevel   :=

SIMSCRIPT.circularity   := rebuild_circularity_check_v2.py
SIMARGS.circularity     :=

SIMSCRIPT.surprisal     := analyze_surprisal.py
SIMARGS.surprisal       :=

SIMSCRIPT.uv2life       := analyze_uv2_lifecycle.py
SIMARGS.uv2life         :=

SIMSCRIPT.massbalance   := check_flux_massbalance.py
SIMARGS.massbalance     :=

SIMSCRIPT.fesize        := run_fe_size_control.py
SIMARGS.fesize          :=

SIMSCRIPT.fig5fe        := rebuild_fig5_and_fe_model.py
SIMARGS.fig5fe          :=


# ---- 4. solver and estimator checks ----

SIMSCRIPT.csolver      := check_c_solver.py
SIMARGS.csolver        :=

SIMSCRIPT.followups    := analyze_master_followups.py
SIMARGS.followups      :=

SIMSCRIPT.cmicross     := run_cmi_crosscheck.py
SIMARGS.cmicross       :=

SIMSCRIPT.nullcal      := run_null_calibration.py
SIMARGS.nullcal        := --n-rep @NullReps --jobs @CmiJobs

SIMSCRIPT.cmiscaledemo := run_cmi_scale_demo.py
SIMARGS.cmiscaledemo   := --ens ensemble:@SeedOne --n-sub @CmiNSub


# ---- 4. CMI: substrate-enriched test, blanket-thickness sweeps, collider checks ----

SIMSCRIPT.uenriched    := run_u_enriched_cmi.py
SIMARGS.uenriched      := --jobs @CmiJobs

SIMSCRIPT.uenrichedfar := run_u_enriched_cmi.py
SIMARGS.uenrichedfar   := --jobs @CmiJobs --target far

SIMSCRIPT.bandsweep    := run_band_sweep_cmi.py
SIMARGS.bandsweep      := --jobs @CmiJobs

SIMSCRIPT.bandsweep1   := run_band_sweep_cmi.py
SIMARGS.bandsweep1     := --jobs @CmiJobs --seed 1

SIMSCRIPT.bandsweep2   := run_band_sweep_cmi.py
SIMARGS.bandsweep2     := --jobs @CmiJobs --seed 2

SIMSCRIPT.randomdisks  := run_random_disks.py
SIMARGS.randomdisks    := --stride 3 --jobs @CmiJobs

SIMSCRIPT.banddisks    := run_band_sweep_cmi.py
SIMARGS.banddisks      := --data random_disks.npz --tag _disks --only-all --jobs @CmiJobs

SIMSCRIPT.levelsetcmi  := run_levelset_cmi.py
SIMARGS.levelsetcmi    := --jobs @CmiJobs

SIMSCRIPT.indepens     := run_cmi_independent_frames.py
SIMARGS.indepens       := --data ensemble:@SeedOne --jobs @CmiJobs

SIMSCRIPT.indepmaster  := run_cmi_independent_frames.py
SIMARGS.indepmaster    := --data master --jobs @CmiJobs

SIMSCRIPT.blanketfigs  := plot_blanket_figures.py
SIMARGS.blanketfigs    :=

SIMSCRIPT.pooledcmi    := analyze_pooled_collider_cmi.py
SIMARGS.pooledcmi      := --n-seeds @CmiSubsamples --null-seeds @CmiNullSubsamples --n-sub @CmiNSub --n-trials @CmiTrials --data master_trajectory_progress.npz

SIMSCRIPT.splithalf    := rebuild_splithalf_check.py
SIMARGS.splithalf      :=


# ---- 4. depth sweep and merge-rounds check ----

SIMSCRIPT.depthsweep  := sweep_from_master.py
SIMARGS.depthsweep    := --stride @DepthStride

SIMSCRIPT.plotdepth   := plot_depth_sensitivity_pc1.py
SIMARGS.plotdepth     := --data sweep_merged_progress.npz

SIMSCRIPT.mergerounds := merge_rounds_check_1024.py
SIMARGS.mergerounds   :=


# ==============================================================================================================
# 6. THE ANALYSES THAT WRITE RESULT MACROS DIRECTLY
# ==============================================================================================================


# Add an analysis by naming it in ANALYSES and giving its script (and arguments).  The rules that use these
# variables are in the Makefile (ANALYSIS_RULE, section 6).

ANALYSES := \
    marginal_screened table1_intervals derived population_acf check_burnin

SCRIPT.marginal_screened  := marginal_screened_ensembles.py
ARGS.marginal_screened    :=

SCRIPT.table1_intervals   := table1_intervals.py
ARGS.table1_intervals     :=

SCRIPT.derived            := derived.py
ARGS.derived              :=

SCRIPT.population_acf     := population_acf.py
ARGS.population_acf       :=

SCRIPT.check_burnin       := check_burnin.py
ARGS.check_burnin         :=
