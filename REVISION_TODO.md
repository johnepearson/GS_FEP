# Revision to-do (written 2026-10-29, after the first complete full run)

Plan: no rerun until the reviewers respond. Then make all code and parameter changes below in ONE batch, rerun
(`make FULL=1`, about a day and a half), then `make pdfs`, then apply the text edits and write the change log.

Reference versions
- Submitted: `reviewers_version_1.zip` (paper.pdf, supplement.pdf, .tex files).
- First rebuilt full run: `build/` (new settled checkpoint, burn-in 7,000, ensemble 3 with 1,000 realisations).

---

## 1. Open questions (not yet looked at)

- [x] `build/logs/stationarity.log` read 2026-10-29: is 3,800 settle steps (`SettleSteps`) long enough? Yes for the PC1 partial r:
      first 50 frames -0.186, first/middle/last third -0.185 / -0.197 / -0.186 (no trend; middle differs by about 0.012, within the
      +-0.01 interval). Organisms per frame 194.0 (SD 9.6) in the first 50 frames against 190-192 (SD 7-8) later: slightly higher
      and wider early, within one SD. Put this table in the supplement as the evidence. The 3,800 comes from the old
      protocol ("eyeballed"). The 7,000 burn-in is a different thing (ensemble copies after branching).
- [ ] Decide the burn-in. `check_burnin`: mean t90 = 3,533 (3,300 / 3,600 / 3,700), so 2x = 7,067; 7,000 is 1% short.
      Planned: raise `BurninOne/Two/Three` to 8,000 (about 2.3x). Decide whether `ThreeVarBurnin` follows (rule was measured
      on the two-variable model; changing it restarts both three-variable stages from scratch, about 10 h).
      If kept at 7,000, reword the paper to "about twice" and give the ratio (1.98).

## 2. Code and parameter changes for the one batch rerun

- [ ] Burn-in parameters (see 1).
- [ ] Level-set threshold: replace the typed `V_THRESHOLD = 0.1356` in `run_level_set_individuation.py` by a value
      computed from the data (median boundary v of clean organisms; `anmaster` log gives 0.1358) and written as a macro.
      Update the supplement sentence to say it is measured from this run's data.
- [ ] Level-set threshold sweep (new stage, runs on archived frames, minutes): r and CI at thresholds about 0.10 to 0.18.
      Replace the "magnitude is not independent of method" sentence by the measured sensitivity.
- [ ] Frozen-lattice run (Dw = 0): save per-frame time and organism count in the npz (`run_3var_shell_data.py`), then make
      "283 organisms" and "t = 108,000" macros. (This forces both three-variable stages to rerun; cheap if batched.)
      New run: 289-290 organisms, final t about 110,500, N = 100,027.
- [ ] Fix the restart bug in the script's "minutes remaining" (rate uses the instances already on file).
- [ ] Add a check in `run_3var_shell_data.py`: only exit cleanly if N >= target.
- [ ] Add `export GS_FFT_WORKERS ?= $(JOBS)` to the Makefile (note: 12 threads did NOT speed up the three-variable stages;
      per-frame organism analysis is single-threaded. Running stages in parallel with `make -j` would; needs a check for stages
      sharing files before use).
- [ ] `src/trace_run.py` already fixed on 2026-10-29: a stage that is interrupted can no longer leave a `.done` file
      (a script that swallowed the stop signal and exited 0 did this).

## 3. Wording fixes: paper

- [ ] Section 3.5 lifespans: median death lifespan is now 660, equal to Pearson's 660, so "longer than ~660" is false for deaths
      (old value 555; division median 1270 is still longer). Understand why 555 -> 660.
- [ ] Sign flip: "r = 0.255" needs an explicit plus sign ("r = +0.255").
- [ ] Burn-in: "at least twice the time (on average 3533 steps)" is false for 7,000 (needs 7,067). Fixed by 8,000, else "about twice".
- [ ] Organism-count decorrelation: new data give 0.05 +- 0.16 at burn-in, zero from about step 2,400 (old: 0.46 +- 0.16 at 3,000,
      zero by 4,800). Paper no longer says the pooled intervals are too narrow, but Section 6.4 Limitations still says the realisations
      "are not fully independent for the population count". Reconcile. Note this contradicts the earlier assumption that the count
      decorrelates more slowly than the field; only 12 copies, so the uncertainty is large (consider more copies).
- [ ] Level-set check (3.3.4): "about a third of the watershed value" is now about half (-0.094 vs -0.189).
- [x] Figure 2, right panel: arrows in both directions (done 2026-10-30 in the working copy, not yet in the submitted version). `src/plotscripts/make_figure2.py` (stage `plotschematic`) draws `build/figure2_blanket.png` from `src/TeX/figures/schematic_left.png` (left half of the old `schematic_combined.png`) plus the diagram; `paper.tex` now includes `figure2_blanket.png`. Check the caption (right panel now has a dashed red arrow). Plain-lines variant: `--lines`. The diagram is not a DAG (loops are not allowed in one), so do not call it a causal diagram. Optional point for the text: the flux depends on both sides and the nonlinear CMI rises when conditioned on flux (+71%), which is what a collider would do, so conditioning on flux can induce dependence. Decide whether to say so in Section 3 or the Limitations.
- [ ] Check the ensemble figure (left panel): "falls to the unconditioned value or below only in the lowest-flux slice and the two
      highest" (still 17 of 20 slices above).
- [ ] "t90 agrees to within about 11%", "growth rates agree to within about 37%" (was 30%): confirm the word "agree" is still fair.

## 4. Wording fixes: supplement

- [ ] Section S1.2: "falls outside the range at 5 of the five depth-fractions" -> "at all five".
- [ ] Section S3: "(-0.28 to -0.28)" -> "about -0.28".
- [ ] Level-set (S2.2): recovered pairs alone are now r = -0.024, 95% CI [-0.058, 0.010], which includes zero (old -0.071). State this.
      Pooled -0.094 is still clearly nonzero. Describe the excluded components as touching organisms the threshold cannot separate
      (not "merged").
- [ ] Section S13 (frozen lattice) and the Figure S7 caption: "283" and "t = 108,000" are old; new about 290 and about 110,500.
      "at every frame printed" is no longer exact (289 at start, 290 later).
- [ ] Section S17.4: Tigramite null: observed value 31.5 SDs above Tigramite's null vs 33.2 for ours (old 25.3 vs 24.2). Check any wording
      about which is larger.
- [ ] Stale sentence: the checkpoint's initial condition "was not recorded, so it is distributed as data". It is now known: 160x160
      central patch (PatchHalf = 80), u = 0.5, v = 0.25, +-1% noise, seed 0, 3,800 steps.
- [ ] Wall-clock times (typed, labelled "a previous run") are old.

## 5. Old (submitted) versus new numbers, for the change log

Cause of every difference: new settled checkpoint, burn-in 3,000/5,000 -> 7,000, ensemble 3 500 -> 1,000 realisations
(same seeds do not give the same realisations from a different starting field).

| | Submitted | New |
|---|---|---|
| Ensemble 1 (seed 1000): burn-in, N | 3,000, 566,689 | 7,000, 576,765 |
| Ensemble 2 (seed 9000): burn-in, N | 3,000, 566,012 | 7,000, 577,101 |
| Ensemble 3 (seed 20000): realisations, burn-in, N | 500, 5,000, 288,286 | 1,000, 7,000, 577,670 |
| Single trajectory N | 96,315 | 95,767 |
| PC1 partial r: ens 1 / 2 / 3 / trajectory | -0.1815 / -0.1843 / -0.1889 / -0.1896 | -0.1900 / -0.1840 / -0.1842 / -0.1893 |
| Marginal r: ens 1 / 2 / 3 / traj | -0.233 / -0.235 / -0.244 / -0.240 | -0.240 / -0.237 / -0.235 / -0.239 |
| Screened %: ens 1 / 2 / 3 / traj | 39.1 / 38.3 / 40.2 / 37.7 | 37.5 / 39.5 / 38.3 / 37.3 |
| Resampled CI ens 1 | [-0.1850, -0.1781] | [-0.1939, -0.1861] |
| Resampled CI trajectory | [-0.1992, -0.1803] | [-0.1988, -0.1793] |
| Joint two-flux partial r (ens 1 / traj) | -0.289 / -0.2822 | -0.2819 / -0.2804 |
| Realisations sharing pooled sign (ens 1) | 999 of 1000 | 1000 of 1000 |
| Time for copies to reach 90% of independent separation | 3,300-3,600 (about 10% spread) | 3,300-3,700, mean 3,533 (11%) |
| Divergence growth rate spread | about 30% | about 37% |
| Organism-count correlation at burn-in | 0.46 +- 0.16 (3,000 steps); zero by about 4,800 | 0.05 +- 0.16 (7,000 steps); zero from about 2,400 |
| Median lifespan, death / division | 555 / 1265 | 660 / 1270 |
| Confirmed divisions | 689 | 677 |
| Level-set: recovered pairs alone | -0.071 | -0.024 (CI includes zero) |
| Level-set: pooled / single-maximum | -0.068 / -0.259 | -0.094 / -0.272 |
| Organism cores with a growing local mode | 26% | 13% |
| Frozen-lattice (Dw = 0) organisms; run length | 283; t = 108,000 | about 290; t about 110,500 |
| CMI given flux raises dependence by | 73% | 71% |

Other small shifts are in the macros and follow from the same causes.

## 6. Housekeeping

- [ ] `tests/golden/marginal_screened.txt` still holds the SUBMITTED numbers (`make check` reports a difference). Save a copy of the old file
      (change log), then replace it with the new reference once the numbers are accepted:
      `cp build/logs/marginal_screened.log tests/golden/marginal_screened.txt`.
- [ ] 22 of the 38 files in `src/TeX/figures/` are not used by paper.tex or supplement.tex (they are still dependencies of the PDF build).
      Decide: move to `figures/unused/` or delete.
- [ ] Nine macros are `nan` but not used in the text (CountWinSeven/Eight*, TigTwentyFourRankLowPercent). Harmless; could be removed.
- [ ] 240 `OTHER` typed numbers remain (build/typed_numbers.txt): method constants, one-off results. Classify when time allows.
- [ ] GitHub: tag the submitted state (`submitted-v2`), keep the rebuild on a separate branch, merge to main only with the revised paper.
- [ ] Delete the GitHub token. Check the arXiv v2 poster and ancillary links.
- [ ] README update after the revised run.
- [x] Readable Makefile is now the real Makefile (rules merged, stage data in stages.mk, figure/table map in section 10). Verified
      identical to the original by `make -n -B` on eight targets. Section 10's table sources are "main source" only; do the exact
      macro-to-stage mapping when time allows.
- [ ] Move `params.py` from `work_dir` to `src/tools/` (change the path in `paramstamps.py`; grep `src/` for `params` first; do it in the batch rerun).
- [ ] Optional: `CHANGELOG.md` (build changes so far written; fill in as the revision proceeds).
- [x] Run times: `trace_run.py` logs every stage to `build/logs/times.tsv`; `make times` lists them (2026-10-29). Use it on the final full run to document the time per stage (expect the three-variable stages to dominate; threevar1 took about 6 h in the first full run).
- [x] `make CONF=reduced FULL=1`: full chain with the three-variable target cut to 10,000 (`src/TeX/initialize_reduced.tex`). Not yet run; check that the run time scales as expected.
- [ ] Optional: one line per stage in stages.mk (script and arguments together) instead of separate SIMSCRIPT / SIMARGS lines.

## 7. Layout notes (2026-10-29)

- Parameters are source: edit `src/TeX/initialize.tex` (full run) and `src/TeX/initialize_NAME.tex` (quick, medium, seed1, ...).
  The build folders hold read-only copies (`build/initialize.tex`); make replaces them whenever they differ from `src/TeX`,
  so never edit the copy.
- `Makefile` holds the rules; `stages.mk` holds the stage data (lists, SIMSCRIPT/SIMARGS, ANALYSES). Copy both together.
- Build tools are in `src/tools/` (trace_run.py, paramstamps.py, texout.py, check_typed_numbers.py, compare_checkpoint.py).
  The audit is `python3 src/tools/check_typed_numbers.py build`.
- If a script in `src/tools` changes, `build/deps/*.d` files that still list it make that stage look stale. The runner
  no longer lists itself, so only a `.d` written by an older runner (here `settle.d`) needed editing.
- Delete `Makefile.prevtools` once the revision run has worked.

## 8. Before the final full run (measured 2026-10-31)

Measured with QUICK (same code, same settings), 12-core MacBook Pro:

| Condition | Elapsed | Notes |
|---|---|---|
| Plugged in (AC) | 9 min 14 s | every compute-heavy stage about 40-60% faster |
| On battery | 13 min 31 s | same stages: csolver 106 s vs 70 s, threevar1 60 s vs 37 s |
| 1, 2, 4, 8, 12 workers (battery) | 17.2, 13.9, 13.0, 13.0, 12.7 min (without the `settle` stall) | QUICK has only 4 realisations per ensemble, so it saturates at 4 workers; it says nothing about 8 vs 12 on a full run |

Checklist for the final full run:

- [ ] Plugged in, and checked the plug before leaving the machine (a knocked-out plug costs about 40% more time; the log flipped between AC and battery several times on 2026-10-28).
- [ ] Lid OPEN. Closing the lid sleeps a MacBook on battery whatever `caffeinate` says.
- [ ] ONE `caffeinate` around the whole run, not one per pass: `caffeinate -dims make FULL=1 2>&1 | tee build/full_run.log`.
      A gap of about 1 s between two caffeinate processes let an idle Mac start sleeping (2026-10-31 09:55:47 to 10:05:48, 596 s lost).
- [ ] Dropbox paused (or build folders excluded from sync); quit the browser and other apps; no movie streaming.
- [ ] Check Energy Mode (System Settings > Battery): battery and power-adapter settings can differ.
- [ ] Afterwards, check for sleeps in the run window: `pmset -g log | grep -E "Entering Sleep|Wake from|Using Batt|Using AC" | grep -E "^YYYY-MM-DD"`.
      No "Entering Sleep" lines means the timings are clean. Then `make times` (note: it shows only the latest run per stage).
- Earlier unexplained stalls (24 min in `posgap`, an 18 min gap) were most likely idle sleeps while running on battery (pmset log, 2026-10-30).
- The full run completed 2026-10-28; the pmset log shows no sleep events from 00:00 to 14:59 that day, so its timings are not affected.

## 9. Run times in the supplement table, generated by the build (decided 2026-11-01)

The Time column of the reproducibility table (`tab:S_repro_table`) is typed by hand (some "---", "minutes", "seconds"; numbers from earlier runs).
Replace it by macros written by the final full run, so the table cannot disagree with the run:

- [x] `src/tools/timemacros.py` (written, tested on QUICK data): reads `build/logs/times.tsv` and a mapping file, writes `build/results/times.tex`
      with one macro per table row (sum of that row's stages). Rounding: under 3 min `$<$3 min`; 3 to 60 min whole minutes; over 60 min hours to one decimal.
      Also `\TimeTotal` (sum of stages) and `\TimeWall` (first start to last finish). A stage that ran more than once counts all its runs.
- [ ] Write the mapping file `src/tools/time_rows.txt` (`TimeName: stage stage ...`, one line per table row). Stage names: `stages.mk`.
- [ ] Makefile rule: `build/results/times.tex` depends on all `.done` markers, the mapping file and `timemacros.py`; `results.tex` joins it like the other fragments.
- [ ] Supplement: put `\TimeName` in each Time cell; caption states the machine and conditions (12-core MacBook Pro, plugged in; battery was 40-60% slower).
- [ ] Supplement intro of the table: say that `make FULL=1` regenerates every number, figure, table, the paper and the supplement; `make QUICK=1` for a short test.
- [ ] The final full run must be a CLEAN build (so `times.tsv` holds exactly that run), plugged in, single `caffeinate` (section 8).
- [ ] The typed-number audit (`check_typed_numbers.py`) should then no longer list the typed times.

### Text about the Makefile (drafts, 2026-11-01; for the revision batch, not the submitted version)

**Main paper, Data Accessibility** (replace "and a README giving the command for each result"):

> ... together with the settled starting field from which every simulation is run. A \texttt{Makefile} runs the whole chain: \texttt{make FULL=1} reruns every simulation and analysis, writes the numbers quoted in this paper and its supplement into a file that \LaTeX\ reads, and builds both documents, so nearly all printed numbers are produced by the code and not typed in. \texttt{make QUICK=1} runs the same chain at reduced size as a test. A full run took about \TimeWall\ on a 12-core laptop (supplement, \S\ref{sec:S_reproducibility}). A README gives the requirements and the command for each result.

**Supplement, start of the reproducibility section** (before "This section lists ..."):

> \paragraph{Automated build.} The repository contains a \texttt{Makefile} that regenerates everything in this paper. A \emph{stage} is one run of one script. The Makefile records, for each stage, the script, its parameters (all in one file, \texttt{initialize.tex}) and the files it read, and reruns a stage only when one of them has changed. The analysis scripts write every number quoted in the text, as a \LaTeX\ macro, into a file (\texttt{results.tex}) that the paper and supplement read, so the printed numbers are the output of the run. Random seeds are fixed. \texttt{make FULL=1} reruns the full chain and builds the paper and the supplement; \texttt{make QUICK=1} runs the same chain at reduced size (about ten minutes) as a test. Package versions are pinned in \texttt{requirements.txt}; FFTW 3 and a C compiler are needed (see below). The Time column below is written by the run itself (\texttt{make times} lists every stage); the times are for a 12-core 2024 MacBook Pro on mains power. Run times depend on the machine's power state: the same quick run took 40--60\% longer on battery.

- [ ] Check wording against what is true at the time of the final run (percentage of numbers that are typed, number of macros, time of QUICK).

## 10. Data-processing-inequality (bottleneck) check (started 2026-11-01)

If B were a blanket, I -> B -> E would give MI(I;E) <= MI(I;B) and MI(I;E) <= MI(B;E). A violation would reject B without conditioning;
passing proves nothing (flux is a collider and should pass). Of interest mainly for the boundary concentration (it screens only about a fifth).

- [x] `src/analysis/analyze_dpi_bottleneck.py` written; self-test passes (chain: gaps negative; bypass: gaps positive; collider: gaps negative).
      KSG estimates run about 0.012 nats high at N=8000 (scalar-scalar and scalar-2D alike), so the gaps are roughly unbiased.
- [x] Run once on the existing full-run data (2026-11-01; N=95,767 instances, 20 subsamples of 8,000, k=5). Result: the inequality is NOT violated for either candidate.
      MI(I;E)=0.076 nats. Flux: MI(I;B)=0.836, MI(B;E)=0.089 (gap_BE = -0.014 +- 0.010, thin margin; 1 of 20 subsamples positive).
      Boundary concentration: MI(I;B)=0.420, MI(B;E)=0.450 (gaps -0.344 and -0.374, far outside the noise).
      Reading: no extra rejection. The boundary concentration is a strong bottleneck (5-6 times more information about each side than the sides share)
      and still leaks: conditioning on it removes only about a fifth of the I-E dependence. Answers the objection that it is "too uninformative to screen".
      Do not lean on the flux row (margin of about 1 SD).
- [ ] Decide whether to keep it: stage in `stages.mk` (after `master`), `--macros` writes `results/dpi.tex`. Suggested text, supplement only (one or two sentences):
      "The boundary concentration is an information bottleneck in the data-processing sense: it shares \\DpiBdryMIIB\ and \\DpiBdryMIBE\ nats with the interior and the
      exterior, against \\DpiMIIE\ nats between them. It nevertheless screens only a fifth of the dependence, so the failure of conditional independence is not
      due to an uninformative blanket variable." (The inequalities are necessary, not sufficient: passing them is not evidence for a blanket.)
- [ ] Thick band as a third candidate B (needs the band summaries from `run_band_sweep_cmi.py`); not in the script yet.
- [ ] Subsample SDs are estimator/subsample variation only; for a published claim add the cluster/block bootstrap used elsewhere.

## 11. Curiosity (not for the paper): spectral content and Lyapunov dimension (2026-11-03)
Decision: John does not plan to put any of this in the paper. Kept here so the scripts and numbers are not lost. Nothing here needs rerunning for the revision batch.

Scripts (in `src/analysis/`, not in the Makefile; run from `build/`):
- `analyze_power_spectrum.py`: radially binned |FFT|^2 of u and v over the 500 archived frames. Fourier coefficients (real dof, of 1024^2) needed for 90 / 99 / 99.9 / 99.99% of the variance: u 950 / 2073 / 4951 / 8399; v 1354 / 4447 / 8294 / 13634. Cutoff wavelength at 99.99% is about one organism radius (0.048 for u). Upper bound only; by translation invariance this is the PCA spectrum.
- `lyapunov_spectrum.py`: Benettin/QR with the exact tangent map of the exponential-Euler step (same map as `PseudoSpectralGS`; the C solver with dealias off is the same map, not compared numerically). Prints spots and D_KY at each report.

Result, one run (256^2 patch, L = 0.625, dx as production, corner 0 0 of master_fields/frame_0400.npz, m = 60, transient 2000, 40000 steps, 92 min on the MacBook):
- lambda_1 = +2.28e-3 (Lyapunov time ~ 440), 13 positive exponents, sum of the 60 = -0.100.
- D_KY = 29.8 (still creeping: 28.5 at t = 10000). Per unit area 76; scaled to L = 2.5 gives about 477 (assumes proportional to area; untested).
- Mean spots t = 15000-40000: 15.0, so D_KY / spots about 2.0. Hypothesis (one run only): about 2 dimensions per spot, as if positions. Not established.
- Mean v rises loosely with the spot count (r = 0.62 over 28 snapshots).

More runs (2026-11-03/04), all 256^2 unless noted, from master_fields/frame_0400.npz:
| run | corner | steps | m | D_KY | mean spots | D_KY/spot |
|---|---|---|---|---|---|---|
| lyap_a | 0 0 | 40000 | 60 | 29.8 | 15.0 | 2.0 |
| lyap_b | 512 512 | 40000 | 60 | 27.0 (still wandering 26.7-27.3) | 15.6 | 1.7 |
| lyap_d | 118 65 | 200000 | 50 | 29.9 (t = 1e5-2e5: mean 29.94, SD 0.08) | 15.8 (SD 2.3, range 8-19) | 1.9 |
- lyap_d is the best converged: lambda_1 = +2.16e-3, 13 positive, sum of the 50 = -0.053; scaled to L = 2.5 gives 479. 5 h 35 min. (m = 30 earlier hit the cap: D_KY >= m; script now prints ">= m (raise --m)" instead of nan.)
- 128^2 box (corner 0 0): population died within 2000 steps (mean v = 0, lambda_1 = -0.02), killed.
- 192^2 box (corner 512 512, m = 30): D_KY 16.2-16.7 with about 9.6 spots (1.7/spot) for about 22000 steps, then extinct by t = 24000, killed. Area scaling from the 256^2 boxes predicts 15.2-16.8. Small boxes are chaotic transients with a finite lifetime (a chaotic saddle), growing with box size; lifetimes of the 256^2 boxes are not known beyond 200000 steps.
- Seeds: the tangent vectors do not feed back, so --seed changes nothing about the trajectory; a new trajectory needs a new corner or frame.

Open, only if the curiosity continues:
- [ ] Second 192^2 corner (e.g. `--corner 256 768`, m = 30) for the lifetime distribution; optional auto-stop on extinction.
- [ ] Rerun lyap_b longer to see whether 27.0 is real or unconverged.
- [ ] Full-domain spot count from `f_pop` in the master data, to turn the per-spot number into a full-domain estimate.
- [ ] Print mean nearest-neighbour distance and mean radius at each progress line (c may depend on d/r).
- [ ] Numerical comparison of the Python and C solver maps (`--check-c`) before quoting any number.
- Caveats: Kaplan-Yorke is a conjecture-based estimate, not a theorem; the average includes the first stretch after the transient; one box size.

## 12. Should the script table in the supplement go? (asked 2026-11-04; decision pending)
`tab:S_repro_table` (28 rows) maps each reported quantity (with its paper location) to its script, command and run time. It does more than describe scripts: it is the only place a reader of the PDF can see which script produced which number. A reviewer reading the PDF will not open stages.mk.
Suggested: keep the table, shrink it. (a) Drop the Command column (stages.mk is the single source; give the stage name or `make <stage>` instead). (b) Time column generated from `times.tex` (section 9). (c) Optionally generate the whole table from stages.mk plus a one-line description and paper-location field per stage, so it cannot drift from the code. Do not delete outright without asking John; no change to the paper before the reviewers respond.

## 13. "Flux" means two different conditioning sets: abstract and section 3.3.1 (found 2026-11-11; for the revision batch, no change before the reviewers respond)
What the paper's own numbers show (single trajectory, N = 96,315; marginal r = -0.2402; Table S2, S9.1, S9.2, Table 2):
| test | conditioning on | result |
|---|---|---|
| linear | PC1 of the flux pair (99.2% of flux variance) | r = -0.1896, "37.7% screened" |
| linear | both fluxes jointly (uses the < 1% orthogonal direction) | r = -0.2822, "-38% screened" (dependence up) |
| nonlinear kNN (k = 5) | both fluxes jointly | MI 0.075 -> 0.130 nats, up 73% |
| linear | boundary concentration (u_d, v_d) | r = +0.1175, "76% removed" |
| nonlinear kNN | boundary concentration | 0.075 -> 0.061 nats, down 19% |
Reading: the linear and nonlinear tests agree when they condition on the same variables (both fluxes: both go up). The "flux removes part of the linear dependence" result is specific to the PC1 convention, not to linear versus nonlinear. For the genuine state (boundary concentration) the linear test overstates screening a great deal (76% versus 19%), so the nonlinear table is the one to lean on.
Rough Gaussian-equivalent information, -1/2 ln(1 - r^2) (not a bound): marginal 0.030 nats, PC1 partial 0.018, joint partial 0.042, against kNN MI 0.075 and CMI given flux 0.130.
Items (none done; the submitted paper is unchanged):
- [ ] Abstract sentence "Flux removes only part of the linear dependence, and a nonlinear test shows that conditioning on flux increases the dependence." can mislead: the two clauses use different conditioning sets, and it implies the nonlinear test found what the linear one missed. Suggested: "Conditioning on the leading component of the flux removes only part of the linear dependence, but conditioning on both flux components increases it, in a linear test and more strongly in a nonlinear one." Abstract goes from 173 to 187 words. Shorter option (179 words): "Conditioning on the leading flux component removes only part of the linear dependence, but conditioning on both components increases it, more strongly in a nonlinear test." Journal word limit: check.
- [ ] Section 3.3.1 and Table 1: "flux removes about 38-40% of the marginal shared variance" should say it is the PC1 convention, and give the joint value (the increase) beside it.
- [ ] Section 3.3.1 sentence treating "joint value outside the single-flux range" as "a further sign of the collider structure": the boundary concentration (a genuine state) shows the same property (S9.2: joint +0.1175 against +0.11 and -0.27), so the property is not specific to a collider (two nearly collinear regressors can do it alone). Soften, or drop the claim that it is a sign.
- [ ] Possible reviewer point: the marginal correlation is negative, so conditioning on a difference-type variable (flux is a difference of concentrations on the two sides) moves r toward zero and looks like screening, while a sum-type variable makes it more negative. Part of the PC1 "screening" could therefore be a collider effect. Untested.
Optional checks (none run; the master fields are not in the sandbox):
- [ ] kNN CMI conditioning on PC1 alone (like with like against the linear PC1 result).
- [ ] Gaussian surrogate with the same covariance run through the same kNN estimator (separates estimator artefact from real non-Gaussian structure).
- [ ] Correlate PC1 with (E - I) and the orthogonal component with (E + I) to test the difference-type / sum-type reading above.
Not for the paper: gauss_cmi_note.pdf (linear 1-D chain, exact Gaussian CMI). It conditioned on the state at the blanket sites, not on flux, so it says nothing about flux; there the CMI was 1e-3 to 1e-5 of the marginal MI (strong screening).

## 14. Conda environment, unit tests and the Makefile (started 2026-11-15; Makefile edits: Claude applies them when John says)
Decision (John, 2026-11-15): use a conda (Miniforge) environment, as in Andy's tutorial, instead of the "make quick" venv route, with short instructions for friends. Tested on John's Mac (Apple chip) on 2026-11-15: `conda env create -f setup/environment.yml`, `conda activate gsfep`, `bash setup/check_setup.sh` gave 15 ok, 0 warnings, 0 failures (Python 3.11.18, make 4.4.1, numpy 1.26.4, scipy 1.17.1, scikit-image 0.26.0, scikit-learn 1.9.1, matplotlib 3.8.3, tigramite 5.2.10.1, numba 0.68.0); `make test` gave 18 passed.
Files delivered (setup_pkg.zip, tests_pkg.zip); John copies them into work_dir:
- `setup/environment.yml` (conda: python 3.11, pip, make, fftw, pytest, coverage; pip installs `-r ../requirements.txt`, so requirements.txt stays the single record of the full-run versions), `setup/check_setup.sh`, `setup/README.md`.
- `pytest.ini` (project root) and `tests/test_solver.py`, `tests/test_analysis.py`, `tests/test_tools.py` (18 tests, a few seconds).
Makefile edits still to apply (not done; no change to the Makefile yet):
- [x] DONE 2026-11-15 (applied with apply_fftw_patch.py; original saved as /tmp/Makefile.bak_fftw, not in a git repo): replace the `src/simulation/libgs_spectral.so` rule (about line 172) with `Makefile.fftw_patch.txt` (adds -I/-L/-rpath for $(CONDA_PREFIX) when a conda environment is active; unchanged otherwise). Then delete the old libgs_spectral.so (built against Homebrew's FFTW) and run `make src/simulation/libgs_spectral.so` and `make test`; the C-solver test should pass. Tested 2026-11-15 on the Mac: the first gcc line used -I/-L/-rpath of the gsfep environment, the library rebuilt, and `make test` gave 18 passed (0.67 s).
- [x] `test` and `coverage` block (Makefile.testblock) is in John's Makefile (`make test` works). Still to do: a line for each in the header comment, and check that the block sits near `check`.
- [ ] Header comment: mention `conda activate gsfep` and setup/README.md.
Still to decide or do:
- [ ] Friends' route: replace the venv route in run_quick_test.sh and make_test_package.sh with the conda route (the zip then needs setup/, requirements.txt, pytest.ini, tests/). Keep or drop the QUICK end-to-end run? (Leaning: keep a tiny end-to-end run beside the unit tests.)
- [ ] More tests: the scripts that have `main(argv)`; a tiny-grid end-to-end run; a check that the numbers in tests/golden still match (make check already exists).
- [ ] Python version: requirements.txt says the full run used 3.11.8; the environment installs 3.11.18 (same minor version). Decide whether to record the patch version.
- [ ] Mac gotcha: an `alias python3=...` in ~/.zshrc is invisible to make and can bypass the environment; check `type python3` after `conda activate gsfep`.
- [ ] Re-run the dry-run check of pins after any change to requirements.txt (the sandbox copy of requirements.txt lacked the numba line that John's has).

### 14a. tests/test_scripts.py (added 2026-11-15; not yet in work_dir until John unzips it)
Pre-QUICK checks on all scripts, about 13 s: scripts named in Makefile/stages.mk exist; every script compiles; every
safe-to-import script imports; every argparse script answers --help; no LaTeX macro name written by two scripts.
`pytest.ini` pythonpath gains `src/plotscripts`. Sandbox result: 155 passed, 18 skipped, 1 failed (stage_times.py not
in the sandbox copy; expected to pass in work_dir). Static check only: it does not prove QUICK passes.

### 14b. Deferred to the final full run (added 2026-11-15; nothing to do before the reviews arrive)
- `make QUICK=1` from clean in the conda environment: finished with no errors in 9 min 11 s; 182 tests passed, 18 skipped.
- `make check` (no QUICK) compares the existing full `build/` with `tests/golden/` and shows a difference for
  marginal_screened. Golden: N=566689/566012/288286, screened 39.1/38.3/40.2%. Current build: N=576765/577101/577670,
  screened 37.5/39.5/38.3%. The golden file looks stale (seed 20000 has about half the instances). Not touched.
  At the final full run: regenerate `tests/golden/` from the final-environment run and compare with the screened
  figures in the submitted paper (sec 3.3.1, Table 1).
- nan in the full build: `TigTwentyFourRankLowPercent`, `CountWinSevenR/Err/Start/End`, `CountWinEightR/Err/Start/End`.
  Find out why, and whether the paper or supplement uses them.
- Dealiasing check: in QUICK the no-dealias and 2/3-rule arms give identical results (N=279, partial r 0.3685,
  difference 0.0000, CI [0, 0]). Confirm the dealias flag reaches the C solver (`--fast`) before the final run.
- Caution: `check` depends on `results`; use `make -n check` first so it cannot start a rerun by accident.
