# =====================================================================================================================
# MAKEFILE for the paper "Does a Flux-Defined Markov Blanket Screen Self-Replicating Gray-Scott Spots?"
# It builds the paper and the supplement from the code (see "WHAT THIS MAKEFILE DOES" below).
# =====================================================================================================================
#
# WHAT THIS MAKEFILE DOES
#   It builds the paper from the code:  settle a starting field -> run the simulations -> analyse them -> write the
#   numbers into results.tex -> run LaTeX.  A "stage" is one run of one Python script.
#
#   make              run every stage whose inputs changed and write build/results.tex     (no PDFs)
#   make pdfs         the same, then build paper.pdf and supplement.pdf
#   make -n           show what would run, run nothing
#   make times        list how long each stage took, longest first   (from build/logs/times.tsv, written by trace_run.py)
#   make check        compare the printed output with the validated output in tests/golden/
#   make QUICK=1      the same chain with a few realisations and frames  (src/TeX/initialize_quick.tex,  folder build_quick/)
#   make MEDIUM=1     100 realisations per ensemble, real burn-in         (src/TeX/initialize_medium.tex, folder build_medium/)
#   make FULL=1       allow the full-size simulations (many hours); without it make stops before starting one
#   make CONF=seed1 FULL=1   the full chain from a second settled field (src/TeX/initialize_seed1.tex, folder build_seed1/)
#   make CONF=reduced FULL=1   the full chain, but the two three-variable stages run to 10,000 frames, not 100,000  (src/TeX/initialize_reduced.tex, folder build_reduced/)
#   make JOBS=12      worker processes for the ensembles (default: the number of cores; does not change the results)
#   make clean-results | clean-quick | clean-medium       delete results (see the end of this file)
#
# WHEN A STAGE RERUNS
#   A stage reruns when its script, a library it imports, a data file it reads, or a NUMBER in initialize.tex that it
#   reads has changed.  Each run records these in build/deps/NAME.d (src/tools/trace_run.py), and make includes that file.
#   A stamp file per parameter (build/params/) changes only when that parameter's value changes.
#
# A FEW MAKE SYMBOLS USED BELOW
#   NAME := text         set a variable (the value is the text after :=)
#   $(NAME)              the value of a variable;   $(NAME.x) is just a variable whose name contains a dot
#   target: needs        "target" is a file; it is rebuilt when any file in "needs" is newer
#   target: | needs      "needs" must exist first, but being newer does NOT trigger a rebuild  (order only)
#   <TAB>command         the command that builds the target; a recipe line must start with a TAB, not spaces
#   $@  $<  $^           the target; the first prerequisite; all prerequisites
#   $$                   a literal $ in a recipe (or inside a define that is expanded twice)
#   define X ... endef   a template; $(call X,a) fills in $(1) with a;  $(eval ...) turns the text into real rules
#   $(foreach v,list,..) repeat for every word of the list
#   -include file        include a file, and do not complain if it does not exist yet


# ==============================================================================================================
# 1. CONFIGURATION: which folder, how many worker processes
# ==============================================================================================================


# pipefail: a failing script must stop make even though its output goes through tee.  (Written without .SHELLFLAGS,
# which the make 3.81 that ships with macOS does not understand.)

# bash is looked up on the PATH, not assumed to be /bin/bash (NixOS has no /bin/bash).
BASH  := $(shell command -v bash || echo /bin/bash)
SHELL := $(BASH) -eo pipefail
PY    := python3 -u -B

# QUICK=1 or MEDIUM=1 (or CONF=name) selects a configuration; B is the folder where that configuration is built.

ifdef QUICK
CONF := quick
else ifdef MEDIUM
CONF := medium
endif

ifdef CONF
B := build_$(CONF)
else
B := build
endif

JOBS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc)
export JOBS
# Do not write __pycache__ folders into the source tree.
export PYTHONDONTWRITEBYTECODE := 1
export PYTHONPATH := $(CURDIR):$(CURDIR)/src/tools:$(CURDIR)/src/analysis:$(CURDIR)/src/simulation:$(CURDIR)/src/plotscripts

.PHONY: all results pdfs check sims times clean-results
all: results


# ==============================================================================================================
# 2-4. THE STAGES: lists, scripts and arguments are in stages.mk
# ==============================================================================================================


# stages.mk has the data: which stages exist, in which order they run, and which script and arguments each one uses.
# The rules that turn that data into make rules (sections 5 and 6) stay in this file.  It must be included before
# the loops over SIMS, POSTS and ANALYSES below.

include stages.mk


# First-build order inside the batch.  After the first run the recorded dependencies take over.
# (target: | what must be finished before it)

$(B)/sim/plotdecay.done:    | $(B)/sim/legacy.done
$(B)/sim/levelsetcmi.done:  | $(B)/sim/levelset.done
$(B)/sim/banddisks.done:    | $(B)/sim/randomdisks.done $(B)/sim/bandsweep.done
$(B)/sim/uenrichedfar.done: | $(B)/sim/uenriched.done
$(B)/sim/blanketfigs.done:  | $(B)/sim/bandsweep.done $(B)/sim/uenriched.done
$(B)/sim/plotdepth.done:    | $(B)/sim/depthsweep.done
$(B)/sim/fecomp.done   $(B)/sim/instancelevel.done $(B)/sim/circularity.done $(B)/sim/fesize.done $(B)/sim/fig5fe.done: | $(B)/sim/fetemplate.done
$(B)/sim/plotlife.done $(B)/sim/lifeautocorr.done  $(B)/sim/lifeage.done     $(B)/sim/lineagemix.done:                  | $(B)/sim/anlife.done


# ==============================================================================================================
# 5. HOW A STAGE IS TURNED INTO RULES
# ==============================================================================================================


SIMDONE  := $(addprefix $(B)/sim/,$(addsuffix .done,$(SIMS)))      # e.g. build/sim/ensOne.done, build/sim/master.done ...
POSTDONE := $(addprefix $(B)/sim/,$(addsuffix .done,$(POSTS)))

# SIM_RULE is a template that writes the rule for ONE stage.  $(1) stands for the stage's name.
# For the stage "shapetail" it produces this rule:
#
#   build/sim/shapetail.done : src/<folder>/run_shape_tail_test.py       <- rerun if the script changed
#       (make the build folders)
#       (refuse to start unless FULL=1, MEDIUM=1 or QUICK=1)             <- the guard
#       cd build ; python3 trace_run.py  <script> <arguments>  | tee logs/shapetail.log
#
# The stage name is linked to its script and arguments only through the variables
# SIMSCRIPT.shapetail and SIMARGS.shapetail (section 3).
#
# Three small helpers, so the templates themselves stay short.  SIM_RULE (simulation stages) and ANALYSIS_RULE
# (section 6) share the first two.

# find_script: the full path of a script, searched in every src/ subfolder.
# $(1) is the NAME OF THE VARIABLE that holds the script's file name, for example SIMSCRIPT.shapetail.

find_script = $(firstword $(wildcard src/*/$($(1))))

# traced: the command that runs one script under trace_run.py, which records what the script depended on.
# $(1) stage name   $(2) the file the run produces, inside the build folder   $(3) script variable   $(4) arguments variable

traced = cd $(B) && TOP=$(CURDIR) $(PY) ../src/tools/trace_run.py deps/$(1).d $(2) ../$(call find_script,$(3)) $($(4)) 2>&1 | tee logs/$(1).log

# full_run_guard: a shell test, run before a stage starts, that stops make unless a full run was asked for.
# It stops when ALL THREE of these hold:
#   1. the stage is not "settle"          (settle is short, and every configuration needs it)
#   2. CONF is not quick or medium        (those configurations are meant to be short)
#   3. FULL is not set                    (you did not type  make FULL=1)
# In words: "do not start a many-hour simulation by accident".

define full_run_guard
if [ "$(1)" != settle ] \
   && [ -z "$(filter quick medium,$(CONF))" ] \
   && [ -z "$(FULL)" ]; then \
     echo "$(1) is a full-size simulation (many hours)."; \
     echo "Run  make FULL=1  to start it, or  make MEDIUM=1 / make QUICK=1  to test the chain."; \
     exit 1; \
fi
endef

# A word on the dollar signs.  make reads the template twice:
#   $(1)   is replaced at once, by the stage name.
#   $$(x)  becomes $(x) and is replaced later, when the rule is used (by then SIMSCRIPT.name etc. exist).

define SIM_RULE
$(B)/sim/$(1).done: $$(call find_script,SIMSCRIPT.$(1))
	@mkdir -p $(B)/logs $(B)/sim $(B)/deps
	@$(call full_run_guard,$(1))
	$$(call traced,$(1),sim/$(1).done,SIMSCRIPT.$(1),SIMARGS.$(1))
-include $(B)/deps/$(1).d
endef

# Apply the template to every name in SIMS and POSTS.

$(foreach s,$(SIMS) $(POSTS),$(eval $(call SIM_RULE,$(s))))

# Order only (the | sign): the settled checkpoint first, then the simulations, and the C solver must exist.

$(filter-out $(B)/sim/settle.done,$(SIMDONE)) $(POSTDONE): | $(B)/sim/settle.done
$(POSTDONE): | $(SIMDONE)
$(SIMDONE) $(POSTDONE): | src/simulation/libgs_spectral.so

# Link flags for FFTW: when a conda environment is active, look for the headers and library there (and record the
# library folder in the .so, so it is found at run time).  Empty otherwise, so nothing changes without conda.
FFTW_CONDA = -I$(CONDA_PREFIX)/include -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
FFTW_ENV := $(if $(CONDA_PREFIX),$(FFTW_CONDA))

src/simulation/libgs_spectral.so: src/simulation/gs_spectral_c.c
	gcc -O3 -march=native -fPIC -shared $(FFTW_ENV) -o $@ $< -lfftw3_threads -lfftw3 -lpthread -lm || \
	gcc -O3 -fPIC -shared -I/opt/homebrew/include -L/opt/homebrew/lib -o $@ $< -lfftw3_threads -lfftw3 -lpthread -lm

sims: $(SIMDONE) $(POSTDONE)


# ==============================================================================================================
# 6. THE ANALYSES THAT WRITE RESULT MACROS DIRECTLY
# ==============================================================================================================


# Add an analysis by naming it in ANALYSES and giving its script (and arguments).  Nothing else is listed by hand:
# the first run records what the script imported, which data files it opened and which parameters it read, in
# build/deps/NAME.d, and make includes that file.  After that a result reruns only when one of those changes.
# (Same idea as the stages above, but the product is build/results/NAME.tex, a file of \newcommand lines.)
# The list ANALYSES and the SCRIPT.NAME / ARGS.NAME lines are in stages.mk.


# The parameters of each configuration are kept in src/TeX (they are source, not build output):
#     src/TeX/initialize.tex           the full run
#     src/TeX/initialize_NAME.tex      the configuration NAME (quick, medium, seed1, ...)
# Each build folder gets a read-only copy, build/initialize.tex or build_NAME/initialize.tex, which is what the scripts
# and LaTeX read.  Edit the file in src/TeX, never the copy: make replaces the copy whenever the two differ, so a
# "make clean" of a build folder never loses the parameters.

INIT := src/TeX/initialize$(if $(CONF),_$(CONF)).tex

ifeq ($(wildcard $(INIT)),)
$(error $(INIT) not found: it holds the parameters of this configuration)
endif

_ := $(shell mkdir -p $(B) && (cmp -s $(INIT) $(B)/initialize.tex || (cp -f $(INIT) $(B)/initialize.tex && chmod a-w $(B)/initialize.tex)))

# Results must be real files: a symbolic link in build/ could make a simulation overwrite data somewhere else.

_links := $(shell find $(B) -maxdepth 1 -type l 2>/dev/null)

ifneq ($(strip $(_links)),)
$(error $(B)/ contains symbolic links: $(_links). Remove them: find $(B) -maxdepth 1 -type l -delete)
endif

# One stamp file per parameter, rewritten only when that value changes (see src/tools/paramstamps.py).

_ := $(shell python3 -B src/tools/paramstamps.py $(B)/initialize.tex $(B)/params)

# ANALYSIS_RULE: the same idea as SIM_RULE, for the analyses.  $(1) is the analysis name.
# Differences from SIM_RULE: the script and arguments are named SCRIPT.NAME and ARGS.NAME,
# the product is build/results/NAME.tex (not a .done marker), and there is no full-run guard.
# For the analysis "derived" it produces:
#
#   build/results/derived.tex : src/<folder>/<SCRIPT.derived>             <- rerun if the script changed
#       (make the build folders)
#       cd build ; python3 trace_run.py  <script> <ARGS.derived>  | tee logs/derived.log
#
# (find_script and traced are defined in section 5.)

define ANALYSIS_RULE
$(B)/results/$(1).tex: $$(call find_script,SCRIPT.$(1))
	@mkdir -p $(B)/logs $(B)/results $(B)/deps
	$$(call traced,$(1),results/$(1).tex,SCRIPT.$(1),ARGS.$(1))
-include $(B)/deps/$(1).d
endef

# Apply the template to every name in ANALYSES.

$(foreach a,$(ANALYSES),$(eval $(call ANALYSIS_RULE,$(a))))

FRAGMENTS := $(addprefix $(B)/results/,$(addsuffix .tex,$(ANALYSES)))
$(FRAGMENTS): | $(SIMDONE) $(POSTDONE)


# ==============================================================================================================
# 7. results.tex: ALL THE MACROS JOINED
# ==============================================================================================================


# The paper does \input{results}.  results.tex is every build/results/*.tex joined: the fragments above and the files
# that the simulation stages write themselves.

$(B)/results.tex: $(FRAGMENTS) $(SIMDONE) $(POSTDONE)
	@echo "% All results, generated by make. Do not edit." > $@
	@cat $(sort $(wildcard $(B)/results/*.tex)) >> $@
	@echo "wrote $@"

results: $(B)/results.tex


# ==============================================================================================================
# 8. THE PDFs
# ==============================================================================================================


# pdflatex runs in the build folder (so the .aux and .log files stay there); TEXINPUTS lets it find the sources in src/TeX.

TEXSRC   := src/TeX
TEXFILES := $(filter-out $(TEXSRC)/initialize%.tex,$(wildcard $(TEXSRC)/*.tex $(TEXSRC)/*.bib $(TEXSRC)/*.sh $(TEXSRC)/figures/*.png))
# (the parameter files initialize*.tex are left out: the copy $(B)/initialize.tex is a prerequisite of the PDFs instead)
TEXENV   := TEXINPUTS=.:$(CURDIR)/$(TEXSRC)//: BIBINPUTS=.:$(CURDIR)/$(TEXSRC): BSTINPUTS=.:$(CURDIR)/$(TEXSRC):

pdfs: $(B)/paper.pdf $(B)/supplement.pdf

# figure2_blanket.png is drawn by the stage plotschematic (in build/, where LaTeX looks first); this line makes it a prerequisite.
$(B)/figure2_blanket.png: $(B)/sim/plotschematic.done

$(B)/paper.pdf: $(TEXFILES) $(B)/initialize.tex $(B)/results.tex $(B)/figure2_blanket.png
	cd $(B) && $(TEXENV) sh ../$(TEXSRC)/build_pdfs.sh

# build_pdfs.sh makes both PDFs in one go; the empty command (the ;) just says supplement.pdf comes with paper.pdf.

$(B)/supplement.pdf: $(B)/paper.pdf ;


# ==============================================================================================================
# 9. CHECK AND CLEAN
# ==============================================================================================================


check: results
	@if grep -n 'nan' $(B)/results.tex > /dev/null; then echo 'These results are nan (not computed):'; grep -n 'nan' $(B)/results.tex; else echo 'no nan results'; fi
	@if [ -n "$(CONF)" ]; then echo "check compares with the validated output, which is for the full-size run: use make check without QUICK or MEDIUM"; exit 0; fi; \
	for n in marginal_screened; do \
	  if [ ! -f tests/golden/$$n.txt ]; then echo "$$n: no tests/golden/$$n.txt yet"; continue; fi; \
	  if diff tests/golden/$$n.txt $(B)/logs/$$n.log > /dev/null; then echo "$$n: matches the validated output"; \
	  else echo "$$n: DIFFERS from the validated output"; diff tests/golden/$$n.txt $(B)/logs/$$n.log; fi; \
	done

# times: print the run time of every stage that has run in this folder (the record is src/tools/trace_run.py's).
times:
	@$(PY) src/tools/stage_times.py $(B)

# Remove the results, logs, dependency records and stage markers of this configuration (NOT the data files).
clean-results:
	rm -rf $(B)/results $(B)/results.tex $(B)/logs $(B)/deps $(B)/sim

# clean-NAME removes the folder build_NAME, for example  make clean-quick  or  make clean-medium  (any configuration).

clean-%:
	rm -rf build_$*


# ==============================================================================================================
# 10. WHICH STAGE MAKES WHICH FIGURE AND TABLE  (comments only; make does not read this)
# ==============================================================================================================
#
# FIGURES.  Format:  figure file  <-  stage  (script).  The stage names are those in section 2; the script is
# found in src/analysis, src/simulation or src/plotscripts.  A plotting stage reads the data made by the
# simulation stages before it.
#
#   spots_snapshot.png            <-  spotsnapshot   (plot_snapshot.py)
#   merge_before.png, merge_after.png  <-  mergefig   (make_merge_figure.py)
#   lyapunov_comparison_fixed.png <-  divergence     (run_divergence_rate.py)
#   decay_curve_1024.png          <-  plotdecay      (plot_decay_curve_pc1.py)         via legacy (master_to_legacy.py)
#   ensemble_1024.png             <-  plotens        (plot_ensemble_summary.py)
#   screening_curve.png           <-  plotscreen     (plot_screening_curve.py)
#   lifespan_1024.png             <-  plotlife       (plot_lifespan.py)
#   fe_four_curves_1024.png       <-  uv2life        (analyze_uv2_lifecycle.py)
#   fe_surprisal_1024.png         <-  surprisal      (analyze_surprisal.py)
#   memory_comparison_1024.png    <-  shells3var     (analyze_3var_shells.py)
#   depth_sensitivity.png         <-  plotdepth      (plot_depth_sensitivity_pc1.py)
#   band_sweep_cmi.png            <-  blanketfigs    (plot_blanket_figures.py)
#   u_enriched_cmi.png            <-  blanketfigs    (plot_blanket_figures.py)
#
#   figure2_blanket.png           <-  plotschematic  (make_figure2.py), with the left half in src/TeX/figures/schematic_left.png
#
# Not made by make (they are drawn by hand and kept in src/TeX/figures/):
#   schematic_left.png (left half of the earlier schematic_combined.png), split_duration_histogram.png
#
# TABLES.  The numbers in a table cell are macros, and a table usually draws on several stages, so only the
# main source is given.
#
#   tab:ensemble_summary          (paper)       table1_intervals (table1_intervals.py), with anmaster and anens1-3
#   tab:cmi                       (paper)       pooledcmi        (analyze_pooled_collider_cmi.py)
#   tab:band                      (paper)       blanketfigs      (plot_blanket_figures.py), from bandsweep
#   tab:S_band, tab:S_band_disks, tab:S_u_enriched (supplement)
#                                               blanketfigs      (plot_blanket_figures.py), from bandsweep,
#                                                                 randomdisks and uenriched
#   tab:S_resolution_definition_2x2 (supplement) res2x2          (analyze_resolution_2x2_merged.py)
#   tab:pc1, tab:correlation_provenance, tab:provenance (supplement)
#                                               values from many stages (followups, lifeage, res2x2, diagsize2,
#                                               pc1details, nullcal and others); not traced one by one here.
#
# To see exactly which stage defines a macro, look it up in the finished build:
#       grep -l "newcommand{.MacroName}" build/results/*.tex
# ---- unit tests (add near the "check" target) ------------------------------------------------------------
#   make test        run the unit tests in tests/ (pytest, a few seconds; no simulation is run)
#   make coverage    the same, with a report of which lines of code the tests ran: build/htmlcov/index.html
# (make check, above, is different: it compares the printed output of a full run with tests/golden/.)

.PHONY: test coverage

test: src/simulation/libgs_spectral.so
	$(PY) -m pytest

coverage: src/simulation/libgs_spectral.so
	rm -rf build/htmlcov build/.coverage
	COVERAGE_FILE=build/.coverage $(PY) -m coverage run --source tests,src/analysis,src/simulation,src/tools -m pytest
	COVERAGE_FILE=build/.coverage $(PY) -m coverage report
	COVERAGE_FILE=build/.coverage $(PY) -m coverage html -d build/htmlcov
	@echo "To view: open build/htmlcov/index.html"
