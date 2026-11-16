"""
Analysis for the "cancellation between pairs" hypothesis (run this
after run_touching_pairs.py has produced touching_pairs_progress.npz).

Two things tested, both frame-cluster bootstrapped (resampling whole
frames, matching this project's existing convention -- see
run_flux_redundancy_check.py's bootstrap_ci_by_frame):

  (a) COMPETITION: corr(I_A, I_B) across touching pairs. Negative and
      away from 0 supports "the two organisms draw on the same locally
      limited resource, so one's gain is partly the other's loss."

  (b) CONTAMINATION: does E_A correlate with I_B (the SPECIFIC
      neighbor's interior state) as much as, or more than, it
      correlates with I_A (A's own interior)? And: how much of the
      marginal I_A-E_A correlation survives controlling for I_B alone,
      compared to controlling for flux alone (the paper's existing
      blanket variable) and for both together? If controlling for I_B
      removes comparable ground to controlling for flux, that's
      evidence the neighbor's own state -- not just the flux crossing
      the shared boundary -- is doing real work in explaining why I_A
      and E_A look related in the first place.

Since pair_id already picks ONE canonical (A,B) labeling per pair (not
a physically meaningful "A vs B" distinction -- just whichever member
sorted first), every correlation here is by construction symmetric
under swapping the A/B labels only for quantities that don't care
about direction (e.g. corr(I_A,I_B) is symmetric already); this
script does not assume A is somehow special.

Usage:
    python analyze_pair_cancellation.py
    python analyze_pair_cancellation.py touching_pairs_progress.npz
"""
import sys
import numpy as np

from partial_corr_utils import _residualize, pc1_partial_corr, check_partial_corr_stability
from run_flux_redundancy_check import bootstrap_ci_by_frame

PROGRESS_FILE = sys.argv[1] if len(sys.argv) > 1 else "touching_pairs_progress.npz"


def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def partial_corr_on(y1, y2, *controls):
    """Partial correlation of y1, y2 controlling for one or more
    control variables (each same length), via OLS residualization."""
    n = len(y1)
    B = np.column_stack([np.ones(n)] + [c for c in controls])
    return corr(_residualize(y1, B), _residualize(y2, B))


def main():
    print(f"loading {PROGRESS_FILE} ...")
    d = np.load(PROGRESS_FILE)
    I_A, E_A, fu_A, fv_A = d["I_A"], d["E_A"], d["fu_A"], d["fv_A"]
    I_B, E_B, fu_B, fv_B = d["I_B"], d["E_B"], d["fu_B"], d["fv_B"]
    frame_id = d["frame_id"]

    good = (np.isfinite(I_A) & np.isfinite(E_A) & np.isfinite(fu_A) & np.isfinite(fv_A) &
            np.isfinite(I_B) & np.isfinite(E_B) & np.isfinite(fu_B) & np.isfinite(fv_B))
    I_A, E_A, fu_A, fv_A = I_A[good], E_A[good], fu_A[good], fv_A[good]
    I_B, E_B, fu_B, fv_B = I_B[good], E_B[good], fu_B[good], fv_B[good]
    frame_id = frame_id[good]
    N = len(I_A)
    print(f"N touching pairs (finite, deduplicated one row per pair): {N}, "
          f"{len(np.unique(frame_id))} frames")

    def report(name, fn, *arrays):
        val = fn(*arrays)
        lo, hi, boot = bootstrap_ci_by_frame(fn, frame_id, *arrays)
        n_clusters = len(np.unique(frame_id))
        print(f"  {name}: {val:.4f}   95% CI [{lo:.4f}, {hi:.4f}]  ({n_clusters} frames, {len(boot)} boot reps)")

    print(f"\n{'='*70}\n(a) competition: are the pair's own interior states anticorrelated?\n{'='*70}")
    report("corr(I_A, I_B)", corr, I_A, I_B)
    report("corr(E_A, E_B)  [for comparison]", corr, E_A, E_B)

    print(f"\n{'='*70}\n(b) contamination: does E_A track the NEIGHBOR's I_B as much as its own I_A?\n{'='*70}")
    report("corr(E_A, I_A)  [A's own exterior vs own interior -- the usual marginal]", corr, E_A, I_A)
    report("corr(E_A, I_B)  [A's exterior vs the SPECIFIC neighbor's interior]", corr, E_A, I_B)
    report("corr(E_B, I_A)  [symmetric check: B's exterior vs A's interior]", corr, E_B, I_A)

    print(f"\n{'='*70}\nhow much of corr(I_A, E_A) survives controlling for I_B vs for flux?\n{'='*70}")
    report("raw marginal corr(I_A, E_A)", corr, I_A, E_A)
    report("controlling for I_B alone", partial_corr_on, I_A, E_A, I_B)

    r_flux, n_flux, explained = pc1_partial_corr(I_A, E_A, fu_A, fv_A)
    print(f"  controlling for flux (PC1 of fu_A,fv_A, {explained*100:.1f}% variance explained): "
          f"{r_flux:.4f}   (N={n_flux}, not frame-bootstrapped here -- see paper's existing "
          f"touching-subset figure for the CI'd version)")

    report("controlling for I_B AND flux jointly (PC1 of fu_A,fv_A + I_B)",
           lambda I_A_, E_A_, I_B_, fu_A_, fv_A_: partial_corr_on(
               I_A_, E_A_, I_B_,
               (fu_A_ - fu_A_.mean()) / fu_A_.std() + (fv_A_ - fv_A_.mean()) / fv_A_.std()),
           I_A, E_A, I_B, fu_A, fv_A)

    print(f"\ninterpretation guide:")
    print(f"  if 'controlling for I_B alone' removes comparable ground to 'controlling")
    print(f"  for flux', the neighbor's own state is doing real work, not just the flux")
    print(f"  crossing the shared boundary. If 'I_B AND flux jointly' removes much more")
    print(f"  than either alone, they're carrying at least partly independent information.")
    print(f"  if corr(I_A,I_B) is negative and away from 0, that's the competition signature;")
    print(f"  if corr(E_A,I_B) exceeds corr(E_A,I_A) in magnitude, A's 'exterior' reading is")
    print(f"  substantially just reading its neighbor.")


if __name__ == "__main__":
    main()
