"""
Template-circularity check for the life-cycle comparison (main text §5.4; ESM §S_fe_comparison): does the agreement
between the template-deviation free energy and the Friston free energy survive when the template is built from a
separate trajectory?

  - Build an independent radial template T(r) from the MASTER trajectory's merged organisms (master_profiles.npz, from
    build_template_from_master.py), unconditioned on division status. The master run is a separate simulation from the
    lifecycle run, so the template is disjoint from the data it is evaluated on.
  - Evaluate F_template against it using only the PRE-division portion of each confirmed-division trajectory in
    lifecycle_progress.pkl.
  - Bin by relative time (several bin widths) and correlate the z-scored, matched-bin means against the Friston F
    (generative model refit live from the Ensemble 1 worker files, fe_model.py).

Requires: master_profiles.npz, lifecycle_progress.pkl, and ensemble_merged_seed1000_worker{0..11}.npz (via fe_model.py).
Run from the directory holding these files.

Usage:
    python3 rebuild_circularity_check_v2.py
"""
import pickle
import sys
import numpy as np

# --- same tolerant-unpickling workaround as the other \S7 rebuilds ---
class _TolerantUnpickler(pickle._Unpickler):
    def load_build(self):
        stack = self.stack
        state = stack.pop()
        inst = stack[-1]
        setstate = getattr(inst, "__setstate__", None)
        try:
            if setstate is not None:
                setstate(state)
                return
            slotstate = None
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            if state:
                inst.__dict__.update(state)
            if slotstate:
                for k, v in slotstate.items():
                    setattr(inst, k, v)
        except Exception as e:
            print(f"  (tolerated a __setstate__ failure on {type(inst).__name__}: {e})")
    dispatch = dict(pickle._Unpickler.dispatch)
    dispatch[pickle.BUILD[0]] = load_build


def tolerant_load(path):
    with open(path, "rb") as f:
        return _TolerantUnpickler(f).load()


import numpy.random._pickle as _np_random_pickle
_orig_bitgen_ctor = _np_random_pickle.__bit_generator_ctor
def _patched_bitgen_ctor(bit_generator_name="MT19937"):
    if isinstance(bit_generator_name, type):
        bit_generator_name = bit_generator_name.__name__
    elif not isinstance(bit_generator_name, str):
        return bit_generator_name
    return _orig_bitgen_ctor(bit_generator_name)
_np_random_pickle.__bit_generator_ctor = _patched_bitgen_ctor

def _patched_generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=None):
    bg = _patched_bitgen_ctor(bit_generator_name)
    return np.random.Generator(bg)
_np_random_pickle.__generator_ctor = _patched_generator_ctor

import run_lifecycle
from run_lifecycle import LifecycleTracker
sys.modules["__main__"].LifecycleTracker = LifecycleTracker

# Generative-model constants (main text §5.2), refit live from the Ensemble 1 worker files (see fe_model.py).
from fe_model import load_constants
MU_P, SIGMA_P2, A, B, SIGMA, SLOPE, INTERCEPT, SIGMA_Q2 = load_constants()

PROFILE_WORKERS = 4
BIN_WIDTHS = [5, 10, 15, 20]


def friston_F(I, E, fu, fv):
    if not (np.isfinite(I) and np.isfinite(E) and np.isfinite(fu) and np.isfinite(fv)):
        return np.nan
    mu_q = SLOPE * I + INTERCEPT
    KL = (np.log(np.sqrt(SIGMA_P2) / np.sqrt(SIGMA_Q2))
          + (SIGMA_Q2 + (mu_q - MU_P) ** 2) / (2 * SIGMA_P2) - 0.5)
    Sigma_inv = np.linalg.inv(SIGMA)
    _, logdet = np.linalg.slogdet(SIGMA)
    const = 0.5 * (2 * np.log(2 * np.pi) + logdet)
    o = np.array([fu, fv])
    pred = mu_q * A + B
    resid = o - pred
    quad = resid @ Sigma_inv @ resid
    extra = SIGMA_Q2 * (A @ Sigma_inv @ A)
    acc = const + 0.5 * (quad + extra)
    return KL + acc


def free_energy_template(profile, template):
    good = np.isfinite(profile) & np.isfinite(template)
    if not good.any():
        return np.nan
    return np.mean((profile[good] - template[good]) ** 2)


def build_disjoint_template():
    """Independent template from the MASTER trajectory's merged organisms (master_profiles.npz, written by
    build_template_from_master.py). The master run is a separate simulation from the lifecycle run, so it is disjoint
    from the data the template is evaluated on."""
    d = np.load("master_profiles.npz")
    stacked = d["profiles"]
    print(f"  master_profiles.npz: {len(stacked)} merged-organism profiles from {len(np.unique(d['frame_id']))} frames")
    with np.errstate(invalid="ignore"):
        template = np.nanmean(np.where(np.isfinite(stacked), stacked, np.nan), axis=0)
    return template


def bin_examples_pre(examples, template, use_friston, bin_width):
    """Pre-division only: rel = h['t'] - e['death'], rel < 0."""
    bins = np.arange(-150, 5, bin_width)
    tol = bin_width * 0.6
    binned = {b: [] for b in bins}
    for e in examples:
        if e['cause'] != 'split' or not e['history']:
            continue
        split_t = e['death']
        for h in e['history']:
            rel = h['t'] - split_t
            if rel > 0 or rel < bins[0] - tol:
                continue
            idx = np.argmin(np.abs(bins - rel))
            if abs(bins[idx] - rel) > tol:
                continue
            if use_friston:
                val = friston_F(h['I'], h['E'], h['fu'], h['fv'])
            else:
                val = free_energy_template(h['profile'], template)
            if np.isfinite(val):
                binned[bins[idx]].append(val)
    return binned


def matched_bin_correlation(examples, template, bin_width):
    results = {}
    for name, use_friston in [("template", False), ("friston", True)]:
        binned = bin_examples_pre(examples, template, use_friston, bin_width)
        centers, means = [], []
        for b in sorted(binned):
            if len(binned[b]) > 3:
                centers.append(b); means.append(np.mean(binned[b]))
        results[name] = (np.array(centers), np.array(means))
    ct, mt = results["template"]
    cf, mf = results["friston"]
    common = sorted(set(ct.tolist()) & set(cf.tolist()))
    if len(common) < 4:
        return None, len(common)
    mt_m = np.array([mt[list(ct).index(c)] for c in common])
    mf_m = np.array([mf[list(cf).index(c)] for c in common])
    zt = (mt_m - mt_m.mean()) / mt_m.std()
    zf = (mf_m - mf_m.mean()) / mf_m.std()
    return np.corrcoef(zt, zf)[0, 1], len(common)


def main():
    print("building disjoint template from master_profiles.npz ...")
    template_disjoint = build_disjoint_template()

    print("\nloading lifecycle tracker from lifecycle_progress.pkl ...")
    tr = tolerant_load("lifecycle_progress.pkl")
    examples = [e for e in tr.examples if e['cause'] == 'split']
    print(f"confirmed divisions: N={len(examples)}")

    print("\nmatched-bin z-scored correlation (pre-division only), "
          "F_template(disjoint template) vs. Friston F, by bin width:")
    rs = []
    for bw in BIN_WIDTHS:
        r, n_bins = matched_bin_correlation(examples, template_disjoint, bw)
        if r is None:
            print(f"  bin_width={bw}: too few matched bins (n={n_bins}), skipped")
            continue
        rs.append(r)
        print(f"  bin_width={bw}: r={r:.3f}  (n={n_bins} matched bins)")

    if rs:
        print(f"\nrange across bin widths: r={min(rs):.3f}-{max(rs):.3f}")
    else:
        print("\nno bin width produced enough matched bins")


if __name__ == "__main__":
    main()
