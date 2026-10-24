"""
Life-cycle comparison figure (main text §5.4, the 1024 x 1024 panel): the template-deviation free energy and the
Friston free energy, aligned to each confirmed division and averaged, with their matched-bin z-scored correlation.

The Friston generative model (mu_p, sigma_p^2, A, b, Sigma, slope, intercept, sigma_q^2) is refit live from the
Ensemble 1 worker files (fe_model.py). Uses lifecycle_progress.pkl (from run_lifecycle.py) and fe_template_1024.npz
(the radial template, built from v-profiles alone by build_template_from_master.py); no new simulation.

Requires: lifecycle_progress.pkl, fe_template_1024.npz, ensemble_merged_seed1000_worker{0..11}.npz.
Run from the directory holding these files.

Usage:
    python3 rebuild_fe_comparison_1024.py
"""
import pickle
import numpy as np

from run_lifecycle import LifecycleTracker

# Work around numpy-version pickle incompatibilities: lifecycle_progress.pkl
# carries the solver's RNG state (inside self.sim), and the numpy version
# that wrote it isn't the one reading it here. We don't need the RNG itself
# (nothing downstream draws random numbers -- we only read tr.examples), so
# rather than chase each incompatibility in numpy's BitGenerator pickling
# format one error at a time, make the unpickler tolerate a failed
# __setstate__ on any single object and move on, instead of aborting the
# whole load. BitGenerator is a Cython extension type and can't be
# monkeypatched directly, so this uses pickle's pure-Python Unpickler
# (which is overridable) instead of the default C-accelerated one.
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


# Also patch the BitGenerator constructor helper itself: some numpy versions'
# __reduce__ passes the class object where the unpickler expects a string
# name, raising "... is not a known BitGenerator module" before __setstate__
# even runs. Harmless to patch regardless of which numpy version is active.
import numpy.random._pickle as _np_random_pickle
_orig_bitgen_ctor = _np_random_pickle.__bit_generator_ctor
def _patched_bitgen_ctor(bit_generator_name="MT19937"):
    if isinstance(bit_generator_name, type):
        bit_generator_name = bit_generator_name.__name__
    elif not isinstance(bit_generator_name, str):
        # Already a constructed BitGenerator instance (happens when the
        # outer Generator's reduce call is handed the inner BitGenerator
        # object after ITS reduce/build already ran) -- nothing to
        # construct, just pass it through as-is.
        return bit_generator_name
    return _orig_bitgen_ctor(bit_generator_name)
_np_random_pickle.__bit_generator_ctor = _patched_bitgen_ctor

# The constructor for the outer Generator object has the SAME problem, but
# in a more stubborn form: __generator_ctor's own signature is
#   def __generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=__bit_generator_ctor):
# -- its second parameter's default value is bound to the ORIGINAL
# __bit_generator_ctor function object at the moment numpy.random._pickle
# was first imported, permanently, regardless of what we assign to the
# module attribute afterward. Patching the module attribute (above) fixes
# calls pickle dispatches directly via a fresh GLOBAL/getattr lookup, but
# has zero effect on this internal pre-bound reference. So replace the
# whole function instead, routing it through our already-fixed
# _patched_bitgen_ctor explicitly rather than relying on its own default.
def _patched_generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=None):
    bg = _patched_bitgen_ctor(bit_generator_name)
    return np.random.Generator(bg)
_np_random_pickle.__generator_ctor = _patched_generator_ctor

# Generative-model constants (main text §5.2), refit live from the Ensemble 1 worker files (see fe_model.py).
from fe_model import load_constants
MU_P, SIGMA_P2, A, B, SIGMA, SLOPE, INTERCEPT, SIGMA_Q2 = load_constants()

PRE_BINS = np.arange(-150, 5, 5)
POST_BINS = np.arange(0, 155, 5)


def free_energy_template(profile, template):
    good = np.isfinite(profile) & np.isfinite(template)
    if not good.any():
        return np.nan
    return np.mean((profile[good] - template[good]) ** 2)


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


def bin_examples(examples, is_pre, template, use_friston):
    bins = PRE_BINS if is_pre else POST_BINS
    binned = {b: [] for b in bins}
    for e in examples:
        if e['cause'] != 'split' or not e['history']:
            continue
        split_t = e['death'] if is_pre else e['birth']
        for h in e['history']:
            rel = h['t'] - split_t
            if rel < bins[0] - 3 or rel > bins[-1] + 3:
                continue
            idx = np.argmin(np.abs(bins - rel))
            if abs(bins[idx] - rel) > 3:
                continue
            if use_friston:
                val = friston_F(h['I'], h['E'], h['fu'], h['fv'])
            else:
                val = free_energy_template(h['profile'], template)
            if np.isfinite(val):
                binned[bins[idx]].append(val)
    return binned


def main():
    tr = tolerant_load("lifecycle_progress.pkl")
    examples = [e for e in tr.examples if e['cause'] == 'split']
    print(f"confirmed divisions (cause=='split'): N={len(examples)}")

    d = np.load("fe_template_1024.npz")
    template = d["template"]

    results = {}
    for name, use_friston in [("template", False), ("friston", True)]:
        pre = bin_examples(examples, True, template, use_friston)
        post = bin_examples(examples, False, template, use_friston)
        centers, means, sems = [], [], []
        for b in sorted(pre):
            if len(pre[b]) > 3:
                centers.append(b); means.append(np.mean(pre[b])); sems.append(np.std(pre[b]) / np.sqrt(len(pre[b])))
        for b in sorted(post):
            if len(post[b]) > 3:
                centers.append(b); means.append(np.mean(post[b])); sems.append(np.std(post[b]) / np.sqrt(len(post[b])))
        results[name] = (np.array(centers), np.array(means), np.array(sems))
        print(f"\n{name}: {len(centers)} bins with >3 observations")
        print(f"  value range: {np.min(means):.5g} to {np.max(means):.5g}")
        centers_arr, means_arr = np.array(centers), np.array(means)
        order = np.argsort(centers_arr)
        c_sorted, m_sorted = centers_arr[order], means_arr[order]
        pre_mask = c_sorted < 0
        post_mask = c_sorted >= 0
        c_pre, m_pre = c_sorted[pre_mask], m_sorted[pre_mask]
        c_post, m_post = c_sorted[post_mask], m_sorted[post_mask]
        print(f"  t={c_sorted[0]:.0f} (earliest): {m_sorted[0]:.5g}")
        if len(m_pre):
            i_peak = np.argmax(m_pre) if name == "template" else np.argmax(m_pre)
            print(f"  pre-division peak at t={c_pre[i_peak]:.0f}: {m_pre[i_peak]:.5g}")
        if len(m_post):
            i_trough = np.argmin(m_post)
            print(f"  post-division trough at t={c_post[i_trough]:.0f}: {m_post[i_trough]:.5g}")
            # plateau: mean over the first ~45 post-division bins
            plateau_mask = (c_post >= 0) & (c_post <= 45)
            if plateau_mask.any():
                print(f"  plateau mean (t=0-45): {m_post[plateau_mask].mean():.5g}")
        print(f"  t={c_sorted[-1]:.0f} (latest): {m_sorted[-1]:.5g}")

    # matched-bin correlation (z-scored), same construction as the paper text
    ct, mt, _ = results["template"]
    cf, mf, _ = results["friston"]
    common = sorted(set(ct.tolist()) & set(cf.tolist()))
    mt_m = np.array([mt[list(ct).index(c)] for c in common])
    mf_m = np.array([mf[list(cf).index(c)] for c in common])
    zt = (mt_m - mt_m.mean()) / mt_m.std()
    zf = (mf_m - mf_m.mean()) / mf_m.std()
    r = np.corrcoef(zt, zf)[0, 1]
    print(f"\nmatched bins: n={len(common)}  z-scored correlation r={r:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(ct, mt, 'o-', color="steelblue", label=r"$\mathcal{F}_{\mathrm{template}}$", markersize=3)
    ax1.set_xlabel("time relative to division")
    ax1.set_ylabel(r"$\mathcal{F}_{\mathrm{template}}$", color="steelblue")
    ax1.tick_params(axis='y', labelcolor="steelblue")
    ax1.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(cf, mf, 's-', color="crimson", label="Friston $\\mathcal{F}$", markersize=3)
    ax2.set_ylabel("Friston $\\mathcal{F}$", color="crimson")
    ax2.tick_params(axis='y', labelcolor="crimson")

    ax1.set_title(f"$1024\\times1024$, $N={len(examples)}$ confirmed divisions, "
                  f"$z$-scored correlation $r={r:.3f}$")
    fig.tight_layout()
    fig.savefig("fe_comparison_1024_rebuilt.png", dpi=130)
    print("\nsaved fe_comparison_1024_rebuilt.png (send this back)")


if __name__ == "__main__":
    main()
