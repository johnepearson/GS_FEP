"""
Rebuilds the paper's \S7.4 instance-level check: every tracked
organism-instance (confirmed divisions AND still-active lineages),
pooled with no relative-time binning or averaging, correlating
F_template against the Friston F directly -- using the REFIT Friston
generative model from the merge-corrected primary ensemble
(N=555,572), in place of the stale pre-merge-fix model this check was
originally run with. (Reported in the paper as "r=0.783"; the existing
script's own docstring/prints call it "r=0.777" -- this run settles
which is current.)

Based directly on compare_fe_instance_level.py (already in the
codebase), with two differences:
  1. Uses the hardcoded REFIT constants from rebuild_fig5_and_fe_model.py
     instead of refitting live from --ensemble, since the only
     ensemble_merged_final.npz files on hand are stale (pre-merge-fix,
     wrong N) -- see rebuild_fig5_and_fe_model.py's own header for why.
  2. Reuses the same tolerant-unpickling workaround as
     rebuild_fe_comparison_1024.py for lifecycle_progress.pkl's
     numpy-version-incompatible embedded RNG state.

Requires (both already present, same as the Figure 13 rebuild):
  - lifecycle_progress.pkl
  - fe_template_1024.npz

Run this from local_package/.

Usage:
    python3 rebuild_instance_level_check.py
"""
import pickle
import sys
import numpy as np

# --- same tolerant-unpickling workaround as rebuild_fe_comparison_1024.py ---
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

# lifecycle_progress.pkl refers to __main__.LifecycleTracker (it was
# pickled from inside run_lifecycle.py run as __main__) -- alias it in,
# same fix compare_fe_instance_level.py already uses.
import run_lifecycle
from run_lifecycle import LifecycleTracker
sys.modules["__main__"].LifecycleTracker = LifecycleTracker

# --- refit generative model constants (rebuild_fig5_and_fe_model.py's
# real output, merge-corrected primary ensemble, N=555,572) ---
MU_P = 0.0387
SIGMA_P2 = 1.611e-04
A = np.array([2.369e-04, -7.994e-05])
B = np.array([-8.080e-05, 2.718e-05])
SIGMA = np.array([[3.392e-10, -1.098e-10], [-1.098e-10, 3.676e-11]])
SLOPE = -0.0696
INTERCEPT = 0.0586
SIGMA_Q2 = 1.514e-04
# OVERRIDE: the constants above are the stale spacing-32 fit; refit live (spacing 24) from the seed-1000 ensemble.
from fe_model import load_constants
MU_P, SIGMA_P2, A, B, SIGMA, SLOPE, INTERCEPT, SIGMA_Q2 = load_constants()

MAX_R, N_BINS = 80, 20


def kl_term(mu_q, sigma_q2, mu_p, sigma_p2):
    sigma_q, sigma_p = np.sqrt(sigma_q2), np.sqrt(sigma_p2)
    return np.log(sigma_p / sigma_q) + (sigma_q2 + (mu_q - mu_p) ** 2) / (2 * sigma_p2) - 0.5


def accuracy_term(o, mu_q, sigma_q2, A, b, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    _, logdet = np.linalg.slogdet(Sigma)
    const = 0.5 * (2 * np.log(2 * np.pi) + logdet)
    pred = np.outer(mu_q, A) + b
    resid = o - pred
    quad = np.einsum('ni,ij,nj->n', resid, Sigma_inv, resid)
    extra = sigma_q2 * (A @ Sigma_inv @ A)
    return const + 0.5 * (quad + extra)


def f_template(profile, template):
    valid = np.isfinite(profile) & np.isfinite(template)
    if valid.sum() == 0:
        return np.nan
    return np.mean((profile[valid] - template[valid]) ** 2)


def size_proxy(profile):
    if not np.isfinite(profile).any():
        return np.nan
    bins = np.linspace(0, MAX_R, N_BINS + 1)
    mids = 0.5 * (bins[:-1] + bins[1:])
    valid = np.isfinite(profile)
    idx = np.argmin(np.where(valid, profile, np.inf))
    return mids[idx] if valid[idx] else np.nan


def partial_corr_2var(x, y, controls):
    n = len(x)
    B_ = np.column_stack([np.ones(n), controls])

    def resid(v):
        beta, *_ = np.linalg.lstsq(B_, v, rcond=None)
        return v - B_ @ beta

    return np.corrcoef(resid(x), resid(y))[0, 1]


def bootstrap_ci(x, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = np.corrcoef(x[idx], y[idx])[0, 1]
    return np.percentile(boot, [2.5, 97.5])


def collect_instances(tr):
    rows = []
    for ex in tr.examples:
        birth = ex.get("birth", 0)
        for rec in ex.get("history", []):
            rows.append(dict(rec, age=rec["t"] - birth))
    for lid, hist in tr.history.items():
        birth = tr.birth_time.get(lid, 0)
        for rec in hist:
            rows.append(dict(rec, age=rec["t"] - birth))
    return rows


def main():
    print("loading lifecycle tracker from lifecycle_progress.pkl ...")
    tr = tolerant_load("lifecycle_progress.pkl")
    print(f"  t={tr.t}, {len(tr.examples)} completed lineages, "
          f"{len(tr.history)} still-active lineages")

    rows = collect_instances(tr)
    print(f"  pooled {len(rows)} per-frame organism-instance records")

    td = np.load("fe_template_1024.npz")
    template = td["template"]
    print(f"loaded template from fe_template_1024.npz (N={int(td['N'])} organisms)")

    I_all, E_all, fu_all, fv_all, age_all, size_all, ft_all = [], [], [], [], [], [], []
    for rec in rows:
        I, E, fu, fv, profile, age = rec["I"], rec["E"], rec["fu"], rec["fv"], rec["profile"], rec["age"]
        if not (np.isfinite(I) and np.isfinite(E) and np.isfinite(fu) and np.isfinite(fv)):
            continue
        ft = f_template(profile, template)
        sz = size_proxy(profile)
        if not (np.isfinite(ft) and np.isfinite(sz)):
            continue
        I_all.append(I); E_all.append(E); fu_all.append(fu); fv_all.append(fv)
        age_all.append(age); size_all.append(sz); ft_all.append(ft)

    I_all = np.array(I_all); E_all = np.array(E_all)
    fu_all = np.array(fu_all); fv_all = np.array(fv_all)
    age_all = np.array(age_all); size_all = np.array(size_all); ft_all = np.array(ft_all)
    N = len(I_all)
    print(f"\n{N} instances have valid I, E, flux, profile, and size-proxy measurements")

    mu_q = SLOPE * I_all + INTERCEPT
    KL = kl_term(mu_q, SIGMA_Q2, MU_P, SIGMA_P2)
    o = np.column_stack([fu_all, fv_all])
    acc = accuracy_term(o, mu_q, SIGMA_Q2, A, B, SIGMA)
    F_friston = KL + acc

    r = np.corrcoef(ft_all, F_friston)[0, 1]
    lo, hi = bootstrap_ci(ft_all, F_friston)
    print(f"\nInstance-level correlation, F_template vs. Friston F:")
    print(f"  r = {r:.4f}  (bootstrap 95% CI [{lo:.4f}, {hi:.4f}])")

    r_age = partial_corr_2var(ft_all, F_friston, age_all.reshape(-1, 1))
    r_size = partial_corr_2var(ft_all, F_friston, size_all.reshape(-1, 1))
    r_both = partial_corr_2var(ft_all, F_friston, np.column_stack([age_all, size_all]))
    print(f"\nControlling for age alone:          r = {r_age:.4f}")
    print(f"Controlling for size proxy alone:   r = {r_size:.4f}")
    print(f"Controlling for both:               r = {r_both:.4f}")

    r_ft_age = np.corrcoef(ft_all, age_all)[0, 1]
    r_friston_age = np.corrcoef(F_friston, age_all)[0, 1]
    r_ft_size = np.corrcoef(ft_all, size_all)[0, 1]
    r_friston_size = np.corrcoef(F_friston, size_all)[0, 1]
    r_age_size = np.corrcoef(age_all, size_all)[0, 1]
    print(f"\nConfound check (correlation of age and size proxy with each quantity):")
    print(f"  F_template vs age:  {r_ft_age:.4f}   Friston F vs age:  {r_friston_age:.4f}")
    print(f"  F_template vs size: {r_ft_size:.4f}   Friston F vs size: {r_friston_size:.4f}")
    print(f"  age vs size proxy:  {r_age_size:.4f}")


if __name__ == "__main__":
    main()
