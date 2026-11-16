"""Unit tests for the statistics: the partial correlation and the conditional mutual information.

Each case is built so that the right answer is known from theory.  For jointly Gaussian variables,
    I(X;Y|Z) = -1/2 ln(1 - rho^2),   rho = the partial correlation of X and Y given Z,
so a partial correlation of 0.5 corresponds to 0.1438 nats.

    make test        (or, from the project root:  python -m pytest tests/test_analysis.py)
"""
import numpy as np

from conditional_mi import cmi_ksg
from partial_corr_utils import pc1_partial_corr, check_partial_corr_stability


def make_data(n, shared_var, seed=0):
    """I and E both depend on the flux Z (through fu and fv, which are nearly collinear); they also share
    an independent term of variance shared_var.  The partial correlation given Z is
    shared_var / (shared_var + 1)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    s = np.sqrt(shared_var) * rng.standard_normal(n)
    I = z + s + rng.standard_normal(n)
    E = z + s + rng.standard_normal(n)
    fu = z
    fv = -z + 0.05 * rng.standard_normal(n)
    return I, E, fu, fv, z


def test_partial_correlation_is_zero_when_flux_explains_everything():
    I, E, fu, fv, _ = make_data(20000, shared_var=0.0)
    r, n, explained = pc1_partial_corr(I, E, fu, fv)
    assert n == 20000
    assert abs(r) < 0.03
    assert explained > 0.99          # the two fluxes are nearly collinear


def test_partial_correlation_recovers_a_known_value():
    I, E, fu, fv, _ = make_data(20000, shared_var=1.0)   # theory: 1 / (1 + 1) = 0.5
    r, _, _ = pc1_partial_corr(I, E, fu, fv)
    assert abs(r - 0.5) < 0.03


def test_partial_correlation_ignores_nonfinite_rows():
    I, E, fu, fv, _ = make_data(2000, shared_var=1.0)
    I2 = I.copy()
    I2[:10] = np.nan
    _, n, _ = pc1_partial_corr(I2, E, fu, fv)
    assert n == 1990


def test_stability_check_runs_and_returns_five_values():
    I, E, fu, fv, _ = make_data(2000, shared_var=1.0)
    out = check_partial_corr_stability(I, E, fu, fv)
    assert len(out) == 5


def test_cmi_is_near_zero_when_conditionally_independent():
    I, E, _, _, z = make_data(3000, shared_var=0.0)
    assert abs(cmi_ksg(I, E, z, k=5)) < 0.03


def test_cmi_matches_the_gaussian_formula():
    I, E, _, _, z = make_data(3000, shared_var=1.0)       # rho = 0.5
    expected = -0.5 * np.log(1 - 0.5**2)                   # 0.1438 nats
    assert abs(cmi_ksg(I, E, z, k=5) - expected) < 0.04


def test_cmi_accepts_two_dimensional_conditioning():
    I, E, fu, fv, _ = make_data(3000, shared_var=1.0)
    est = cmi_ksg(I, E, np.column_stack([fu, fv]), k=5)
    assert 0.05 < est < 0.25
