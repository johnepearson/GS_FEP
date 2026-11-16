"""Unit tests for the pseudo-spectral Gray-Scott solver (small grids, a fraction of a second).

Each test has a known answer that does not depend on any simulation run:
  - the uniform state (u, v) = (1, 0) is a fixed point;
  - pure diffusion of a single cosine mode decays by exactly exp(-D k^2 dt);
  - a uniform state follows the reaction terms alone (explicit Euler for the k = 0 mode);
  - shifting the initial field shifts the result (translation invariance);
  - a checkpoint saves and loads without change;
  - the C solver gives the same answer as the Python one (skipped when libgs_spectral.so is not built).

    make test        (or, from the project root:  python -m pytest tests/test_solver.py)
"""
import numpy as np
import pytest

from gray_scott_spectral import PseudoSpectralGS

N = 32
L = 2.5


def make(**kw):
    return PseudoSpectralGS(n=N, L=L, **kw)


def test_uniform_state_is_a_fixed_point():
    sim = make()
    sim.run(10)
    assert np.allclose(sim.u, 1.0, atol=1e-14)
    assert np.allclose(sim.v, 0.0, atol=1e-14)
    assert sim.step_count == 10


def test_pure_diffusion_of_one_mode_decays_exactly():
    # With u = 0 and F = k = 0 the reaction terms vanish, so v only diffuses.
    sim = make(F=0.0, k=0.0, Dv=1e-3, dt=1.0)
    x = np.arange(N) * (L / N)
    mode = np.cos(2 * np.pi * x / L)[:, None] * np.ones((1, N))
    sim.from_checkpoint_fields(np.zeros((N, N)), mode)
    sim.step()
    kk = 2 * np.pi / L
    expected = np.exp(-1e-3 * kk**2 * 1.0)
    assert np.allclose(sim.v, expected * mode, atol=1e-12)


def test_uniform_state_follows_the_reaction_terms_alone():
    # For a uniform field only the k = 0 mode is present, where the exponential Euler step is
    # u + dt*R(u, v).
    u0, v0 = 0.5, 0.25
    sim = make()
    sim.from_checkpoint_fields(np.full((N, N), u0), np.full((N, N), v0))
    sim.step()
    Ru = -u0 * v0**2 + sim.F * (1 - u0)
    Rv = u0 * v0**2 - (sim.F + sim.k) * v0
    assert np.allclose(sim.u, u0 + sim.dt * Ru, atol=1e-13)
    assert np.allclose(sim.v, v0 + sim.dt * Rv, atol=1e-13)


def test_translation_invariance():
    rng = np.random.default_rng(1)
    u0 = 1.0 - 0.1 * rng.random((N, N))
    v0 = 0.1 * rng.random((N, N))
    a, b = make(), make()
    a.from_checkpoint_fields(u0, v0)
    b.from_checkpoint_fields(np.roll(u0, (5, 7), axis=(0, 1)), np.roll(v0, (5, 7), axis=(0, 1)))
    a.run(5)
    b.run(5)
    assert np.allclose(np.roll(a.u, (5, 7), axis=(0, 1)), b.u, atol=1e-12)
    assert np.allclose(np.roll(a.v, (5, 7), axis=(0, 1)), b.v, atol=1e-12)


def test_checkpoint_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    sim = make()
    sim.from_checkpoint_fields(1.0 - 0.1 * rng.random((N, N)), 0.1 * rng.random((N, N)))
    sim.run(3)
    path = tmp_path / "ckpt.npz"
    sim.save_checkpoint(path)
    back = PseudoSpectralGS.from_checkpoint(path)
    assert back.n == N and back.step_count == 3
    assert np.array_equal(back.u, sim.u) and np.array_equal(back.v, sim.v)
    assert (back.F, back.k, back.Du, back.Dv, back.dt) == (sim.F, sim.k, sim.Du, sim.Dv, sim.dt)


def test_c_solver_matches_python_solver():
    try:
        from gray_scott_spectral_c import PseudoSpectralGSFast
    except OSError as err:  # libgs_spectral.so is missing or was built for another machine
        pytest.skip(f"C solver not available: {err}")
    rng = np.random.default_rng(3)
    u0 = 1.0 - 0.1 * rng.random((N, N))
    v0 = 0.1 * rng.random((N, N))
    py = make()
    py.from_checkpoint_fields(u0, v0)
    c = PseudoSpectralGSFast(n=N, L=L)
    c.from_checkpoint_fields(u0, v0)
    py.run(20)
    c.run(20)
    cu, cv = c.get_uv()
    assert np.allclose(cu, py.u, atol=1e-10)
    assert np.allclose(cv, py.v, atol=1e-10)
