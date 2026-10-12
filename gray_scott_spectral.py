"""
Pseudo-spectral Gray-Scott solver on a periodic domain.

Diffusion is linear, so in Fourier space it is solved EXACTLY via an
integrating factor -- no stability restriction on dt from diffusion at
all, regardless of grid resolution. Only the (explicit) reaction term
retains any stability constraint, and that constraint depends on the
reaction rate constants (F, k), not on dx. This is what makes
refining the mesh cheap: cost scales as N^2 log N (the FFT), not the
4th-power penalty of an explicit finite-difference scheme.

Scheme: standard exponential Euler / ETDRK1.
  du/dt = Du*Lap(u) + Ru(u,v),   Ru = -u*v^2 + F*(1-u)
  dv/dt = Dv*Lap(v) + Rv(u,v),   Rv =  u*v^2 - (F+k)*v

In Fourier space, Lap -> -k^2. For each mode:
  u_hat(t+dt) = E_u * u_hat(t) + dt * phi1(-Du*k^2*dt) * Ru_hat(t)
where E_u = exp(-Du*k^2*dt) and phi1(z) = (exp(z)-1)/z, phi1(0)=1.
This is exact for the linear part and first-order (explicit Euler) for
the nonlinear reaction part.

Uses scipy.fft's real FFT (rfft2/irfft2) with workers=-1 (use all
available CPU cores) rather than numpy.fft, since scipy's pocketfft
backend supports multi-threading -- worthwhile on a multi-core
laptop. Falls back to numpy.fft automatically if scipy is unavailable.
"""
import numpy as np

try:
    from scipy.fft import rfft2, irfft2
    import os

    _WORKERS = int(os.environ.get("GS_FFT_WORKERS", os.cpu_count() or 1))

    def _rfft2(x):
        return rfft2(x, workers=_WORKERS)

    def _irfft2(x, s):
        return irfft2(x, s=s, workers=_WORKERS)
except ImportError:
    def _rfft2(x):
        return np.fft.rfft2(x)

    def _irfft2(x, s):
        return np.fft.irfft2(x, s=s)


class PseudoSpectralGS:
    def __init__(self, n=256, L=2.5, Du=2e-5, Dv=1e-5, F=0.018, k=0.055, dt=1.0):
        self.n = n
        self.L = L
        self.dx = L / n
        self.Du, self.Dv, self.F, self.k, self.dt = Du, Dv, F, k, dt

        self.u = np.ones((n, n), dtype=np.float64)
        self.v = np.zeros((n, n), dtype=np.float64)
        self.step_count = 0
        self.time = 0.0

        kx = 2 * np.pi * np.fft.fftfreq(n, d=self.dx)
        ky = 2 * np.pi * np.fft.rfftfreq(n, d=self.dx)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = KX**2 + KY**2

        self._precompute_factors()

    def _precompute_factors(self):
        dt = self.dt
        Lu = -self.Du * self.K2
        Lv = -self.Dv * self.K2
        self.Eu = np.exp(Lu * dt)
        self.Ev = np.exp(Lv * dt)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi1_u = (self.Eu - 1.0) / (Lu * dt)
            phi1_v = (self.Ev - 1.0) / (Lv * dt)
        zero_mask = (self.K2 == 0)
        phi1_u[zero_mask] = 1.0
        phi1_v[zero_mask] = 1.0
        self.phi1_u = phi1_u
        self.phi1_v = phi1_v

    def from_checkpoint_fields(self, u, v):
        self.u = u.copy()
        self.v = v.copy()
        return self

    def step(self):
        u, v = self.u, self.v
        Ru = -u * v * v + self.F * (1.0 - u)
        Rv = u * v * v - (self.F + self.k) * v

        u_hat = _rfft2(u)
        v_hat = _rfft2(v)
        Ru_hat = _rfft2(Ru)
        Rv_hat = _rfft2(Rv)

        u_hat_new = self.Eu * u_hat + self.dt * self.phi1_u * Ru_hat
        v_hat_new = self.Ev * v_hat + self.dt * self.phi1_v * Rv_hat

        self.u = _irfft2(u_hat_new, s=(self.n, self.n))
        self.v = _irfft2(v_hat_new, s=(self.n, self.n))
        self.step_count += 1
        self.time += self.dt

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def save_checkpoint(self, path):
        np.savez(path, u=self.u, v=self.v, step_count=self.step_count,
                 time=self.time, n=self.n, L=self.L, Du=self.Du, Dv=self.Dv,
                 F=self.F, k=self.k, dt=self.dt)

    @classmethod
    def from_checkpoint(cls, path, dt=None):
        d = np.load(path)
        sim = cls(n=int(d["n"]), L=float(d["L"]), Du=float(d["Du"]), Dv=float(d["Dv"]),
                  F=float(d["F"]), k=float(d["k"]), dt=float(d["dt"]) if dt is None else dt)
        sim.u = d["u"].copy()
        sim.v = d["v"].copy()
        sim.step_count = int(d["step_count"])
        sim.time = float(d["time"]) if "time" in d else float(sim.step_count)
        return sim
