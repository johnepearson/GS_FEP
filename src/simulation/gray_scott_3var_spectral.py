"""
Pseudo-spectral solver for the three-variable chemical-memory Gray-Scott
model (see gray_scott_3var.py for the finite-difference version and the
model derivation).

Two distinct sources of numerical stiffness are present, and they are
diagonal in different bases, so they are handled separately in a
first-order (Lie/Godunov) operator split each step:

  1. Diffusion (Du, Dv, Dw) is diagonal in Fourier space and is handled
     there exactly via the ETD integrating factor, with no timestep
     restriction from diffusion at all -- exactly as in the plain
     two-variable pseudo-spectral solver (gray_scott_spectral.py).

  2. The stiff local reverse-reaction term (-w/eps^2) is diagonal in
     real space (it is purely local, no spatial coupling) and is
     handled there via the same semi-implicit elementwise update
     already used in the finite-difference version: the linear reverse
     term is evaluated at the new sub-step, which is an elementwise
     division needing no matrix solve.

Per step: react fully in real space (with w's stiff term implicit,
matching the finite-difference scheme's own convention of using the
new w for the stiff reverse-reaction contribution to dv/dt, and the old
w for the non-stiff catalytic contribution) -- then diffuse exactly in
Fourier space. This is first-order in time overall, matching the
accuracy already documented for the two-variable pseudo-spectral scheme
(the reaction part there is also explicit/first-order); no new
accuracy-order inconsistency is introduced.
"""
import os
import numpy as np
import scipy.fft

FFT_WORKERS = int(os.environ.get("GS_FFT_WORKERS", "1"))


class ThreeVarPseudoSpectralGS:
    def __init__(self, n=1024, L=2.5, Du=2e-5, Dv=1e-5, Dw=1e-5,
                 F=0.018, k=0.055, eps=0.1, dt=1.0):
        self.n = n
        self.L = L
        self.dx = L / n
        self.Du, self.Dv, self.Dw = Du, Dv, Dw
        self.F, self.k, self.eps, self.dt = F, k, eps, dt

        self.u = np.ones((n, n), dtype=np.float64)
        self.v = np.zeros((n, n), dtype=np.float64)
        self.w = np.zeros((n, n), dtype=np.float64)
        self.time = 0.0

        freqs_x = 2 * np.pi * np.fft.fftfreq(n, d=self.dx)
        freqs_y = 2 * np.pi * np.fft.rfftfreq(n, d=self.dx)
        kx, ky = np.meshgrid(freqs_x, freqs_y, indexing='ij')
        kappa2 = kx**2 + ky**2
        self.kappa2 = kappa2
        self.E_u = np.exp(-Du * kappa2 * dt)
        self.E_v = np.exp(-Dv * kappa2 * dt)
        self.E_w = np.exp(-Dw * kappa2 * dt)

        def phi1(z):
            out = np.ones_like(z)
            nonzero = z != 0
            out[nonzero] = (np.exp(z[nonzero]) - 1.0) / z[nonzero]
            return out

        self.phi1_u = phi1(-Du * kappa2 * dt)
        self.phi1_v = phi1(-Dv * kappa2 * dt)
        self.phi1_w = phi1(-Dw * kappa2 * dt)

    @classmethod
    def from_gs_checkpoint(cls, path, eps=0.1, dt=1.0, Dw=None):
        d = np.load(path)
        Du, Dv = float(d["Du"]), float(d["Dv"])
        if Dw is None:
            Dw = Dv
        sim = cls(n=int(d["n"]), L=float(d["L"]), Du=Du, Dv=Dv, Dw=Dw,
                  F=float(d["F"]), k=float(d["k"]), eps=eps, dt=dt)
        sim.u = d["u"].copy()
        sim.v = d["v"].copy()
        sim.w = eps * sim.v**2  # quasi-steady-state guess, avoids an initial shock
        sim.time = 0.0
        return sim

    @classmethod
    def from_checkpoint(cls, path):
        d = np.load(path)
        sim = cls(n=int(d["n"]), L=float(d["L"]), Du=float(d["Du"]), Dv=float(d["Dv"]),
                  Dw=float(d["Dw"]), F=float(d["F"]), k=float(d["k"]),
                  eps=float(d["eps"]), dt=float(d["dt"]))
        sim.u = d["u"].copy()
        sim.v = d["v"].copy()
        sim.w = d["w"].copy()
        sim.time = float(d["time"])
        return sim

    def save_checkpoint(self, path):
        np.savez(path, u=self.u, v=self.v, w=self.w, n=self.n, L=self.L,
                  Du=self.Du, Dv=self.Dv, Dw=self.Dw, F=self.F, k=self.k,
                  eps=self.eps, dt=self.dt, time=self.time)

    def _step(self):
        u, v, w = self.u, self.v, self.w
        eps, dt, F, k = self.eps, self.dt, self.F, self.k

        # Stiff local reaction-only update for w (no diffusion folded in
        # here). w's diffusion is instead treated exactly via the ETD step
        # below (E_w / phi1_w), same as u and v -- unlike an explicit
        # diffusion treatment, this carries no timestep restriction at any
        # resolution, which is the entire point of using this solver.
        # (An explicit alternative that adds the diffusion term directly
        # inside this division, mirroring the finite-difference formula,
        # gives closer one-step agreement with that scheme at n=256, but
        # is unconditionally unstable at dt=1 once n=1024 pushes kappa_max^2
        # high enough -- Dw*kappa_max^2*dt ~ 16.6 there, far past the ~2
        # stability limit for explicit diffusion. That approach was tried
        # and rejected for exactly this reason.)
        w_react = (eps**2 * w + dt * eps * v**2) / (eps**2 + dt)
        R_w = (w_react - w) / dt  # effective rate, for the ETD update below

        R_u = -(1.0/eps) * u * w + F * (1.0 - u)
        R_v = ((1.0/eps) * u * w - (2.0/eps) * v**2
               + (2.0/eps**2) * w_react - (F + k) * v)

        u_hat = scipy.fft.rfft2(u, workers=FFT_WORKERS)
        v_hat = scipy.fft.rfft2(v, workers=FFT_WORKERS)
        w_hat = scipy.fft.rfft2(w, workers=FFT_WORKERS)
        Ru_hat = scipy.fft.rfft2(R_u, workers=FFT_WORKERS)
        Rv_hat = scipy.fft.rfft2(R_v, workers=FFT_WORKERS)
        Rw_hat = scipy.fft.rfft2(R_w, workers=FFT_WORKERS)

        u_hat_new = self.E_u * u_hat + dt * self.phi1_u * Ru_hat
        v_hat_new = self.E_v * v_hat + dt * self.phi1_v * Rv_hat
        w_hat_new = self.E_w * w_hat + dt * self.phi1_w * Rw_hat

        self.u = scipy.fft.irfft2(u_hat_new, s=(self.n, self.n), workers=FFT_WORKERS)
        self.v = scipy.fft.irfft2(v_hat_new, s=(self.n, self.n), workers=FFT_WORKERS)
        self.w = scipy.fft.irfft2(w_hat_new, s=(self.n, self.n), workers=FFT_WORKERS)
        self.time += dt

    def run(self, n_steps):
        for _ in range(n_steps):
            self._step()
