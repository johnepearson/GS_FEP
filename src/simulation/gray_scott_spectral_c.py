"""
Python wrapper (ctypes) around the C pseudo-spectral solver
(gs_spectral_c.c / libgs_spectral.so), which reduces the per-step FFT
count from 6 to 3 (see the C file's docstring for the derivation) and
avoids numpy's per-step temporary-array allocation overhead.

Drop-in replacement for PseudoSpectralGS (gray_scott_spectral.py):
same constructor signature, same .u/.v/.time/.n/.L/.Du/.Dv/.F/.k/.dt
attributes (computed lazily from the persistent Fourier-space state on
first access after a run, then cached), same .run()/.step(),
from_checkpoint()/save_checkpoint() read and write the identical
checkpoint format, so it can be swapped into any of the existing
run_*.py scripts by changing only the import line.

Build (already done once for this package; rerun if gs_spectral_c.c
changes or on a machine without libgs_spectral.so):
    gcc -O3 -march=native -fPIC -shared -o libgs_spectral.so \
        gs_spectral_c.c -lfftw3 -lm
"""
import ctypes
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_here, "libgs_spectral.so"))

_lib.gs_create.restype = ctypes.c_void_p
_lib.gs_create.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_int]
_lib.gs_destroy.argtypes = [ctypes.c_void_p]
_lib.gs_set_field.argtypes = [ctypes.c_void_p,
                               np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
                               np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")]
_lib.gs_get_field.argtypes = [ctypes.c_void_p,
                               np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
                               np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")]
_lib.gs_run.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.gs_get_time.restype = ctypes.c_double
_lib.gs_get_time.argtypes = [ctypes.c_void_p]
_lib.gs_get_step_count.restype = ctypes.c_long
_lib.gs_get_step_count.argtypes = [ctypes.c_void_p]
_lib.gs_set_time.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_long]


class PseudoSpectralGSFast:
    def __init__(self, n=256, L=2.5, Du=2e-5, Dv=1e-5, F=0.018, k=0.055, dt=1.0, dealias=False):
        self.n, self.L, self.Du, self.Dv, self.F, self.k, self.dt = n, L, Du, Dv, F, k, dt
        self.dealias = bool(dealias)
        self.dx = L / n
        self._handle = _lib.gs_create(n, L, Du, Dv, F, k, dt, int(self.dealias))
        u0 = np.ones((n, n), dtype=np.float64)
        v0 = np.zeros((n, n), dtype=np.float64)
        _lib.gs_set_field(self._handle, np.ascontiguousarray(u0), np.ascontiguousarray(v0))
        self._dirty = False  # True once .u/.v setters are used and need re-pushing to C state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            _lib.gs_destroy(handle)

    def __getstate__(self):
        # self._handle is a raw C pointer -- meaningless (or worse, a
        # dangling/reused address) in any process other than the one
        # that created it, so it must never be pickled directly.
        # run_lifecycle.py pickles its whole LifecycleTracker
        # (including .sim) for --resume, so this matters in practice,
        # not just in principle. Serialize as plain field data instead
        # and rebuild the C simulation object on unpickling.
        u, v = self.get_uv()
        return dict(n=self.n, L=self.L, Du=self.Du, Dv=self.Dv, F=self.F, k=self.k,
                    dt=self.dt, dealias=self.dealias, u=u, v=v, time=self.time,
                    step_count=self.step_count)

    def __setstate__(self, state):
        self.n, self.L, self.Du, self.Dv = state["n"], state["L"], state["Du"], state["Dv"]
        self.F, self.k, self.dt = state["F"], state["k"], state["dt"]
        self.dealias = bool(state.get("dealias", False))
        self.dx = self.L / self.n
        self._handle = _lib.gs_create(self.n, self.L, self.Du, self.Dv, self.F, self.k,
                                        self.dt, int(self.dealias))
        _lib.gs_set_field(self._handle, np.ascontiguousarray(state["u"], dtype=np.float64),
                           np.ascontiguousarray(state["v"], dtype=np.float64))
        _lib.gs_set_time(self._handle, float(state["time"]), int(state["step_count"]))
        self._dirty = False

    def from_checkpoint_fields(self, u, v):
        _lib.gs_set_field(self._handle, np.ascontiguousarray(u, dtype=np.float64),
                           np.ascontiguousarray(v, dtype=np.float64))
        return self

    def step(self):
        _lib.gs_run(self._handle, 1)

    def run(self, n_steps):
        _lib.gs_run(self._handle, int(n_steps))

    def get_uv(self):
        """Both fields in one call (one pair of inverse FFTs total,
        instead of two pairs from accessing .u then .v separately)."""
        u_out = np.empty((self.n, self.n), dtype=np.float64)
        v_out = np.empty((self.n, self.n), dtype=np.float64)
        _lib.gs_get_field(self._handle, u_out, v_out)
        return u_out, v_out

    @property
    def u(self):
        u_out = np.empty((self.n, self.n), dtype=np.float64)
        v_out = np.empty((self.n, self.n), dtype=np.float64)
        _lib.gs_get_field(self._handle, u_out, v_out)
        self._u_cache, self._v_cache = u_out, v_out
        return u_out

    @property
    def v(self):
        # if .u was just accessed this call is redundant but cheap
        # relative to a simulation run; keep it simple/correct rather
        # than adding cross-property caching subtlety.
        u_out = np.empty((self.n, self.n), dtype=np.float64)
        v_out = np.empty((self.n, self.n), dtype=np.float64)
        _lib.gs_get_field(self._handle, u_out, v_out)
        return v_out

    @u.setter
    def u(self, value):
        _lib.gs_set_field(self._handle, np.ascontiguousarray(value, dtype=np.float64),
                           np.ascontiguousarray(self.v, dtype=np.float64))

    @v.setter
    def v(self, value):
        _lib.gs_set_field(self._handle, np.ascontiguousarray(self.u, dtype=np.float64),
                           np.ascontiguousarray(value, dtype=np.float64))

    @property
    def time(self):
        return _lib.gs_get_time(self._handle)

    @time.setter
    def time(self, value):
        _lib.gs_set_time(self._handle, float(value), _lib.gs_get_step_count(self._handle))

    @property
    def step_count(self):
        return _lib.gs_get_step_count(self._handle)

    def save_checkpoint(self, path):
        np.savez(path, u=self.u, v=self.v, step_count=self.step_count,
                 time=self.time, n=self.n, L=self.L, Du=self.Du, Dv=self.Dv,
                 F=self.F, k=self.k, dt=self.dt, dealias=self.dealias)

    @classmethod
    def from_checkpoint(cls, path, dt=None, dealias=False):
        # NOTE: dealiasing is a per-run choice, not a property of a
        # checkpoint file (the same settled field is a valid starting
        # point for either arm of the dealiasing comparison), so this
        # does NOT read d["dealias"] even when present -- it always
        # uses the caller's explicit `dealias` argument.
        d = np.load(path)
        sim = cls(n=int(d["n"]), L=float(d["L"]), Du=float(d["Du"]), Dv=float(d["Dv"]),
                   F=float(d["F"]), k=float(d["k"]), dt=float(d["dt"]) if dt is None else dt,
                   dealias=dealias)
        _lib.gs_set_field(sim._handle, np.ascontiguousarray(d["u"], dtype=np.float64),
                           np.ascontiguousarray(d["v"], dtype=np.float64))
        step_count = int(d["step_count"])
        time_val = float(d["time"]) if "time" in d else float(step_count)
        _lib.gs_set_time(sim._handle, time_val, step_count)
        return sim
