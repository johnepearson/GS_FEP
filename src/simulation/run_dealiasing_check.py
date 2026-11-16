"""
Referee-requested dealiasing check (both ChatGPT technical reviews of
the current draft ask for this independently): the existing pseudo-
spectral solver (gray_scott_spectral.py / gray_scott_spectral_c.py)
computes the cubic reaction term w = u*v^2 pointwise in real space and
never truncates its Fourier transform, so it does not apply the
standard 2/3-rule dealiasing pseudo-spectral methods normally use to
suppress aliased high-wavenumber content from the nonlinear product.
The paper currently defends this only indirectly (short-window
agreement between the pseudo-spectral and finite-difference solvers).
This script runs the SAME production configuration twice -- once with
the existing (un-dealiased) behavior, once with 2/3-rule dealiasing
newly added to gs_spectral_c.c -- from the SAME settled starting field,
and records the same (I, E, flux) data the paper's headline analysis
uses, so the two arms can be compared on the actual headline
partial-correlation statistic rather than on short-window field
agreement.

The two runs are expected to diverge from each other after a handful
of steps (this is a chaotic system -- that is not a bug and is not
what is being tested). What matters is whether the *distributional*/
statistical result computed over many frames and organism-instances
(the partial correlation of I and E controlling for flux) is
materially different between the two arms, not whether the two
trajectories stay close to each other pointwise.

Dealiasing implementation: gs_spectral_c.c's gs_create() now takes an
extra trailing `dealias` (0/1) argument. When 1, the Fourier transform
of w=u*v^2 (`what`) is masked to zero every mode outside the lowest 2/3
of each axis's represented wavenumbers, immediately after its forward
FFT and before it is combined into the update -- the standard,
separable (per-axis, not circular) 2/3 rule. When 0 (the default, and
what every existing script in this package still uses since
PseudoSpectralGSFast's `dealias` constructor argument defaults to
False), the new code path is a no-op and reproduces the original
solver bit-for-bit (verified: see the regression check in this
project's chat history / commit message for this script).

REQUIRES REBUILDING libgs_spectral.so first -- the .c file changed:
    gcc -O3 -march=native -fPIC -shared -o libgs_spectral.so \\
        gs_spectral_c.c -lfftw3_threads -lfftw3 -lpthread -lm
(Linux command shown; see README.md's "C solver" section for the
macOS/Homebrew variant if needed.) This OVERWRITES the existing
libgs_spectral.so in place -- every other script that uses
PseudoSpectralGSFast keeps working exactly as before, since dealias
defaults to False and reproduces the old behavior exactly.

Usage (run both arms; each is independent and can run in either order
or in parallel in two terminals -- they do not share state):
    python run_dealiasing_check.py --dealias 0 --n-frames 400
    python run_dealiasing_check.py --dealias 1 --n-frames 400

Then compare with:
    python compare_dealiasing_check.py
"""
import argparse
import time
import numpy as np
from scipy import ndimage

from gray_scott import GrayScottSim, laplacian_periodic
from run_confound_checks import organism_interior_exterior_states_fast

try:
    from gray_scott_spectral_c import PseudoSpectralGSFast
except OSError as e:
    raise SystemExit(
        "libgs_spectral.so isn't built (or is stale -- this script needs the "
        "version of gs_spectral_c.c with the `dealias` argument). Build it "
        "with:\n"
        "  gcc -O3 -march=native -fPIC -shared -o libgs_spectral.so \\\n"
        "      gs_spectral_c.c -lfftw3_threads -lfftw3 -lpthread -lm\n"
        f"(original error: {e})"
    )

CHECKPOINT_FILE = "checkpoint_1024_settled.npz"  # same settled field for both arms
SAMPLE_EVERY_STEPS = 100   # matches the paper's production single-trajectory spacing
SAVE_EVERY_FRAMES = 10
DEPTH_FRACTION = 0.5
EXTERIOR_RING_PX = 24
CROP_IE = 90
PAD = 80
MIN_PEAK_DISTANCE = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dealias", type=int, required=True, choices=[0, 1],
                     help="0 = existing (un-dealiased) solver behavior, "
                          "1 = 2/3-rule dealiasing enabled")
    ap.add_argument("--n-frames", type=int, default=400,
                     help="matches the paper's production single-trajectory "
                          "run (400 frames, SAMPLE_EVERY_STEPS=100 apart)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dealias = bool(args.dealias)
    tag = "dealias" if dealias else "nodealias"
    progress_file = f"dealiasing_check_{tag}_progress.npz"
    keys = ["I", "E", "fu", "fv", "frame_id"]

    if args.resume:
        d = np.load(progress_file)
        sim = PseudoSpectralGSFast.from_checkpoint(CHECKPOINT_FILE, dealias=dealias)
        sim.from_checkpoint_fields(d["u"], d["v"])
        sim.time = float(d["time"])
        lists = {k: [d[k]] for k in keys}
        frames_done = int(d["frames_done"])
        print(f"resumed dealias={dealias} at t={sim.time:.0f}, {frames_done} frames done")
    else:
        sim = PseudoSpectralGSFast.from_checkpoint(CHECKPOINT_FILE, dealias=dealias)
        lists = {k: [] for k in keys}
        frames_done = 0
        print(f"loaded {CHECKPOINT_FILE} at n={sim.n}, t={sim.time:.0f}, dealias={dealias}")

    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)

    t_start = time.time()
    while frames_done < args.n_frames:
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.get_uv()

        if not (np.all(np.isfinite(proxy.u)) and np.all(np.isfinite(proxy.v))):
            raise SystemExit(
                f"dealias={dealias}: non-finite field at frame {frames_done+1} "
                f"(t={sim.time:.0f}) -- simulation blew up. This would itself "
                f"be a reportable result (dealiasing destabilizing production "
                f"parameters, or the un-dealiased run's aliasing being the "
                f"instability's source), so stop and report which arm failed "
                f"and at what step rather than restarting past it."
            )

        labels, n_org = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)

        if n_org > 0:
            proxy.Lu = laplacian_periodic(proxy.u, proxy.dx)
            proxy.Lv = laplacian_periodic(proxy.v, proxy.dx)
            flux_u, flux_v = proxy.per_organism_flux(labels, n_org)
            centroids, _radii = proxy.organism_geometry(labels, n_org)
            I, E = organism_interior_exterior_states_fast(
                proxy, labels, n_org, centroids, depth_fraction=DEPTH_FRACTION,
                exterior_ring_px=EXTERIOR_RING_PX, crop=CROP_IE)
            frame_id = np.full(n_org, frames_done, dtype=np.int64)
            for k, v_ in zip(keys, [I, E, flux_u, flux_v, frame_id]):
                lists[k].append(v_)

        frames_done += 1
        elapsed = time.time() - t_start
        if frames_done % 10 == 0:
            n_so_far = sum(len(x) for x in lists["I"])
            print(f"dealias={dealias} frame {frames_done}/{args.n_frames}  t={sim.time:.0f}  "
                  f"organisms={n_org}  N so far={n_so_far}  "
                  f"({elapsed:.0f}s elapsed, ~{elapsed/frames_done:.2f}s/frame)")

        if frames_done % SAVE_EVERY_FRAMES == 0 or frames_done == args.n_frames:
            save_kwargs = {k: np.concatenate(v_) if len(v_) else np.array([])
                            for k, v_ in lists.items()}
            u, v = sim.get_uv()
            np.savez(progress_file, u=u, v=v, time=sim.time, n=sim.n, dealias=dealias,
                      frames_done=frames_done, **save_kwargs)

    n_total = sum(len(x) for x in lists["I"])
    print(f"\ndone: dealias={dealias}, {frames_done} frames, N={n_total} "
          f"organism-instances, saved to {progress_file}")


if __name__ == "__main__":
    main()
