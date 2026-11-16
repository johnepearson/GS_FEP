"""
Movie of the v field: a central crop of the 1024x1024 Gray-Scott simulation, written straight to an mp4.

Memory use is one 1024x1024 field plus one small RGB frame: each frame is rendered and piped into ffmpeg
and then discarded, so no frame stack is ever held in memory and no intermediate PNG files are written.

Defaults: 150 frames, 20 solver steps (time units, dt = 1) between frames, central 512 x 512 crop,
30 frames/s (5 s of video), starting from checkpoint_1024_settled.npz.

Usage:
    python3 make_v_movie.py
    python3 make_v_movie.py --every 20 --n-frames 150 --crop 512 --fps 30 --out v_movie.mp4
    python3 make_v_movie.py --start-field master_fields/frame_0200.npz      # start later in the master run
    python3 make_v_movie.py --burn-in 5000                                   # run on before recording

Also writes <out>_poster.png (the first frame) for use as the still image of Figure 1.
Requires ffmpeg on the PATH and libgs_spectral.so (built as described in README.md).
"""
import argparse
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
from PIL import Image

from gray_scott_spectral_c import PseudoSpectralGSFast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint_1024_settled.npz")
    ap.add_argument("--start-field", default=None, help="npz with u and v (e.g. master_fields/frame_0200.npz)")
    ap.add_argument("--burn-in", type=int, default=0, help="solver steps to run before the first frame")
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--every", type=int, default=20, help="solver steps (time units) between frames")
    ap.add_argument("--crop", type=int, default=512, help="side of the central square crop, in pixels")
    ap.add_argument("--scale", type=int, default=1, help="integer upscaling of the output (bilinear, as in the snapshot figure)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cmap", default="byr", help="byr = the blue-yellow-red scale of Figure 1, or any matplotlib name")
    ap.add_argument("--vmax", type=float, default=None, help="colour scale maximum (default: from first frame)")
    ap.add_argument("--crf", type=int, default=16, help="mp4 quality, lower is better")
    ap.add_argument("--out", default="v_movie.mp4")
    a = ap.parse_args()

    sim = PseudoSpectralGSFast.from_checkpoint(a.checkpoint)
    if a.start_field:
        d = np.load(a.start_field)
        sim.from_checkpoint_fields(d["u"], d["v"])
        if "time" in d.files:
            sim.time = float(d["time"])
    if a.burn_in:
        sim.run(a.burn_in)

    n = sim.n
    lo = (n - a.crop) // 2
    sl = slice(lo, lo + a.crop)
    side = a.crop * a.scale
    side -= side % 2
    cmap = (LinearSegmentedColormap.from_list("byr", ["#0000ff", "#ffff00", "#ff0000"])
            if a.cmap == "byr" else colormaps[a.cmap])

    def render(v, vmax):
        f = v[sl, sl]
        if a.scale > 1:
            f = zoom(f, a.scale, order=1)
        img = (cmap(np.clip(f / vmax, 0, 1))[:, :, :3] * 255).astype(np.uint8)
        return np.ascontiguousarray(img[:side, :side])

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{side}x{side}", "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-movflags", "+faststart", a.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    vmax = a.vmax
    try:
        for i in range(a.n_frames):
            if i:
                sim.run(a.every)
            u, v = sim.get_uv()
            if vmax is None:
                vmax = float(v.max())
                print(f"colour scale: 0 to {vmax:.4f}")
            img = render(v, vmax)
            if i == 0:
                Image.fromarray(img).save(a.out.rsplit(".", 1)[0] + "_poster.png")
            ff.stdin.write(img.tobytes())
            if (i + 1) % 25 == 0:
                print(f"frame {i + 1}/{a.n_frames}  t={sim.time:.0f}  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        ff.stdin.close()
        ff.wait()
    print(f"wrote {a.out}: {a.n_frames} frames, {a.every} time units apart "
          f"({a.n_frames * a.every} total), {a.n_frames / a.fps:.1f} s at {a.fps} fps")


if __name__ == "__main__":
    main()
