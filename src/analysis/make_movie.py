#!/usr/bin/env python3
"""Make an animated GIF of the v (or u) field, continuing from a saved state.

Run from build/:
  python3 -B ../src/analysis/make_movie.py --state tri200_final/final_triangle_n200_s120_seed103.npz --n 200 \
      --steps 6000 --every 20 --out movie_tri200.gif
Needs numpy, scipy, Pillow and matplotlib (already required by the rest of the code); no ffmpeg.
The colour scale is fixed for the whole movie, so brightness changes are real.  --tile 2 shows the
periodic box repeated 2 x 2, which makes spots that cross the edge easier to follow.
Frames are 'steps/every'; at --fps 20 and --every 20, one second of movie is 400 solver steps.
"""
import os, sys, argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation'))

def make_solver(n, fast):
    L = n * 2.5 / 1024
    if fast:
        from gray_scott_spectral_c import PseudoSpectralGSFast
        return PseudoSpectralGSFast(n=n, L=L)
    from gray_scott_spectral import PseudoSpectralGS
    return PseudoSpectralGS(n=n, L=L)

def get_uv(sim):
    return sim.get_uv() if hasattr(sim, 'get_uv') else (sim.u, sim.v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', required=True, help='npz with u, v (from basin_scan --save-final)')
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--every', type=int, default=20, help='solver steps between frames')
    ap.add_argument('--field', choices=['u', 'v'], default='v')
    ap.add_argument('--vmax', type=float, default=None, help='top of the colour scale (default: 1.05 x the maximum seen in frame 0)')
    ap.add_argument('--scale', type=int, default=2, help='pixels per grid cell in the output')
    ap.add_argument('--tile', type=int, default=1, help='repeat the periodic box tile x tile times')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--cmap', default='inferno')
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--out', default='movie.gif')
    a = ap.parse_args()
    d = np.load(a.state)
    sim = make_solver(a.n, a.fast)
    sim.from_checkpoint_fields(d['u'].astype(np.float64), d['v'].astype(np.float64))
    t0 = int(d['time']) if 'time' in d.files else 0
    cmap = matplotlib.colormaps[a.cmap]
    frames = []; vmax = a.vmax
    for k in range(a.steps // a.every + 1):
        if k: sim.run(a.every)
        u, v = get_uv(sim)
        x = v if a.field == 'v' else u
        if vmax is None: vmax = 1.05 * float(x.max())
        x = np.tile(x, (a.tile, a.tile))
        rgb = (cmap(np.clip(x / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
        im = Image.fromarray(rgb).resize((x.shape[1] * a.scale, x.shape[0] * a.scale), Image.BILINEAR)
        ImageDraw.Draw(im).text((6, 4), f'{a.field}  t = {t0 + k * a.every}   mean {a.field} {x.mean():.3f}', fill=(255, 255, 255))
        frames.append(im.convert('P', palette=Image.ADAPTIVE, colors=256))
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0)
    print(f'wrote {a.out}: {len(frames)} frames, {frames[0].size[0]}x{frames[0].size[1]}, {os.path.getsize(a.out)/1e6:.1f} MB, colour scale 0..{vmax:.3f}')

if __name__ == '__main__':
    main()
