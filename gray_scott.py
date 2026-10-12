"""
Gray-Scott reaction-diffusion solver, replicating Pearson (1993),
"Complex Patterns in a Simple System," Science 261:189-192.

Model:
    du/dt = Du * Laplacian(u) - u*v^2 + F*(1-u)
    dv/dt = Dv * Laplacian(v) + u*v^2 - (F+k)*v

Exact parameters from the paper:
    domain size      : 2.5 x 2.5
    Du               : 2e-5
    Dv               : 1e-5
    grid             : 256 x 256
    dt               : 1
    boundary         : periodic
    initial condition: trivial state (u=1, v=0) everywhere, except a
                        20x20 mesh-point square at the center set to
                        (u=0.5, v=0.25), plus +/-1% random noise on
                        that whole initial field to break symmetry.

F, k for the self-replicating-spot regime (Munafo's extension of
Pearson's classification, "lambda" type): F=0.0392, k=0.0649.

We additionally:
  - store dv/dt at every recorded frame (it's already computed as
    part of the update step, so this is free)
  - track connected components of {dv/dt > 0} over time via
    scipy.ndimage.label, which gives us a per-frame count and labeling
    of individual "organisms" (spots), including split/merge events
    showing up as component-count changes between frames.
"""

import numpy as np
from scipy import ndimage


def laplacian_periodic(field, dx):
    """5-point stencil Laplacian with periodic boundary conditions."""
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4.0 * field
    ) / (dx * dx)


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _merge_periodic_labels(labeled, n_features):
    """
    scipy.ndimage.label doesn't know the grid is periodic, so a single
    spot straddling the top/bottom or left/right seam gets split into
    two labels. This merges labels that touch across the periodic
    boundary, then relabels 1..n_merged contiguously.
    """
    uf = _UnionFind(n_features + 1)  # labels are 1..n_features, 0 = background

    top, bottom = labeled[0, :], labeled[-1, :]
    for a, b in zip(top, bottom):
        if a > 0 and b > 0 and a != b:
            uf.union(a, b)

    left, right = labeled[:, 0], labeled[:, -1]
    for a, b in zip(left, right):
        if a > 0 and b > 0 and a != b:
            uf.union(a, b)

    roots = {}
    remap = np.zeros(n_features + 1, dtype=labeled.dtype)
    next_id = 1
    for lbl in range(1, n_features + 1):
        r = uf.find(lbl)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        remap[lbl] = roots[r]

    merged = remap[labeled]
    return merged, next_id - 1


class GrayScottSim:
    def __init__(self, n=256, L=2.5, Du=2e-5, Dv=1e-5,
                 F=0.0392, k=0.0649, dt=1.0, seed=0):
        self.n = n
        self.L = L
        self.dx = L / n
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.u = np.ones((n, n), dtype=np.float64)
        self.v = np.zeros((n, n), dtype=np.float64)

        # 20x20 center square perturbation, exactly as in Pearson (1993)
        c = n // 2
        half = 10  # 20x20 square => 10 on each side of center
        sl = slice(c - half, c + half)
        self.u[sl, sl] = 0.5
        self.v[sl, sl] = 0.25

        # +/-1% random noise on the whole field to break square symmetry
        self.u *= 1.0 + 0.01 * self.rng.uniform(-1, 1, size=(n, n))
        self.v_noise_mask = self.v > 0  # only v is nonzero in the square
        self.v[self.v_noise_mask] *= (
            1.0 + 0.01 * self.rng.uniform(-1, 1, size=(n, n))[self.v_noise_mask]
        )

        # dv/dt from the most recent step, kept around for analysis
        self.dvdt = np.zeros((n, n), dtype=np.float64)
        self.Lu = np.zeros((n, n), dtype=np.float64)
        self.Lv = np.zeros((n, n), dtype=np.float64)
        self.step_count = 0
        self.time = 0.0

    def save_checkpoint(self, path):
        np.savez(path, u=self.u, v=self.v, step_count=self.step_count,
                 time=self.time, n=self.n, L=self.L, Du=self.Du, Dv=self.Dv,
                 F=self.F, k=self.k, dt=self.dt)

    @classmethod
    def from_checkpoint(cls, path, dt=None):
        d = np.load(path)
        sim = cls(n=int(d["n"]), L=float(d["L"]), Du=float(d["Du"]),
                   Dv=float(d["Dv"]), F=float(d["F"]), k=float(d["k"]),
                   dt=float(d["dt"]) if dt is None else dt)
        sim.u = d["u"].copy()
        sim.v = d["v"].copy()
        sim.step_count = int(d["step_count"])
        sim.time = float(d["time"]) if "time" in d else float(sim.step_count)
        return sim

    def step(self, noise_sigma_u=0.0, noise_sigma_v=0.0, rng=None):
        u, v = self.u, self.v
        Lu = laplacian_periodic(u, self.dx)
        Lv = laplacian_periodic(v, self.dx)

        uvv = u * v * v
        du = self.Du * Lu - uvv + self.F * (1.0 - u)
        dv = self.Dv * Lv + uvv - (self.F + self.k) * v

        u_new = u + self.dt * du
        v_new = v + self.dt * dv

        if noise_sigma_u > 0 or noise_sigma_v > 0:
            if rng is None:
                rng = np.random.default_rng()
            sqdt = np.sqrt(self.dt)
            if noise_sigma_u > 0:
                u_new = u_new + noise_sigma_u * sqdt * rng.standard_normal(u.shape)
            if noise_sigma_v > 0:
                v_new = v_new + noise_sigma_v * sqdt * rng.standard_normal(v.shape)

        self.u = u_new
        self.v = v_new
        self.dvdt = dv  # store the just-computed dv/dt for this state
        self.Lu = Lu
        self.Lv = Lv
        self.step_count += 1
        self.time += self.dt

    def run(self, n_steps, noise_sigma_u=0.0, noise_sigma_v=0.0, rng=None):
        for _ in range(n_steps):
            self.step(noise_sigma_u=noise_sigma_u, noise_sigma_v=noise_sigma_v, rng=rng)

    def active_mask(self, rel_floor=1e-6):
        """
        Boolean mask of points where |dv/dt| is dynamically significant,
        i.e. not just the exponentially-small diffusive tail bleeding
        into nominally-quiescent background. The gap between real
        dynamics (~1e-5 to 1e-8 near a spot) and the numerical tail
        (<1e-20 a few pixels further out) spans dozens of orders of
        magnitude, so rel_floor is not a delicate parameter -- almost
        any choice between 1e-3 and 1e-15 (relative to the current
        frame's peak |dv/dt|) gives the same partition.
        """
        mag = np.abs(self.dvdt)
        peak = mag.max()
        if peak == 0:
            return np.zeros_like(mag, dtype=bool)
        return mag > rel_floor * peak

    def partition(self, rel_floor=1e-6, blanket_radius=2):
        """
        Three-way partition of the domain into interior / blanket /
        exterior, restricted to the dynamically active region so the
        deep quiescent background (technically dv/dt>0 by a hair, but
        physically inert) is excluded from all three categories:

          interior : active and dv/dt > 0  (net producing V)
          exterior : active and dv/dt < 0  (net losing V)
          blanket  : quiescent points immediately adjacent to the
                     active region (a thin shell around each spot's
                     zone of activity, NOT the whole inactive background)

        Returns (interior, blanket, exterior) boolean arrays.
        """
        active = self.active_mask(rel_floor=rel_floor)
        interior = active & (self.dvdt > 0)
        exterior = active & (self.dvdt < 0)

        padded = np.pad(active, blanket_radius, mode="wrap")
        structure = ndimage.generate_binary_structure(2, 2)
        dilated = ndimage.binary_dilation(
            padded, structure=structure, iterations=blanket_radius
        )
        dilated = dilated[blanket_radius:-blanket_radius,
                           blanket_radius:-blanket_radius]
        blanket = dilated & ~active

        return interior, blanket, exterior

    def bands(self, fast_rel=1e-3, noise_rel=1e-8, blanket_radius=2):
        """
        Split dv/dt into two physically distinct bands:

          fast   : |dv/dt| > fast_rel * peak
                   -- genuine growth/division dynamics. Used for
                   individuation (interior/blanket/exterior, connected
                   components).
          slow   : noise_rel*peak < |dv/dt| <= fast_rel*peak
                   -- small but non-noise residual left over once
                   reaction and decay have nearly (not exactly)
                   cancelled. This is the inter-spot substrate-
                   competition signal: doesn't respect individual
                   spot boundaries, exists nearly everywhere at once.
          (below noise_rel*peak is treated as genuinely negligible --
           the exponential diffusive tail into quiescent background.)

        Returns a dict with:
          fast_interior, fast_blanket, fast_exterior : bool arrays
          competition : float array, dv/dt where in the slow band,
                        0.0 elsewhere (so its sign/magnitude can be
                        plotted directly)
        """
        mag = np.abs(self.dvdt)
        peak = mag.max()
        if peak == 0:
            z = np.zeros_like(mag, dtype=bool)
            return dict(fast_interior=z, fast_blanket=z, fast_exterior=z,
                        competition=np.zeros_like(self.dvdt))

        fast_active = mag > fast_rel * peak
        fast_interior = fast_active & (self.dvdt > 0)
        fast_exterior = fast_active & (self.dvdt < 0)

        padded = np.pad(fast_active, blanket_radius, mode="wrap")
        structure = ndimage.generate_binary_structure(2, 2)
        dilated = ndimage.binary_dilation(
            padded, structure=structure, iterations=blanket_radius
        )
        dilated = dilated[blanket_radius:-blanket_radius,
                           blanket_radius:-blanket_radius]
        fast_blanket = dilated & ~fast_active

        slow_mask = (mag > noise_rel * peak) & (~fast_active)
        competition = np.where(slow_mask, self.dvdt, 0.0)

        return dict(fast_interior=fast_interior, fast_blanket=fast_blanket,
                    fast_exterior=fast_exterior, competition=competition)

    def interior_components(self, rel_floor=1e-6):
        """
        Connected components of the (significance-filtered) interior,
        wrapped periodically (components touching opposite edges of the
        periodic domain are merged). Returns (labeled_array, num_features).

        NOTE: this dv/dt-sign-based individuation is meaningful mainly
        in the sparse/early regime, where there's still genuinely
        quiescent background to contrast against (a real, huge
        magnitude gap). Once the domain fills with a dense population,
        the whole dv/dt field becomes one continuous, unimodal
        distribution (confirmed empirically) and this notion of
        individuation stops being well-defined. Use segment_organisms()
        below for individuation that stays valid throughout.
        """
        interior, _, _ = self.partition(rel_floor=rel_floor)
        structure = np.ones((3, 3), dtype=int)
        labeled, n_features = ndimage.label(interior, structure=structure)
        if n_features == 0:
            return labeled, 0
        labeled, n_merged = _merge_periodic_labels(labeled, n_features)
        return labeled, n_merged

    def segment_organisms(self, pad=20, min_peak_distance=8):
        """
        Individuate organisms geometrically from the V field rather
        than from dv/dt: threshold V (Otsu, computed fresh each frame
        so it adapts as the population's V distribution evolves),
        find local maxima (one per spot core) within the thresholded
        region, and run a marker-controlled watershed on -V to split
        any touching/merged spots at their natural saddle boundary.

        Periodic boundary conditions are handled by padding the field
        with a periodic wrap before segmenting, then cropping back to
        the original domain -- this lets watershed see real neighbor
        data across the seam instead of an artificial edge.

        Returns (labels, n_organisms) where labels is an (n,n) int
        array, 0 = background, 1..n_organisms = organism id.
        """
        from skimage.filters import threshold_otsu
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        v = self.v
        n = self.n
        vp = np.pad(v, pad, mode="wrap")

        thresh = threshold_otsu(v)
        mask = vp > thresh

        coords = peak_local_max(vp, min_distance=min_peak_distance,
                                 labels=mask.astype(int))
        markers = np.zeros(vp.shape, dtype=int)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        labeled = watershed(-vp, markers=markers, mask=mask)
        labeled = labeled[pad:pad + n, pad:pad + n]

        # relabel contiguously in case some markers only existed in the
        # padding region and never survived into the cropped tile
        present = np.unique(labeled)
        present = present[present != 0]
        remap = np.zeros(markers.max() + 1, dtype=int)
        for new_id, old_id in enumerate(present, start=1):
            remap[old_id] = new_id
        labeled = remap[labeled]

        return labeled, len(present)

    def per_organism_flux(self, labels, n_organisms):
        """
        Net OUTWARD diffusive flux of U and V across each organism's
        boundary, via the divergence theorem: for a diffusive flux
        F = -D*grad(X), the net outward flux through a closed boundary
        equals -D * integral of Laplacian(X) over the enclosed region.
        We already have the Laplacians (self.Lu, self.Lv) stored from
        the last step. Sign convention: negative = net inward flow.

        Hypothesis being tested: U flows net INWARD (organisms consume
        it) while V flows net OUTWARD (organisms produce it, and it's
        the shared resource neighbors compete over) -- i.e.
        flux_u < 0 and flux_v > 0 for most organisms.

        Returns (flux_u, flux_v), each an array of length n_organisms.
        """
        if n_organisms == 0:
            return np.array([]), np.array([])
        cell_area = self.dx * self.dx
        idx = np.arange(1, n_organisms + 1)
        flux_u = -self.Du * ndimage.sum(self.Lu, labels=labels, index=idx) * cell_area
        flux_v = -self.Dv * ndimage.sum(self.Lv, labels=labels, index=idx) * cell_area
        return flux_u, flux_v

    def organism_geometry(self, labels, n_organisms):
        """
        Centroid (row, col) and equivalent radius (from pixel area) for
        each organism. Centroids are computed with periodic wrap-around
        awareness is NOT applied here -- for organisms that straddle a
        seam this centroid can be off; fine for the bulk statistics we
        use it for, since only a small fraction of organisms sit on the
        seam at any given time.
        Returns (centroids: (n,2) array, radii: (n,) array).
        """
        if n_organisms == 0:
            return np.zeros((0, 2)), np.zeros(0)
        idx = np.arange(1, n_organisms + 1)
        centroids = np.array(ndimage.center_of_mass(labels, labels=labels, index=idx))
        areas = ndimage.sum(np.ones_like(labels), labels=labels, index=idx)
        radii = np.sqrt(areas / np.pi) * self.dx
        return centroids, radii


    def per_organism_dvdt(self, labels, n_organisms):
        """
        For each segmented organism, the net (unfiltered, real) rate
        of change of its total V content: sum of dv/dt over its
        footprint. No threshold needed -- this is well-posed whether
        the organism is isolated or embedded in a dense population.
        Returns an array of length n_organisms (index i = organism i+1).
        """
        if n_organisms == 0:
            return np.array([])
        totals = ndimage.sum(self.dvdt, labels=labels,
                              index=np.arange(1, n_organisms + 1))
        return totals


def organism_interior_exterior_states_adaptive(sim, labels, n_organisms,
                                                keep_fraction=0.5,
                                                exterior_ring_px=6):
    """
    Same as organism_interior_exterior_states, but the interior erosion
    amount is scaled to each organism's OWN equivalent radius (erode
    until keep_fraction of the radius remains) rather than a fixed
    pixel count for every organism regardless of size. This avoids the
    empty-core fallback that a fixed erosion hits for smaller organisms.
    """
    if n_organisms == 0:
        return np.array([]), np.array([])

    pad = exterior_ring_px + 12
    labels_p = np.pad(labels, pad, mode="wrap")
    v_p = np.pad(sim.v, pad, mode="wrap")
    structure = ndimage.generate_binary_structure(2, 2)

    I = np.full(n_organisms, np.nan)
    E = np.full(n_organisms, np.nan)

    for i in range(1, n_organisms + 1):
        mask = labels_p == i
        if not mask.any():
            continue
        area = mask.sum()
        radius = np.sqrt(area / np.pi)
        erode_px = max(1, int(round((1 - keep_fraction) * radius)))

        core = ndimage.binary_erosion(mask, structure=structure, iterations=erode_px)
        while not core.any() and erode_px > 0:
            erode_px -= 1
            core = ndimage.binary_erosion(mask, structure=structure, iterations=erode_px) \
                   if erode_px > 0 else mask
        I[i - 1] = v_p[core].mean()

        dil = ndimage.binary_dilation(mask, structure=structure,
                                       iterations=exterior_ring_px)
        ring = dil & (~mask)
        if ring.any():
            E[i - 1] = v_p[ring].mean()

    return I, E


def organism_interior_exterior_states(sim, labels, n_organisms,
                                       erode_px=2, exterior_ring_px=6):
    """
    For each organism: mean V in its eroded core (interior state) and
    mean V in a ring just beyond its boundary (exterior state, reaching
    into neighboring organisms' territory and shared background).
    Periodic-safe via wrap-padding, matching segment_organisms().
    Returns (I, E), each length n_organisms; E entries can be nan if
    an organism has no exterior ring pixels (shouldn't happen in a
    filled domain, but guarded anyway).
    """
    if n_organisms == 0:
        return np.array([]), np.array([])

    pad = exterior_ring_px + erode_px + 2
    labels_p = np.pad(labels, pad, mode="wrap")
    v_p = np.pad(sim.v, pad, mode="wrap")
    structure = ndimage.generate_binary_structure(2, 2)

    I = np.full(n_organisms, np.nan)
    E = np.full(n_organisms, np.nan)

    for i in range(1, n_organisms + 1):
        mask = labels_p == i
        if not mask.any():
            continue
        core = ndimage.binary_erosion(mask, structure=structure, iterations=erode_px)
        if not core.any():
            core = mask
        I[i - 1] = v_p[core].mean()

        dil = ndimage.binary_dilation(mask, structure=structure,
                                       iterations=exterior_ring_px)
        ring = dil & (~mask)
        if ring.any():
            E[i - 1] = v_p[ring].mean()

    return I, E


def organism_ring_state(sim, labels, n_organisms, r_inner, r_outer):
    """
    Mean V in the annulus between r_inner and r_outer pixels beyond
    each organism's own boundary (r_inner=0 means starting right at
    the boundary). Periodic-safe via wrap-padding.
    Use r_inner=0, r_outer=6 for the "immediate neighbor" ring;
    r_inner=6, r_outer=18 for a "third-organism-away" ring that skips
    over the immediate neighbor entirely.
    """
    if n_organisms == 0:
        return np.array([])

    pad = r_outer + 2
    labels_p = np.pad(labels, pad, mode="wrap")
    v_p = np.pad(sim.v, pad, mode="wrap")
    structure = ndimage.generate_binary_structure(2, 2)

    R = np.full(n_organisms, np.nan)
    for i in range(1, n_organisms + 1):
        mask = labels_p == i
        if not mask.any():
            continue
        outer = ndimage.binary_dilation(mask, structure=structure, iterations=r_outer)
        if r_inner > 0:
            inner = ndimage.binary_dilation(mask, structure=structure, iterations=r_inner)
        else:
            inner = mask
        ring = outer & (~inner)
        if ring.any():
            R[i - 1] = v_p[ring].mean()
    return R


def organism_shell_profile(sim, labels, n_organisms, radii):
    """
    Mean V in each annular shell [radii[i], radii[i+1]) beyond each
    organism's boundary, for ALL shells at once via a single Euclidean
    distance transform per organism (much cheaper than repeating a
    dilation per radius). Periodic-safe via wrap-padding.

    Returns an (n_organisms, len(radii)-1) array.
    """
    if n_organisms == 0:
        return np.zeros((0, len(radii) - 1))

    pad = int(radii[-1]) + 2
    labels_p = np.pad(labels, pad, mode="wrap")
    v_p = np.pad(sim.v, pad, mode="wrap")

    n_shells = len(radii) - 1
    out = np.full((n_organisms, n_shells), np.nan)

    for i in range(1, n_organisms + 1):
        mask = labels_p == i
        if not mask.any():
            continue
        dist = ndimage.distance_transform_edt(~mask)
        for s in range(n_shells):
            shell = (dist > radii[s]) & (dist <= radii[s + 1])
            if shell.any():
                out[i - 1, s] = v_p[shell].mean()

    return out


def organism_distant_state(sim, labels, n_organisms, min_dist_px=60,
                            patch_radius=3, rng=None):
    """
    Mean V in a small random patch, far from each organism's own
    centroid (periodic-safe minimum-image distance >= min_dist_px).
    Fully vectorized (no per-organism morphology), for estimating the
    correlation length of the residual interior/exterior coupling.
    """
    if n_organisms == 0:
        return np.array([])
    if rng is None:
        rng = np.random.default_rng()
    centroids, _ = sim.organism_geometry(labels, n_organisms)
    n = sim.n
    result = np.full(n_organisms, np.nan)
    for i in range(n_organisms):
        for _ in range(20):  # a few tries to satisfy the distance constraint
            r = rng.integers(0, n)
            c = rng.integers(0, n)
            d = np.array([r, c]) - centroids[i]
            d = (d + n / 2) % n - n / 2
            if np.hypot(*d) >= min_dist_px:
                break
        r0, r1 = r - patch_radius, r + patch_radius + 1
        c0, c1 = c - patch_radius, c + patch_radius + 1
        patch = sim.v.take(range(r0, r1), axis=0, mode="wrap") \
                     .take(range(c0, c1), axis=1, mode="wrap")
        result[i] = patch.mean()
    return result


def organism_local_density(sim, labels, n_organisms, radius_px=20):
    """
    Number of OTHER organism centroids within radius_px pixels of each
    organism's own centroid, using minimum-image periodic distance.
    A simple, cheap proxy for local crowding.
    """
    if n_organisms == 0:
        return np.array([])
    centroids, _ = sim.organism_geometry(labels, n_organisms)
    n = sim.n
    density = np.zeros(n_organisms)
    for i in range(n_organisms):
        d = centroids - centroids[i]
        d = (d + n / 2) % n - n / 2  # minimum-image periodic correction, pixel units
        dist = np.sqrt((d ** 2).sum(axis=1))
        density[i] = ((dist > 0) & (dist <= radius_px)).sum()
    return density


def match_organisms_by_overlap(labels_prev, labels_curr, n_prev, n_curr):
    """
    Match organisms between two frames by maximum pixel overlap.
    Returns a dict: prev_id -> (curr_id or None, overlap_fraction).
    curr_id is None if no meaningful overlap was found (organism died,
    or this is a newly-appeared organism with no prev counterpart).
    A prev organism whose best-overlap curr organism is also the best
    match for some OTHER prev organism indicates a split (one prev
    organism's territory got claimed by multiple future organisms) --
    we flag this by also returning, for each curr_id, the list of
    prev_ids that overlapped it above a small threshold.
    """
    if n_prev == 0 or n_curr == 0:
        return {}, {}

    # build overlap counts via a joint histogram
    joint = labels_prev.astype(np.int64) * (n_curr + 1) + labels_curr.astype(np.int64)
    vals, counts = np.unique(joint, return_counts=True)
    prev_ids = vals // (n_curr + 1)
    curr_ids = vals % (n_curr + 1)

    prev_areas = ndimage.sum(np.ones_like(labels_prev), labels=labels_prev,
                              index=np.arange(1, n_prev + 1))

    forward = {}  # prev_id -> (best curr_id, overlap_fraction)
    backward = {}  # curr_id -> list of prev_ids overlapping it (excluding background)

    best_overlap = {}
    for p, c, cnt in zip(prev_ids, curr_ids, counts):
        if p == 0:
            continue
        if c == 0:
            continue
        if p not in best_overlap or cnt > best_overlap[p][1]:
            best_overlap[p] = (c, cnt)
        backward.setdefault(c, []).append(p)

    for p in range(1, n_prev + 1):
        if p in best_overlap:
            c, cnt = best_overlap[p]
            frac = cnt / prev_areas[p - 1] if prev_areas[p - 1] > 0 else 0.0
            forward[p] = (int(c), float(frac))
        else:
            forward[p] = (None, 0.0)

    return forward, backward
