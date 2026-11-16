"""
Interior/exterior organism state via the LOCAL distance transform,
rather than uniform whole-mask erosion.

The previous adaptive-erosion approach (organism_interior_exterior_states_adaptive
in gray_scott.py) erodes the whole mask inward by a single depth
computed from the organism's AREA-EQUIVALENT radius. For an elongated
("peanut") organism, that one global depth is a poor match for a
narrow waist: pixels there can still count as "interior" even though
they sit close to the TRUE LOCAL boundary -- exactly the boundary
we've shown (Section 3.5) carries neighbor-dependent contamination.

This version instead computes, for every pixel in the mask, its own
local distance to the nearest point outside the mask (the Euclidean
distance transform), and keeps only pixels whose local depth exceeds
depth_fraction * (organism's equivalent radius). This respects the
true local geometry everywhere -- a narrow waist is correctly
excluded even near the centroid, since its LOCAL distance to boundary
is small there regardless of the organism's overall size.

Convention: depth_fraction is the erosion DEPTH itself, as a fraction
of the equivalent radius (not the fraction of radius RETAINED, as in
the old keep_fraction parameter). depth_fraction=0.5 gives the same
numeric threshold (0.5*radius) as the old keep_fraction=0.5, since
(1 - 0.5) = 0.5 too -- a coincidence at exactly this value, and the
reason the two conventions agree there but nowhere else.
depth_fraction=0.9 leaves only the deepest, most central pixels.
"""
import numpy as np
from scipy import ndimage


def organism_interior_exterior_states_distance(sim, labels, n_organisms,
                                                 depth_fraction=0.5,
                                                 exterior_ring_px=6):
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
        threshold = depth_fraction * radius

        dist = ndimage.distance_transform_edt(mask)
        core = dist >= threshold
        if not core.any():
            # graceful fallback: relax to whatever the deepest point
            # in this organism actually reaches, rather than an
            # arbitrary full-mask fallback
            max_depth = dist.max()
            core = dist >= (max_depth * 0.99) if max_depth > 0 else mask

        I[i - 1] = v_p[core].mean()

        dil = ndimage.binary_dilation(mask, structure=structure,
                                       iterations=exterior_ring_px)
        ring = dil & (~mask)
        if ring.any():
            E[i - 1] = v_p[ring].mean()

    return I, E
