"""Figure 2 of the paper: left = the field snapshot with the interior, boundary and exterior rings (src/TeX/figures/schematic_left.png,
the left half of the earlier hand-made figure), right = the blanket diagram, drawn here.
Writes figure2_blanket.png in the current folder (build/).   Option --lines draws plain lines instead of two-way arrows."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

LINES = "--lines" in sys.argv
GREEN = ("#f0fff0", "#006400"); ORANGE = ("#ffefd5", "#8b4513"); BLUE = ("#f0f8ff", "#4682b4"); RED = "#dc143c"
X = {"I": 0.17, "B": 0.5, "E": 0.83}
FS = 0.8   # font scale: the right half is narrower than the candidate sheet

def node(ax, key, y, label, face_edge):
    f, e = face_edge
    ax.add_patch(FancyBboxPatch((X[key] - 0.055, y - 0.07), 0.11, 0.14, boxstyle="round,pad=0.005,rounding_size=0.02", fc=f, ec=e, lw=2))
    ax.text(X[key], y, label, ha="center", va="center", fontsize=15 * FS, style="italic", fontweight="bold" if key == "B" else None)

def outer(ax, x0, x1, y, text):
    ax.add_patch(Rectangle((x0, y - 0.19), x1 - x0, 0.33, fill=False, ec="gray", ls="--", lw=1.2))
    ax.text((x0 + x1) / 2, y + 0.17, text, ha="center", fontsize=8.5 * FS, color="dimgray")

def link(ax, a, b, y):
    xa = X[a] + (0.06 if X[a] < X[b] else -0.06); xb = X[b] + (-0.06 if X[a] < X[b] else 0.06)
    ax.add_patch(FancyArrowPatch((xa, y), (xb, y), arrowstyle="-" if LINES else "<|-|>", mutation_scale=16, color="black", lw=1.8, shrinkA=0, shrinkB=0))

def red_arc(ax, y):
    ax.add_patch(FancyArrowPatch((X["I"], y - 0.22), (X["E"], y - 0.22), connectionstyle="arc3,rad=0.2",
                 arrowstyle="-" if LINES else "<|-|>", mutation_scale=16, color=RED, lw=2, ls="--", shrinkA=2, shrinkB=2))

def panel(ax, kind):
    ax.set_xlim(0, 1); ax.set_ylim(-0.14, 1); ax.axis("off")
    y = 0.58
    outer(ax, 0.06, 0.28, y, r"$\mu_{\rm full}$ (full internal state)"); outer(ax, 0.72, 0.94, y, r"$\eta_{\rm full}$ (full external state)")
    node(ax, "I", y, "I", GREEN); node(ax, "B", y, "B", ORANGE); node(ax, "E", y, "E", BLUE)
    ax.text(X["I"], y - 0.14, r"$I\subsetneq\mu_{\rm full}$", ha="center", fontsize=10 * FS)
    ax.text(X["B"], y - 0.14, r"$(\Phi_u,\Phi_v)$", ha="center", fontsize=10 * FS)
    ax.text(X["E"], y - 0.14, r"$E\subsetneq\eta_{\rm full}$", ha="center", fontsize=10 * FS)
    link(ax, "I", "B", y); link(ax, "B", "E", y)
    if kind == "hyp":
        ax.text(0.5, 0.22, r"hypothesized: $I \perp E \mid B$", ha="center", fontsize=12 * FS)
    else:
        red_arc(ax, y)
        ax.text(0.5, 0.0, r"observed: $\neg\,(I \perp E \mid B)$", ha="center", fontsize=12 * FS, color=RED, fontweight="bold")
        ax.text(0.5, -0.09, "dashed: dependence that remains after conditioning on B", ha="center", fontsize=8.5 * FS, color="dimgray")

W, H, DPI = 1860, 1108, 200          # same size as the earlier figure, so the page layout does not change
left = mpimg.imread(os.path.join(os.environ.get("TOP", "."), "src", "TeX", "figures", "schematic_left.png"))
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
axl = fig.add_axes([0, 0, left.shape[1] / W, 1]); axl.imshow(left, aspect="auto"); axl.axis("off")
x0 = 1000 / W
for kind, (b, h, title) in {"hyp": (0.5, 0.5, "Markov blanket hypothesis"), "find": (0.0, 0.5, "What we find")}.items():
    ax = fig.add_axes([x0, b, 1 - x0, h * 0.86]); panel(ax, kind)
    fig.text(x0 + (1 - x0) / 2, b + h * 0.93, title, ha="center", fontsize=14 * FS, fontweight="bold")
fig.savefig("figure2_blanket.png", dpi=DPI)
