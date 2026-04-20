"""Plot (t_before, goal, action, t_after) samples from debug_dataset.npz.

Each sample becomes one subplot: goal T (gray outline), before T (blue),
after T (red), and an arrow for the action angle anchored at t_before.

Run:
    python scripts/generate_data.py   # produce debug_dataset.npz
    python scripts/visualize_data.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

DATA_PATH = Path(__file__).resolve().parent.parent / "debug_dataset.npz"
OUT_PATH = Path(__file__).resolve().parent.parent / "debug_dataset.png"
WORKSPACE = 1.5
N_SHOW = 16  # first N samples

# T-block geometry (matches core.py T_TOP_BAR / T_STEM, in body frame)
TOP_C, TOP_H = np.array([0.0, 0.10]), np.array([0.10, 0.025])
STEM_C, STEM_H = np.array([0.0, -0.025]), np.array([0.025, 0.10])


def _rect(center, half):
    cx, cy = center
    hx, hy = half
    return np.array([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]])


def _t_polys(pose):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return [(_rect(TOP_C, TOP_H) @ R.T) + [x, y], (_rect(STEM_C, STEM_H) @ R.T) + [x, y]]


def _draw_t(ax, pose, color, alpha=0.5, lw=1.5):
    for poly in _t_polys(pose):
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=color, alpha=alpha, lw=lw))


def main():
    d = np.load(DATA_PATH)
    n = min(N_SHOW, d["t_before"].shape[0])
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes).ravel()

    for i in range(n):
        ax = axes[i]
        _draw_t(ax, d["goal"][i], "gray", alpha=0.3)
        _draw_t(ax, d["t_before"][i], "tab:blue", alpha=0.6)
        _draw_t(ax, d["t_after"][i], "tab:red", alpha=0.4)

        x, y = d["t_before"][i, :2]
        a = float(d["angle"][i])
        L = 0.15
        ax.arrow(x, y, L * np.cos(a), L * np.sin(a), head_width=0.03, color="black", lw=1.2)

        ax.set_xlim(0, WORKSPACE)
        ax.set_ylim(0, WORKSPACE)
        ax.set_aspect("equal")
        ax.set_title(
            f"f={int(d['face'][i])} c={d['contact'][i]:.2f} a={a:.2f}",
            fontsize=8,
        )
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("gray=goal  blue=before  red=after  arrow=action angle", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
