# MIT License
#
# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/usr/bin/env python3
"""Regenerate all PNG figures used in the documentation.

Run from the repo root after installing visualization dependencies:
    pip install -e ".[viz]"
    python3 docs/generate_figures.py
Or, after the same install step:
    make docs
"""

import shutil
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from tensor_layouts import Layout, Swizzle
from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN
from tensor_layouts.layout_utils import tile_mma_grid
from tensor_layouts.viz import (
    draw_composite,
    draw_layout,
    draw_slice,
    draw_swizzle,
    draw_tiled_grid,
    draw_tv_layout,
    draw_mma_layout,
)

IMAGES = Path(__file__).resolve().parent / "images"


def _generate_intile_oftile(path: Path) -> None:
    """intile/oftile coordinate diagram: manual index math vs layout algebra.

    Three panels on a 4×8 matrix tiled by (2,4):
      Left:   cells show linear index i
      Center: cells show 2D index (r,c)
      Right:  cells show intile coords — the output of logical_divide
    Tile coloring (shared across all panels) shows the oftile grouping.
    Formulas below each panel explain the conversion.
    """
    M, K = 4, 8
    tm, tk = 2, 4
    tile_colors = {
        (0, 0): "#DBEAFE", (0, 1): "#FEE2E2",
        (1, 0): "#D1FAE5", (1, 1): "#EDE9FE",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    def _draw_grid(ax, cell_text_fn, title, subtitle):
        """Draw M×K grid with colored tiles and per-cell text."""
        for r in range(M):
            for c in range(K):
                om, im = r // tm, r % tm
                ok, ik = c // tk, c % tk
                y = M - 1 - r
                rect = patches.Rectangle(
                    (c, y), 1, 1,
                    facecolor=tile_colors[(om, ok)],
                    edgecolor="#D1D5DB", linewidth=0.5,
                )
                ax.add_patch(rect)
                ax.text(
                    c + 0.5, y + 0.5, cell_text_fn(r, c),
                    ha="center", va="center", fontsize=9,
                    color="#374151", family="monospace",
                )
        # thick tile borders
        for i in range(0, M + 1, tm):
            ax.plot([0, K], [i, i], color="#1F2937", lw=2.5,
                    solid_capstyle="butt")
        for j in range(0, K + 1, tk):
            ax.plot([j, j], [0, M], color="#1F2937", lw=2.5,
                    solid_capstyle="butt")
        # oftile margin labels
        for om in range(M // tm):
            y_c = M - om * tm - tm / 2
            ax.text(-0.3, y_c, f"oftile\u2080={om}", ha="right", va="center",
                    fontsize=8, fontweight="bold", color="#7C3AED",
                    family="monospace")
        for ok in range(K // tk):
            x_c = ok * tk + tk / 2
            ax.text(x_c, M + 0.15, f"oftile\u2081={ok}", ha="center",
                    va="bottom", fontsize=8, fontweight="bold", color="#7C3AED",
                    family="monospace")
        ax.set_xlim(-2.5, K + 0.5)
        ax.set_ylim(-1.8, M + 0.8)
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
        ax.text(K / 2, -0.3, subtitle, ha="center", va="top", fontsize=8,
                color="#6B7280", family="monospace", linespacing=1.6)

    # ── Panel 1: linear index ────────────────────────────────────
    _draw_grid(
        axes[0],
        lambda r, c: str(r * K + c),
        "Linear index i",
        "row = i // 8,  col = i % 8\n"
        "intile = (row % 2, col % 4)\n"
        "oftile = (row // 2, col // 4)",
    )

    # ── Panel 2: 2D index ────────────────────────────────────────
    _draw_grid(
        axes[1],
        lambda r, c: f"{r},{c}",
        "2D index (row, col)",
        "intile = (row % 2, col % 4)\n"
        "oftile = (row // 2, col // 4)",
    )

    # ── Panel 3: layout algebra ──────────────────────────────────
    _draw_grid(
        axes[2],
        lambda r, c: f"{r % tm},{c % tk}",
        "logical_divide((4,8):(8,1), (2,4))",
        "result: ((2,2),(4,2)) : ((8,16),(1,4))\n"
        "mode 0: (intile\u2080, oftile\u2080)\n"
        "mode 1: (intile\u2081, oftile\u2081)",
    )
    axes[2].text(K / 2, -1.45, "cells show (intile\u2080, intile\u2081)",
                 ha="center", va="top", fontsize=9, fontweight="bold",
                 color="#2563EB", family="monospace")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    IMAGES.mkdir(exist_ok=True)

    layout_8x8 = Layout((8, 8), (8, 1))

    # -- draw_layout (README + viz_api) --
    draw_layout(
        layout_8x8,
        IMAGES / "draw_layout.png",
        title="Row-Major 8x8",
        colorize=True,
    )
    # README uses a separate file with identical content
    shutil.copy(IMAGES / "draw_layout.png", IMAGES / "row_major_8x8.png")

    # -- draw_swizzle (README + viz_api) --
    draw_swizzle(
        layout_8x8,
        Swizzle(3, 0, 3),
        IMAGES / "draw_swizzle.png",
        colorize=True,
    )
    shutil.copy(IMAGES / "draw_swizzle.png", IMAGES / "swizzle_8x8.png")

    # -- color_by presets --
    draw_layout(
        layout_8x8,
        IMAGES / "color_by_row.png",
        color_layout=Layout((8, 8), (1, 0)),
        colorize=True,
    )
    draw_layout(
        layout_8x8,
        IMAGES / "color_by_col.png",
        color_layout=Layout((8, 8), (0, 1)),
        colorize=True,
    )
    draw_layout(
        layout_8x8,
        IMAGES / "color_uniform.png",
        color_layout=Layout(1, 0),
    )

    # -- hierarchical --
    hier = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))
    draw_layout(
        hier,
        IMAGES / "hierarchical.png",
        flatten_hierarchical=False,
        title="With explicit nested coordinates",
    )

    # -- draw_tv_layout --
    atom = SM80_16x8x16_F16F16F16F16_TN
    draw_tv_layout(
        atom.c_layout,
        IMAGES / "draw_tv_layout.png",
        title="SM80 16x8x16 C (Thread-Value)",
        colorize=True,
    )

    # -- draw_mma_layout --
    draw_mma_layout(
        atom.a_layout,
        atom.b_layout,
        atom.c_layout,
        IMAGES / "draw_mma_layout.png",
        tile_mnk=atom.shape_mnk,
        main_title=atom.name,
    )

    # -- draw_slice --
    slice_layout = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    draw_slice(
        slice_layout,
        ((1, None), ((None, 0), None)),
        IMAGES / "draw_slice.png",
        title="((1,:),((:,0),:))",
    )

    # -- draw_composite --
    panels = [Layout((4, 4), (4, 1)), Layout((4, 4), (1, 4))]
    draw_composite(
        panels,
        IMAGES / "draw_composite.png",
        titles=["Row-Major", "Column-Major"],
        main_title="Layout Comparison",
        colorize=True,
    )

    # -- draw_tiled_grid --
    atom_layout = Layout((2, 2), (1, 2))
    grid, tile_shape = tile_mma_grid(atom, atom_layout, matrix="C")
    draw_tiled_grid(
        grid,
        tile_shape[0],
        tile_shape[1],
        IMAGES / "draw_tiled_grid.png",
        title="SM80 16x8x16 C \u2014 2x2 atoms",
    )

    # -- intile / oftile (applications.ipynb §3.3.5) --
    _generate_intile_oftile(IMAGES / "intile_oftile.png")

    print(f"Generated {len(list(IMAGES.glob('*.png')))} figures in {IMAGES}")


if __name__ == "__main__":
    main()
