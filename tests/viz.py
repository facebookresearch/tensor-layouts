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

from collections import defaultdict
import tempfile

import pytest

from tensor_layouts import *
from tensor_layouts.tensor import Tensor
from tensor_layouts.atoms_amd import (
    CDNA3P_16x16x32_F32F16F16_MFMA,
    CDNA3_32x32x16_F32F8F8_MFMA,
)
from tensor_layouts.atoms_nv import (
    SM80_16x8x16_F16F16F16F16_TN,
    SM90_16x8x4_F64F64F64F64_TN,
    SM120_16x8x32_F32E4M3E4M3F32_TN,
)
from tensor_layouts.layout_utils import tile_mma_grid

try:
    import matplotlib.figure
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.transforms import Bbox
    from tensor_layouts.viz import *
    import tensor_layouts.viz as viz_mod
    from tensor_layouts.viz import (
        _build_combined_grid_figure,
        _build_composite_figure,
        _build_copy_figure,
        _build_layout_figure,
        _build_mma_figure,
        _build_slice_figure,
        _build_swizzle_figure,
        _build_tiled_grid_figure,
        _build_tv_figure,
        _compute_tv_mapping,
        _draw_hierarchical_grid,
        _format_hierarchical_cell_lines,
        _format_nested_coord,
        _coord_levels,
        _get_slice_highlight_mask_2d,
        _level_block_sizes,
        _level_spans,
        _get_hierarchical_cell_coords_2d,
        _get_hierarchical_indices_2d,
        _get_indices_2d,
        _get_color_indices_2d,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

requires_viz = pytest.mark.skipif(
    not HAS_VIZ,
    reason="tensor_layouts.viz not available (needs matplotlib)"
)


def _call_draw_hierarchical_grid(ax, layout, **kwargs):
    """Test helper: extract data from layout and call _draw_hierarchical_grid."""
    indices, rows, cols, _, _ = _get_hierarchical_indices_2d(layout)
    cell_coords = _get_hierarchical_cell_coords_2d(layout)
    row_shape = mode(layout.shape, 0)
    col_shape = mode(layout.shape, 1)
    return _draw_hierarchical_grid(
        ax, indices, rows, cols,
        cell_coords=cell_coords,
        row_shape=row_shape, col_shape=col_shape,
        **kwargs,
    )

MIXED_VIZ_ATOMS = [
    # Representative cross-section for visualization smoke tests:
    # - NVIDIA: Ampere (SM80), Hopper-era scalar/legacy-style atom (SM90),
    #   and Blackwell (SM120)
    # - AMD: CDNA3 fp8 and CDNA3+ fp16
    # Keep this list small so viz tests stay lightweight while still covering
    # multiple layout families and thread/value mapping styles.
    SM80_16x8x16_F16F16F16F16_TN,
    SM90_16x8x4_F64F64F64F64_TN,
    SM120_16x8x32_F32E4M3E4M3F32_TN,
    CDNA3_32x32x16_F32F8F8_MFMA,
    CDNA3P_16x16x32_F32F16F16_MFMA,
]



@requires_viz
def test_draw_layout_returns_figure_without_raising():
    """Smoke test for draw_layout helper."""
    fig = _build_layout_figure(Layout((8, 8), (8, 1)))
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)



@requires_viz
def test_draw_swizzle_returns_figure_without_raising():
    """Regression test for draw_swizzle helper."""
    fig = _build_swizzle_figure(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3))
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)



@requires_viz
def test_draw_layout_accepts_tensor():
    """draw_layout/draw_layout accept Tensor and display offset-adjusted values."""
    layout = Layout((4, 8), (8, 1))
    tensor = Tensor(layout, offset=16)
    fig = _build_layout_figure(tensor)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        # Cell (0,0) at position (0.5, 0.5) should show 16 (offset), not 0
        ax = fig.axes[0]
        cell_texts = {
            (round(t.get_position()[0], 1), round(t.get_position()[1], 1)): t.get_text()
            for t in ax.texts
            if t.get_position()[0] > 0 and t.get_position()[1] > 0
        }
        assert cell_texts[(0.5, 0.5)] == "16"
    finally:
        plt.close(fig)



@requires_viz
def test_draw_layout_tensor_zero_offset():
    """Tensor with offset=0 produces same values as bare Layout."""
    layout = Layout((4, 4), (4, 1))
    tensor = Tensor(layout, offset=0)
    fig_layout = _build_layout_figure(layout)
    fig_tensor = _build_layout_figure(tensor)
    try:
        def _cell_values(fig):
            ax = fig.axes[0]
            return sorted(
                [(t.get_position(), t.get_text()) for t in ax.texts if t.get_text().isdigit()],
            )
        assert _cell_values(fig_layout) == _cell_values(fig_tensor)
    finally:
        plt.close(fig_layout)
        plt.close(fig_tensor)



@requires_viz
def test_draw_layout_swizzled_tensor():
    """Swizzled Tensor renders without error."""
    sw = Swizzle(3, 0, 3)
    layout = Layout((8, 8), (8, 1), swizzle=sw)
    tensor = Tensor(layout, offset=0)
    fig = _build_layout_figure(tensor)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_layout_smoke():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_layout(Layout((8, 8), (8, 1)), filename=f.name)



@requires_viz
def test_color_by_row_matches_color_layout():
    """color_by='row' produces the same color indices as the manual color_layout."""
    layout = Layout((4, 8), (8, 1))
    fig_by = _build_layout_figure(layout, color_by="row")
    fig_manual = _build_layout_figure(layout, color_layout=Layout((4, 8), (1, 0)),
                             colorize=True)
    try:
        # Both should have the same cell background colors
        patches_by = [p for p in fig_by.axes[0].patches]
        patches_manual = [p for p in fig_manual.axes[0].patches]
        colors_by = [p.get_facecolor() for p in patches_by]
        colors_manual = [p.get_facecolor() for p in patches_manual]
        assert colors_by == colors_manual
    finally:
        plt.close(fig_by)
        plt.close(fig_manual)



@requires_viz
def test_color_by_column():
    """color_by='column' renders without error."""
    fig = _build_layout_figure(Layout((4, 8), (8, 1)), color_by="column")
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)



@requires_viz
def test_color_by_and_color_layout_exclusive():
    """Providing both color_by and color_layout raises ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        draw_layout(Layout((4, 4), (4, 1)),
                    color_by="row",
                    color_layout=Layout((4, 4), (1, 0)))



@requires_viz
def test_rank3_layout_produces_multi_panel():
    """Rank-3 layout from flat_divide renders as multiple 2D panels."""
    matrix = Layout((8, 8), (8, 1))
    divided = flat_divide(matrix, Layout(2, 1))
    assert rank(divided) == 3
    fig = _build_layout_figure(divided)
    try:
        # shape=(2, 4, 8) → modes 0,1 are 2×4 grid, mode 2 = 8 panels
        with_content = [ax for ax in fig.axes if len(ax.patches) > 0]
        assert len(with_content) == 8
    finally:
        plt.close(fig)



@requires_viz
def test_rank3_panel_values_match_layout():
    """Each rank-3 panel shows correct offset values."""
    matrix = Layout((8, 8), (8, 1))
    divided = flat_divide(matrix, Layout(2, 1))
    fig = _build_layout_figure(divided)
    try:
        def _cell_val(ax, x, y):
            for t in ax.texts:
                tx = round(t.get_position()[0], 1)
                ty = round(t.get_position()[1], 1)
                if tx == x and ty == y:
                    try:
                        return int(t.get_text())
                    except ValueError:
                        pass
            return None

        # Panel 0: divided(0, 0, 0) — mode[2]=0
        assert _cell_val(fig.axes[0], 0.5, 0.5) == divided(0, 0, 0)
        # Panel 1: divided(0, 0, 1) — mode[2]=1
        assert _cell_val(fig.axes[1], 0.5, 0.5) == divided(0, 0, 1)
    finally:
        plt.close(fig)



@requires_viz
def test_rank4_layout_renders():
    """Rank-4 layout renders with multiple panels (outer modes flattened)."""
    # tiled_divide produces rank-3 with nested shape in mode 0;
    # flat_product can produce rank-3. Stack to get rank-4+.
    # Use a simple manual rank-4 layout:
    L = Layout((2, 3, 4, 5), (60, 20, 5, 1))
    assert rank(L) == 4
    fig = _build_layout_figure(L)
    try:
        # modes 0,1 = 2×3 grid, modes 2,3 = 4×5 = 20 panels
        with_content = [ax for ax in fig.axes if len(ax.patches) > 0]
        assert len(with_content) == 20
    finally:
        plt.close(fig)



@requires_viz
def test_draw_swizzle_smoke():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), filename=f.name)



@requires_viz
def test_draw_slice_smoke():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_slice(Layout((4, 8), (8, 1)), (2, None), filename=f.name)



@pytest.mark.parametrize("atom", MIXED_VIZ_ATOMS, ids=lambda a: a.name)
@requires_viz
def test_draw_tv_layout_smoke(atom):
    m, n, _ = atom.shape_mnk
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_tv_layout(
            atom.c_layout,
            filename=f.name,
            grid_shape=(m, n),
            colorize=True,
            thr_id_layout=atom.thr_id,
        )



@pytest.mark.parametrize("atom", MIXED_VIZ_ATOMS, ids=lambda a: a.name)
@requires_viz
def test_draw_mma_layout_smoke(atom):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_mma_layout(
            atom.a_layout,
            atom.b_layout,
            atom.c_layout,
            filename=f.name,
            tile_mnk=atom.shape_mnk,
            colorize=True,
            thr_id_layout=atom.thr_id,
        )



@requires_viz
def test_draw_mma_layout_raises_for_incompatible_panel_shape():
    a_layout = Layout((2, 2), (1, 2))
    b_layout = Layout((2, 1), (1, 0))
    c_layout = Layout((2, 2), (1, 2))

    try:
        with pytest.raises(ValueError, match=r"A .*panel shape .*out of bounds"):
            draw_mma_layout(
                a_layout,
                b_layout,
                c_layout,
                tile_mnk=(2, 2, 1),
            )
    finally:
        plt.close("all")



@pytest.mark.parametrize("atom", MIXED_VIZ_ATOMS, ids=lambda a: a.name)
@requires_viz
def test_draw_tiled_grid_smoke(atom):
    atom_layout = Layout((2, 2), (1, 2))
    grid, tile_shape = tile_mma_grid(atom, atom_layout, matrix="C")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_tiled_grid(
            grid,
            tile_shape[0],
            tile_shape[1],
            filename=f.name,
            title="tiled grid smoke",
        )



@requires_viz
def test_draw_composite_smoke():
    panels = [Layout((4, 4), (4, 1)), Layout((4, 4), (1, 4))]
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_composite(
            panels,
            filename=f.name,
            titles=["Row-Major", "Column-Major"],
            main_title="Composite Smoke",
            colorize=True,
        )



@requires_viz
def test_draw_composite_mixed_tv_and_offset():
    """Composite figure with per-panel tv_mode: one offset grid, one TV grid."""
    atom = SM80_16x8x16_F16F16F16F16_TN
    panels = [
        Layout((4, 4), (4, 1)),  # offset grid (default)
        (atom.c_layout, {'tv_mode': True}),  # TV grid
    ]
    fig = _build_composite_figure(panels, titles=["Offset", "TV"])
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_hierarchical_panel():
    """Composite figure with flatten_hierarchical=False renders hierarchy lines."""
    hier = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))
    flat = Layout((4, 4), (4, 1))
    panels = [
        (hier, {'flatten_hierarchical': False}),
        flat,
    ]
    fig = _build_composite_figure(panels, titles=["Hierarchical", "Flat"])
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        # The hierarchical panel should have hierarchy boundary lines
        hier_ax = fig.axes[0]
        assert len(hier_ax.lines) > 0
        # The flat panel should have no hierarchy lines
        flat_ax = fig.axes[1]
        assert len(flat_ax.lines) == 0
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_hierarchical_top_level_default():
    """flatten_hierarchical=False as top-level default applies to all panels."""
    hier = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))
    fig = _build_composite_figure([hier], flatten_hierarchical=False)
    try:
        ax = fig.axes[0]
        assert len(ax.lines) > 0
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_warns_on_panel_truncation():
    """Warning emitted when panels exceed grid capacity."""
    panels = [Layout((2, 2), (2, 1)) for _ in range(5)]
    with pytest.warns(UserWarning, match="5 panels.*4 cells"):
        fig = _build_composite_figure(panels, arrangement="grid:2x2")
        plt.close(fig)



@requires_viz
def test_draw_copy_layout_smoke():
    src = Layout((4, 2), (2, 1))
    dst = Layout((4, 2), (1, 4))
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_copy_layout(src, dst, filename=f.name,
                         title="copy smoke", colorize=True)



@requires_viz
def test_draw_copy_layout_rejects_rank1():
    with pytest.raises(ValueError, match="rank 2"):
        draw_copy_layout(Layout(8, 1), Layout((4, 2), (2, 1)),
                         filename="ignored.png")



@requires_viz
def test_draw_copy_layout_returns_figure():
    src = Layout((4, 2), (2, 1))
    dst = Layout((4, 2), (1, 4))
    fig = _build_copy_figure(src, dst, title="copy show")
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_copy_atom_smoke():
    """draw_copy_atom handles the upcast from bit coordinates automatically."""
    from tensor_layouts.atoms_nv import SM75_U32x1_LDSM_N
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_copy_atom(SM75_U32x1_LDSM_N, element_bits=16, filename=f.name)



@requires_viz
def test_draw_copy_atom_returns_figure():
    """draw_copy_atom renders without raising."""
    from tensor_layouts.atoms_nv import SM90_U32x4_STSM_N
    draw_copy_atom(SM90_U32x4_STSM_N, element_bits=16)



@requires_viz
def test_draw_tv_layout_returns_figure():
    fig = _build_tv_figure(Layout((4, 2), (2, 1)))
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_mma_layout_returns_figure():
    from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN
    atom = SM80_16x8x16_F16F16F16F16_TN
    fig = _build_mma_figure(atom.a_layout, atom.b_layout, atom.c_layout,
                          tile_mnk=atom.shape_mnk, colorize=True,
                          thr_id_layout=atom.thr_id)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_tiled_grid_returns_figure():
    from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN
    atom = SM80_16x8x16_F16F16F16F16_TN
    atom_layout = Layout((2, 2), (1, 2))
    grid, tile_shape = tile_mma_grid(atom, atom_layout, matrix="C")
    fig = _build_tiled_grid_figure(grid, tile_shape[0], tile_shape[1], title="tiled")
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_slice_returns_figure():
    fig = _build_slice_figure(Layout((4, 8), (8, 1)), (2, None))
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_returns_figure():
    l1 = Layout((4, 4), (4, 1))
    l2 = Layout((4, 4), (1, 4))
    fig = _build_composite_figure([l1, l2], titles=["Row", "Col"])
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_tensor_data_labels():
    """Tensor panels show data values, not raw offsets, in cell text."""
    t = Tensor(Layout(4, 1), data=list("WXYZ"))
    fig = _build_composite_figure([t], titles=["Data"])
    try:
        ax = fig.axes[0]
        cell_texts = [c.get_text() for c in ax.texts
                      if c.get_text() in ("W", "X", "Y", "Z")]
        assert len(cell_texts) == 4, f"expected data labels, got {cell_texts}"
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_layout_shows_offsets():
    """Layout panels show offset integers, not data."""
    fig = _build_composite_figure([Layout(4, 1)], titles=["Offsets"])
    try:
        ax = fig.axes[0]
        cell_texts = sorted(c.get_text() for c in ax.texts
                            if c.get_text().isdigit())
        assert "0" in cell_texts and "3" in cell_texts
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_auto_panel_size_compact_for_1d():
    """Auto panel_size produces compact height for rank-1 layouts."""
    fig = _build_composite_figure([Layout(8, 1)])
    try:
        _, h = fig.get_size_inches()
        assert h < 3.0, f"1-row layout should be compact, got height={h}"
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_auto_panel_size_scales_with_rows():
    """Auto panel_size grows taller for layouts with more rows."""
    fig_1d = _build_composite_figure([Layout(8, 1)])
    fig_2d = _build_composite_figure([Layout((4, 8), (8, 1))])
    try:
        _, h_1d = fig_1d.get_size_inches()
        _, h_2d = fig_2d.get_size_inches()
        assert h_2d > h_1d, f"4×8 should be taller than 1×8: {h_2d} vs {h_1d}"
    finally:
        plt.close(fig_1d)
        plt.close(fig_2d)



@requires_viz
def test_draw_composite_explicit_panel_size_overrides_auto():
    """Explicit panel_size takes precedence over auto-compute."""
    fig = _build_composite_figure([Layout(8, 1)], panel_size=(5.0, 5.0))
    try:
        w, h = fig.get_size_inches()
        assert abs(w - 5.0) < 0.1 and abs(h - 5.0) < 0.1
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_cell_labels_offset_kwarg():
    """cell_labels='offset' passed as kwarg forces offset display for Tensors."""
    t = Tensor(Layout(4, 1), data=list("WXYZ"))
    fig = _build_composite_figure([t], cell_labels="offset")
    try:
        ax = fig.axes[0]
        cell_texts = [c.get_text() for c in ax.texts]
        # Should show offsets (0, 1, 2, 3), not letters
        assert "0" in cell_texts and "3" in cell_texts
        assert "W" not in cell_texts
    finally:
        plt.close(fig)



@requires_viz
def test_draw_composite_per_panel_override_wins():
    """Per-panel option dict overrides top-level kwarg."""
    t = Tensor(Layout(4, 1), data=list("WXYZ"))
    # Top-level says offset, but per-panel says show data (True = auto)
    fig = _build_composite_figure(
        [(t, {"cell_labels": True})],
        cell_labels="offset",
    )
    try:
        ax = fig.axes[0]
        cell_texts = [c.get_text() for c in ax.texts
                      if c.get_text() in ("W", "X", "Y", "Z")]
        assert len(cell_texts) == 4
    finally:
        plt.close(fig)



@requires_viz
def test_draw_copy_layout_same_thread_colors_both_panels():
    """Src and dst panels should use the same color for the same thread."""
    src = Layout((4, 2), (2, 1))
    dst = Layout((4, 2), (1, 4))
    fig = _build_copy_figure(src, dst, colorize=True)
    try:
        ax = fig.axes[0]
        patches_list = ax.patches
        # 4×2=8 cells per panel, 2 panels = 16 patches
        assert len(patches_list) == 16
        src_colors = [p.get_facecolor() for p in patches_list[:8]]
        dst_colors = [p.get_facecolor() for p in patches_list[8:]]
        # Thread 0 should get the same color on both sides
        assert src_colors[0] == dst_colors[0]
    finally:
        plt.close(fig)



@requires_viz
def test_get_indices_2d_row_major_matches_logical_coordinates():
    layout = Layout((4, 3), (3, 1))
    indices = _get_indices_2d(layout)
    assert indices.tolist() == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]



@requires_viz
def test_get_indices_2d_column_major_matches_logical_coordinates():
    layout = Layout((4, 3), (1, 4))
    indices = _get_indices_2d(layout)
    assert indices.tolist() == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
    ]



@requires_viz
def test_get_color_indices_2d_by_row_matches_logical_coordinates():
    layout = Layout((4, 3), (3, 1))
    color_layout = Layout((4, 3), (1, 0))
    color_indices = _get_color_indices_2d(layout, color_layout)
    assert color_indices.tolist() == [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
    ]



@requires_viz
def test_get_color_indices_2d_by_column_matches_logical_coordinates():
    layout = Layout((4, 3), (3, 1))
    color_layout = Layout((4, 3), (0, 1))
    color_indices = _get_color_indices_2d(layout, color_layout)
    assert color_indices.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ]



@requires_viz
def test_get_color_indices_2d_uniform_layout_is_uniform():
    layout = Layout((4, 3), (3, 1))
    color_layout = Layout(1, 0)
    color_indices = _get_color_indices_2d(layout, color_layout)
    assert color_indices.tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]



@requires_viz
def test_get_color_indices_2d_1d_layout_is_not_treated_as_uniform():
    layout = Layout(4, 1)
    color_layout = Layout(4, 1)
    color_indices = _get_color_indices_2d(layout, color_layout)
    assert color_indices.tolist() == [[0, 1, 2, 3]]



@requires_viz
def test_get_hierarchical_cell_coords_2d_preserves_nested_coordinates():
    layout = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))
    coords = _get_hierarchical_cell_coords_2d(layout)
    assert coords[0, 0] == ((0, 0), (0, 0))
    assert coords[1, 0] == ((1, 0), (0, 0))
    assert coords[2, 0] == ((0, 1), (0, 0))
    assert coords[0, 1] == ((0, 0), (1, 0))
    assert coords[0, 2] == ((0, 0), (0, 1))



@requires_viz
def test_format_nested_coord_formats_hierarchical_labels():
    assert _format_nested_coord(3) == "3"
    assert _format_nested_coord((1, 2)) == "(1,2)"
    assert _format_nested_coord(((1, 2), 3)) == "((1,2),3)"



@requires_viz
def test_format_hierarchical_cell_lines_is_explicit_and_pedagogical():
    assert _format_hierarchical_cell_lines((1, 2), (3, 4), 17) == (
        "row=(1,2)",
        "col=(3,4)",
        "offset=17",
    )



@requires_viz
def test_coord_levels_flattens_nested_coordinates_for_axis_labels():
    assert _coord_levels(3) == (3,)
    assert _coord_levels((1, 2)) == (1, 2)
    assert _coord_levels(((1, 2), 3)) == (1, 2, 3)



@requires_viz
def test_level_spans_supports_three_level_hierarchy():
    assert _level_spans((2, 3, 4)) == (2, 6, 24)



@requires_viz
def test_level_block_sizes_supports_three_level_hierarchy():
    assert _level_block_sizes((2, 3, 4)) == (1, 2, 6)



@requires_viz
def test_draw_layout_nested_passes_color_indices_to_hierarchical_renderer(monkeypatch):
    layout = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))
    color_layout = Layout(layout.shape, ((1, 2), (0, 0)))
    seen = {}

    def fake_draw(ax, indices, rows, cols, **kwargs):
        seen["indices"] = indices
        seen["rows"] = rows
        seen["cols"] = cols
        seen["color_indices"] = kwargs.get("color_indices")

    monkeypatch.setattr(viz_mod, "_draw_hierarchical_grid", fake_draw)
    monkeypatch.setattr(viz_mod, "_save_figure", lambda fig, filename, dpi=150: plt.close(fig))

    draw_layout(
        layout,
        filename="ignored.png",
        flatten_hierarchical=False,
        color_layout=color_layout,
    )

    assert seen["rows"] == 4
    assert seen["cols"] == 4
    assert seen["color_indices"] is not None
    assert seen["color_indices"].shape == (4, 4)



@requires_viz
def test_draw_hierarchical_grid_uses_supplied_color_indices():
    layout = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))
    color_layout = Layout(layout.shape, ((1, 2), (0, 0)))
    color_indices = _get_color_indices_2d(layout, color_layout)

    fig, ax = plt.subplots()
    try:
        _call_draw_hierarchical_grid(
            ax,
            layout,
            colorize=True,
            color_indices=color_indices,
            flatten_hierarchical=False,
        )
        facecolors = [patch.get_facecolor() for patch in ax.patches]
        assert len(facecolors) == 16
        assert facecolors[0] == facecolors[1] == facecolors[2] == facecolors[3]
        assert facecolors[0] != facecolors[4]
    finally:
        plt.close(fig)


def _hierarchy_line_positions(ax, color):
    """Return horizontal/vertical hierarchy line positions for a given color."""
    horizontal = set()
    vertical = set()

    for line in ax.lines:
        if line.get_color() != color:
            continue
        x0, x1 = [float(x) for x in line.get_xdata()]
        y0, y1 = [float(y) for y in line.get_ydata()]
        if y0 == y1:
            horizontal.add(y0)
        elif x0 == x1:
            vertical.add(x0)

    return horizontal, vertical


def _hierarchy_line_zorders(ax, color):
    """Return zorders for hierarchy lines of a given color."""
    return [line.get_zorder() for line in ax.lines if line.get_color() == color]


def _label_bboxes(ax):
    """Return rendered bounding boxes for row[...] and col[...] margin labels."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row_boxes = []
    col_boxes = []
    for text in ax.texts:
        label = text.get_text()
        bbox = text.get_window_extent(renderer=renderer)
        if label.startswith("row["):
            row_boxes.append((label, bbox))
        elif label.startswith("col["):
            col_boxes.append((label, bbox))
    return row_boxes, col_boxes


def _has_bbox_overlap(boxes):
    """Return True if any pair of bounding boxes overlaps."""
    for i, (_, bbox_i) in enumerate(boxes):
        for _, bbox_j in boxes[i + 1:]:
            if Bbox.overlaps(bbox_i, bbox_j):
                return True
    return False


def _offset_label_value_bboxes(ax):
    """Return rendered bbox pairs for each 'offset=' label and its value."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pairs = []
    for i, text in enumerate(ax.texts[:-1]):
        if text.get_text() != "offset=":
            continue
        value_text = ax.texts[i + 1]
        if not value_text.get_text().lstrip("-").isdigit():
            continue
        pairs.append(
            (
                text.get_window_extent(renderer=renderer),
                value_text.get_window_extent(renderer=renderer),
            )
        )
    return pairs


def _cell_patch_bboxes(ax):
    """Return rendered bounding boxes for unit cell rectangles by (row, col)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = {}
    for patch in ax.patches:
        if not all(hasattr(patch, attr) for attr in ("get_x", "get_y", "get_width", "get_height")):
            continue
        if patch.get_width() != 1.0 or patch.get_height() != 1.0:
            continue
        boxes[(int(round(patch.get_y())), int(round(patch.get_x())))] = patch.get_window_extent(renderer=renderer)
    return boxes


def _cell_text_bboxes(ax, rows: int, cols: int):
    """Return rendered in-cell monospace text boxes grouped by (row, col)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = defaultdict(list)
    for text in ax.texts:
        if "monospace" not in text.get_fontfamily():
            continue
        x, y = text.get_position()
        if not (0.0 <= x < cols and 0.0 <= y < rows):
            continue
        boxes[(int(y), int(x))].append(text.get_window_extent(renderer=renderer))
    return boxes



@requires_viz
def test_draw_hierarchical_grid_draws_outer_perimeter_for_coarse_tiles():
    layout = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))

    fig, ax = plt.subplots()
    try:
        _call_draw_hierarchical_grid(ax, layout, flatten_hierarchical=False)
        blue = viz_mod.HIERARCHY_LEVEL_COLORS[0]
        horizontal, vertical = _hierarchy_line_positions(ax, blue)
        assert horizontal == {0.0, 2.0, 4.0}
        assert vertical == {0.0, 2.0, 4.0}
    finally:
        plt.close(fig)



@requires_viz
def test_draw_hierarchical_grid_cecka_hier_col_margin_labels_do_not_overlap():
    layout = Layout((4, (4, 2)), (4, (1, 16)))

    fig, ax = plt.subplots(figsize=(8 * 0.8 + 1, 4 * 0.8 + 1))
    try:
        _call_draw_hierarchical_grid(ax, layout, flatten_hierarchical=False,
                                label_hierarchy_levels=True)
        row_boxes, col_boxes = _label_bboxes(ax)
        assert row_boxes
        assert col_boxes
        assert not _has_bbox_overlap(row_boxes)
        assert not _has_bbox_overlap(col_boxes)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_hierarchical_grid_offset_values_clear_offset_equals_label():
    layout = Layout((4, (4, 2)), (4, (1, 16)))

    fig, ax = plt.subplots(figsize=(8 * 0.8 + 1, 4 * 0.8 + 1))
    try:
        _call_draw_hierarchical_grid(ax, layout, flatten_hierarchical=False,
                                label_hierarchy_levels=True)
        pairs = _offset_label_value_bboxes(ax)
        assert pairs
        min_gap = min(value_bbox.x0 - label_bbox.x1 for label_bbox, value_bbox in pairs)
        # With advance-width text measurement, the value starts at the correct
        # cursor position after "offset=".  The ink bounding boxes of the "="
        # glyph and a bold digit may overlap slightly, so we check that the
        # overlap is within one advance width (no full-character overlap).
        assert min_gap >= -1.0
    finally:
        plt.close(fig)



@requires_viz
def test_draw_layout_small_nested_hierarchy_keeps_text_inside_cells(monkeypatch):
    layout = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))
    seen = {}

    def fake_save(fig, filename, dpi=150):
        seen["fig"] = fig

    monkeypatch.setattr(viz_mod, "_save_figure", fake_save)
    draw_layout(
        layout,
        filename="ignored.svg",
        flatten_hierarchical=False,
        label_hierarchy_levels=True,
    )

    fig = seen["fig"]
    try:
        cell_boxes = _cell_patch_bboxes(fig.axes[0])
        text_boxes = _cell_text_bboxes(fig.axes[0], rows=4, cols=4)
        assert len(cell_boxes) == 16
        assert len(text_boxes) == 16

        for cell, boxes in text_boxes.items():
            cell_bbox = cell_boxes[cell]
            union_bbox = Bbox.union(boxes)
            assert union_bbox.x0 >= cell_bbox.x0 - 1.0
            assert union_bbox.x1 <= cell_bbox.x1 + 1.0
            assert union_bbox.y0 >= cell_bbox.y0 - 1.0
            assert union_bbox.y1 <= cell_bbox.y1 + 1.0
    finally:
        plt.close(fig)



@requires_viz
def test_draw_hierarchical_grid_leaves_corner_gap_between_axis_label_bands():
    layout = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))

    fig, ax = plt.subplots(figsize=(12 * 0.8 + 1, 6 * 0.8 + 1))
    try:
        _call_draw_hierarchical_grid(ax, layout, flatten_hierarchical=False,
                                label_hierarchy_levels=True)
        row_boxes, col_boxes = _label_bboxes(ax)
        assert row_boxes
        assert col_boxes

        max_row_x1 = max(bbox.x1 for _, bbox in row_boxes)
        min_col_x0 = min(bbox.x0 for _, bbox in col_boxes)
        max_row_y1 = max(bbox.y1 for _, bbox in row_boxes)
        min_col_y0 = min(bbox.y0 for _, bbox in col_boxes)

        assert min_col_x0 - max_row_x1 >= 10.0
        assert min_col_y0 - max_row_y1 >= 10.0
        assert not _has_bbox_overlap(row_boxes)
        assert not _has_bbox_overlap(col_boxes)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_hierarchical_grid_draws_outer_perimeter_for_multiple_levels():
    layout = Layout(((3, 2, 2, 2), (4, 2, 2, 2)),
                    ((1, 3, 6, 12), (24, 96, 192, 384)))

    fig, ax = plt.subplots()
    try:
        _call_draw_hierarchical_grid(ax, layout, flatten_hierarchical=False)
        blue = viz_mod.HIERARCHY_LEVEL_COLORS[0]
        orange = viz_mod.HIERARCHY_LEVEL_COLORS[1]
        green = viz_mod.HIERARCHY_LEVEL_COLORS[2]

        blue_horizontal, blue_vertical = _hierarchy_line_positions(ax, blue)
        orange_horizontal, orange_vertical = _hierarchy_line_positions(ax, orange)
        green_horizontal, green_vertical = _hierarchy_line_positions(ax, green)

        assert {0.0, 24.0}.issubset(blue_horizontal)
        assert {0.0, 32.0}.issubset(blue_vertical)
        assert {0.0, 24.0}.issubset(orange_horizontal)
        assert {0.0, 32.0}.issubset(orange_vertical)
        assert {0.0, 24.0}.issubset(green_horizontal)
        assert {0.0, 32.0}.issubset(green_vertical)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_hierarchical_grid_closes_boxes_for_column_only_hierarchy():
    layout = Layout((4, (4, 2)), (4, (1, 16)))

    fig, ax = plt.subplots()
    try:
        _call_draw_hierarchical_grid(ax, layout, flatten_hierarchical=False)
        blue = viz_mod.HIERARCHY_LEVEL_COLORS[0]
        blue_horizontal, blue_vertical = _hierarchy_line_positions(ax, blue)
        assert blue_horizontal == {0.0, 4.0}
        assert blue_vertical == {0.0, 4.0, 8.0}
    finally:
        plt.close(fig)



@requires_viz
def test_draw_hierarchical_grid_closes_boxes_for_coarse_column_only_level():
    layout = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))

    fig, ax = plt.subplots()
    try:
        _call_draw_hierarchical_grid(ax, layout, flatten_hierarchical=False)
        orange = viz_mod.HIERARCHY_LEVEL_COLORS[1]
        orange_horizontal, orange_vertical = _hierarchy_line_positions(ax, orange)
        assert orange_horizontal == {0.0, 6.0}
        assert orange_vertical == {0.0, 6.0, 12.0}
    finally:
        plt.close(fig)



@requires_viz
def test_draw_hierarchical_grid_draws_coarser_lines_above_finer_lines():
    layout = Layout(
        ((2, 3, 2), (3, 2, 2)),
        ((1, 2, 6), (12, 36, 72)),
    )

    fig, ax = plt.subplots()
    try:
        _call_draw_hierarchical_grid(ax, layout, flatten_hierarchical=False)
        blue = viz_mod.HIERARCHY_LEVEL_COLORS[0]
        orange = viz_mod.HIERARCHY_LEVEL_COLORS[1]
        blue_zorders = _hierarchy_line_zorders(ax, blue)
        orange_zorders = _hierarchy_line_zorders(ax, orange)
        assert blue_zorders
        assert orange_zorders
        assert max(blue_zorders) < min(orange_zorders)
    finally:
        plt.close(fig)



@requires_viz
def test_draw_slice_hierarchical_keeps_flat_grid_and_highlights_on_top(monkeypatch):
    layout = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))
    seen = {}

    def fake_save(fig, filename, dpi=150):
        ax = fig.axes[0]
        seen["line_count"] = len(ax.lines)
        red_edge = to_rgba(viz_mod.HIGHLIGHT_EDGE)
        seen["highlight_zorders"] = [
            patch.get_zorder()
            for patch in ax.patches
            if patch.get_edgecolor() == red_edge
        ]
        seen["base_zorders"] = [
            patch.get_zorder()
            for patch in ax.patches
            if patch.get_edgecolor() != red_edge
        ]
        plt.close(fig)

    monkeypatch.setattr(viz_mod, "_save_figure", fake_save)
    draw_slice(layout, (0, None), filename="ignored.png")

    assert seen["line_count"] == 0
    assert seen["highlight_zorders"]
    assert max(seen["base_zorders"]) < min(seen["highlight_zorders"])



@requires_viz
def test_slice_highlight_mask_tracks_logical_cells_not_offsets():
    layout = Layout((2, 2), (0, 1))
    mask = _get_slice_highlight_mask_2d(layout, (0, None))
    assert mask.tolist() == [
        [True, True],
        [False, False],
    ]



@requires_viz
def test_slice_highlight_mask_1d_tuple_spec():
    """1D layout with tuple slice_spec should highlight the correct elements."""
    layout = Layout(8, 1)
    mask = _get_slice_highlight_mask_2d(layout, (slice(2, 5),))
    assert mask.shape == (1, 8)
    assert mask.tolist() == [[False, False, True, True, True, False, False, False]]



@requires_viz
def test_slice_highlight_mask_1d_tuple_spec_rank1():
    """Rank-1 layout with tuple slice_spec should highlight the correct elements."""
    layout = Layout((8,), (1,))
    mask = _get_slice_highlight_mask_2d(layout, (slice(2, 5),))
    assert mask.shape == (1, 8)
    assert mask.tolist() == [[False, False, True, True, True, False, False, False]]



@requires_viz
def test_slice_highlight_mask_1d_tuple_int_spec():
    """1D layout with tuple (int,) slice_spec highlights a single element."""
    layout = Layout(8, 1)
    mask = _get_slice_highlight_mask_2d(layout, (3,))
    assert mask.shape == (1, 8)
    assert mask.tolist() == [[False, False, False, True, False, False, False, False]]



@requires_viz
def test_slice_highlight_mask_1d_tuple_none_spec():
    """1D layout with tuple (None,) selects all elements."""
    layout = Layout(4, 1)
    mask = _get_slice_highlight_mask_2d(layout, (None,))
    assert mask.tolist() == [[True, True, True, True]]



@requires_viz
def test_slice_highlight_mask_1d_wrong_tuple_length_raises():
    """1D layout with 2-element tuple slice_spec raises ValueError."""
    layout = Layout(4, 1)
    with pytest.raises(ValueError, match="1-element tuple"):
        _get_slice_highlight_mask_2d(layout, (1, 2))



@requires_viz
def test_compute_tv_mapping_uses_first_wins_for_duplicate_cells():
    layout = Layout((2, 2), (0, 0))
    tv_map = _compute_tv_mapping(layout, grid_rows=1, grid_cols=1)
    assert tv_map == {(0, 0): (0, 0, 0)}



@requires_viz
def test_compute_tv_mapping_raises_for_out_of_bounds_grid():
    layout = Layout((2, 2), (1, 2))
    with pytest.raises(ValueError, match="out of bounds"):
        _compute_tv_mapping(layout, grid_rows=1, grid_cols=1)



@requires_viz
def test_draw_swizzle_delegates_to_shared_builder(monkeypatch):
    layout = Layout((8, 64), (64, 1))
    swizzle = Swizzle(3, 4, 3)
    fig = plt.figure()

    def fake_builder(*args, **kwargs):
        return fig

    monkeypatch.setattr(viz_mod, "_build_swizzle_figure", fake_builder)
    try:
        assert draw_swizzle(layout, swizzle) is fig
    finally:
        plt.close(fig)



@requires_viz
def test_draw_swizzle_delegates_to_shared_builder(monkeypatch):
    layout = Layout((8, 64), (64, 1))
    swizzle = Swizzle(3, 4, 3)
    fig = plt.figure()
    seen = {}

    def fake_builder(*args, **kwargs):
        return fig

    def fake_save(passed_fig, filename, dpi=150):
        seen["fig"] = passed_fig
        seen["filename"] = filename
        plt.close(passed_fig)

    monkeypatch.setattr(viz_mod, "_build_swizzle_figure", fake_builder)
    monkeypatch.setattr(viz_mod, "_save_figure", fake_save)

    draw_swizzle(layout, swizzle, filename="out.png")
    assert seen["fig"] is fig
    assert seen["filename"] == "out.png"


# ── draw_combined_mma_grid / draw_combined_mma_grid ──────────────────────



@requires_viz
def test_draw_combined_mma_grid_smoke():
    atom = SM80_16x8x16_F16F16F16F16_TN
    atom_layout = Layout((2, 2), (1, 2))
    c_grid, _ = tile_mma_grid(atom, atom_layout, "C")
    a_grid, _ = tile_mma_grid(atom, atom_layout, "A")
    b_grid, _ = tile_mma_grid(atom, atom_layout, "B")

    # Transpose B for display (N×K → K×N)
    b_display = {(c, r): v for (r, c), v in b_grid.items()}

    M_a, N_a, K_a = atom.shape_mnk
    M, N, K = M_a * 2, N_a * 2, K_a

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_combined_mma_grid(a_grid, b_display, c_grid, M, N, K,
                               filename=f.name, title="test")



@requires_viz
def test_draw_combined_mma_grid_returns_figure():
    atom = SM80_16x8x16_F16F16F16F16_TN
    atom_layout = Layout((2, 2), (1, 2))
    c_grid, _ = tile_mma_grid(atom, atom_layout, "C")
    a_grid, _ = tile_mma_grid(atom, atom_layout, "A")
    b_grid, _ = tile_mma_grid(atom, atom_layout, "B")

    b_display = {(c, r): v for (r, c), v in b_grid.items()}

    M_a, N_a, K_a = atom.shape_mnk
    M, N, K = M_a * 2, N_a * 2, K_a

    fig = _build_combined_grid_figure(a_grid, b_display, c_grid, M, N, K,
                                 title="test")
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        plt.close(fig)
