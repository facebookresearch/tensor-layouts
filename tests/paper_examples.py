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

"""Tests derived from concrete examples in:

    Cecka, C. "CuTe Layout Representation and Algebra." arXiv:2603.02298v1 (2026).

Each test cites the specific figure, table, equation, or section it comes from.

Paper reference / coverage guide:
    Figure 1  -> test_fig1_*
    Figure 2  -> test_fig2_*
    Figure 3  -> test_fig3_*
    Figure 4  -> test_fig4_*
    Figure 5  -> test_fig5_*
    Figure 6  -> test_fig6_*
    Figure 7  -> test_fig7_*
    Figure 8  -> test_fig8_*
    Figure 9  -> test_fig9_*
    Figure 10 -> test_fig10_*
    Figure 11 -> test_fig11_*
    Figure 12 -> test_fig12_*
    Table 1   -> test_table1_*
    Table 2   -> test_table2_*
    Table 3   -> test_table3_*
    Table 4   -> test_table4_*
    Table 5   -> test_table5_*
    Table 6   -> test_table6_*
    Table 7   -> test_table7_*

Translation notes:
    Coordinate-semimodule examples in Figure 4, Tables 1, 4, 5, 6, and 7 are
    covered with explicit coordinate-label proxies, compatible-domain proxies,
    pointwise coordinate formulas, and complement-shift proxies checked
    against Eqs. (28) and (29) because ``Layout`` models affine integer
    strides directly.
    Binary-semimodule examples in Figure 4 and Tables 1, 4, 5, and 6 are
    covered with equivalent ``Swizzle``-based affine layouts and their
    associated F2 matrices.
    Table 3's grouped-reduction GETT shorthand is instantiated with explicit
    compact nested strides where this Python API requires congruent tuples.

Run with --draw to generate the corresponding paper figures into tests/figures/:

    pytest tests/paper_examples.py --draw
"""

import os
import subprocess
import sys
from itertools import product

import pytest

from tensor_layouts import *
from tensor_layouts.analysis import *


# ---------------------------------------------------------------------------
# Figure drawing infrastructure
# ---------------------------------------------------------------------------
#
# When pytest is invoked with --draw, figure tests also render the layout
# into tests/figures/ using the public tensor_layouts.viz functions.


def _format_coord(coord):
    if isinstance(coord, tuple):
        return "(" + ",".join(_format_coord(c) for c in coord) + ")"
    return str(coord)


def _grid_labels(row_shape, col_shape, label_fn):
    return [
        label_fn(idx2crd(i, row_shape), idx2crd(j, col_shape))
        for i in range(size(row_shape))
        for j in range(size(col_shape))
    ]


def _scale_structure(value, factor):
    if is_tuple(value):
        return tuple(_scale_structure(v, factor) for v in value)
    return value * factor


def display_layout(row_shape, col_shape):
    """Return a row-major display proxy over the flattened 2D grid."""
    return Layout(
        (row_shape, col_shape),
        (
            _scale_structure(compute_col_major_strides(row_shape), size(col_shape)),
            compute_col_major_strides(col_shape),
        ),
    )


def _build_tensor(layout, value_fn):
    tensor = Tensor(layout, data=[0] * max(cosize(layout), 1))
    rows = size(mode(layout.shape, 0))
    cols = size(mode(layout.shape, 1))
    for row in range(rows):
        for col in range(cols):
            tensor[row, col] = value_fn(row, col)
    return tensor


def _gemm(a_tensor, b_tensor, c_tensor):
    """Reference GEMM from §2.6.2 with the paper's loop order."""
    for k in range(size(mode(b_tensor.shape, 1))):
        for n in range(size(mode(b_tensor.shape, 0))):
            for m in range(size(mode(a_tensor.shape, 0))):
                c_tensor[m, n] = c_tensor[m, n] + a_tensor[m, k] * b_tensor[n, k]


def _copy(src_tensor, dst_tensor):
    """Reference COPY from Table 2: dst[i] = src[i]."""
    assert size(src_tensor.layout) == size(dst_tensor.layout)
    for i in range(size(dst_tensor.layout)):
        dst_tensor[i] = src_tensor[i]


def _flat_tensor(layout, value_fn):
    """Build a tensor by assigning values through flat indexing."""
    tensor = Tensor(layout, data=[None] * max(cosize(layout), 1))
    for i in range(size(layout)):
        tensor[i] = value_fn(i)
    return tensor


def _selected_cells(layout, tiler):
    selected = compose(layout, tiler)
    return {idx2crd(selected(i), layout.shape) for i in range(size(selected))}


def _expand_slice_component(spec, shape):
    if is_tuple(shape):
        if not is_tuple(spec):
            raise ValueError(
                f"Nested shape {shape!r} requires nested spec, got {spec!r}"
            )
        parts = [
            _expand_slice_component(subspec, subshape)
            for subspec, subshape in zip(spec, shape)
        ]
        return {coords for coords in product(*parts)}
    if spec is None:
        return set(range(shape))
    if isinstance(spec, slice):
        return set(range(*spec.indices(shape)))
    return {spec}


def _infer_slice_component(selected, shape):
    selected = list(selected)
    if is_tuple(shape):
        if not selected:
            raise ValueError(
                f"Cannot infer slice for empty selection in shape {shape!r}"
            )
        spec = tuple(
            _infer_slice_component([coord[i] for coord in selected], subshape)
            for i, subshape in enumerate(shape)
        )
        if _expand_slice_component(spec, shape) != set(selected):
            raise ValueError(
                f"Selection {selected!r} is not representable as a slice for shape {shape!r}"
            )
        return spec

    values = sorted(set(selected))
    if len(values) == 1:
        return values[0]
    if values == list(range(shape)):
        return None
    if values == list(range(values[0], values[-1] + 1)):
        return slice(values[0], values[-1] + 1)
    raise ValueError(
        f"Selection {values!r} is not representable as a flat slice of {shape}"
    )


def _common_subvector_slice_spec(layout, common):
    selected = [idx2crd(common(i), layout.shape) for i in range(size(common))]
    return _infer_slice_component(selected, layout.shape)


def _selected_coords_from_slice_spec(layout, slice_spec):
    row_spec, col_spec = slice_spec
    row_coords = _expand_slice_component(row_spec, mode(layout.shape, 0))
    col_coords = _expand_slice_component(col_spec, mode(layout.shape, 1))
    return {
        (row_coord, col_coord)
        for row_coord, col_coord in product(row_coords, col_coords)
    }


def _fig1_tensor(layout=None):
    tensor = Tensor(FIG1_BASE_LAYOUT, data=FIG1_TENSOR_DATA)
    if layout is None:
        return tensor
    return tensor.view(layout)


FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")


@pytest.fixture
def viz(request):
    """Return tensor_layouts.viz when drawing is enabled, else None."""
    if not request.config.getoption("draw"):
        return None
    try:
        import tensor_layouts.viz as viz_module
    except ImportError:
        pytest.skip("matplotlib not installed — cannot draw")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return viz_module


def _figure_path(name):
    return os.path.join(FIGURE_DIR, f"{name}.svg")


# =============================================================================
# Figure 1 — Tensor folding: 2×2×2 viewed as matrices
# =============================================================================

FIG1_BASE_LAYOUT = Layout((2, 2, 2), (2, 1, 4))
FIG1_TENSOR_DATA = tuple("abcdefgh")
FIG1_COLOR_LAYOUT_RANK3 = Layout((2, 2, 2), (1, 0, 2))
FIG1_COLOR_LAYOUT_4X2 = Layout((4, 2), (1, 0))
FIG1_COLOR_LAYOUT_2X4 = Layout((2, (2, 2)), (1, (0, 2)))


def test_fig1_rank3_tensor(viz):
    """Figure 1, row 1: a 2×2×2 tensor with Shape (2,2,2) : Stride (2,1,4)."""
    L = FIG1_BASE_LAYOUT
    tensor = _fig1_tensor()
    assert size(L) == 8
    # Offsets from Figure 1: a=0,b=1,c=2,d=3,e=4,f=5,g=6,h=7
    # mapping: (0,0,0)->0, (1,0,0)->2, (0,1,0)->1, (1,1,0)->3,
    #          (0,0,1)->4, (1,0,1)->6, (0,1,1)->5, (1,1,1)->7
    assert L(0, 0, 0) == 0
    assert L(1, 0, 0) == 2
    assert L(0, 1, 0) == 1
    assert L(1, 1, 0) == 3
    assert L(0, 0, 1) == 4
    assert L(1, 1, 1) == 7
    assert tensor[0, 0, 0] == "a"
    assert tensor[0, 1, 1] == "f"
    assert tensor[1, 1, 1] == "h"
    if not viz:
        return
    viz.draw_layout(
        tensor,
        filename=_figure_path("fig1_rank3_tensor"),
        title="Fig 1: (2,2,2):(2,1,4)",
        color_layout=FIG1_COLOR_LAYOUT_RANK3,
        colorize=True,
        num_colors=4,
    )


def test_fig1_fold_mode2_into_mode0(viz):
    """Figure 1, row 2: fold mode 2 into mode 0 → ((2,2),2):((2,4),1).

    This is a 4×2 matrix view. The flat representation is (4,2):(2,1)
    which is the coalesced version.
    """
    L = Layout(((2, 2), 2), ((2, 4), 1))
    assert size(L) == 8
    C = coalesce(L)
    assert C == Layout((4, 2), (2, 1))
    assert functionally_equal(L, C)
    tensor = _fig1_tensor(C)
    assert [tensor[row, col] for row in range(4) for col in range(2)] == list(
        "abcdefgh"
    )
    if not viz:
        return
    viz.draw_layout(
        tensor,
        filename=_figure_path("fig1_fold_mode2_into_mode0"),
        title="Fig 1: ((2,2),2):((2,4),1) — coalesced to (4,2):(2,1)",
        color_layout=FIG1_COLOR_LAYOUT_4X2,
        colorize=True,
        num_colors=4,
    )


def test_fig1_fold_mode2_into_mode1(viz):
    """Figure 1, row 3: fold mode 2 into mode 1 → (2,(2,2)):(2,(1,4)).

    This is a 2×4 matrix view. No flat representation exists (no single stride).
    """
    L = Layout((2, (2, 2)), (2, (1, 4)))
    tensor = _fig1_tensor(L)
    assert size(L) == 8
    # Verify offsets match the original tensor
    original = FIG1_BASE_LAYOUT
    for i in range(8):
        assert L(i) == original(i)
    assert [tensor[row, col] for row in range(2) for col in range(4)] == [
        "a",
        "b",
        "e",
        "f",
        "c",
        "d",
        "g",
        "h",
    ]
    if not viz:
        return
    viz.draw_layout(
        tensor,
        filename=_figure_path("fig1_fold_mode2_into_mode1"),
        title="Fig 1: (2,(2,2)):(2,(1,4))",
        color_layout=FIG1_COLOR_LAYOUT_2X4,
        colorize=True,
        num_colors=4,
    )


# =============================================================================
# Figure 2 — Coordinate sets for shapes
# =============================================================================


def test_fig2_shape_4():
    """Figure 2: S = 4, coordinate table Z4 = {0,1,2,3}."""
    S = 4
    for i in range(4):
        assert idx2crd(i, S) == i


def test_fig2_shape_2_3():
    """Figure 2: S = (2,3), coordinate table Z6 → Z(2,3)."""
    S = (2, 3)
    expected = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    for i, crd in enumerate(expected):
        assert idx2crd(i, S) == crd
        assert crd2idx(crd, S) == i


def test_fig2_shape_2_3_by_2():
    """Figure 2: S = ((2,3), 2), coordinate table Z12 → Z(6,2) → Z((2,3),2)."""
    S = ((2, 3), 2)
    # First few entries from the table
    expected = [
        ((0, 0), 0),  # 0
        ((1, 0), 0),  # 1
        ((0, 1), 0),  # 2
        ((1, 1), 0),  # 3
        ((0, 2), 0),  # 4
        ((1, 2), 0),  # 5
        ((0, 0), 1),  # 6
        ((1, 0), 1),  # 7
        ((0, 1), 1),  # 8
        ((1, 1), 1),  # 9
        ((0, 2), 1),  # 10
        ((1, 2), 1),  # 11
    ]
    for i, crd in enumerate(expected):
        assert idx2crd(i, S) == crd


# =============================================================================
# Figure 3 — Layout examples compatible with shape (4, 8)
# =============================================================================


def test_fig3a_col_major(viz):
    """Figure 3a: Col-Major (4,8):(1,4)."""
    L = Layout((4, 8), (1, 4))
    assert L(0, 0) == 0
    assert L(1, 0) == 1
    assert L(0, 1) == 4
    assert L(3, 7) == 31
    if not viz:
        return
    viz.draw_layout(
        L,
        filename=_figure_path("fig3a_col_major"),
        title="Fig 3a: Col-Major (4,8):(1,4)",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig3b_row_major(viz):
    """Figure 3b: Row-Major (4,8):(8,1)."""
    L = Layout((4, 8), (8, 1))
    assert L(0, 0) == 0
    assert L(1, 0) == 8
    assert L(0, 1) == 1
    assert L(3, 7) == 31
    if not viz:
        return
    viz.draw_layout(
        L,
        filename=_figure_path("fig3b_row_major"),
        title="Fig 3b: Row-Major (4,8):(8,1)",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig3c_col_major_padded(viz):
    """Figure 3c: Col-Major Padded (4,8):(1,5)."""
    L = Layout((4, 8), (1, 5))
    assert L(0, 0) == 0
    assert L(1, 0) == 1
    assert L(0, 1) == 5
    assert L(3, 7) == 38
    assert cosize(L) == 39
    assert not is_bijective(L)  # padded → gaps
    if not viz:
        return
    viz.draw_layout(
        L,
        filename=_figure_path("fig3c_col_major_padded"),
        title="Fig 3c: Col-Major Padded (4,8):(1,5)",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig3d_col_major_interleave(viz):
    """Figure 3d: Col-Major Interleave (4,(4,2)):(4,(1,16))."""
    L = Layout((4, (4, 2)), (4, (1, 16)))
    assert size(L) == 32
    assert L(0, (0, 0)) == 0
    assert L(0, (1, 0)) == 1
    assert L(0, (0, 1)) == 16
    if not viz:
        return
    viz.draw_layout(
        L,
        filename=_figure_path("fig3d_col_major_interleave"),
        title="Fig 3d: Col-Major Interleave (4,(4,2)):(4,(1,16))",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig3e_mixed(viz):
    """Figure 3e: Mixed ((2,2),(4,2)):((1,8),(2,16)).

    Paper gives L(22) = L(2,5) = L((0,1),(1,1)) = 26.
    """
    L = Layout(((2, 2), (4, 2)), ((1, 8), (2, 16)))
    assert size(L) == 32
    assert L(22) == 26
    # Verify the intermediate coordinates: 22 in shape (4,8) = (2, 5)
    assert L(2, 5) == 26
    if not viz:
        return
    viz.draw_layout(
        L,
        filename=_figure_path("fig3e_mixed"),
        title="Fig 3e: Mixed ((2,2),(4,2)):((1,8),(2,16))",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig3f_blocked_broadcast(viz):
    """Figure 3f: Blocked Broadcast ((2,2),(2,4)):((0,2),(0,4))."""
    L = Layout(((2, 2), (2, 4)), ((0, 2), (0, 4)))
    assert size(L) == 32
    assert not is_injective(L)  # broadcast → aliasing
    assert L(0, 0) == 0
    assert L(1, 0) == 0  # broadcast in first sub-mode
    if not viz:
        return
    viz.draw_layout(
        L,
        filename=_figure_path("fig3f_blocked_broadcast"),
        title="Fig 3f: Blocked Broadcast ((2,2),(2,4)):((0,2),(0,4))",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


# =============================================================================
# Figure 4 — Coordinate and binary-semimodule layouts
# =============================================================================


def test_fig4a_identity_coordinate(viz):
    """Figure 4a: coordinate tensor for (4,8):(e0,e1), computed via idx2crd()."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(
        layout,
        data=[
            _format_coord(idx2crd(offset, layout.shape))
            for offset in range(size(layout))
        ],
    )
    assert tensor[0, 0] == "(0,0)"
    assert tensor[3, 7] == "(3,7)"
    if not viz:
        return
    viz.draw_layout(
        tensor,
        filename=_figure_path("fig4a_identity_coordinate"),
        title="Fig 4a: (4,8):(e0,e1)",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig4b_transposed_block_coordinate(viz):
    """Figure 4b: coordinate tensor for (4,(4,2)):(e1,(e0,6e1)), via idx2crd()."""
    layout = Layout((4, (2, 4)), (4, (24, 1)))
    tensor = Tensor(
        layout,
        data=[
            _format_coord(idx2crd(offset, (4, 10))) for offset in range(cosize(layout))
        ],
    )
    assert [tensor[0, j] for j in range(8)] == [
        "(0,0)",
        "(0,6)",
        "(1,0)",
        "(1,6)",
        "(2,0)",
        "(2,6)",
        "(3,0)",
        "(3,6)",
    ]
    assert tensor[3, 7] == "(3,9)"
    if not viz:
        return
    viz.draw_layout(
        tensor,
        filename=_figure_path("fig4b_transposed_block_coordinate"),
        title="Fig 4b: (4,(4,2)):(e1,(e0,6e1))",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig4c_binary_swizzle(viz):
    """Figure 4c: (4,(4,3)):(f1,(f5,f16)) via a Swizzle-equivalent affine layout."""
    L = compose(Swizzle(2, 0, 2), Layout((4, (4, 3)), (1, (4, 16))))
    expected = [
        [0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47],
        [1, 4, 11, 14, 17, 20, 27, 30, 33, 36, 43, 46],
        [2, 7, 8, 13, 18, 23, 24, 29, 34, 39, 40, 45],
        [3, 6, 9, 12, 19, 22, 25, 28, 35, 38, 41, 44],
    ]
    for row in range(4):
        assert [L(row, col) for col in range(12)] == expected[row]
    if not viz:
        return
    viz.draw_layout(
        L,
        filename=_figure_path("fig4c_binary_swizzle"),
        title="Fig 4c: (4,(4,3)):(f1,(f5,f16))",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


# =============================================================================
# Table 1 — Layouts and their associated linear forms
# =============================================================================

FIG4_BINARY_SWIZZLE_4X4 = compose(Swizzle(2, 0, 2), Layout((4, 4), (1, 4)))


def test_table1_integer_linear_form():
    """Table 1: integer strides are columns of a 1×n Z-matrix."""
    L = Layout(((2, 2), (4, 2)), ((1, 8), (2, 16)))
    assert flatten(L).stride == (1, 8, 2, 16)
    for c0 in range(2):
        for c1 in range(2):
            for c2 in range(4):
                for c3 in range(2):
                    assert L((c0, c1), (c2, c3)) == c0 + 8 * c1 + 2 * c2 + 16 * c3


def test_table1_coordinate_linear_form():
    """Table 1: coordinate strides form the 2×3 matrix [e1, e0, 6e1]."""

    def coord_layout(row, col0, col1):
        return (col0, row + 6 * col1)

    assert coord_layout(1, 0, 0) == (0, 1)
    assert coord_layout(0, 1, 0) == (1, 0)
    assert coord_layout(0, 0, 1) == (0, 6)
    for row in range(4):
        for col0 in range(4):
            for col1 in range(2):
                assert coord_layout(row, col0, col1) == (col0, row + 6 * col1)


def test_table1_binary_linear_form():
    """Table 1: (4,4):(f1,f5) is the 4×4 F2 matrix shown in the paper."""
    assert to_F2_matrix(FIG4_BINARY_SWIZZLE_4X4) == [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


# =============================================================================
# Figure 5 — Tensor slicing
# =============================================================================


def test_fig5_tensor_A(viz):
    """Figure 5: Tensor A = {0} ◦ ((3,2),((2,3),2)) : ((4,1),((2,15),100))."""
    L = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    assert size(L) == 72  # 6 × 12
    assert L(0) == 0
    assert L(1) == 4  # stride 4 in first sub-mode
    if not viz:
        return
    viz.draw_layout(
        L,
        filename=_figure_path("fig5_tensor_A"),
        title="Fig 5: Tensor A",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig5_slice_row(viz):
    """Figure 5: A(2, _) = {8} ◦ ((2,3),2) : ((2,15),100).

    Slicing at row=2: offset = L(2, 0) = 8, remaining layout for the columns.
    """
    L = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    assert L(2, 0) == 8
    remaining = Layout(((2, 3), 2), ((2, 15), 100))
    for j in range(size(remaining)):
        assert L(2, j) == 8 + remaining(j)
    if not viz:
        return
    viz.draw_slice(
        L,
        (2, None),
        filename=_figure_path("fig5_slice_row"),
        title="Fig 5: A(2, _)",
        color_layout=Layout(1, 0),
        num_colors=1,
        highlight_facecolor="#8C8C8C",
        highlight_edgecolor="black",
        base_facecolor="white",
        show_text=True,
    )


def test_fig5_slice_col(viz):
    """Figure 5: A(_, 5) = {32} ◦ (3,2) : (4,1).

    Slicing at col=5: offset = L(0, 5) = 32, remaining layout for the rows.
    """
    L = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    assert L(0, 5) == 32
    remaining = Layout((3, 2), (4, 1))
    for i in range(size(remaining)):
        assert L(i, 5) == 32 + remaining(i)
    if not viz:
        return
    viz.draw_slice(
        L,
        (None, 5),
        filename=_figure_path("fig5_slice_col"),
        title="Fig 5: A(_, 5)",
        color_layout=Layout(1, 0),
        num_colors=1,
        highlight_facecolor="#8C8C8C",
        highlight_edgecolor="black",
        base_facecolor="white",
        show_text=True,
    )


def test_fig5_slice_nested_col_block(viz):
    """Figure 5: A(2, ((0, _), _)) = {8} ◦ (3,2) : (15,100)."""
    L = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    spec = (2, ((0, None), None))
    remaining, offset = slice_and_offset(spec, L)
    assert offset == 8
    assert remaining == Layout((3, 2), (15, 100))
    tensor_slice = Tensor(L)[spec]
    assert tensor_slice == Tensor(remaining, offset=offset)
    for j0 in range(3):
        for j1 in range(2):
            assert L(2, ((0, j0), j1)) == offset + remaining(j0, j1)
    if not viz:
        return
    viz.draw_slice(
        L,
        spec,
        filename=_figure_path("fig5_slice_nested_col_block"),
        title="Fig 5: A(2, ((0, _), _))",
        color_layout=Layout(1, 0),
        num_colors=1,
        highlight_facecolor="#8C8C8C",
        highlight_edgecolor="black",
        base_facecolor="white",
        show_text=True,
    )


def test_fig5_slice_fixed_inner_row_and_col(viz):
    """Figure 5: A((_, 1), (_, 0)) = {1} ◦ (3,(2,3)) : (4,(2,15))."""
    L = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    spec = ((None, 1), (None, 0))
    remaining, offset = slice_and_offset(spec, L)
    assert offset == 1
    assert remaining == Layout((3, (2, 3)), (4, (2, 15)))
    tensor_slice = Tensor(L)[spec]
    assert tensor_slice == Tensor(remaining, offset=offset)
    for i0 in range(3):
        for j in range(6):
            assert L((i0, 1), (j, 0)) == offset + remaining(i0, j)
    if not viz:
        return
    viz.draw_slice(
        L,
        spec,
        filename=_figure_path("fig5_slice_fixed_inner_row_and_col"),
        title="Fig 5: A((_, 1), (_, 0))",
        color_layout=Layout(1, 0),
        num_colors=1,
        highlight_facecolor="#8C8C8C",
        highlight_edgecolor="black",
        base_facecolor="white",
        show_text=True,
    )


def test_fig5_slice_outer_row_first_col_second_block(viz):
    """Figure 5: A((_, 0), ((0, _), 1)) = {100} ◦ (3,3) : (4,15)."""
    L = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    spec = ((None, 0), ((0, None), 1))
    remaining, offset = slice_and_offset(spec, L)
    assert offset == 100
    assert remaining == Layout((3, 3), (4, 15))
    tensor_slice = Tensor(L)[spec]
    assert tensor_slice == Tensor(remaining, offset=offset)
    for i0 in range(3):
        for j0 in range(3):
            assert L((i0, 0), ((0, j0), 1)) == offset + remaining(i0, j0)
    if not viz:
        return
    viz.draw_slice(
        L,
        spec,
        filename=_figure_path("fig5_slice_outer_row_first_col_second_block"),
        title="Fig 5: A((_, 0), ((0, _), 1))",
        color_layout=Layout(1, 0),
        num_colors=1,
        highlight_facecolor="#8C8C8C",
        highlight_edgecolor="black",
        base_facecolor="white",
        show_text=True,
    )


def test_fig5_slice_inner_row_and_first_col_modes(viz):
    """Figure 5: A((1, _), ((_, 0), _)) = {4} ◦ (2,(2,2)) : (1,(2,100))."""
    L = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    spec = ((1, None), ((None, 0), None))
    remaining, offset = slice_and_offset(spec, L)
    assert offset == 4
    assert remaining == Layout((2, (2, 2)), (1, (2, 100)))
    tensor_slice = Tensor(L)[spec]
    assert tensor_slice == Tensor(remaining, offset=offset)
    for i1 in range(2):
        for j0 in range(2):
            for j1 in range(2):
                assert L((1, i1), ((j0, 0), j1)) == offset + remaining(i1, (j0, j1))
    if not viz:
        return
    viz.draw_slice(
        L,
        spec,
        filename=_figure_path("fig5_slice_inner_row_and_first_col_modes"),
        title="Fig 5: A((1, _), ((_, 0), _))",
        color_layout=Layout(1, 0),
        num_colors=1,
        highlight_facecolor="#8C8C8C",
        highlight_edgecolor="black",
        base_facecolor="white",
        show_text=True,
    )


# =============================================================================
# Table 2 — COPY application layouts
# =============================================================================


def test_table2_copy_1d_arrays():
    """Table 2: 1D Arrays src=8:1, dst=8:1."""
    layout = Layout(8, 1)
    src = _flat_tensor(layout, lambda i: i)
    dst = Tensor(layout, data=[None] * max(cosize(layout), 1))
    _copy(src, dst)
    assert dst.data == list(range(8))


def test_table2_copy_nd_arrays():
    """Table 2: ND Arrays src=(8,2,3):(1,16,32), dst=(8,2,3):(1,16,32)."""
    layout = Layout((8, 2, 3), (1, 16, 32))
    src = _flat_tensor(layout, lambda i: i)
    dst = Tensor(layout, data=[None] * max(cosize(layout), 1))
    _copy(src, dst)
    assert dst.data == src.data
    assert sum(v is not None for v in dst.data) == 48


def test_table2_copy_gather():
    """Table 2: Gather src=(2,3,2):(42,1,128), dst=12:1."""
    src = _flat_tensor(Layout((2, 3, 2), (42, 1, 128)), lambda i: i)
    dst = Tensor(Layout(12, 1), data=[None] * 12)
    _copy(src, dst)
    assert dst.data == list(range(12))


def test_table2_copy_scatter():
    """Table 2: Scatter src=12:1, dst=(2,3,2):(42,1,128)."""
    src = _flat_tensor(Layout(12, 1), lambda i: i)
    dst_layout = Layout((2, 3, 2), (42, 1, 128))
    dst = Tensor(dst_layout, data=[None] * max(cosize(dst_layout), 1))
    _copy(src, dst)
    assert sum(v is not None for v in dst.data) == 12
    for i in range(12):
        assert dst.data[dst_layout(i)] == i


def test_table2_copy_broadcast():
    """Table 2: Broadcast src=7:0, dst=7:1."""
    src = Tensor(Layout(7, 0), data=[42])
    dst = Tensor(Layout(7, 1), data=[None] * 7)
    _copy(src, dst)
    assert dst.data == [42] * 7


def test_table2_copy_constant():
    """Table 2: Constant src=7:0, dst=7:0."""
    src = Tensor(Layout(7, 0), data=[17])
    dst = Tensor(Layout(7, 0), data=[-1])
    _copy(src, dst)
    assert dst.data == [17]
    for i in range(7):
        assert dst[i] == 17


def test_table2_copy_transpose():
    """Table 2: Transpose src=(8,3):(1,8), dst=(8,3):(3,1)."""
    src_layout = Layout((8, 3), (1, 8))
    dst_layout = Layout((8, 3), (3, 1))
    src = _build_tensor(src_layout, lambda row, col: 10 * row + col)
    dst = Tensor(dst_layout, data=[None] * max(cosize(dst_layout), 1))
    _copy(src, dst)
    assert dst.data == [10 * row + col for row in range(8) for col in range(3)]
    for row in range(8):
        for col in range(3):
            assert dst[row, col] == src[row, col]


def test_table2_copy_tensor_transpose():
    """Table 2: Tensor Transpose src=(8,(3,5)):(1,(57,8)), dst=(8,15):(1,8)."""
    src_layout = Layout((8, (3, 5)), (1, (57, 8)))
    dst_layout = Layout((8, 15), (1, 8))
    src = Tensor(src_layout, data=[None] * max(cosize(src_layout), 1))
    for row in range(8):
        for col0 in range(3):
            for col1 in range(5):
                src[row, (col0, col1)] = 100 * row + 10 * col0 + col1
    dst = Tensor(dst_layout, data=[None] * max(cosize(dst_layout), 1))
    _copy(src, dst)
    assert compatible(dst_layout.shape, src_layout.shape)
    for row in range(8):
        for col0 in range(3):
            for col1 in range(5):
                assert dst[row, col0 + 3 * col1] == 100 * row + 10 * col0 + col1


# =============================================================================
# Table 3 — GEMM applications
# =============================================================================


def test_table3_gemm_applications():
    """Table 3: the generic GEMM loop handles each listed application layout."""
    cases = [
        (
            "NT GEMM",
            Layout((3, 4), (1, 5)),
            Layout((2, 4), (1, 4)),
            Layout((3, 2), (1, 3)),
        ),
        (
            "TN GEMM",
            Layout((3, 4), (6, 1)),
            Layout((2, 4), (5, 1)),
            Layout((3, 2), (1, 4)),
        ),
        (
            "NTT GEMM",
            Layout((2, 4), (1, 4)),
            Layout((3, 4), (1, 5)),
            Layout((2, 3), (1, 3)),
        ),
        (
            "BLIS GEMM",
            Layout((3, 4), (2, 7)),
            Layout((2, 4), (3, 11)),
            Layout((3, 2), (5, 13)),
        ),
        (
            "GETT row modes",
            Layout(((2, 3), 2), ((1, 4), 17)),
            Layout((4, 2), (2, 1)),
            Layout(((2, 3), 4), ((1, 5), 23)),
        ),
        (
            "GETT grouped K modes",
            # Expand the paper's compact grouped-K shorthand to explicit
            # congruent nested strides: ((W,X),1) -> ((W,X),(1,K1)).
            Layout(((2, 3), (2, 2)), ((5, 19), (1, 2))),
            Layout(((2, 2), (2, 2)), ((7, 23), (1, 2))),
            Layout(((2, 3), (2, 2)), ((1, 2), (6, 12))),
        ),
        (
            "CONV",
            Layout((2, (1, 1, 2, 2)), (1, (2, 2, 2, 4))),
            Layout(
                (((1, 1, 2, 2)), (1, 1, 2, 2)),
                (((4, 4, 4, 8)), (1, 1, 1, 2)),
            ),
            Layout((2, (1, 1, 2, 2)), (1, (2, 2, 2, 4))),
        ),
    ]

    for name, a_layout, b_layout, c_layout in cases:
        a_tensor = _build_tensor(a_layout, lambda m, k: 10 * (m + 1) + (k + 1))
        b_tensor = _build_tensor(b_layout, lambda n, k: 100 * (n + 1) + 3 * (k + 1))
        c_tensor = _build_tensor(c_layout, lambda m, n: 0)
        _gemm(a_tensor, b_tensor, c_tensor)

        rows = size(mode(c_layout.shape, 0))
        cols = size(mode(c_layout.shape, 1))
        k_extent = size(mode(a_layout.shape, 1))
        for m in range(rows):
            for n in range(cols):
                expected = sum(
                    (10 * (m + 1) + (k + 1)) * (100 * (n + 1) + 3 * (k + 1))
                    for k in range(k_extent)
                )
                assert c_tensor[m, n] == expected, (name, m, n)


# =============================================================================
# §3.1 — Concatenation
# =============================================================================


def test_s3_1_concatenation():
    """Eq. 11: L(c) = L0(c0) + L1(c1) + ... + Ln(cn)."""
    L = Layout((3, 4), (2, 6))
    L0 = Layout(3, 2)
    L1 = Layout(4, 6)
    for c0 in range(3):
        for c1 in range(4):
            assert L(c0, c1) == L0(c0) + L1(c1)


# =============================================================================
# §3.2 — Coalesce
# =============================================================================


def test_s3_2_coalesce_example1():
    """§3.2: coalesce((2,(1,6)) : (1,(6,2))) = 12:1."""
    L = Layout((2, (1, 6)), (1, (6, 2)))
    C = coalesce(L)
    assert C == Layout(12, 1)
    assert functionally_equal(L, C)


def test_s3_2_coalesce_example2():
    """§3.2: ((4,3),5):((15,1),3) coalesces to (4,15):(15,1)."""
    L = Layout(((4, 3), 5), ((15, 1), 3))
    C = coalesce(L)
    assert C == Layout((4, 15), (15, 1))
    assert functionally_equal(L, C)


def test_s3_2_coalesce_by_mode():
    """§3.2: (2,(1,6)):(1,(6,2)) coalesced by-mode = (2,6):(1,2)."""
    L = Layout((2, (1, 6)), (1, (6, 2)))
    C = coalesce(L, profile=(None, None))
    assert C == Layout((2, 6), (1, 2))
    assert functionally_equal(L, C)


# =============================================================================
# §3.3 — Composition
# =============================================================================


def test_s3_3_1_identity_layouts():
    """§3.3.1: Identity layouts I_24 satisfy L(i) = i for all i ∈ Z24."""
    identities = [
        Layout(24, 1),
        Layout((4, 6), (1, 4)),
        Layout((3, (4, 2)), (1, (3, 12))),
    ]
    for I in identities:
        for i in range(24):
            assert I(i) == i


def test_s3_3_1_associativity_holds():
    """§3.3.1: A ◦ (B ◦ C) = (A ◦ B) ◦ C when image(C) ⊆ Z(B)."""
    A = Layout((4, 8), (1, 4))
    B = Layout(16, 2)
    C = Layout(4, 1)
    lhs = compose(A, compose(B, C))
    rhs = compose(compose(A, B), C)
    for i in range(size(C)):
        assert lhs(i) == rhs(i)


def test_s3_3_1_associativity_fails():
    """§3.3.1: Associativity fails when image(C) ⊄ Z(B).

    (5,3):(1,7) ◦ [4:1 ◦ 2:5] = 2:7, but
    [(5,3):(1,7) ◦ 4:1] ◦ 2:5 = 2:5
    """
    A = Layout((5, 3), (1, 7))
    B = Layout(4, 1)
    C = Layout(2, 5)
    lhs = compose(A, compose(B, C))
    rhs = compose(compose(A, B), C)
    # The paper says these yield different results
    lhs_val = lhs(1)
    rhs_val = rhs(1)
    assert lhs_val == 7  # A(B(C(1))) = A(B(5)) = A(5) = 1*5 mod.. = 7
    assert rhs_val == 5  # (A◦B)(C(1)) = (A◦B)(5) = 5


def test_s3_3_3_rank1_compose():
    """§3.3.3: Compositions with rank-1 A are trivial: S0:D0 ◦ s:d = s:(D0·d)."""
    assert compose(Layout(7, 11), Layout(3, 4)) == Layout(3, 44)


def test_s3_3_3_rank1_distributes():
    """§3.3.3: 7:11 ◦ (3,5):(6,3) = (3,5):(66,33)."""
    result = compose(Layout(7, 11), Layout((3, 5), (6, 3)))
    assert result == Layout((3, 5), (66, 33))


def test_s3_3_3_intuition_example():
    """§3.3.3: (4,6,8,10):(2,3,5,7) ◦ 6:12 = (2,3):(9,5)."""
    A = Layout((4, 6, 8, 10), (2, 3, 5, 7))
    B = Layout(6, 12)
    C = compose(A, B)
    assert C == Layout((2, 3), (9, 5))
    for i in range(6):
        assert C(i) == A(B(i))


def test_s3_3_3_stride_violation_raises():
    """§3.3.3: (4,6,8):(2,3,5) ◦ 6:3 is rejected as a divisibility violation.

    CuTe C++ 4.5.0 accepts this via a weak stride check (rest_stride < curr_shape)
    but produces functionally incorrect results (C(i) ≠ A(B(i)) for 4/6 inputs).
    pycute correctly rejects it; we follow pycute.
    """
    A = Layout((4, 6, 8), (2, 3, 5))
    B = Layout(6, 3)

    try:
        compose(A, B)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        pytest.fail("compose((4,6,8):(2,3,5), 6:3) should raise ValueError")


def test_s3_3_3_apparent_violation():
    """§3.3.3: (4,2,8):(3,12,97) ◦ 3:3 = 3:9 after coalescing.

    The paper notes this "seemingly violates" stride divisibility but is
    resolved by coalescing A first, then truncating unreachable modes.
    CuTe C++ handles this via the rest_stride < curr_shape path in
    composition_impl (layout.hpp:1077).
    """
    A = Layout((4, 2, 8), (3, 12, 97))
    B = Layout(3, 3)
    C = compose(A, B)
    assert C == Layout(3, 9)
    for i in range(3):
        assert C(i) == A(B(i))


def test_s3_3_3_apparent_violation_fail_left_raises():
    """§3.3.3: (4,2,8):(3,12,97) ◦ 4:3 cannot be sufficiently truncated."""
    A = Layout((4, 2, 8), (3, 12, 97))
    B = Layout(4, 3)
    with pytest.raises(ValueError, match="divisible"):
        compose(A, B)


def test_s3_3_3_apparent_violation_fail_right_raises():
    """§3.3.3: (4,2,8):(3,15,97) ◦ 3:3 cannot be coalesced to a valid composition."""
    A = Layout((4, 2, 8), (3, 15, 97))
    B = Layout(3, 3)
    with pytest.raises(ValueError, match="divisible"):
        compose(A, B)


# =============================================================================
# Figure 6 — Thread-value partitioning
# =============================================================================

THR_VAL_LAYOUT_C = Layout(((4, 8), 2), ((16, 1), 8))


def test_fig6_thread_value_partitioning(viz):
    """Figure 6: inverse display of ThrValLayoutC over the 8×8 C-matrix."""
    t_shape = mode(THR_VAL_LAYOUT_C.shape, 0)
    v_shape = mode(THR_VAL_LAYOUT_C.shape, 1)
    assert size(t_shape) == 32
    assert size(v_shape) == 2
    for flat_t in range(size(t_shape)):
        t_coord = idx2crd(flat_t, t_shape)
        for flat_v in range(size(v_shape)):
            offset = THR_VAL_LAYOUT_C(t_coord, flat_v)
            assert offset % 8 == t_coord[1]
            assert offset // 8 == 2 * t_coord[0] + flat_v
    if not viz:
        return
    viz.draw_tv_layout(
        THR_VAL_LAYOUT_C,
        filename=_figure_path("fig6_thread_value_partitioning"),
        title="Fig 6: ThrValLayoutC inverse over 8×8",
        grid_shape=(8, 8),
        colorize=True,
        num_colors=4,
    )


# =============================================================================
# §3.3.4 — Application: Partitioning (Table 4)
# =============================================================================


def test_table4_colmajor():
    """Table 4: ColMajor (8,8):(1,8) composed with TV layout."""
    data = Layout((8, 8), (1, 8))
    result = compose(data, THR_VAL_LAYOUT_C)
    assert result == THR_VAL_LAYOUT_C
    for i in range(size(THR_VAL_LAYOUT_C)):
        assert result(i) == data(THR_VAL_LAYOUT_C(i))


def test_table4_rowmajor():
    """Table 4: RowMajor (8,8):(8,1) composed with TV layout."""
    data = Layout((8, 8), (8, 1))
    result = compose(data, THR_VAL_LAYOUT_C)
    assert result == Layout(((4, 8), 2), ((2, 8), 1))
    for i in range(size(THR_VAL_LAYOUT_C)):
        assert result(i) == data(THR_VAL_LAYOUT_C(i))


def test_table4_padded():
    """Table 4: Padded (8,8):(1,9) composed with TV layout."""
    data = Layout((8, 8), (1, 9))
    result = compose(data, THR_VAL_LAYOUT_C)
    assert result == Layout(((4, 8), 2), ((18, 1), 9))
    for i in range(size(THR_VAL_LAYOUT_C)):
        assert result(i) == data(THR_VAL_LAYOUT_C(i))


def test_table4_colinterleaved():
    """Table 4: ColInterleaved row from the paper."""
    data = Layout(((4, 2), (2, 4)), ((2, 16), (1, 8)))
    result = compose(data, THR_VAL_LAYOUT_C)
    assert result == Layout(((4, (4, 2)), 2), ((8, (2, 16)), 1))
    for i in range(size(THR_VAL_LAYOUT_C)):
        assert result(i) == data(THR_VAL_LAYOUT_C(i))


def test_table4_swizzled():
    """Table 4: Swizzled (8,8):(f1,f9) via a Swizzle-equivalent affine layout."""
    data = compose(Swizzle(3, 0, 3), Layout((8, 8), (1, 8)))
    result = compose(data, THR_VAL_LAYOUT_C)
    expected = compose(Swizzle(3, 0, 3), THR_VAL_LAYOUT_C)
    assert result == expected
    for i in range(size(THR_VAL_LAYOUT_C)):
        assert result(i) == data(THR_VAL_LAYOUT_C(i))


def test_table4_coordinate():
    """Table 4: Coordinate (8,8):(e0,e1) gives ((2e1,e0),e1) over (Thr,Val)."""
    for thread in range(size(mode(THR_VAL_LAYOUT_C.shape, 0))):
        t0, t1 = idx2crd(thread, (4, 8))
        for value in range(size(mode(THR_VAL_LAYOUT_C.shape, 1))):
            assert idx2crd(THR_VAL_LAYOUT_C(thread, value), (8, 8)) == (
                t1,
                2 * t0 + value,
            )


# =============================================================================
# §3.3.5 — Tilers (Figure 7)
# =============================================================================


def test_s3_3_5_shape_as_tiler():
    """§3.3.5: (4,8) ≡ <4,8> ≡ <4:1, 8:1>. All equivalent as tilers."""
    L = Layout((8, 16), (1, 8))
    r1 = compose(L, (4, 8))
    r2 = compose(L, Tile(Layout(4, 1), Layout(8, 1)))
    assert r1 == r2


@pytest.mark.parametrize(
    ("name", "title", "tiler", "display_row_shape", "slice_spec", "expected"),
    [
        (
            "fig7a_tiler_4_8",
            "Fig 7a: <4:1, 8:1> ≡ <4,8>",
            (Layout(4, 1), Layout(8, 1)),
            8,
            (slice(0, 4), slice(0, 8)),
            {(row, col) for row in range(4) for col in range(8)},
        ),
        (
            "fig7b_tiler_2x2_1x4_8_1",
            "Fig 7b: <(2,2):(1,4), 8:1>",
            (Layout((2, 2), (1, 4)), Layout(8, 1)),
            (2, 4),
            ((None, slice(0, 4, 2)), slice(0, 8)),
            {(row, col) for row in (0, 1, 4, 5) for col in range(8)},
        ),
        (
            "fig7c_tiler_2x2_1x4_8_2",
            "Fig 7c: <(2,2):(1,4), 8:2>",
            (Layout((2, 2), (1, 4)), Layout(8, 2)),
            (2, 4),
            ((None, slice(0, 4, 2)), slice(0, 16, 2)),
            {(row, col) for row in (0, 1, 4, 5) for col in range(0, 16, 2)},
        ),
        (
            "fig7d_tiler_4_2_8_2",
            "Fig 7d: <4:2, 8:2>",
            (Layout(4, 2), Layout(8, 2)),
            8,
            (slice(0, 8, 2), slice(0, 16, 2)),
            {(row, col) for row in range(0, 8, 2) for col in range(0, 16, 2)},
        ),
    ],
)
def test_fig7_tilers_extract_expected_sublayouts(
    viz, name, title, tiler, display_row_shape, slice_spec, expected
):
    """Figure 7: tilers select the expected 4×8 sublayout from an 8×16 layout."""
    base = Layout((8, 16), (1, 8))
    assert _selected_cells(base, tiler) == expected
    if not viz:
        return
    viz.draw_slice(
        display_layout(display_row_shape, 16),
        slice_spec,
        filename=_figure_path(name),
        title=title,
        color_layout=Layout(1, 0),
        num_colors=1,
        highlight_facecolor="#8C8C8C",
        highlight_edgecolor="black",
        base_facecolor="white",
    )


# =============================================================================
# §3.4 — Inverses (Tables 5 and 6)
# =============================================================================


def test_table5_right_inverse_colmajor():
    """Table 5: right_inverse((4,8):(1,4)) = 32:1."""
    L = Layout((4, 8), (1, 4))
    R = right_inverse(L)
    assert R == Layout(32, 1)
    for i in range(size(R)):
        assert L(R(i)) == i


def test_table5_right_inverse_rowmajor():
    """Table 5: right_inverse((4,8):(8,1)) = (8,4):(4,1)."""
    L = Layout((4, 8), (8, 1))
    R = right_inverse(L)
    assert R == Layout((8, 4), (4, 1))
    for i in range(size(R)):
        assert L(R(i)) == i


def test_table5_right_inverse_padded():
    """Table 5: right_inverse((4,8):(1,5)) = 4:1. Smaller for non-contiguous."""
    L = Layout((4, 8), (1, 5))
    R = right_inverse(L)
    assert R == Layout(4, 1)
    for i in range(size(R)):
        assert L(R(i)) == i


def test_table5_right_inverse_broadcast_even_stride():
    """Table 5: right_inverse(((2,2),(2,4)):((0,2),(0,4))) = 1:0. Trivial."""
    L = Layout(((2, 2), (2, 4)), ((0, 2), (0, 4)))
    R = right_inverse(L)
    assert R == Layout(1, 0)


def test_table5_right_inverse_rank3():
    """Table 5: right_inverse((3,7,5):(5,15,1)) = (5,21):(21,1)."""
    L = Layout((3, 7, 5), (5, 15, 1))
    R = right_inverse(L)
    assert R == Layout((5, 21), (21, 1))
    for i in range(size(R)):
        assert L(R(i)) == i


def test_table5_right_inverse_nested_colmajor():
    """Table 5: right_inverse((4,(4,2)):(4,(1,16))) = (4,4,2):(4,1,16)."""
    L = Layout((4, (4, 2)), (4, (1, 16)))
    R = right_inverse(L)
    assert R == Layout((4, 4, 2), (4, 1, 16))
    for i in range(size(R)):
        assert L(R(i)) == i


def test_table5_right_inverse_nested_mixed():
    """Table 5: right_inverse(((2,2),(4,2)):((1,8),(2,16))) = (2,4,2,2):(1,4,2,16)."""
    L = Layout(((2, 2), (4, 2)), ((1, 8), (2, 16)))
    R = right_inverse(L)
    assert R == Layout((2, 4, 2, 2), (1, 4, 2, 16))
    for i in range(size(R)):
        assert L(R(i)) == i


def test_table5_right_inverse_broadcast_unit_stride():
    """Table 5 (corrected): right_inverse(((2,2),(2,4)):((0,1),(0,2))) = (2,4):(2,8)."""
    L = Layout(((2, 2), (2, 4)), ((0, 1), (0, 2)))
    R = right_inverse(L)
    assert R == Layout((2, 4), (2, 8))
    for i in range(size(R)):
        assert L(R(i)) == i


def test_table5_right_inverse_coordinate_identity():
    """Table 5: (4,8):(e0,e1) is inverted by the affine proxy (4,8):(1,4)."""
    R = Layout((4, 8), (1, 4))
    for row in range(4):
        for col in range(8):
            coord = (row, col)
            assert R(coord) == crd2flat(coord, R.shape)


def test_table5_right_inverse_coordinate_blocked():
    """Table 5: (4,(4,2)):(e1,(e0,6e1)) is inverted on a compatible 4×4 proxy domain."""
    R = Layout((4, 4), (4, 1))
    src_shape = (4, (4, 2))
    for col in range(4):
        for row in range(4):
            natural = R(col, row)
            src_coord = idx2crd(natural, src_shape)
            mapped = (src_coord[1][0], src_coord[0] + 6 * src_coord[1][1])
            assert mapped == (col, row)


def test_table5_right_inverse_binary_swizzle():
    """Table 5: (4,(4,3)):(f1,(f5,f16)) matches the paper's swizzled 4×4×3 proxy."""
    swizzle = Swizzle(2, 0, 2)
    L = compose(swizzle, Layout((4, (4, 3)), (1, (4, 16))))
    R = right_inverse(L)
    paper_proxy = compose(swizzle, Layout((4, 4, 3), (1, 4, 16)))
    for i in range(size(R)):
        assert L(R(i)) == i
        assert R(i) == paper_proxy(i)


def test_table6_left_inverse_colmajor():
    """Table 6: left_inverse((4,8):(1,4)) = 32:1."""
    L = Layout((4, 8), (1, 4))
    Li = left_inverse(L)
    assert Li == Layout(32, 1)
    for i in range(size(L)):
        assert Li(L(i)) == i


def test_table6_left_inverse_rowmajor():
    """Table 6: left_inverse((4,8):(8,1)) = (8,4):(4,1)."""
    L = Layout((4, 8), (8, 1))
    Li = left_inverse(L)
    assert Li == Layout((8, 4), (4, 1))
    for i in range(size(L)):
        assert Li(L(i)) == i


def test_table6_left_inverse_padded():
    """Table 6: left_inverse((4,8):(1,5)) = (5,8):(1,4).

    For padded (non-contiguous) layouts, the CuTe C++ algorithm builds
    an inverse that covers the full codomain. Gap positions map to
    stride-0 (don't care), in-image positions map back correctly.
    """
    L = Layout((4, 8), (1, 5))
    Li = left_inverse(L)
    assert Li == Layout((5, 8), (1, 4))
    for k in range(size(L)):
        assert Li(L(k)) == k


def test_table6_left_inverse_rank3():
    """Table 6: left_inverse((3,7,5):(5,15,1)) = (5,21):(21,1)."""
    L = Layout((3, 7, 5), (5, 15, 1))
    Li = left_inverse(L)
    assert Li == Layout((5, 21), (21, 1))
    for k in range(size(L)):
        assert Li(L(k)) == k


def test_table6_left_inverse_nested_colmajor():
    """Table 6: left_inverse((4,(4,2)):(4,(1,16))) = (4,4,2):(4,1,16)."""
    L = Layout((4, (4, 2)), (4, (1, 16)))
    Li = left_inverse(L)
    assert Li == Layout((4, 4, 2), (4, 1, 16))
    for k in range(size(L)):
        assert Li(L(k)) == k


def test_table6_left_inverse_nested_mixed():
    """Table 6: left_inverse(((2,2),(4,2)):((1,8),(2,16))) = (2,4,2,2):(1,4,2,16)."""
    L = Layout(((2, 2), (4, 2)), ((1, 8), (2, 16)))
    Li = left_inverse(L)
    assert Li == Layout((2, 4, 2, 2), (1, 4, 2, 16))
    for k in range(size(L)):
        assert Li(L(k)) == k


def test_table6_left_inverse_broadcast_even_stride():
    """Table 6: left_inverse(((2,2),(2,4)):((0,2),(0,4))) = (2,2,4):(0,2,8)."""
    L = Layout(((2, 2), (2, 4)), ((0, 2), (0, 4)))
    Li = left_inverse(L)
    assert Li == Layout((2, 2, 4), (0, 2, 8))
    for k in range(size(L)):
        assert L(Li(L(k))) == L(k)


def test_table6_left_inverse_broadcast_unit_stride():
    """Table 6 (corrected): left_inverse(((2,2),(2,4)):((0,1),(0,2))) = (2,4):(2,8)."""
    L = Layout(((2, 2), (2, 4)), ((0, 1), (0, 2)))
    Li = left_inverse(L)
    assert Li == Layout((2, 4), (2, 8))
    for k in range(size(L)):
        assert L(Li(L(k))) == L(k)


def test_table6_left_inverse_coordinate_identity():
    """Table 6: (4,8):(e0,e1) is inverted by the affine proxy (4,8):(1,4)."""
    Li = Layout((4, 8), (1, 4))
    for k in range(size(Li)):
        assert Li(idx2crd(k, Li.shape)) == k


def test_table6_left_inverse_coordinate_blocked():
    """Table 6: (4,(4,2)):(e1,(e0,6e1)) is inverted by (4,(6,2)):(4,(1,16))."""
    src_shape = (4, (4, 2))
    Li = Layout((4, (6, 2)), (4, (1, 16)))
    for k in range(size(src_shape)):
        row, (col, block) = idx2crd(k, src_shape)
        coord = (col, idx2crd(row + 6 * block, (6, 2)))
        assert Li(coord) == k


def test_table6_left_inverse_binary_swizzle():
    """Table 6: (4,(4,3)):(f1,(f5,f16)) is inverted by (4,4,3):(f1,f5,f16)."""
    swizzle = Swizzle(2, 0, 2)
    L = compose(swizzle, Layout((4, (4, 3)), (1, (4, 16))))
    Li = left_inverse(L)
    paper_proxy = compose(swizzle, Layout((4, 4, 3), (1, 4, 16)))
    for k in range(size(L)):
        assert Li(L(k)) == k
        assert Li(k) == paper_proxy(k)


# =============================================================================
# Figure 8 — Common subvectors between layouts
# =============================================================================


def test_fig8a_two_element_common_subvector(viz):
    """Figure 8a: source/destination share a 2-element common subvector."""
    src = Layout((4, 4), (1, 4))
    dst = Layout(((2, 2), 4), ((1, 8), 2))
    common = max_common_layout(src, dst)
    assert max_common_vector(src, dst) == 2
    assert common == Layout(2, 1)
    src_spec = _common_subvector_slice_spec(src, common)
    dst_spec = _common_subvector_slice_spec(dst, common)
    common_coords_src = {idx2crd(common(i), src.shape) for i in range(size(common))}
    common_coords_dst = {idx2crd(common(i), dst.shape) for i in range(size(common))}
    assert _selected_coords_from_slice_spec(src, src_spec) == common_coords_src
    assert _selected_coords_from_slice_spec(dst, dst_spec) == common_coords_dst
    if not viz:
        return
    viz.draw_composite(
        [(src, {"slice_spec": src_spec}), (dst, {"slice_spec": dst_spec})],
        filename=_figure_path("fig8a_two_element_common_subvector"),
        titles=["Source", "Destination"],
        main_title="Fig 8a: A 2-element common subvector",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


def test_fig8b_four_element_common_subvector(viz):
    """Figure 8b: source/destination share a 4-element common subvector."""
    src = Layout(((2, 2), (2, 2)), ((8, 2), (4, 1)))
    dst = Layout(((2, 2), (2, 2)), ((4, 2), (8, 1)))
    assert max_common_vector(src, dst) == 4
    common = max_common_layout(src, dst)
    for i in range(size(common)):
        assert src(common(i)) == i
        assert dst(common(i)) == i
    src_spec = _common_subvector_slice_spec(src, common)
    dst_spec = _common_subvector_slice_spec(dst, common)
    common_coords_src = {idx2crd(common(i), src.shape) for i in range(size(common))}
    common_coords_dst = {idx2crd(common(i), dst.shape) for i in range(size(common))}
    assert _selected_coords_from_slice_spec(src, src_spec) == common_coords_src
    assert _selected_coords_from_slice_spec(dst, dst_spec) == common_coords_dst
    if not viz:
        return
    viz.draw_composite(
        [(src, {"slice_spec": src_spec}), (dst, {"slice_spec": dst_spec})],
        filename=_figure_path("fig8b_four_element_common_subvector"),
        titles=["Source", "Destination"],
        main_title="Fig 8b: A 4-element common subvector",
        color_layout=Layout(1, 0),
        num_colors=1,
    )


# =============================================================================
# Figure 9 — TMEM load/store register layouts
# =============================================================================


def test_fig9a_tcgen05_32x32b(viz):
    """Figure 9a: the 32×2 register layout for tcgen05{.1d,.st}.32x32b.{x1,x2}."""
    layout = display_layout(32, 2)
    labels = _grid_labels(32, 2, lambda row, col: f"T{row}:r{col}")
    assert labels[0] == "T0:r0"
    assert labels[-1] == "T31:r1"
    if not viz:
        return
    viz.draw_layout(
        layout,
        filename=_figure_path("fig9a_tcgen05_32x32b"),
        title="Fig 9a: tcgen05{.1d,.st}.32x32b.{x1,x2}",
        cell_labels=labels,
        color_layout=Layout((32, 2), (1, 0)),
        colorize=True,
        num_colors=32,
    )


def test_fig9b_tcgen05_16x256b(viz):
    """Figure 9b: the 16×8 register layout for tcgen05{.1d,.st}.16x256b.x1."""
    row_shape = (8, 2)
    col_shape = (2, 4)
    layout = display_layout(row_shape, col_shape)
    labels = _grid_labels(
        row_shape,
        col_shape,
        lambda row, col: f"T{col[1] + 4 * row[0]}:r{col[0] + 2 * row[1]}",
    )
    assert labels[0] == "T0:r0"
    assert labels[7] == "T3:r1"
    assert labels[8 * 8] == "T0:r2"
    color_layout = Layout((row_shape, col_shape), ((1, 0), (0, 0)))
    if not viz:
        return
    viz.draw_layout(
        layout,
        filename=_figure_path("fig9b_tcgen05_16x256b"),
        title="Fig 9b: tcgen05{.1d,.st}.16x256b.x1",
        cell_labels=labels,
        color_layout=color_layout,
        colorize=True,
        num_colors=8,
    )


# =============================================================================
# §3.4.4 — Admissibility example
# =============================================================================


def test_s3_4_4_tcgen05_32x32b_x1_layout():
    """§3.4.4: tcgen05{.ld,.st}.32x32b.x1 uses (1,128):(1,16384)."""
    L = Layout((1, 128), (1, 16384))
    assert L(0, 0) == 0
    assert L(0, 1) == 16384
    assert L(0, 127) == 127 * 16384


def test_s3_4_4_tcgen05_32x32b_x2_layout():
    """§3.4.4: tcgen05{.ld,.st}.32x32b.x2 uses (2,128):(1,16384)."""
    L = Layout((2, 128), (1, 16384))
    assert L(0, 0) == 0
    assert L(1, 0) == 1
    assert L(0, 1) == 16384
    assert L(1, 127) == 1 + 127 * 16384


def test_s3_4_4_tcgen05_16x256b_x1_layout():
    """§3.4.4: tcgen05{.ld,.st}.16x256b.x1 uses (8,(16,4)):(1,(16384,32·16384))."""
    L = Layout((8, (16, 4)), (1, (16384, 32 * 16384)))
    assert L(0, (0, 0)) == 0
    assert L(7, (0, 0)) == 7
    assert L(0, (1, 0)) == 16384
    assert L(0, (0, 1)) == 32 * 16384
    assert L(7, (15, 3)) == 7 + 15 * 16384 + 3 * 32 * 16384


# =============================================================================
# §3.5 — Complement (Table 7)
# =============================================================================


def test_table7_complement_contiguous():
    """Table 7: complement((4,8):(1,4)).

    Bijective layout → complement is trivial (size 1).
    CuTe C++ coalesces the result, so 1:32 becomes 1:0.
    """
    L = Layout((4, 8), (1, 4))
    C = complement(L)
    assert C == Layout(1, 0)


def test_table7_complement_rowmajor():
    """Table 7: complement((4,8):(8,1)).

    Bijective layout → complement is trivial (size 1).
    """
    L = Layout((4, 8), (8, 1))
    C = complement(L)
    assert C == Layout(1, 0)


def test_table7_complement_nested_colmajor():
    """Table 7: complement((4,(4,2)):(4,(1,16))) canonically coalesces to 1:0 in CuTe C++."""
    L = Layout((4, (4, 2)), (4, (1, 16)))
    C = complement(L)
    assert C == Layout(1, 0)


def test_table7_complement_padded():
    """Table 7: complement((4,8):(1,5)).

    Non-contiguous layout → complement is trivial (size 1) since all gaps
    are beyond the last mode. CuTe C++ coalesces the result.
    """
    L = Layout((4, 8), (1, 5))
    C = complement(L)
    assert C == Layout(1, 0)


def test_table7_complement_with_holes():
    """Table 7: complement((4,8):(1,8)) is printed as (2,1):(4,64) and coalesces to 2:4.

    This layout has holes between columns (stride 8 > size 4).
    The complement fills the gaps with stride 4 (= cosize of mode 0).
    """
    L = Layout((4, 8), (1, 8))
    C = complement(L)
    assert C == Layout(2, 4)
    # Verify disjointness: complement generates offsets not in L's image
    L_img = {L(i) for i in range(size(L))}
    for j in range(1, size(C)):
        assert C(j) not in L_img


def test_table7_complement_broadcast_unit_stride():
    """Table 7: complement(((2,2),(2,4)):((0,1),(0,2))) canonically coalesces to 1:0 in CuTe C++."""
    L = Layout(((2, 2), (2, 4)), ((0, 1), (0, 2)))
    C = complement(L)
    assert C == Layout(1, 0)


def test_table7_complement_broadcast_even_stride():
    """Table 7: complement(((2,2),(2,4)):((0,2),(0,4))) coalesces to 2:1."""
    L = Layout(((2, 2), (2, 4)), ((0, 2), (0, 4)))
    C = complement(L)
    assert C == Layout(2, 1)


def test_table7_complement_coordinate_identity_proxy():
    """Table 7: (4,8):(e0,e1) is complemented by the proxy (1,1):(4e0,4e1)."""

    def complement_proxy(coord):
        block_row, block_col = coord
        return (4 * block_row, 4 * block_col)

    # The paper's comment for this row is only that the complement's domain is
    # congruent with the codomain Z(*,*).  We therefore verify the stated
    # 4-step coordinate basis directly, rather than asserting arbitrary finite
    # extensions are disjoint from the original 4×8 block.
    sample_shape = (2, 2)
    sample_image = [
        complement_proxy(idx2crd(i, sample_shape)) for i in range(size(sample_shape))
    ]
    assert sample_image == [(0, 0), (4, 0), (0, 4), (4, 4)]
    assert [crd2idx(coord, (8, 8)) for coord in sample_image] == [0, 4, 32, 36]


def test_table7_complement_coordinate_blocked_proxy():
    """Table 7: (4,(4,2)):(e1,(e0,12e1)) is complemented by (1,(3,1)):(4e0,(4e1,24e1))."""

    def coord_layout(coord):
        row, (col, block) = coord
        return (col, row + 12 * block)

    def complement_proxy(coord):
        block_row, (block_col, block_group) = coord
        return (4 * block_row, 4 * block_col + 24 * block_group)

    original_shape = (4, (4, 2))
    original_image = {
        coord_layout(idx2crd(i, original_shape)) for i in range(size(original_shape))
    }

    paper_shape = (1, (3, 1))
    assert [
        complement_proxy(idx2crd(i, paper_shape)) for i in range(size(paper_shape))
    ] == [
        (0, 0),
        (0, 4),
        (0, 8),
    ]

    # Eq. (28) uses the infinite extended domain of the complement.  Sample a
    # larger compatible extension to check the paper's "disjoint and ordered"
    # claim concretely.
    sample_shape = (2, (3, 2))
    sample_image = [
        complement_proxy(idx2crd(i, sample_shape)) for i in range(size(sample_shape))
    ]
    for coord in sample_image[1:]:
        assert coord not in original_image
    assert [crd2idx(coord, (8, 48)) for coord in sample_image] == sorted(
        crd2idx(coord, (8, 48)) for coord in sample_image
    )


# =============================================================================
# §3.5.1 — Logical Product examples
# =============================================================================


def test_s3_5_1_logical_product_example1():
    """§3.5.1: (3,4):(4,1) ⊗ (2,5):(1,2) = ((3,4),(2,5)):((4,1),(12,24))."""
    A = Layout((3, 4), (4, 1))
    B = Layout((2, 5), (1, 2))
    R = logical_product(A, B)
    assert R == Layout(((3, 4), (2, 5)), ((4, 1), (12, 24)))


def test_s3_5_1_logical_product_example2():
    """§3.5.1: (4,8):(20,2) ⊗ (3,2):(2,1) = ((4,8),(3,2)):((20,2),(80,1)).

    complement((4,8):(20,2)) includes (2,1):(1,80).
    """
    A = Layout((4, 8), (20, 2))
    B = Layout((3, 2), (2, 1))
    R = logical_product(A, B)
    assert R == Layout(((4, 8), (3, 2)), ((20, 2), (80, 1)))


# =============================================================================
# §3.5.1 — Blocked product (Figure 10)
# =============================================================================


def test_fig10_blocked_product(viz):
    """Figure 10: blocked_product of (3,4):(4,1) with (2,5):(1,2).

    Result is ((3,2),(4,5)):((4,12),(1,24)), a 6×20 layout.
    """
    tile = Layout((3, 4), (4, 1))
    grid = Layout((2, 5), (1, 2))
    result = blocked_product(tile, grid)
    assert result == Layout(((3, 2), (4, 5)), ((4, 12), (1, 24)))
    assert size(result) == 6 * 20
    # Verify Figure 10's first column: offsets 0,4,8,12,16,20
    col0 = [result(i, 0) for i in range(6)]
    assert col0 == [0, 4, 8, 12, 16, 20]
    if not viz:
        return
    viz.draw_layout(
        result,
        filename=_figure_path("fig10_blocked_product"),
        title="Fig 10: blocked_product((3,4):(4,1), (2,5):(1,2))",
        color_layout=Layout(((3, 2), (4, 5)), ((0, 1), (0, 2))),
        colorize=True,
        num_colors=10,
    )


# =============================================================================
# Figure 11 — Raked product
# =============================================================================


def test_fig11_raked_product(viz):
    """Figure 11: raked_product of (3,4):(4,1) with (2,5):(1,2)."""
    block = Layout((3, 4), (4, 1))
    tiler = Layout((2, 5), (1, 2))
    result = raked_product(block, tiler)
    assert result == Layout(((2, 3), (5, 4)), ((12, 4), (24, 1)))
    assert size(result) == 6 * 20
    if not viz:
        return
    viz.draw_layout(
        result,
        filename=_figure_path("fig11_raked_product"),
        title="Fig 11: raked_product((3,4):(4,1), (2,5):(1,2))",
        color_layout=Layout(((2, 3), (5, 4)), ((1, 0), (2, 0))),
        colorize=True,
        num_colors=10,
    )


# =============================================================================
# §3.5.2 — Logical Divide examples
# =============================================================================


def test_s3_5_2_every_third_element():
    """§3.5.2: 24:3 ◦ 8:3 = 8:9."""
    assert compose(Layout(24, 3), Layout(8, 3)) == Layout(8, 9)


def test_s3_5_2_logical_divide_1d():
    """§3.5.2: 24:3 ⊘ 8:3 = (8,3):(9,3).

    complement(8:3, 24) = 3:1, so divisor = (8,3):(3,1).
    With a Layout tiler the coordinates are reordered by the tiler's
    access pattern — same set of offsets, different flat-index order.
    """
    A = Layout(24, 3)
    B = Layout(8, 3)
    R = logical_divide(A, B)
    assert R == Layout((8, 3), (9, 3))
    # Same set of offsets (reordered by tiler)
    R_offsets = sorted(R(i) for i in range(size(R)))
    A_offsets = sorted(A(i) for i in range(size(A)))
    assert R_offsets == A_offsets


def test_s3_5_2_logical_divide_hierarchical():
    """§3.5.2: (6,2,2):(2,1,20) ⊘ 8:3 = ((2,2,2),3):((6,1,20),2)."""
    A = Layout((6, 2, 2), (2, 1, 20))
    B = Layout(8, 3)
    R = logical_divide(A, B)
    assert R == Layout(((2, 2, 2), 3), ((6, 1, 20), 2))
    # Same set of offsets
    R_offsets = sorted(R(i) for i in range(size(R)))
    A_offsets = sorted(A(i) for i in range(size(A)))
    assert R_offsets == A_offsets


def test_s3_5_2_by_mode_divide():
    """§3.5.2: (8,16):(20,1) ⊘ <4:1, 8:2> = ((4,2),(8,2)):((20,80),(2,1))."""
    A = Layout((8, 16), (20, 1))
    R = logical_divide(A, (Layout(4, 1), Layout(8, 2)))
    assert R == Layout(((4, 2), (8, 2)), ((20, 80), (2, 1)))
    R_offsets = sorted(R(i) for i in range(size(R)))
    A_offsets = sorted(A(i) for i in range(size(A)))
    assert R_offsets == A_offsets


def test_s3_5_2_zipped_divide():
    """§3.5.2: zipped_divide of above = ((4,8),(2,2)):((20,2),(80,1))."""
    A = Layout((8, 16), (20, 1))
    R = zipped_divide(A, (Layout(4, 1), Layout(8, 2)))
    assert R == Layout(((4, 8), (2, 2)), ((20, 2), (80, 1)))
    R_offsets = sorted(R(i) for i in range(size(R)))
    A_offsets = sorted(A(i) for i in range(size(A)))
    assert R_offsets == A_offsets


# =============================================================================
# Figure 12 — Tilers splitting an 8×16 layout into 4×8 tiles
# =============================================================================


def test_fig12_tiler_split_patterns(viz):
    """Figure 12: draw the four 2×2 tilings induced by the paper's tilers."""
    fig12a = Layout(((4, 2), (8, 2)), ((0, 1), (0, 2)))
    fig12b = Layout((((2, 2), 2), (8, 2)), (((0, 1), 0), (0, 2)))
    fig12c = Layout((((2, 2), 2), (2, 8)), (((0, 1), 0), (2, 0)))
    fig12d = Layout(((2, 4), (2, 8)), ((1, 0), (2, 0)))
    for layout in [fig12a, fig12b, fig12c, fig12d]:
        assert sorted({layout(i) for i in range(size(layout))}) == [0, 1, 2, 3]
    if not viz:
        return
    viz.draw_composite(
        [fig12a, fig12b, fig12c, fig12d],
        filename=_figure_path("fig12_tiler_split_patterns"),
        arrangement="grid:2x2",
        titles=[
            "<4,8> ≡ <4:1, 8:1>",
            "<(2,2):(1,4), 8:1>",
            "<(2,2):(1,4), 8:2>",
            "<4:2, 8:2>",
        ],
        main_title="Fig 12: tilers splitting an 8×16 layout into 4×8 tiles",
        cell_labels=False,
        num_colors=4,
        panel_size=(6.0, 3.8),
    )


# =============================================================================
# Cross-cutting invariants from the paper
# =============================================================================


def test_complement_disjoint_images():
    """Eq. 28: ∀b ∈ Z(L), ∀a ∈ Z(L*)/{0}, L(b) ≠ L*(a).

    The complement's non-zero image is disjoint from L's image.
    """
    test_cases = [
        Layout(4, 2),
        Layout((2, 4), (1, 4)),
        Layout((4, 8), (1, 8)),
    ]
    for L in test_cases:
        C = complement(L)
        L_img = {L(i) for i in range(size(L))}
        for j in range(1, size(C)):
            assert C(j) not in L_img, f"complement({L}) image not disjoint at j={j}"


def test_logical_divide_surjective():
    """§3.5.2: B★ = (B, complement(B, |A|)) is surjective onto Z|A|."""
    A = Layout(24, 1)
    B = Layout(4, 1)
    C = complement(B, size(A))
    Bstar = Layout((B.shape, C.shape), (B.stride, C.stride))
    # Surjective: every element in [0, |A|) is hit
    offsets = sorted(Bstar(i) for i in range(size(Bstar)))
    assert offsets == list(range(size(A)))


def test_compose_functional_property():
    """Eq. 17: ∀c ∈ Z(B), R(c) = A(B(c))."""
    cases = [
        (Layout((4, 8), (1, 4)), Layout((2, 4), (1, 2))),
        (Layout((4, 6, 8, 10), (2, 3, 5, 7)), Layout(6, 12)),
        (Layout(7, 11), Layout((3, 5), (6, 3))),
    ]
    for A, B in cases:
        R = compose(A, B)
        for i in range(size(B)):
            assert R(i) == A(B(i)), f"compose({A}, {B}) failed at i={i}"


def test_right_inverse_property():
    """Eq. 24 (integer case): L(L‡(k)) = k for all k."""
    layouts = [
        Layout((4, 8), (1, 4)),
        Layout((4, 8), (8, 1)),
        Layout(16, 1),
        Layout((3, 7, 5), (5, 15, 1)),
    ]
    for L in layouts:
        R = right_inverse(L)
        for k in range(size(R)):
            assert L(R(k)) == k, f"right_inverse failed for {L} at k={k}"


def test_left_inverse_property():
    """Injective case: L†(L(k)) = k for all k.

    This is the paper's integer-stride left-inverse property and also holds
    for the padded example from Table 6.
    """
    for L in [
        Layout((4, 8), (1, 4)),
        Layout((4, 8), (8, 1)),
        Layout((4, 8), (1, 5)),
        Layout((3, 7, 5), (5, 15, 1)),
        Layout(16, 1),
    ]:
        Li = left_inverse(L)
        for k in range(size(L)):
            assert Li(L(k)) == k, f"left_inverse failed for {L} at k={k}"


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
