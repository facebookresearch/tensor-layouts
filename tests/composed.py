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

from __future__ import annotations

import dataclasses

import pytest

from tensor_layouts import *
from tensor_layouts.analysis import (
    bank_conflicts,
    coalescing_efficiency,
    contiguity,
    cycles,
    fixed_points,
    footprint,
    functionally_equal,
    image,
    is_bijective,
    is_contiguous,
    is_injective,
    is_surjective,
    mode_contiguity,
    offset_table,
    order,
    per_group_bank_conflicts,
    per_group_coalescing,
    segment_analysis,
    slice_contiguity,
    to_F2_matrix,
)
from tensor_layouts.tensor import Tensor
from tensor_layouts.viz import draw_layout, draw_slice


def _assert_pointwise_equal(a, b):
    lhs_size = size(a) if is_layout(a) else None
    rhs_size = size(b) if is_layout(b) else None
    if lhs_size is not None and rhs_size is not None:
        assert lhs_size == rhs_size
    n = lhs_size if lhs_size is not None else rhs_size
    assert n is not None
    for flat_idx in range(n):
        lhs = a(flat_idx) if callable(a) else a[flat_idx]
        rhs = b(flat_idx) if callable(b) else b[flat_idx]
        assert lhs == rhs, f"Mismatch at flat_idx={flat_idx}"


def test_composed_layout_boilerplate():
    outer = Layout(32, 2)
    inner = Layout((2, 4), (1, 2))
    layout = ComposedLayout(outer, inner, preoffset=3)

    assert layout.shape == inner.shape
    assert size(layout) == size(inner)
    assert rank(layout) == rank(inner)
    assert depth(layout) == depth(inner)
    assert cosize(layout) == cosize(inner)
    assert repr(layout) == f"ComposedLayout({outer!r}, {inner!r}, preoffset=3)"
    assert str(layout) == f"({outer}) o {{3}} o ({inner})"
    assert layout == ComposedLayout(outer, inner, preoffset=3)
    assert hash(layout) == hash(ComposedLayout(outer, inner, preoffset=3))
    assert layout((1, 2)) == outer(3 + inner((1, 2)))

    # Hash must also work when outer is a Swizzle (not just Layout).
    swz_composed = ComposedLayout(Swizzle(3, 0, 3), Layout(16, 1))
    assert hash(swz_composed) == hash(ComposedLayout(Swizzle(3, 0, 3), Layout(16, 1)))

    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.preoffset = 4


def test_mode_on_composed_layout_uses_inner_domain():
    inner = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    composed = compose(Layout((4, 4), (4, 1)), inner)

    m0 = mode(composed, 0)
    assert isinstance(m0, ComposedLayout)

    for i in range(size(m0)):
        assert m0(i) == composed(i, 0)


def test_compose_double_swizzle_is_exact():
    base = Layout((8, 8), (8, 1))
    inner = compose(Swizzle(3, 0, 3), base)
    outer = Swizzle(1, 0, 3)
    result = compose(outer, inner)

    assert isinstance(result, ComposedLayout)
    for i in range(8):
        for j in range(8):
            assert result(i, j) == outer(inner(i, j))


def test_compose_affine_with_swizzled_layout_is_exact():
    outer = Layout((4, 4), (4, 1))
    inner = compose(Swizzle(3, 0, 3), Layout(16, 1))
    result = compose(outer, inner)

    assert isinstance(result, ComposedLayout)
    for i in range(size(result)):
        assert result(i) == outer(inner(i))


def test_compose_layout_with_swizzle_rhs_falls_back_to_exact_composed_layout():
    outer = Layout((4, 4), (4, 1))
    swizzle = Swizzle(2, 1, 3)
    result = compose(outer, swizzle)
    expected = ComposedLayout(outer, compose(swizzle, Layout(outer.shape)))

    assert isinstance(result, ComposedLayout)
    _assert_pointwise_equal(result, expected)
    assert [result(i) for i in range(size(result))] == [
        0,
        4,
        8,
        12,
        1,
        5,
        9,
        13,
        2,
        6,
        10,
        14,
        3,
        7,
        11,
        15,
    ]


def test_compose_layout_with_swizzle_rhs_keeps_representable_fast_path():
    outer = Layout(16, 2)
    swizzle = Swizzle(2, 0, 2)
    result = compose(outer, swizzle)

    assert isinstance(result, Layout)
    assert result.swizzle is not None
    _assert_pointwise_equal(result, lambda i: outer(swizzle(i)))


def test_compose_layout_with_swizzle_rhs_nonpower_stride_stays_exact():
    outer = Layout(16, 3)
    swizzle = Swizzle(2, 0, 2)
    result = compose(outer, swizzle)
    expected = ComposedLayout(outer, compose(swizzle, Layout(outer.shape)))

    assert isinstance(result, ComposedLayout)
    _assert_pointwise_equal(result, expected)


def test_compose_layout_on_zero_preoffset_composed_layout_can_collapse():
    outer = Layout(32, 2)
    inner = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), preoffset=0)
    result = compose(outer, inner)

    assert isinstance(result, Layout)
    assert result.swizzle is not None
    _assert_pointwise_equal(result, lambda i: outer(inner(i)))


def test_compose_layout_on_nonzero_preoffset_composed_layout_stays_exact():
    outer = Layout(32, 2)
    inner = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), preoffset=4)
    result = compose(outer, inner)

    assert isinstance(result, ComposedLayout)
    _assert_pointwise_equal(result, lambda i: outer(inner(i)))


def test_compose_swizzled_layout_outer_preserves_exactness():
    outer = compose(Swizzle(2, 0, 2), Layout(16, 1))
    inner = Layout(8, 2)
    result = compose(outer, inner)

    assert isinstance(result, Layout)
    assert result.swizzle == outer.swizzle
    _assert_pointwise_equal(result, lambda i: outer(inner(i)))


def test_logical_divide_forwards_through_composed_layout():
    composed = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout(16, 1)))
    result = logical_divide(composed, 4)
    expected = ComposedLayout(composed.outer, logical_divide(composed.inner, 4), preoffset=0)

    assert isinstance(result, ComposedLayout)
    assert result.outer == composed.outer
    assert result.preoffset == 0
    _assert_pointwise_equal(result, expected)


def test_logical_product_forwards_through_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    result = logical_product(composed, Layout(3, 1))
    expected = ComposedLayout(composed.outer, logical_product(composed.inner, Layout(3, 1)))

    assert isinstance(result, ComposedLayout)
    _assert_pointwise_equal(result, expected)


def test_slice_and_offset_on_composed_layout_keeps_offset_internal():
    composed = compose(
        Layout((4, 4), (4, 1)),
        compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1))),
    )

    sub, offset = slice_and_offset((2, None), composed)
    assert offset == 0
    assert isinstance(sub, ComposedLayout)

    for j in range(8):
        assert sub(j) == composed(2, j)


def test_tensor_accepts_composed_layout_with_storage():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), preoffset=4)
    tensor = Tensor(composed, offset=100, data=list(range(256)))

    for i in range(size(composed)):
        assert tensor(i) == 100 + composed(i)
        assert tensor[i] == tensor.data[tensor(i)]


def test_tensor_slice_on_composed_layout_keeps_external_offset():
    composed = compose(
        Layout((4, 4), (4, 1)),
        compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1))),
    )
    tensor = Tensor(composed, offset=100)

    row = tensor[2, :]
    assert isinstance(row, Tensor)
    assert row.offset == 100
    assert isinstance(row.layout, ComposedLayout)

    for j in range(8):
        assert row(j) == tensor(2, j)


def test_tensor_stride_rejects_composed_layout():
    tensor = Tensor(ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), preoffset=4))
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        _ = tensor.stride


def test_right_inverse_of_zero_preoffset_swizzled_composed_layout():
    composed = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), preoffset=0)
    inv = right_inverse(composed)

    assert isinstance(inv, Layout)
    for i in range(size(inv)):
        assert composed(inv(i)) == i


def test_left_inverse_of_zero_preoffset_swizzled_composed_layout():
    composed = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), preoffset=0)
    inv = left_inverse(composed)

    assert isinstance(inv, Layout)
    for i in range(size(composed)):
        assert inv(composed(i)) == i


def test_max_common_vector_for_swizzled_composed_layout_is_capped_by_swizzle_base():
    composed = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), preoffset=0)
    plain = Layout(32, 1)

    assert max_common_vector(composed, plain) == 2
    common = max_common_layout(composed, plain)
    assert size(common) == 2
    for i in range(size(common)):
        assert composed(common(i)) == i
        assert plain(common(i)) == i


def test_affine_only_boundaries_reject_composed_layout():
    composed = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout(16, 1)))

    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        as_affine_layout(composed)
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        to_F2_matrix(composed)
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        contiguity(composed)
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        mode_contiguity(composed)
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        slice_contiguity(composed, (None,))


def test_draw_layout_and_draw_slice_smoke_for_composed_layout(tmp_path):
    composed = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout((4, 4), (4, 1))))

    fig1 = draw_layout(composed, tmp_path / "composed_layout.png")
    fig2 = draw_slice(composed, (None, 1), tmp_path / "composed_slice.png")
    assert fig1 is None
    assert fig2 is None
    assert (tmp_path / "composed_layout.png").exists()
    assert (tmp_path / "composed_slice.png").exists()


# ---------------------------------------------------------------------------
# Generative / search-based differential tests
# ---------------------------------------------------------------------------

_SMALL_AFFINE_LAYOUTS = [
    Layout(8, 1),
    Layout(8, 2),
    Layout(16, 1),
    Layout((4, 4), (4, 1)),
    Layout((4, 4), (1, 4)),
    Layout((2, 4), (4, 1)),
    Layout((2, 2, 4), (1, 4, 2)),
]

_SMALL_SWIZZLES = [
    Swizzle(1, 0, 2),
    Swizzle(2, 0, 2),
    Swizzle(3, 0, 3),
    Swizzle(1, 0, 3),
    Swizzle(2, 1, 3),
]

_SMALL_PREOFFSETS = [0, 1, 4, 7]


def _brute_force_compose(a, b):
    """Evaluate compose(a, b)(i) as a(b(i)) for all flat indices."""
    n = size(b) if is_layout(b) else None
    if n is None:
        raise ValueError("RHS must have a known size")
    return [a(b(i)) for i in range(n)]


def test_generative_compose_affine_over_swizzled():
    """compose(affine, compose(Swizzle, affine)) matches brute-force for many combos."""
    for outer in _SMALL_AFFINE_LAYOUTS:
        for swz in _SMALL_SWIZZLES:
            for inner in _SMALL_AFFINE_LAYOUTS:
                swizzled = compose(swz, inner)
                # outer must consume at most as many elements as swizzled produces
                if size(outer) > size(swizzled):
                    continue
                result = compose(outer, swizzled)
                expected = _brute_force_compose(outer, swizzled)
                actual = [result(i) for i in range(size(result))]
                assert actual == expected, (
                    f"Mismatch for compose({outer}, compose({swz}, {inner}))"
                )


def test_generative_compose_double_swizzle():
    """compose(Swizzle, compose(Swizzle, base)) matches brute-force."""
    for sw_outer in _SMALL_SWIZZLES:
        for sw_inner in _SMALL_SWIZZLES:
            for base in _SMALL_AFFINE_LAYOUTS:
                inner = compose(sw_inner, base)
                result = compose(sw_outer, inner)
                expected = [sw_outer(inner(i)) for i in range(size(inner))]
                actual = [result(i) for i in range(size(result))]
                assert actual == expected, (
                    f"Mismatch for compose({sw_outer}, compose({sw_inner}, {base}))"
                )


def test_generative_compose_with_preoffsets():
    """ComposedLayout with various preoffsets evaluates correctly."""
    for swz in _SMALL_SWIZZLES[:2]:
        for inner in _SMALL_AFFINE_LAYOUTS[:3]:
            for po in _SMALL_PREOFFSETS:
                composed = ComposedLayout(swz, inner, preoffset=po)
                for i in range(size(composed)):
                    assert composed(i) == swz(po + inner(i)), (
                        f"Mismatch at i={i} for ComposedLayout({swz}, {inner}, preoffset={po})"
                    )


# ---------------------------------------------------------------------------
# Divide / product cascade on composed inputs
# ---------------------------------------------------------------------------

def test_zipped_divide_forwards_through_composed_layout():
    composed = compose(
        Layout((4, 4), (4, 1)),
        compose(Swizzle(2, 0, 2), Layout((4, 4), (4, 1))),
    )
    result = zipped_divide(composed, (2, 2))
    assert isinstance(result, ComposedLayout)
    expected = ComposedLayout(composed.outer, zipped_divide(composed.inner, (2, 2)))
    _assert_pointwise_equal(result, expected)


def test_tiled_divide_forwards_through_composed_layout():
    composed = compose(
        Layout((4, 4), (4, 1)),
        compose(Swizzle(2, 0, 2), Layout((4, 4), (4, 1))),
    )
    result = tiled_divide(composed, (2, 2))
    assert isinstance(result, ComposedLayout)
    expected = ComposedLayout(composed.outer, tiled_divide(composed.inner, (2, 2)))
    _assert_pointwise_equal(result, expected)


def test_flat_divide_forwards_through_composed_layout():
    composed = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout(16, 1)))
    result = flat_divide(composed, 4)
    assert isinstance(result, ComposedLayout)
    expected = ComposedLayout(composed.outer, flat_divide(composed.inner, 4))
    _assert_pointwise_equal(result, expected)


def test_zipped_product_forwards_through_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    result = zipped_product(composed, Layout(3, 1))
    assert isinstance(result, ComposedLayout)
    expected = ComposedLayout(composed.outer, zipped_product(composed.inner, Layout(3, 1)))
    _assert_pointwise_equal(result, expected)


def test_tiled_product_forwards_through_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    result = tiled_product(composed, Layout(3, 1))
    assert isinstance(result, ComposedLayout)
    expected = ComposedLayout(composed.outer, tiled_product(composed.inner, Layout(3, 1)))
    _assert_pointwise_equal(result, expected)


def test_flat_product_forwards_through_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    result = flat_product(composed, Layout(3, 1))
    assert isinstance(result, ComposedLayout)
    expected = ComposedLayout(composed.outer, flat_product(composed.inner, Layout(3, 1)))
    _assert_pointwise_equal(result, expected)


# ---------------------------------------------------------------------------
# Recursive composition and push-through
# ---------------------------------------------------------------------------

def test_compose_composed_layout_as_outer_pushes_through():
    """compose(ComposedLayout, Layout) pushes into the inner."""
    base_inner = compose(Swizzle(2, 0, 2), Layout(16, 1))
    composed = compose(Layout(8, 2), base_inner)
    rhs = Layout(4, 2)

    result = compose(composed, rhs)
    assert isinstance(result, ComposedLayout)
    # Verify pointwise: result(i) == composed(rhs(i))
    for i in range(size(rhs)):
        assert result(i) == composed(rhs(i))


def test_recursive_compose_chain_stays_exact():
    """compose(A2, compose(A1, composed)) chains remain exact."""
    base = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))

    # Build a chain: outer2(outer1(base(i, j)))
    outer1 = Layout((4, 4), (4, 1))
    step1 = compose(outer1, base)
    assert isinstance(step1, ComposedLayout)

    outer2 = Layout(8, 2)
    step2 = compose(outer2, step1)
    assert isinstance(step2, ComposedLayout)

    for i in range(size(step2)):
        assert step2(i) == outer2(step1(i))


def test_compose_with_hierarchical_inner_layout():
    """Composed layouts with nested-tuple inner shapes work correctly."""
    inner = compose(Swizzle(2, 0, 2), Layout(((2, 4), (2, 4)), ((1, 4), (2, 8))))
    outer = Layout(16, 2)
    result = compose(outer, inner)

    assert isinstance(result, ComposedLayout)
    for i in range(size(result)):
        assert result(i) == outer(inner(i))


def test_logical_divide_on_hierarchical_composed():
    """logical_divide works on composed layouts with nested shapes."""
    inner = compose(Swizzle(2, 0, 2), Layout((4, 4), (4, 1)))
    composed = compose(Layout(8, 2), inner)
    result = logical_divide(composed, (2, 2))

    assert isinstance(result, ComposedLayout)
    expected = ComposedLayout(composed.outer, logical_divide(composed.inner, (2, 2)))
    _assert_pointwise_equal(result, expected)


# ---------------------------------------------------------------------------
# Full-slice and multi-mode
# ---------------------------------------------------------------------------

def test_full_slice_on_composed_layout_preserves_identity():
    """Slicing with all-None returns the same composed layout with offset 0."""
    composed = compose(
        Layout((4, 4), (4, 1)),
        compose(Swizzle(2, 0, 2), Layout((4, 4), (4, 1))),
    )
    sub, offset = slice_and_offset((None, None), composed)
    assert offset == 0
    assert isinstance(sub, ComposedLayout)
    _assert_pointwise_equal(sub, composed)


def test_mode_on_composed_layout_mode1():
    """mode(composed, 1) also works correctly."""
    inner = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    composed = compose(Layout((4, 4), (4, 1)), inner)

    m1 = mode(composed, 1)
    assert isinstance(m1, ComposedLayout)
    for j in range(size(m1)):
        assert m1(j) == composed(0, j)


# ---------------------------------------------------------------------------
# Tensor.view() with ComposedLayout
# ---------------------------------------------------------------------------

def test_tensor_view_with_composed_layout():
    """Tensor.view() accepts a ComposedLayout."""
    composed = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout(16, 1)))
    t = Tensor(Layout(16, 1), data=list(range(256)))
    v = t.view(composed)
    assert isinstance(v.layout, ComposedLayout)
    for i in range(size(v)):
        assert v(i) == composed(i)


# ---------------------------------------------------------------------------
# Generic analysis functions with ComposedLayout
# ---------------------------------------------------------------------------

def test_image_on_composed_layout():
    composed = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout(16, 1)))
    img = image(composed)
    expected = sorted(set(composed(i) for i in range(size(composed))))
    assert img == expected


def test_offset_table_on_composed_layout():
    composed = compose(
        Layout((4, 4), (4, 1)),
        compose(Swizzle(2, 0, 2), Layout((4, 4), (4, 1))),
    )
    table = offset_table(composed)
    assert isinstance(table, dict)
    # Every element in the domain maps to some offset
    for i in range(4):
        for j in range(4):
            offset = composed(i, j)
            assert offset in table
            assert (i, j) in table[offset]


def test_footprint_on_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    fp = footprint(composed)
    assert isinstance(fp, dict)
    assert "unique_offsets" in fp
    assert fp["unique_offsets"] >= 1


def test_bank_conflicts_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), preoffset=0)
    result = bank_conflicts(
        composed,
        element_bytes=4,
        num_banks=4,
        bank_width_bytes=4,
        group_size=4,
    )
    assert result == {
        "conflict_free": False,
        "max_ways": 2,
        "bank_to_threads": {
            0: [0, 1],
            1: [0, 1],
            2: [2, 3],
            3: [2, 3],
        },
    }


def test_coalescing_efficiency_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), preoffset=4)
    result = coalescing_efficiency(
        composed,
        element_bytes=4,
        warp_size=4,
        cache_line_bytes=16,
    )
    assert result == {
        "transactions": 2,
        "efficiency": 1.0,
        "cache_lines": [0, 1],
    }


def test_segment_analysis_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), preoffset=4)
    result = segment_analysis(
        composed,
        element_bytes=4,
        warp_size=4,
        segment_bytes=8,
        cache_line_bytes=16,
    )
    assert result == {
        "segments": 4,
        "cache_lines": 2,
        "unique_bytes": 32,
        "requested_bytes": 32,
        "transferred_bytes": 32,
        "segment_efficiency": 1.0,
        "first_byte_addr": 4,
        "first_alignment": 4,
    }


def test_per_group_bank_conflicts_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), preoffset=4)
    result = per_group_bank_conflicts(
        composed,
        element_bytes=4,
        group_size=2,
    )
    assert result == {
        "groups": [
            {
                "conflict_free": True,
                "max_ways": 1,
                "bank_to_threads": {
                    5: [0],
                    10: [0],
                    4: [1],
                    11: [1],
                },
            },
            {
                "conflict_free": True,
                "max_ways": 1,
                "bank_to_threads": {
                    7: [2],
                    8: [2],
                    6: [3],
                    9: [3],
                },
            },
        ],
        "worst_group": 0,
        "worst_max_ways": 1,
    }


def test_per_group_coalescing_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), preoffset=4)
    result = per_group_coalescing(
        composed,
        element_bytes=4,
        group_size=2,
        cache_line_bytes=16,
    )
    assert result == {
        "groups": [
            {
                "transactions": 2,
                "efficiency": 0.5,
                "cache_lines": [0, 1],
            },
            {
                "transactions": 1,
                "efficiency": 1.0,
                "cache_lines": [0],
            },
        ],
        "worst_group": 0,
        "worst_efficiency": 0.5,
    }


def test_is_injective_on_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    # Just verify it returns a bool without raising
    result = is_injective(composed)
    assert isinstance(result, bool)


def test_is_surjective_on_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    result = is_surjective(composed)
    assert isinstance(result, bool)


def test_is_bijective_on_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    result = is_bijective(composed)
    assert isinstance(result, bool)


def test_is_contiguous_on_composed_layout():
    composed = compose(Layout(8, 2), compose(Swizzle(2, 0, 2), Layout(8, 1)))
    result = is_contiguous(composed)
    assert isinstance(result, bool)


def test_functionally_equal_on_composed_layout():
    composed1 = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout(16, 1)))
    composed2 = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout(16, 1)))
    assert functionally_equal(composed1, composed2)

    different = compose(Layout(16, 1), compose(Swizzle(2, 0, 2), Layout(16, 1)))
    assert not functionally_equal(composed1, different)


def test_cycles_on_composed_layout():
    # cycles requires a dense injective permutation; use an identity-sized composed layout
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), preoffset=0)
    result = cycles(composed)
    assert isinstance(result, list)


def test_fixed_points_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), preoffset=0)
    result = fixed_points(composed)
    assert isinstance(result, (list, set))


def test_order_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), preoffset=0)
    result = order(composed)
    assert isinstance(result, int)
    assert result >= 1
