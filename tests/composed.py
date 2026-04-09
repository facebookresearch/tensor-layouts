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
    contiguity,
    mode_contiguity,
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
