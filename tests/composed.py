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
