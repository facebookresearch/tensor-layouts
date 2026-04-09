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
