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

"""Pure-Python implementation of NVIDIA CuTe layout algebra.

A Layout is a function from logical coordinates to memory offsets, defined by
a pair (shape, stride). The shape describes the logical domain (how many
elements along each dimension), and the stride describes how coordinates map
to offsets: offset = dot(coord, stride).

    Layout((4, 8), (1, 4))  maps a 4x8 column-major matrix.
    Layout((4, 8), (8, 1))  maps a 4x8 row-major matrix.
    Layout(32, 1)           maps 32 contiguous elements.

What makes CuTe's algebra powerful is that shapes can be hierarchical ---
nested tuples like ((2, 4), (3, 2)) describe multi-level coordinate spaces.
This lets you represent complex GPU memory access patterns (tiles within tiles,
swizzled shared memory banks) as simple shape/stride pairs.

The algebra is built on four key operations:

  compose(A, B)      Function composition: compose(A, B)(i) = A(B(i)).
                     B selects which elements of A to visit, and in what order.

  complement(L)      The "other half": a layout that visits the offsets L skips,
                     so Layout(L, complement(L)) covers every offset once.

  logical_divide(L, T)   Factor L into (tile, rest) using T as the tile shape.
                         Defined as compose(L, Layout(T, complement(T))).

  logical_product(A, B)  Reproduce A's pattern at each position B describes.
                         Defined as Layout(A, compose(complement(A), B)).

Division answers "how do I iterate in tiles?", product answers "how do I
replicate a pattern?", and both are defined in terms of compose + complement.
"""

from __future__ import annotations

from .core import *  # noqa: F401, F403
from .expr import *  # noqa: F401, F403
from .algebra import *  # noqa: F401, F403

# Private re-exports needed by tensor.py and other in-package modules that
# previously did ``from .layouts import _NO_FORWARD`` etc.
from .expr import (  # noqa: F401
    _NO_FORWARD,
    _forward_layout_domain,
    _is_swizzle_inner_composed,
    _reject_swizzle_inner_composed,
)

# The package's public surface is the union of the three submodules' __all__s.
# Each submodule owns the curation of its own exports; this avoids drift
# (an entry added to e.g. core.__all__ would otherwise need a parallel edit
# here to be picked up by ``from tensor_layouts.layouts import *``).
from .core import __all__ as _core_all
from .expr import __all__ as _expr_all
from .algebra import __all__ as _algebra_all

__all__ = [*_core_all, *_expr_all, *_algebra_all]
