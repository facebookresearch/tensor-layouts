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

"""The ``LayoutExpr`` layer: ``ComposedLayout`` and every predicate / coercer
that operates on the union ``LayoutExpr = Layout | ComposedLayout``.

Layered above ``core.py``. The ``LayoutExpr`` type alias is defined here, and
anything that switches behaviour on whether an object is a ``Layout`` versus a
``ComposedLayout`` (``is_layout``, ``is_affine``, ``as_layout``,
``as_layout_expr``, ``as_affine_layout``, ``split_outer_swizzle``,
``_forward_layout_domain``) lives here. Algebraic operations on the resulting
``LayoutExpr`` values live in ``algebra.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core import (
    Layout,
    Swizzle,
    UnsupportedComposedLayoutError,
    is_int,
    is_tuple,
)


__all__ = [
    "ComposedLayout",
    "LayoutExpr",
    "is_layout",
    "is_affine",
    "is_pure_shape",
    "is_empty",
    "as_layout",
    "as_layout_expr",
    "as_affine_layout",
    "split_outer_swizzle",
]


_NO_FORWARD = object()


@dataclass(frozen=True)
class ComposedLayout:
    """An exact layout-expression node for compositions that are not affine.

    Semantics:
        ComposedLayout(outer, inner, offset)(coord) ==
            outer(offset + inner(coord))

    The inner layout defines the logical domain (shape, size, rank, depth).
    The offset remains inside the composition, before the outer nonlinear
    map, which is why ComposedLayout intentionally does not expose .stride.

    Supported inner shapes
    ----------------------

    Two shapes for ``(outer, inner)`` are supported, with very different
    expressiveness:

    1. **Layout (or ComposedLayout) inner** -- the canonical case. All
       layout-algebra operations are defined: ``coalesce``, ``complement``,
       ``flatten``, ``logical_product``, ``logical_divide``, ``compose``,
       ``right_inverse``, ``left_inverse``, etc. This is what arises from
       slicing a swizzled layout (``ComposedLayout(Swizzle, Layout, k)``)
       and from nesting two compositions.

    2. **Swizzle inner** -- the *inverse-form* shape, structurally
       ``ComposedLayout(outer=Layout, inner=Swizzle, offset)``. This form
       arises only as the result of ``right_inverse`` / ``left_inverse``
       applied to an offset-bearing swizzle-fronted ``ComposedLayout`` (see
       CuTe ``swizzle_layout.hpp:348-358``); the inverse swaps the slots and
       negates the offset. Its logical domain is 1-D with extent taken from
       ``outer.shape``.

       **What works:** ``__call__``, ``size``, ``shape``, ``cosize``,
       ``rank``, ``depth``, ``flatten``, ``coalesce`` (no-op: rank-1 with
       no structure to merge), ``right_inverse``, ``left_inverse``,
       ``compose`` (so the inverse-and-cancel round trip is closed).

       **What raises NotImplementedError:** ``complement``,
       ``logical_product``, ``logical_divide``. These ops delegate to the
       inner layout and ``Swizzle`` does not satisfy the Layout interface;
       defining them on the inverse-form would also require a sensible
       answer for ``complement`` of a 1-D non-affine layout, which is
       not just a coding question. CuTe C++ refuses these forms too --
       the corresponding templates don't instantiate. Matching CuTe's
       posture keeps tensor-layouts honest: structurally allowed,
       semantically narrow, errors loud.

       **Beware of negative offsets.** The negation in the inverse rule
       means ``__call__`` can return values below zero on early indices
       (``F6(0) = -4`` for ``ComposedLayout(Layout(32,1), Swizzle(2,1,3),
       offset=-4)``). The inverse-form is intended for composition with
       its forward layout, where the negative term cancels; using it as
       direct buffer addressing is wrong. ``Tensor`` rejects storage that
       would receive negative addresses; see ``tensor.py``
       ``_validate_storage``.
    """

    outer: Any
    inner: "LayoutExpr"
    # ``offset`` is keyword-only -- you must spell it as ``offset=k``. This
    # rules out the silent porting trap where someone copies a CuTe C++
    # ``ComposedLayout<A, Offset, B>`` literal into Python expecting the
    # same positional order. tensor-layouts uses ``ComposedLayout(outer,
    # inner, offset=k)`` so the common zero-offset case can drop the
    # ``offset`` argument entirely; CuTe / pycute place the offset
    # positionally between A and B and require it on every literal. See
    # docs/layout_api.md for a full discussion.
    offset: int = field(default=0, kw_only=True)
    # Lazy O(1) cache for cosize. Populated by cosize() on first call via
    # object.__setattr__ (frozen dataclass blocks normal assignment).
    # Excluded from init/repr/eq/hash so two equal ComposedLayouts with
    # different cache states still compare equal and hash the same.
    _cached_cosize: "int | None" = field(
        default=None, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self):
        if not callable(self.outer):
            raise TypeError(
                f"ComposedLayout outer must be callable, got {type(self.outer).__name__}"
            )
        if not (is_layout(self.inner) or isinstance(self.inner, Swizzle)):
            raise TypeError(
                f"ComposedLayout inner must be Layout, ComposedLayout, or Swizzle, "
                f"got {type(self.inner).__name__}"
            )
        if not is_int(self.offset):
            raise TypeError(
                f"ComposedLayout offset must be int, got {type(self.offset).__name__}"
            )

    @property
    def shape(self):
        if isinstance(self.inner, Swizzle):
            return self.outer.shape
        return self.inner.shape

    def __repr__(self) -> str:
        return f"ComposedLayout({self.outer!r}, {self.inner!r}, offset={self.offset!r})"

    def __str__(self) -> str:
        if self.offset:
            return f"({self.outer}) o {{{self.offset}}} o ({self.inner})"
        return f"({self.outer}) o ({self.inner})"

    def __call__(self, *args):
        from .algebra import slice_and_offset
        from .core import has_none

        if len(args) == 1:
            coords = args[0]
        else:
            coords = args
        if coords is None:
            return self
        if has_none(coords):
            return slice_and_offset(coords, self)[0]
        return self.outer(self.offset + self.inner(coords))

    def __len__(self):
        from .algebra import size
        return size(self)

    def __iter__(self):
        from .algebra import idx2crd, size
        for i in range(size(self)):
            yield idx2crd(i, self.shape)


LayoutExpr = Layout | ComposedLayout


# =============================================================================
# Layout / ComposedLayout-aware predicates and conversions
# =============================================================================


def is_layout(x) -> bool:
    """Check if x is a supported layout object (matches CuTe's is_layout trait)."""
    return isinstance(x, (Layout, ComposedLayout))


def is_affine(obj) -> bool:
    """Return True if obj is (or contains) an affine ``Layout`` node.

    This is a *structural* check: a ``ComposedLayout`` that happens to be
    mathematically affine (e.g. a swizzle-free composition that could
    coalesce to a flat ``Layout``) still returns False, because we have no
    machinery to attempt that normalization. Callers that have an arbitrary
    ``LayoutExpr`` and need direct ``.shape``/``.stride`` access should pair
    this with ``as_affine_layout()`` (which raises on ``ComposedLayout``).

    Works with both ``Layout``/``ComposedLayout`` and any object exposing a
    ``.layout`` attribute (e.g. ``Tensor``).
    """
    layout = obj.layout if hasattr(obj, "layout") and not is_layout(obj) else obj
    return isinstance(layout, Layout)


def is_pure_shape(t) -> bool:
    """Check if t is a pure shape (nested ints with no Layouts).

    A pure shape is an int or a tuple containing only ints (recursively).
    This is used to distinguish shape tuples from tiler tuples that may
    contain Layouts.

    Examples:
        is_pure_shape(4) -> True
        is_pure_shape((2, 3)) -> True
        is_pure_shape(((2, 3), 4)) -> True
        is_pure_shape(Layout(4, 1)) -> False
        is_pure_shape((Layout(4, 1), 3)) -> False
    """
    if is_layout(t):
        return False
    if is_int(t):
        return True
    if is_tuple(t):
        return all(is_pure_shape(elem) for elem in t)
    return False


def is_empty(obj) -> bool:
    """Return True if obj is (or contains) the unit/empty layout.

    The unit layout has the empty-tuple shape ``()``, rank 0, and size 1
    (the empty product). It is the multiplicative identity for layout
    composition and concatenation. This is **distinct** from a zero-sized
    layout such as ``Layout((0,), (0,))``, which has rank 1 and size 0;
    use ``size(L) == 0`` to test for that.

    Matches the conventions in pycute (``product(()) == 1``) and CuTe C++
    (``Product`` returns ``Int<1>{}`` for empty tuples).

    Works with both Layout objects and Tensors (via the ``.layout`` attribute).
    """
    layout = obj.layout if hasattr(obj, "layout") and not is_layout(obj) else obj
    if not is_layout(layout):
        return False
    return is_tuple(layout.shape) and len(layout.shape) == 0


def as_layout(obj):
    """Convert an affine Layout-like object to a Layout.

    Accepts our Layout, or any object with .shape and .stride attributes
    (e.g. pycute Layout). This allows viz and analysis functions to accept
    foreign affine layout objects without requiring them as a dependency.

    ComposedLayout is intentionally rejected here because it has a logical
    domain but no affine stride tree. Generic consumers should use
    as_layout_expr() instead.
    """
    if hasattr(obj, "layout") and not isinstance(obj, (Layout, ComposedLayout)):
        return as_layout(obj.layout)
    if isinstance(obj, Layout):
        return obj
    if isinstance(obj, ComposedLayout):
        raise TypeError("Expected affine Layout, got ComposedLayout")
    if hasattr(obj, "shape") and hasattr(obj, "stride"):
        return Layout(obj.shape, obj.stride)
    raise TypeError(f"Expected Layout, got {type(obj).__name__}")


def as_layout_expr(obj):
    """Convert a layout-like object to Layout or ComposedLayout.

    Accepts our Layout / ComposedLayout, Tensor-like objects with a .layout
    attribute, or foreign affine layout objects with .shape/.stride.
    """
    if hasattr(obj, "layout") and not isinstance(obj, (Layout, ComposedLayout)):
        return as_layout_expr(obj.layout)
    if isinstance(obj, (Layout, ComposedLayout)):
        return obj
    if hasattr(obj, "shape") and hasattr(obj, "stride"):
        return Layout(obj.shape, obj.stride)
    raise TypeError(f"Expected LayoutExpr, got {type(obj).__name__}")


def as_affine_layout(obj):
    """Convert obj to an affine Layout, asserting affinity at the boundary.

    Like ``as_layout()`` but explicit about the contract: callers that need
    direct ``.shape`` / ``.stride`` access (analysis, viz, ``Tensor.stride``,
    affine algebra) should use this so the precondition is documented at the
    call site. Use ``as_layout_expr()`` instead if your caller can handle a
    ``ComposedLayout``.

    Raises ``TypeError`` if ``obj`` cannot be coerced to an affine ``Layout``.
    The ``is_affine`` post-check is belt-and-suspenders: today ``as_layout()``
    already guarantees the result, but the explicit assertion documents the
    contract and protects against future loosening.
    """
    layout = as_layout(obj)
    if not is_affine(layout):
        raise TypeError(
            f"as_affine_layout: not affine after conversion: {layout!r} "
            f"(use as_layout_expr() to accept ComposedLayout)"
        )
    return layout


# =============================================================================
# Swizzle structural recognisers / dispatchers
# =============================================================================


def split_outer_swizzle(layout: LayoutExpr):
    """Recognize the canonical ``ComposedLayout(Swizzle, Layout, offset=0)`` form.

    Returns ``(swizzle, inner_layout)`` if ``layout`` is a zero-offset
    swizzle wrapper around an affine layout, else ``None``. This is the
    structural shape produced by ``Tensor`` swizzling and by
    ``ComposedLayout(Sw, L, 0)`` literals; recognising it lets callers
    take fast paths that exploit the swizzle's linearity (e.g. O(1)
    cosize-based address bounds, see ``max_common_vector``).

    "Outer" refers to the slot the Swizzle occupies. The mirror-image
    inverse-form ``ComposedLayout(Layout, Swizzle, offset)`` -- where
    the Swizzle sits in the *inner* slot -- is a different beast: 1-D,
    non-affine, can address negative storage, and arises only as the
    output of ``right_inverse`` / ``left_inverse``. It is intentionally
    NOT recognised here. The current call sites (``_address_bounds``,
    ``max_common_vector``) only make sense for the outer form, and
    conflating the two would let callers misuse a non-affine result via
    affine-shaped reasoning. The inverse-form has its own private
    predicate ``_is_swizzle_inner_composed``; promote it to a sibling
    ``split_inner_swizzle`` if a public consumer ever materialises.

    Forms NOT recognised:

    - Nonzero offset: an affine shift wrapping the swizzle that callers
      must handle explicitly.
    - Inverse-form (see above).
    - Plain ``Layout``: pre-Path-X ``Layout(..., swizzle=Sw)`` no
      longer exists; bare ``Layout`` is purely affine.
    """
    if (
        isinstance(layout, ComposedLayout)
        and isinstance(layout.outer, Swizzle)
        and layout.offset == 0
    ):
        return layout.outer, layout.inner
    return None


def _forward_layout_domain(layout, transform):
    """Apply a domain-only transform to the inner layout of a layout expression.

    Path X: ``Layout`` is purely affine. ComposedLayout always stays
    composed; bare Layout falls through to the caller (which then applies
    the transform on the whole affine layout).
    """
    if isinstance(layout, ComposedLayout):
        if isinstance(layout.inner, Swizzle):
            return _NO_FORWARD
        return ComposedLayout(layout.outer, transform(layout.inner), offset=layout.offset)
    return _NO_FORWARD


def _is_swizzle_inner_composed(obj: Any) -> bool:
    """True iff obj is a ComposedLayout whose inner slot holds a Swizzle.

    Structurally: ``ComposedLayout(outer=Layout, inner=Swizzle, offset)``.
    Semantically: this form arises only as the result of ``right_inverse`` /
    ``left_inverse`` applied to an offset-bearing swizzle-fronted
    ``ComposedLayout`` (e.g. ``Sw o {+k} o L``); the inverse swaps the outer
    and inner slots and negates the offset, putting the Swizzle on the inner
    side.

    Most layout-algebra operations are not defined on this form -- CuTe C++
    refuses to instantiate ``cosize``, ``coalesce``, ``complement``,
    ``logical_product``, ``logical_divide`` because they delegate to the
    inner layout, and ``Swizzle`` does not satisfy the Layout interface.
    tensor-layouts matches by raising ``NotImplementedError``. The form
    remains usable for ``__call__``, ``size``, ``shape``, ``rank``,
    ``depth``, ``flatten``, and round-tripping through ``right_inverse`` /
    ``left_inverse`` / ``compose`` so that the inverse-and-cancel algebra
    continues to work.
    """
    return isinstance(obj, ComposedLayout) and isinstance(obj.inner, Swizzle)


def _reject_swizzle_inner_composed(obj: Any, op_name: str) -> None:
    """Raise NotImplementedError if obj is the F6 inverse-form."""
    if _is_swizzle_inner_composed(obj):
        raise UnsupportedComposedLayoutError(
            f"{op_name} is not defined on a ComposedLayout with a Swizzle "
            f"in the inner slot (the inverse-form produced by "
            f"right_inverse/left_inverse on an offset-bearing swizzle-fronted "
            f"ComposedLayout). CuTe C++ refuses this form too. The inverse "
            f"is intended for composition with the forward layout, not for "
            f"direct algebraic manipulation. Got: {obj}"
        )
