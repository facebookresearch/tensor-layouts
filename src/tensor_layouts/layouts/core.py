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

"""Core types and pure-tuple utilities for the layout algebra.

Provides exception taxonomy, type predicates, tuple operations, the affine
``Layout`` class, ``Tile``, and the ``Swizzle`` primitive. This module has no
dependency on ``ComposedLayout``; everything that needs ComposedLayout in scope
lives in ``expr.py`` and ``algebra.py``.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableType
from typing import Any, Union

# Tuple of int | tuple
IntOrIntTuple = Union[int, tuple["IntOrIntTuple", ...]]


__all__ = [
    # Exceptions
    "LayoutError",
    "UnsupportedComposedLayoutError",
    "TensorStorageError",
    # Type alias
    "IntOrIntTuple",
    # Type predicates
    "is_tuple",
    "is_int",
    "is_scalar",
    "is_iterable",
    "has_none",
    "coords_all_none",
    # Shape conversions
    "as_tuple",
    "as_shape",
    "as_list",
    "unwrap",
    "normalize",
    # Core types
    "Layout",
    "Tile",
    "Swizzle",
    "make_swizzle",
    # Stride computation
    "compute_col_major_strides",
    "compute_row_major_strides",
    # Tuple operations
    "concat",
    "congruent",
    "weakly_congruent",
    "compatible",
    "tuple_max",
    "transform_tuple",
    "zip_transform",
    "fold",
    "fold_accumulate",
    "elem_scale",
    "inner_product",
    "prefix_product",
    "suffix_product",
    "product_each",
]


# =============================================================================
# Exception taxonomy
# =============================================================================
#
# These three classes give the layout algebra, ComposedLayout structural
# restrictions, and Tensor storage state errors stable identities while
# remaining backwards-compatible with handlers that catch the standard
# Python base classes.
#
# The classes are used by raise sites throughout this package and also
# re-exported via the package's star-import surface so user code can
# write e.g. `except LayoutError` when it cares about the specific kind.


class LayoutError(ValueError):
    """A layout-algebra precondition failed.

    Raised when a Layout / ComposedLayout operation is called with inputs
    that violate the algebra's structural rules (shape/stride congruence,
    rank mismatch, mode out of range, tiler incompatibility, swizzle mask
    overlap, etc.). Subclasses ``ValueError`` so existing
    ``except ValueError`` handlers continue to catch it.
    """


class UnsupportedComposedLayoutError(NotImplementedError):
    """A ComposedLayout form does not support the requested operation.

    Mostly the F6 inverse-form ``ComposedLayout(Layout, Swizzle, offset)``
    produced by ``right_inverse`` / ``left_inverse`` of an offset-bearing
    swizzle-fronted layout. Operations that delegate to the inner Layout
    (``complement``, ``coalesce``, ``logical_product``, ``logical_divide``,
    ``to_F2_matrix``) cannot be defined on this form because the inner
    is a ``Swizzle`` rather than a Layout. Mirrors CuTe C++ -- those
    templates fail to instantiate. Subclasses ``NotImplementedError`` so
    existing handlers continue to catch it.
    """


class TensorStorageError(ValueError):
    """Tensor's storage state is inconsistent with the requested operation.

    Raised when a Tensor without backing storage is asked to perform an
    operation that requires storage (assignment, view), or when a Tensor's
    layout addresses positions outside its storage (negative indices,
    out-of-range upper bound). Subclasses ``ValueError`` because from the
    caller's point of view the offending input is the storage value (or
    its absence) -- existing ``except ValueError`` handlers continue to
    catch it.
    """


# =============================================================================
# Type predicates
# =============================================================================
#
# Simple type checks used throughout the algebra.
#


def is_tuple(x) -> bool:
    """Check if x is a tuple (matches CuTe's is_tuple convention)."""
    return isinstance(x, tuple)


def is_int(x) -> bool:
    """Check if x is an integer (excluding booleans which are int subclasses in Python)."""
    return isinstance(x, int) and not isinstance(x, bool)


def is_scalar(x) -> bool:
    """Check if x represents a scalar shape (int, not tuple)."""
    return is_int(x)


def is_iterable(x) -> bool:
    """Check if x is an iterable collection (excluding strings and bytes)."""
    return isinstance(x, IterableType) and not isinstance(x, (str, bytes))


def has_none(a) -> bool:
    """Determine if None appears at any terminal of an int-tuple.

    Used to detect slice operations in coordinate arguments.

    Examples:
        has_none(3) -> False
        has_none(None) -> True
        has_none((1, None, 3)) -> True
        has_none((1, (2, None))) -> True
    """
    return fold(a, False, lambda acc, v: acc or v is None)


def coords_all_none(a) -> bool:
    """Return True if every terminal coordinate is None."""
    return fold(a, True, lambda acc, v: acc and v is None)


# =============================================================================
# Shape conversions
# =============================================================================
#
#   Function          Direction              When to use
#   ────────────────  ─────────────────────  ──────────────────────────────────
#   as_tuple(x)       int|tuple → tuple      Iterate uniformly over modes
#   as_shape(items)   list → int|tuple       Build result, preserving rank
#   unwrap(t)         (x,) → x               Extract single composed mode
#   normalize(x)      any → int|tuple        Sanitize user input
#


def as_tuple(x) -> tuple:
    """Ensure x is a tuple for uniform iteration.

    Scalars become single-element tuples; tuples pass through unchanged.
    Use this to iterate over modes uniformly:

        for s, d in zip(as_tuple(shape), as_tuple(stride)):
            ...

    Examples:
        as_tuple(8)       → (8,)
        as_tuple((4, 8))  → (4, 8)
    """
    if isinstance(x, int):
        return (x,)
    return tuple(x)


def as_shape(items) -> IntOrIntTuple:
    """Convert a list of modes back to a shape, preserving rank semantics.

    Single-element lists become scalars (rank-0); multi-element become tuples.
    Use this when building computed results from a list of modes:

        result_shapes = [...]  # built up during computation
        return Layout(as_shape(result_shapes), as_shape(result_strides))

    Examples:
        as_shape([8])        → 8        (scalar)
        as_shape([(2, 4)])   → (2, 4)   (still a tuple, just unwrapped from list)
        as_shape([4, 8])     → (4, 8)   (tuple)
    """
    if len(items) == 1:
        return items[0]
    return tuple(items)


def as_list(x) -> list:
    """Convert a shape or stride to a list for mutation.

    Unlike as_tuple (for iteration) or as_shape (for building results),
    as_list is for when you need to modify the structure before creating
    a new Layout.

    Examples:
        as_list(8)              # [8]
        as_list((4, 8))         # [4, 8]
        as_list(((2, 4), 8))    # [(2, 4), 8]  (nested structure preserved)
    """
    return list(as_tuple(x))


def unwrap(t):
    """Unwrap a single-element tuple to its element; pass through otherwise.

    Use this when extracting a single mode from composition, where the result
    might be wrapped in a spurious outer tuple:

        composed = compose(mode_layout, other)
        result_shapes.append(unwrap(composed.shape))

    Examples:
        unwrap((4,))    → 4
        unwrap((4, 8))  → (4, 8)
        unwrap(4)       → 4
    """
    if is_tuple(t) and len(t) == 1:
        return t[0]
    return t


def normalize(x: Any) -> IntOrIntTuple:
    """Normalize user input to a canonical shape: int | tuple[int | tuple, ...].

    - int passes through unchanged
    - iterables (lists, generators) become tuples with normalized elements
    - single-element tuples are preserved (user intent is explicit)

    Used by Layout.__init__ to sanitize user-provided shapes/strides.

    Examples:
        normalize(8)           → 8
        normalize([4, 8])      → (4, 8)
        normalize((4,))        → (4,)      # preserved!
        normalize([[2, 4], 8]) → ((2, 4), 8)
    """
    if is_int(x):
        return x
    if is_iterable(x):
        return tuple(normalize(elem) for elem in x)
    raise TypeError(f"Cannot normalize shape: {type(x).__name__}")


# =============================================================================
# Tuple arithmetic and structural predicates
# =============================================================================
#
# These pure-tuple utilities are needed by Layout's constructor (congruent),
# by stride helpers (prefix_product, suffix_product), and throughout the
# algebra. They live in core because they have no Layout/ComposedLayout
# dependency.
#


def congruent(a: IntOrIntTuple, b: IntOrIntTuple) -> bool:
    """Returns True if two layouts have the same rank and structure.

    Matches CuTe's congruent(): tests if two tuples have the same profile
    (hierarchical rank division).  Congruent shapes can be element-wise
    zipped (like zip_transform).

    Examples:
        congruent((2, 3), (4, 5))     -> True   (same rank)
        congruent((2, 3), 6)          -> False  (int vs tuple)
        congruent(((2, 3), 4), ((5, 6), 7))  -> True   (same nesting)
    """
    if isinstance(a, int) and isinstance(b, int):
        return True
    if is_tuple(a) and is_tuple(b):
        return len(a) == len(b) and all(congruent(sa, sb) for sa, sb in zip(a, b))
    return False


def weakly_congruent(a: IntOrIntTuple, b: IntOrIntTuple) -> bool:
    """Returns True if A's profile is contained in B's profile.

    Matches CuTe's weakly_congruent(): a partial order A <= B where A's
    hierarchical rank division is "at most as deep as" B's.  A scalar on
    the A side matches any sub-tree on the B side, but a tuple on the A
    side requires at least as much structure on the B side.

    Examples:
        weakly_congruent(6, (2, 3))              -> True   (scalar matches anything)
        weakly_congruent((2, 3), 6)              -> False  (tuple vs scalar)
        weakly_congruent((2, 3), (4, 5))         -> True   (same rank)
        weakly_congruent((2, (3, 4)), (5, (6, 7)))  -> True  (same nesting)
        weakly_congruent((2, (3, 4)), (5, 6))    -> False  (A deeper than B)
        weakly_congruent((2, 3), (5, (6, 7)))    -> True   (A flatter than B)
    """
    if isinstance(a, int):
        return True
    if is_tuple(a) and is_tuple(b):
        return len(a) == len(b) and all(weakly_congruent(sa, sb) for sa, sb in zip(a, b))
    return False


def compatible(a: IntOrIntTuple, b: IntOrIntTuple) -> bool:
    """Checks if shape A is compatible with shape B.

    Matches CuTe's compatible(): A is compatible with B if size(A) == size(B)
    and any coordinate into A can also be used as a coordinate into B.
    This is a partial order: A <= B.

    A is compatible with B if A's modes can be grouped to match B's structure.

    Examples:
        compatible((2, 2, 3), (4, 3))  -> True   (2*2 groups into 4)
        compatible(12, (2, 2, 3))      -> True   (scalar is compatible with any shape)
        compatible((2, 2, 3), (5, 2))  -> False  (sizes don't match)
    """
    # inlined shape-product to avoid algebra import
    def _sz(x):
        return fold(x, 1, lambda acc, v: acc * v)

    if _sz(a) != _sz(b):
        return False

    if is_scalar(a):
        return True
    if is_scalar(b):
        return False

    if len(a) == len(b):
        return all(compatible(sa, sb) for sa, sb in zip(a, b))

    return _can_group_a_into_b(list(a), b)


def _can_group_a_into_b(a_modes: list, b) -> bool:
    """Check if A's modes can be consumed/grouped to match B's structure."""
    # inlined shape-product to avoid algebra import
    def _sz(x):
        return fold(x, 1, lambda acc, v: acc * v)

    if is_scalar(b):
        target_size = _sz(b)
        acc_size = 1
        while acc_size < target_size and a_modes:
            acc_size *= _sz(a_modes.pop(0))
        return acc_size == target_size

    if is_tuple(b):
        return all(_can_group_a_into_b(a_modes, sub_b) for sub_b in b) and len(a_modes) == 0

    return False


def tuple_max(a: Any) -> int:
    """Return the maximum value across all terminals of a (possibly nested) int-tuple.

    Examples:
        tuple_max(5) -> 5
        tuple_max((3, 7, 2)) -> 7
        tuple_max(((1, 9), (4, 2))) -> 9
    """
    return fold(a, -float("inf"), lambda acc, x: max(acc, x))


def transform_tuple(t: Any, f) -> Any:
    """Apply f to each leaf element of a (possibly nested) tuple.

    Recursively descends into nested tuples, applying f only to
    non-tuple elements (integers). Preserves the hierarchical structure.

    Examples:
        transform_tuple(5, lambda x: x*2) -> 10
        transform_tuple((3, 4), lambda x: x*2) -> (6, 8)
        transform_tuple(((2, 3), 4), lambda x: x+1) -> ((3, 4), 5)
    """
    if is_tuple(t):
        return tuple(transform_tuple(elem, f) for elem in t)
    return f(t)


def zip_transform(a: Any, b: Any, f) -> Any:
    """Apply f(a_i, b_i) element-wise to two congruent tuples.

    Both arguments must have the same structure (same nesting and lengths).
    Recursively descends into nested tuples, applying f to paired leaf elements.

    Examples:
        zip_transform(2, 3, lambda x, y: x*y) -> 6
        zip_transform((1, 2), (3, 4), lambda x, y: x+y) -> (4, 6)
        zip_transform(((1, 2), 3), ((4, 5), 6), lambda x, y: x*y) -> ((4, 10), 18)
    """
    if is_tuple(a):
        if not is_tuple(b) or len(a) != len(b):
            raise LayoutError(f"Structure mismatch: {a} vs {b}")
        return tuple(zip_transform(ai, bi, f) for ai, bi in zip(a, b))
    return f(a, b)


def fold(t: Any, init: Any, f) -> Any:
    """Left-fold a (possibly nested) tuple with an initial value and binary function.

    Recursively descends into nested tuples, applying f only to leaf elements.
    Reduces from left to right: f(f(f(init, leaf0), leaf1), leaf2)...
    For scalars, returns f(init, t).

    This is useful for accumulating results across all elements of a shape/stride.

    Examples:
        fold(5, 0, lambda acc, x: acc + x) -> 5
        fold((1, 2, 3), 0, lambda acc, x: acc + x) -> 6
        fold(((1, 2), 3), 0, lambda acc, x: acc + x) -> 6
        fold((2, 3, 4), 1, lambda acc, x: acc * x) -> 24
    """
    if is_tuple(t):
        acc = init
        for elem in t:
            acc = fold(elem, acc, f)
        return acc
    return f(init, t)


def fold_accumulate(t: Any, init: Any, f, update) -> Any:
    """Left-fold a tuple, collecting intermediate results while threading state.

    Like fold, but returns a tuple of the same structure containing the result
    at each position. The state is updated via `update` after each element.

    Implements the pattern:
        fold_accumulate((a, b, c), v, f, u) = (f(a, v), f(b, u(a, v)), f(c, u(b, u(a, v))))

    Args:
        t: A (possibly nested) tuple to fold over
        init: Initial state value
        f: (element, state) -> result for each element
        update: (element, state) -> new_state for the next element

    Examples:
        # Prefix product (computing strides from shapes):
        fold_accumulate((2, 3, 4), 1,
                        f=lambda elem, state: state,
                        update=lambda elem, state: state * elem)
        # -> (1, 2, 6)  — each result is the product of all prior elements

        # shape_div uses this to divide a shape by a divisor:
        #   f: ceil(element / divisor)  — divide this mode
        #   update: divisor / size(element)  — carry remainder to next mode
        # shape_div((2, 3, 4), 6) -> (1, 1, 4)
        #   mode 0: ceil(2/6)=1, remaining divisor=6/2=3
        #   mode 1: ceil(3/3)=1, remaining divisor=3/3=1
        #   mode 2: ceil(4/1)=4, done
    """
    if isinstance(t, int):
        return f(t, init)

    if not is_tuple(t) or len(t) == 0:
        return t

    results = []
    state = init
    for elem in t:
        results.append(fold_accumulate(elem, state, f, update))
        state = update(elem, state)

    return tuple(results)


def elem_scale(a: Any, b: Any) -> Any:
    """Element-wise scale of int-tuple a by int-tuple b.

    For scalars: a * b.
    For tuple a, scalar b: error (ambiguous).
    For scalar a, tuple b: a * product(b).
    For tuple a, tuple b: pairwise elem_scale.

    Examples:
        elem_scale(3, 4) -> 12
        elem_scale(2, (3, 4)) -> 24   (2 * 12)
        elem_scale((2, 3), (4, 5)) -> (8, 15)
    """
    # inlined shape-product to avoid algebra import
    def _sz(x):
        return fold(x, 1, lambda acc, v: acc * v)

    if is_tuple(a):
        if is_tuple(b):
            return zip_transform(a, b, elem_scale)
        else:
            raise TypeError("Cannot elem_scale tuple by scalar (ambiguous)")
    else:
        if is_tuple(b):
            return elem_scale(a, _sz(b))
        else:
            return a * b


def inner_product(a: Any, b: Any) -> int:
    """Compute the inner product of two int-tuples.

    For scalars: a * b
    For tuples: sum of pairwise inner products.

    Examples:
        inner_product(2, 3) -> 6
        inner_product((1, 2), (3, 2)) -> 7
        inner_product(((2, 3), 4), ((2, 1), 2)) -> 15
    """
    if is_tuple(a):
        if not is_tuple(b) or len(a) != len(b):
            raise LayoutError(f"Structure mismatch: {a} vs {b}")
        return sum(inner_product(x, y) for x, y in zip(a, b))
    else:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError(f"Expected int, got {type(a).__name__} and {type(b).__name__}")
        return a * b


def prefix_product(a: Any, init: Any = 1) -> Any:
    """Compute the exclusive prefix product of an int-tuple.

    Returns a tuple of the same structure where each element is replaced
    by the product of all preceding elements (starting from init).

    For scalars: returns init (the prefix before the scalar).
    For tuples: recursively computes prefix products with carry.

    Examples:
        prefix_product(2) -> 1
        prefix_product((3, 2)) -> (1, 3)
        prefix_product((3, 2, 4)) -> (1, 3, 6)
        prefix_product(((2, 3), 4)) -> ((1, 2), 6)
        prefix_product(((2, 3), (2, 1, 2), (5, 2, 1))) -> ((1, 2), (6, 12, 12), (24, 120, 240))
    """
    # inlined shape-product to avoid algebra import
    def _sz(x):
        return fold(x, 1, lambda acc, v: acc * v)

    if is_tuple(a):
        if is_tuple(init):
            if len(a) != len(init):
                raise LayoutError(f"Length mismatch: {len(a)} vs {len(init)}")
            return zip_transform(a, init, prefix_product)
        else:
            r = []
            for v in a:
                r.append(prefix_product(v, init))
                init = init * _sz(v)
            return tuple(r)
    else:
        if is_tuple(init):
            raise LayoutError("Cannot apply tuple init to scalar shape")
        return init


def suffix_product(a: Any, init: Any = 1) -> Any:
    """Compute the exclusive suffix product of an int-tuple.

    Returns a tuple of the same structure where each element is replaced
    by the product of all following elements (ending with init).

    For scalars: returns init (the suffix after the scalar).
    For tuples: recursively computes suffix products with carry from the right.

    Examples:
        suffix_product(2) -> 1
        suffix_product((3, 2)) -> (2, 1)
        suffix_product((3, 2, 4)) -> (8, 4, 1)
        suffix_product(((2, 3), 4)) -> ((12, 4), 1)
        suffix_product((3, (2, 4))) -> (8, (4, 1))
    """
    # inlined shape-product to avoid algebra import
    def _sz(x):
        return fold(x, 1, lambda acc, v: acc * v)

    if is_tuple(a):
        if is_tuple(init):
            if len(a) != len(init):
                raise LayoutError(f"Length mismatch: {len(a)} vs {len(init)}")
            return zip_transform(a, init, suffix_product)
        else:
            r = []
            carry = init
            for v in reversed(a):
                r.append(suffix_product(v, carry))
                carry = carry * _sz(v)
            return tuple(reversed(r))
    else:
        if is_tuple(init):
            raise LayoutError("Cannot apply tuple init to scalar shape")
        return init


def product_each(shape: Any) -> tuple:
    """Compute the product of each top-level mode of a shape.

    Flattens nested shape elements to get the size of each top-level mode.
    This is useful when you need the "effective" size of each mode after
    flattening any internal structure.

    Args:
        shape: A shape (int or tuple, possibly nested)

    Returns:
        A tuple where each element is the product of the corresponding
        top-level mode. If input is an int, returns (shape,).

    Examples:
        product_each((4, 8))       -> (4, 8)
        product_each(((2, 2), 8))  -> (4, 8)    # 2*2 = 4
        product_each((3, (2, 4)))  -> (3, 8)    # 2*4 = 8
        product_each(16)           -> (16,)
    """
    # inlined shape-product to avoid algebra import
    def _sz(x):
        return fold(x, 1, lambda acc, v: acc * v)

    if is_int(shape):
        return (shape,)
    return tuple(_sz(s) for s in shape)


# =============================================================================
# Layout
# =============================================================================
#
# A Layout is a function from logical coordinates to memory offsets, defined by
# a pair (shape, stride). Each "mode" (dimension) contributes coord_i * stride_i
# to the offset. When a shape element is itself a tuple, that mode has sub-modes,
# creating the hierarchical coordinate spaces that are CuTe's key innovation.
#


def _validate_shape_type(x, name: str) -> None:
    """Validate that *x* is a valid shape or stride: int or nested tuple of ints.

    Raises TypeError with a clear message naming the offending parameter
    (``name`` should be ``"shape"`` or ``"stride"``).
    """
    if is_int(x):
        return
    if isinstance(x, (list, tuple)):
        for elem in x:
            _validate_shape_type(elem, name)
        return
    raise TypeError(f"Layout {name} must be int or tuple of ints, got {type(x).__name__}")


def _validate_nonnegative_shape(shape: Any) -> None:
    """Validate that every shape extent is nonnegative."""
    if is_int(shape):
        if shape < 0:
            raise LayoutError(f"Layout shape must contain only nonnegative extents, got {shape}")
        return
    for elem in shape:
        _validate_nonnegative_shape(elem)


def _fmt_shape(x):
    """Format a shape/stride without Python's trailing-comma for 1-tuples.

    Python renders ``((4, 2),)`` for a 1-element tuple, but CuTe notation
    uses ``((4, 2))`` which is cleaner for human readers.
    """
    if isinstance(x, int):
        return str(x)
    return "(" + ", ".join(_fmt_shape(e) for e in x) + ")"


class Layout:
    """A function from logical coordinates to memory offsets: offset = sum(coord_i * stride_i).

    A Layout is defined by (shape, stride) where shape describes the logical
    domain and stride describes the memory step for each dimension.

    Examples:
        Layout((4, 8), (1, 4))   -- 4x8 column-major matrix
        Layout((4, 8), (8, 1))   -- 4x8 row-major matrix
        Layout(32, 1)            -- 32 contiguous elements
        Layout((4, 8), (2, 0))   -- strided rows, broadcast columns

    Shapes can be hierarchical (nested tuples):
        Layout(((2, 4), 8), ((1, 2), 8))   -- a 2x4 tile within an 8-column layout

    This hierarchy lets you describe complex GPU memory patterns --- tiles within
    tiles, swizzled banks, interleaved threads --- as simple shape/stride pairs.

    Shapes and strides are stored as int | tuple:
    - int for scalar (1D) shapes
    - tuple for multi-dimensional shapes

    Swizzled layouts:
        When composed with a Swizzle, a Layout stores the swizzle function and
        applies it after computing the linear offset. This keeps composition
        closed: compose(Swizzle, Layout) returns a Layout.

    Construction:
        Layout(shape)              -- column-major strides computed automatically
        Layout(shape, stride)      -- explicit shape and stride
        Layout(layout_a, layout_b) -- bundle two layouts as modes of a new layout
    """

    def __init__(self, *args):
        # Path X: Layout is purely affine. The legacy ``swizzle=`` kwarg
        # and ``_cached_cosize`` slot were removed in C3.

        if len(args) == 0:
            self._shape = ()
            self._stride = ()

        elif all(isinstance(arg, Layout) for arg in args):
            if len(args) == 1:
                # Wrap the inner layout's shape/stride to add one level of nesting
                inner = args[0]
                self._shape = normalize((inner.shape,))
                self._stride = normalize((inner.stride,))
            else:
                shapes = tuple(layout.shape for layout in args)
                strides = tuple(layout.stride for layout in args)
                self._shape = shapes
                self._stride = strides

        elif len(args) == 1:
            shape = args[0]
            _validate_shape_type(shape, "shape")
            self._shape = normalize(shape)
            _validate_nonnegative_shape(self._shape)
            self._stride = compute_col_major_strides(self._shape)

        elif len(args) == 2:
            shape, stride = args
            _validate_shape_type(shape, "shape")
            _validate_shape_type(stride, "stride")
            self._shape = normalize(shape)
            _validate_nonnegative_shape(self._shape)
            self._stride = normalize(stride)

        else:
            raise TypeError(
                "Layout() takes shapes/stride arguments or multiple Layout arguments for bundling"
            )

        if not congruent(self._shape, self._stride):
            raise LayoutError(f"Shape {self._shape} and Stride {self._stride} are not congruent")

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Layout):
            return False
        return (
            self.shape == other.shape
            and self.stride == other.stride
        )

    def __hash__(self):
        return hash((self.shape, self.stride))

    def __repr__(self):
        """Return an eval-safe constructor string: Layout((4, 2), (1, 4))."""
        return f"Layout({self._shape!r}, {self._stride!r})"

    def __str__(self):
        """Return human-readable CuTe notation: (4, 2) : (1, 4)."""
        return f"{_fmt_shape(self._shape)} : {_fmt_shape(self._stride)}"

    @property
    def shape(self) -> IntOrIntTuple:
        return self._shape

    @property
    def stride(self) -> IntOrIntTuple:
        return self._stride

    def __call__(self, *args):
        """Map a logical coordinate to a linear index, or slice the layout.

        If any coordinate is None, returns a sublayout (the sliced dimensions).
        A bare None is a full slice and returns the layout unchanged.
        Otherwise returns the integer offset.

        Path X: Layout is purely affine; swizzled forms live in
        ``ComposedLayout``.

        Examples:
            Layout((4,8))((2,3)) -> 26       # coordinate to index
            Layout((4,8))(None) -> (4, 8) : (1, 4)  # full slice
            Layout((4,8))(None, 3) -> (4,) : (1,)  # slice: fix dim 1 to 3, keep dim 0
        """
        from .algebra import slice_modes, crd2offset

        if len(args) == 1:
            coords = args[0]
        else:
            coords = args
        if coords is None:
            return self
        if has_none(coords):
            sliced_shape = slice_modes(coords, self.shape)
            sliced_stride = slice_modes(coords, self.stride)
            if not sliced_shape:
                return Layout((), ())
            return Layout(sliced_shape, sliced_stride)
        return crd2offset(coords, self.shape, self.stride)

    def squeeze(self) -> "Layout":
        """Removes all dimensions of size 1 and their corresponding strides."""
        new_shape, new_stride = self.filter_shapes(self.shape, self.stride, 1)
        return Layout(new_shape, new_stride)

    def filter(self) -> "Layout":
        """Removes all dimensions with a stride of 0."""
        new_shape, new_stride = self.filter_strides(self.shape, self.stride, 0)
        return Layout(new_shape, new_stride)

    def filter_shapes(self, shape, stride, target):
        """Removes all dimensions of size 'target', and their corresponding strides."""
        if is_int(shape):
            if shape == target:
                return (), ()
            return shape, stride

        s_out = []
        d_out = []
        for s, d in zip(shape, stride):
            if is_tuple(s):
                sub_s, sub_d = self.filter_shapes(s, d, target)
                if sub_s != ():
                    s_out.append(sub_s)
                    d_out.append(sub_d)
            elif s != target:
                s_out.append(s)
                d_out.append(d)
        return as_shape(s_out) if s_out else (), as_shape(d_out) if d_out else ()

    def filter_strides(self, shape, stride, target):
        """Removes all dimensions with a stride of 'target', and their corresponding shapes."""
        if is_int(shape):
            if stride == target:
                return (), ()
            return shape, stride

        s_out = []
        d_out = []
        for s, d in zip(shape, stride):
            if is_tuple(s):
                sub_s, sub_d = self.filter_strides(s, d, target)
                if sub_s != ():
                    s_out.append(sub_s)
                    d_out.append(sub_d)
            elif d != target:
                s_out.append(s)
                d_out.append(d)
        return as_shape(s_out) if s_out else (), as_shape(d_out) if d_out else ()

    def __len__(self):
        """Number of elements in the layout's domain."""
        from .algebra import size
        return size(self)

    def __iter__(self):
        """Yield coordinates in colexicographic order (flat index 0, 1, 2, ...)."""
        from .algebra import idx2crd, size
        for i in range(size(self)):
            yield idx2crd(i, self._shape)


def concat(t1: Any, t2: Any):
    if is_tuple(t1) and is_tuple(t2):
        return t1 + t2
    if isinstance(t1, Layout) and isinstance(t2, Layout):
        return Layout(
            as_tuple(t1.shape) + as_tuple(t2.shape), as_tuple(t1.stride) + as_tuple(t2.stride)
        )
    raise TypeError(f"Cannot concatenate objects of {type(t1).__name__} and {type(t2).__name__}")


# =============================================================================
# Stride helpers
# =============================================================================


def compute_col_major_strides(shape: IntOrIntTuple) -> IntOrIntTuple:
    """Compute column-major (leftmost-fastest) strides for a shape.

    Each element gets stride equal to the product of all preceding elements,
    making the first (leftmost) mode vary fastest --- like Fortran/column-major order.
    """
    strides = prefix_product(shape)
    return _zero_leading_unit_strides(shape, strides)


def compute_row_major_strides(shape: IntOrIntTuple) -> IntOrIntTuple:
    """Compute row-major (rightmost-fastest) strides for a shape.

    Each element gets stride equal to the product of all following elements,
    making the last (rightmost) mode vary fastest --- like C/row-major order.
    """
    return suffix_product(shape)


def _zero_leading_unit_strides(shape, strides):
    """CuTe convention: leading size-1 modes get stride 0 instead of 1."""
    if is_int(shape):
        if shape == 1 and strides == 1:
            return 0
        return strides

    result = []
    still_leading = True
    for s, d in zip(shape, strides):
        if is_tuple(s):
            if still_leading:
                sub = _zero_leading_unit_strides(s, d)
                result.append(sub)
                # inlined shape-product to avoid algebra import
                if fold(s, 1, lambda acc, x: acc * x) != 1:
                    still_leading = False
            else:
                result.append(d)
        else:
            if still_leading and s == 1 and d == 1:
                result.append(0)
            else:
                result.append(d)
                if s != 1:
                    still_leading = False
    return tuple(result)


# =============================================================================
# Tile
# =============================================================================


class Tile(tuple):
    """A Tiler is a tuple-of-Layouts used for mode-by-mode composition.

    Tile is semantically distinct from a plain tuple: it signals mode-by-mode
    composition rather than bundling.  When you compose(L, Tile(A, B)), each
    mode of L is composed independently:

        compose(a, tiler) = Layout(compose(mode(a, 0), tiler[0]),
                                   compose(mode(a, 1), tiler[1]), ...)

    This is different from compose(L, Layout((s0, s1), (d0, d1))) where the
    Layout is treated as a single mapping.  Tile makes the intent explicit:
    "apply these tilers to L's modes, one-by-one."

    Examples:
        # (12,(4,8)):(59,(13,1))
        a = Layout((12, (4, 8)), (59, (13, 1)))

        # <3:4, 8:2>
        tiler = Tile(Layout(3, 4), Layout(8, 2))

        # (3,(2,4)):(236,(26,1))
        result = compose(a, tiler)
    """

    def __new__(cls, *layouts):
        """Create a Tile from one or more Layouts.

        Args:
            *layouts: Layout objects to include in the tile
        """
        for i, layout in enumerate(layouts):
            if not isinstance(layout, Layout):
                raise TypeError(f"Tile element {i} must be a Layout, got {type(layout).__name__}")
        return super().__new__(cls, layouts)

    def __repr__(self):
        contents = ", ".join(repr(layout) for layout in self)
        return f"Tile({contents})"


# =============================================================================
# Swizzle
# =============================================================================
#
# Swizzling is used in GPU shared memory to avoid bank conflicts. Shared memory
# is divided into banks (typically 32), and threads that access the same bank
# in the same cycle must serialize. By XORing row bits into column bits,
# adjacent rows access different banks, enabling full memory bandwidth.
#
# A Swizzle is a nonlinear function (it uses XOR, not multiply-add), so it
# cannot be represented as strides. Instead, compose(Swizzle, Layout) produces
# a Layout with an embedded swizzle that applies the layout first, then swizzles.
#


class Swizzle:
    """A nonlinear index transformation that XORs two bit fields to avoid bank conflicts.

    GPU shared memory is divided into banks (typically 32). When multiple threads
    access the same bank simultaneously, they serialize. Swizzling avoids this by
    XORing row bits into column bits so that adjacent rows map to different banks,
    enabling full memory bandwidth.

    Given an index with bit pattern: 0bxxxYYYxxxxZZZxxxx
    - base: number of least-significant bits to keep constant (rightmost xxxx)
    - bits: number of bits in each mask (ZZZ and YYY width)
    - shift: distance between the two bit fields

    The operation replaces ZZZ with (ZZZ XOR YYY), leaving everything else unchanged.

    Args:
        bits: Number of bits in each mask
        base: Number of least-significant bits to keep constant
        shift: Distance between the two masks (positive: YYY is above ZZZ)

    Examples:
        Swizzle(3, 0, 3)  -- XOR bits [0,3) with bits [3,6)
        Swizzle(2, 1, 3)  -- XOR bits [1,3) with bits [4,6)

    Visual example for Swizzle(3, 0, 3):
        Input index:  0b___YYY___ZZZ   (Y=row bits [3,6), Z=col bits [0,3))
        Output index: 0b___YYY___(ZZZ XOR YYY)

        Concrete: index 19 = 0b010_011  (Y=010=2, Z=011=3)
            -> 0b010_(011 XOR 010) = 0b010_001 = 17

        This causes adjacent rows to access different memory banks,
        avoiding shared memory bank conflicts.
    """

    def __init__(self, bits: int, base: int, shift: int):
        self.bits = bits
        self.base = base
        self.shift = shift

    def __repr__(self) -> str:
        return f"Swizzle({self.bits}, {self.base}, {self.shift})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Swizzle):
            return False
        return self.bits == other.bits and self.base == other.base and self.shift == other.shift

    def __hash__(self) -> int:
        return hash((self.bits, self.base, self.shift))

    @property
    def yyy_msk(self) -> int:
        """Bit mask for the Y (source) bits of the swizzle."""
        return ((1 << self.bits) - 1) << (self.base + max(0, self.shift))

    @property
    def zzz_msk(self) -> int:
        """Bit mask for the Z (destination) bits of the swizzle."""
        return ((1 << self.bits) - 1) << (self.base + max(0, -self.shift))

    def __call__(self, idx: int) -> int:
        """Apply the swizzle to an index."""
        # Create mask for 'bits' number of bits at position 'base'
        mask = ((1 << self.bits) - 1) << self.base

        if self.shift >= 0:
            # Positive shift: XOR higher bits into lower bits
            # Extract bits from [base+shift, base+shift+bits), XOR into [base, base+bits)
            return idx ^ ((idx >> self.shift) & mask)
        else:
            # Negative shift: XOR lower bits into higher bits
            # Extract bits from [base, base+bits), shift left, XOR into higher position
            return idx ^ ((idx & mask) << (-self.shift))


def _popcount(x: int) -> int:
    """Number of set bits in ``x`` (matches C++ ``std::popcount``)."""
    return bin(x).count("1")


def _countr_zero(x: int) -> int:
    """Number of trailing zero bits in ``x`` (matches C++ ``std::countr_zero``).

    Uses the standard ``x & -x`` trick: two's-complement negation isolates
    the lowest set bit, and ``bit_length() - 1`` reads off its position.
    Caller must guarantee ``x != 0``.
    """
    return (x & -x).bit_length() - 1


def make_swizzle(Y: int, Z: int):
    """Create a Swizzle from Y and Z bit positions.

    Given bit masks Y and Z indicating which bits interact, construct
    the Swizzle(bits, base, shift) that performs the corresponding XOR.

    Matches CuTe C++ make_swizzle<Y,Z>() in swizzle.hpp.

    Args:
        Y: Bit mask for the Y (source) bits
        Z: Bit mask for the Z (destination) bits

    Returns:
        A Swizzle, or None if both masks are zero (identity).
    """
    num_bits = _popcount(Y)
    if num_bits != _popcount(Z):
        raise LayoutError(
            f"make_swizzle: bit count mismatch: popcount({Y:#b})={num_bits} "
            f"vs popcount({Z:#b})={_popcount(Z)}"
        )
    if num_bits == 0:
        return None  # Identity swizzle
    tz_y = _countr_zero(Y)
    tz_z = _countr_zero(Z)
    base = min(tz_y, tz_z)
    shift = tz_y - tz_z
    if abs(shift) < num_bits:
        raise LayoutError(
            f"make_swizzle: masks overlap for popcount {num_bits}: Y={Y:#b}, Z={Z:#b}"
        )
    swizzle = Swizzle(num_bits, base, shift)
    if (Y | Z) != (swizzle.yyy_msk | swizzle.zzz_msk):
        raise LayoutError(
            "make_swizzle: masks are not a canonical CuTe swizzle: "
            f"Y={Y:#b}, Z={Z:#b}, candidate={swizzle!r}"
        )
    return swizzle
