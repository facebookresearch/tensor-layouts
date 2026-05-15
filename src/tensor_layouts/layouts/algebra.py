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

"""The CuTe layout algebra: composition, division, product, inverses, and
their structural support functions.

Layered above ``core.py`` (Layout, Swizzle, tuple operations) and
``expr.py`` (ComposedLayout, the LayoutExpr predicates and coercers,
swizzle splitters). Everything in this
module assumes both are in scope.
"""

from __future__ import annotations

import math
from typing import Any

from .core import *  # noqa: F401, F403
from .expr import *  # noqa: F401, F403
from .expr import (
    _NO_FORWARD,
    _forward_layout_domain,
    _is_swizzle_inner_composed,
    _reject_swizzle_inner_composed,
)


__all__ = [
    "size",
    "cosize",
    "rank",
    "depth",
    "mode",
    "iter_layout",
    "append",
    "prepend",
    "replace",
    "group",
    "flatten",
    "unflatten",
    "sort",
    "coalesce",
    "complement",
    "right_inverse",
    "left_inverse",
    "nullspace",
    "max_common_layout",
    "max_common_vector",
    "slice_and_offset",
    "idx2crd",
    "crd2flat",
    "crd2offset",
    "crd2idx",
    "crd2crd",
    "slice_modes",
    "dice_modes",
    "safe_div",
    "shape_div",
    "shape_mod",
    "upcast",
    "downcast",
    "compose",
    "logical_divide",
    "zipped_divide",
    "tiled_divide",
    "flat_divide",
    "logical_product",
    "blocked_product",
    "raked_product",
    "flat_product",
    "zipped_product",
    "tiled_product",
    "hier_unzip",
]


# =============================================================================
# Query functions: size, rank, depth, mode
# =============================================================================
#
# These functions query properties of shapes and layouts:
#   size  -- total number of elements (product of all shape elements)
#   cosize -- memory span (max offset + 1, the size of the codomain)
#   rank  -- number of top-level modes (dimensions)
#   depth -- nesting depth of the shape hierarchy
#   mode  -- extract a single mode (dimension) from a shape or layout
#


def size(obj: Any) -> int:
    """Returns the logical number of elements (product of shape)."""
    if is_layout(obj):
        return size(obj.shape)
    if hasattr(obj, "layout"):  # Tensor or any layout-backed object
        return size(obj.layout)
    if is_tuple(obj) or is_int(obj):
        return fold(obj, 1, lambda acc, x: acc * x)
    raise TypeError(f"Cannot calculate size of {type(obj).__name__}")


def cosize(obj: "LayoutExpr") -> int:
    """Returns the codomain size: max(L(i) for i in [0, size(L))) + 1.

    For affine ``Layout`` this is computed in O(1) via the closed-form
    ``1 + sum_i (s_i - 1) * |d_i|`` (the affine span). For
    ``ComposedLayout`` there is no closed form -- the outer slot can be a
    Swizzle, a non-bijective Layout, or another ComposedLayout that
    permutes or rescales the inner's image -- so cosize must enumerate
    the full domain. That makes cosize on a ComposedLayout **O(size(L))**
    rather than O(1).

    Why enumeration is the only correct rule: see
    ``bug-reports/cute_cosize/cute_cosize_violation.cpp``. CuTe C++'s
    ``cosize(ComposedLayout) = cosize(layout_b())`` is wrong (it ignores
    the outer and the offset). The documented ``L(size-1) + 1`` is also
    wrong (e.g. F7 visits values past ``L(size-1)``). The differential
    survey shows ``max(L(i)) + 1`` is the only definition that matches
    the actual codomain extent for every form we tested.
    """
    if hasattr(obj, "layout") and not is_layout(obj):
        return cosize(obj.layout)
    if isinstance(obj, ComposedLayout):
        # O(n) enumeration -- no closed form for non-affine layouts.
        # Result is memoized on the instance because the layout is frozen
        # (its (outer, inner, offset) tuple is immutable, so cosize never
        # changes after construction).
        if obj._cached_cosize is not None:
            return obj._cached_cosize
        n = size(obj)
        v = 0 if n == 0 else max(obj(i) for i in range(n)) + 1
        object.__setattr__(obj, "_cached_cosize", v)
        return v
    # Layout with embedded swizzle: the swizzle is a bit-permutation that
    # can map an affine offset to a value above the affine max when the
    # affine image is not a full power-of-2 range. Example: image of
    # Layout(5, 1) is [0, 5); Sw o Layout(5, 1) for Swizzle(2, 0, 2) hits
    # 5 because XOR flips bit 2 above the affine max. Enumerate to capture
    # this -- mirrors the ComposedLayout fix in 5fbd19f for the embedded
    # form. For the common power-of-2 case both formulas agree.
    # Path X: Layout is purely affine; the embedded-swizzle cache branch
    # is gone. cosize is closed-form O(1) for affine layouts.
    if is_int(obj.shape):
        return _affine_max_offset(obj.shape, obj.stride) + 1
    if len(obj.shape) == 0:
        return 1
    return _affine_max_offset(obj.shape, obj.stride) + 1


def _affine_max_offset(shape: Any, stride: Any) -> int:
    """Maximum linear offset reachable by an affine (shape, stride) pair.

    Computes ``sum_i (s_i - 1) * |d_i|`` recursively across nested modes.
    Used by ``cosize`` for affine ``Layout``; ``cosize == max_offset + 1``.
    Strides are taken in absolute value so the result reports the affine
    span regardless of stride sign.
    """
    if is_tuple(shape):
        return sum(_affine_max_offset(s, d) for s, d in zip(shape, stride))
    return (shape - 1) * abs(stride)


def rank(obj: Any) -> int:
    if hasattr(obj, "layout") and not is_layout(obj):
        return rank(obj.layout)
    if is_tuple(obj):
        return len(obj)
    if is_layout(obj):
        if is_int(obj.shape):
            return 1
        return len(obj.shape)
    if is_int(obj):
        return 0
    raise TypeError(f"Cannot calculate rank of {type(obj).__name__}")


def depth(obj: Any) -> int:
    """Calculate nesting depth of a shape/layout.

    - int has depth 0
    - tuple has depth 1 + max depth of its elements
    - Layout delegates to its shape
    """
    if is_layout(obj):
        return depth(obj.shape)
    if hasattr(obj, "layout"):
        return depth(obj.layout)
    if is_int(obj):
        return 0
    if is_tuple(obj):
        if not obj:
            return 0
        return 1 + max((depth(elem) for elem in obj), default=0)
    raise TypeError(f"Cannot calculate depth of {type(obj).__name__}")


def mode(obj: Any, idx):
    if hasattr(obj, "layout") and not is_layout(obj):
        return mode(obj.layout, idx)
    if is_tuple(obj):
        if not obj:
            return ()
        return obj[idx]
    if isinstance(obj, ComposedLayout):
        if isinstance(obj.inner, Swizzle):
            if idx != 0:
                raise IndexError(f"Index {idx} out of range for swizzle-inner ComposedLayout")
            return obj
        return ComposedLayout(obj.outer, mode(obj.inner, idx), offset=obj.offset)
    if isinstance(obj, Layout):
        if is_int(obj.shape):
            if idx != 0:
                raise IndexError(f"Index {idx} out of range for scalar layout")
            return obj
        return Layout(obj.shape[idx], obj.stride[idx])
    if is_int(obj):
        if idx != 0:
            raise IndexError(f"Index {idx} out of range for scalar")
        return obj
    raise TypeError(f"Cannot get mode of {type(obj).__name__}")


# =============================================================================
# Iteration
# =============================================================================
#
# iter_layout yields (coordinate, offset) pairs for every element in a
# layout's domain, in colexicographic (column-major) order.  This is the
# most natural traversal: the flat index runs from 0 to size(layout) - 1,
# and coordinates are computed via idx2crd.
#


def iter_layout(layout: "LayoutExpr"):
    """Yield (coordinate, offset) pairs for every element in the layout.

    Iterates in colexicographic order (flat index 0, 1, 2, ...).

    Examples:
        list(iter_layout(Layout(4, 1)))
        # [(0, 0), (1, 1), (2, 2), (3, 3)]

        list(iter_layout(Layout((2, 3), (1, 2))))
        # [((0, 0), 0), ((1, 0), 1), ((0, 1), 2), ((1, 1), 3), ((0, 2), 4), ((1, 2), 5)]
    """
    for i in range(size(layout)):
        yield (idx2crd(i, layout.shape), layout(i))


# =============================================================================
# Layout manipulation: append, prepend, replace, group, flatten, sort
# =============================================================================
#
# These functions restructure a layout's modes without changing the underlying
# mapping. Flatten removes hierarchy, sort reorders by stride, group nests
# adjacent modes. They are the structural building blocks for composition
# and coalescing.
#


def append(a: "LayoutExpr", b: "Layout") -> "LayoutExpr":
    """Appends layout b as a new mode at the end of layout a.

    append(3:1, 4:3) -> (3,4):(1,3)
    append((3,4):(1,3), (3,4):(1,3)) -> (3,4,(3,4)):(1,3,(1,3))
    """
    forwarded = _forward_layout_domain(a, lambda inner: append(inner, b))
    if forwarded is not _NO_FORWARD:
        return forwarded
    return Layout(as_tuple(a.shape) + (b.shape,), as_tuple(a.stride) + (b.stride,))


def prepend(a: "LayoutExpr", b: "Layout") -> "LayoutExpr":
    """Prepends layout b as a new mode at the beginning of layout a.

    prepend(3:1, 4:3) -> (4,3):(3,1)
    """
    forwarded = _forward_layout_domain(a, lambda inner: prepend(inner, b))
    if forwarded is not _NO_FORWARD:
        return forwarded
    return Layout((b.shape,) + as_tuple(a.shape), (b.stride,) + as_tuple(a.stride))


def replace(layout: "LayoutExpr", idx: int, new_layout: "Layout") -> "LayoutExpr":
    """Replaces the mode at index idx with new_layout.

    replace((3,4,(3,4)):(1,3,(1,3)), 2, 4:3) -> (3,4,4):(1,3,3)
    """
    forwarded = _forward_layout_domain(layout, lambda inner: replace(inner, idx, new_layout))
    if forwarded is not _NO_FORWARD:
        return forwarded
    shapes = as_list(layout.shape)
    strides = as_list(layout.stride)

    shapes[idx] = new_layout.shape
    strides[idx] = new_layout.stride

    return Layout(tuple(shapes), tuple(strides))


def group(layout: "LayoutExpr", start: int, end: int) -> "LayoutExpr":
    """Groups modes from index start to end (exclusive) into a nested tuple.

    group((2,3,5,7):(1,2,6,30), 0, 2) -> ((2,3),5,7):((1,2),6,30)
    group(((2,3),5,7):((1,2),6,30), 1, 3) -> ((2,3),(5,7)):((1,2),(6,30))
    """
    forwarded = _forward_layout_domain(layout, lambda inner: group(inner, start, end))
    if forwarded is not _NO_FORWARD:
        return forwarded
    r = rank(layout)
    if start < 0 or end > r or start >= end:
        raise LayoutError(f"Invalid group range [{start}, {end}) for layout of rank {r}")

    shapes = as_list(layout.shape)
    strides = as_list(layout.stride)

    # Extract the modes to group
    grouped_shape = tuple(shapes[start:end])
    grouped_stride = tuple(strides[start:end])

    # Build new layout: [0:start] + [grouped] + [end:]
    new_shapes = shapes[:start] + [grouped_shape] + shapes[end:]
    new_strides = strides[:start] + [grouped_stride] + strides[end:]

    return Layout(tuple(new_shapes), tuple(new_strides))


def flatten(obj: Any) -> Any:
    """Flattens a hierarchical layout into a rank-N flat layout."""

    def _flatten(s):
        if is_int(s):
            return (s,)
        flat = []
        for si in s:
            if is_tuple(si):
                s_rec = _flatten(si)
                flat.extend(s_rec)
            else:
                flat.append(si)
        return tuple(flat)

    if is_int(obj):
        return (obj,)
    if is_tuple(obj):
        return _flatten(obj)
    if hasattr(obj, "layout") and not is_layout(obj):
        return flatten(obj.layout)
    elif isinstance(obj, ComposedLayout):
        if isinstance(obj.inner, Swizzle):
            return obj
        return ComposedLayout(obj.outer, flatten(obj.inner), offset=obj.offset)
    elif isinstance(obj, Layout):
        # Path X: Layout is purely affine; the embedded-swizzle branch is
        # unreachable for in-tree callers and is removed in C3.
        flat_shape = _flatten(obj.shape)
        flat_stride = _flatten(obj.stride)
        return Layout(as_shape(list(flat_shape)), as_shape(list(flat_stride)))
    else:
        raise TypeError(f"Cannot flatten object of type {type(obj).__name__}")


def unflatten(obj, target_profile):
    """Unflatten a flat object to match a target's hierarchical structure.

    This is the inverse of flatten: it reshapes a flat tuple or layout into
    a hierarchical structure matching target_profile.

    Args:
        obj: A flat tuple or Layout
        target_profile: A (possibly nested) tuple or Layout defining the
                        desired structure

    Returns:
        A tuple or Layout with the structure of target_profile

    Examples:
        unflatten((1,2,3,4,5), ((0,0), (0,0,0))) -> ((1,2), (3,4,5))
        unflatten(Layout((2,3,5,7), (1,2,6,30)), (4, 3))
            -> Layout((2,3), (5,7)), ((1,2), (6,30)))

    Preconditions:
        flatten(obj) == obj  (obj must already be flat)
        rank(flatten(target_profile)) == rank(obj)
    """

    def _unflatten_helper(flat_tuple, profile):
        """Consume elements from flat_tuple to match profile's structure."""
        if is_tuple(profile):
            result = []
            remaining = list(flat_tuple)
            for elem in profile:
                sub_result, remaining = _unflatten_helper(remaining, elem)
                result.append(sub_result)
            return tuple(result), remaining
        else:
            return flat_tuple[0], flat_tuple[1:]

    if isinstance(target_profile, Layout):
        target_profile = target_profile.shape

    if isinstance(obj, Layout):
        new_shape, remaining_s = _unflatten_helper(tuple(obj.shape), target_profile)
        new_stride, remaining_d = _unflatten_helper(tuple(obj.stride), target_profile)
        if len(remaining_s) != 0:
            raise LayoutError(f"Rank mismatch: leftover shape elements {remaining_s}")
        if len(remaining_d) != 0:
            raise LayoutError(f"Rank mismatch: leftover stride elements {remaining_d}")
        return Layout(new_shape, new_stride)

    if is_tuple(obj):
        result, remaining = _unflatten_helper(tuple(obj), target_profile)
        if len(remaining) != 0:
            raise LayoutError(f"Rank mismatch: leftover elements {remaining}")
        return result

    raise TypeError(f"Cannot unflatten object of type {type(obj).__name__}")


def sort(obj: "LayoutExpr") -> "LayoutExpr":
    """Returns a new Layout with modes sorted by stride."""
    forwarded = _forward_layout_domain(obj, sort)
    if forwarded is not _NO_FORWARD:
        return forwarded
    if rank(obj) <= 1:
        return obj

    flat = flatten(obj)
    combined = list(zip(flat.stride, flat.shape))
    combined.sort()
    new_stride = tuple(item[0] for item in combined)
    new_shape = tuple(item[1] for item in combined)

    return Layout(new_shape, new_stride)


# =============================================================================
# Coalescing
# =============================================================================
#
# Coalescing merges contiguous modes. Two adjacent modes are contiguous when
# stride[i+1] == shape[i] * stride[i], meaning they cover a contiguous range
# of offsets. Merging them into one larger mode simplifies the layout without
# changing the mapping. Coalescing is the canonical simplification: it is
# always safe and always preserves semantics.
#


def coalesce(obj: "LayoutExpr", profile: Any = None) -> "LayoutExpr":
    """Returns a new Layout where contiguous dimensions are merged.

    Args:
        obj: The layout to coalesce
        profile: Optional shape profile that defines mode boundaries.
                 When provided, coalescing happens within each mode independently,
                 preserving the hierarchical structure defined by the profile.

    Examples:
        coalesce(Layout((2,4), (1,2))) -> Layout(8, 1)
        coalesce(Layout((2,4,2,2), (1,2,8,16)), (4,4)) -> Layout((8,4), (1,8))
    """
    forwarded = _forward_layout_domain(obj, lambda inner: coalesce(inner, profile))
    if forwarded is not _NO_FORWARD:
        return forwarded
    if _is_swizzle_inner_composed(obj):
        # Inverse-form ComposedLayout(Layout, offset, Swizzle): the inner
        # Swizzle has no multi-mode structure to merge, and the domain is
        # rank-1 with no size-1 modes to filter. Coalescing is a no-op.
        # CuTe C++ refuses this form because its template delegates to the
        # inner; we can answer it directly because we know the answer is
        # the input.
        return obj
    if rank(obj) == 0:
        if is_int(obj.shape):
            return Layout(1, 0) if obj.shape == 1 else obj
        return Layout()

    if profile is None:
        return _coalesce_flat(obj)

    return _coalesce_by_mode(obj, profile if is_tuple(profile) else (profile,))


def _coalesce_flat(obj: "Layout") -> "Layout":
    """Coalesce a layout by filtering trivial modes and merging contiguous ones."""
    flat = flatten(obj)

    if is_int(flat.shape):
        return Layout(1, 0) if flat.shape == 1 else flat

    shapes = list(flat.shape)
    strides = list(flat.stride)

    # Filter and merge in one pass: skip size-1 modes, merge contiguous ones
    merged_s, merged_d = [], []
    for s, d in zip(shapes, strides):
        if s == 1:
            continue
        if merged_s and d == merged_s[-1] * merged_d[-1]:
            merged_s[-1] *= s
        else:
            merged_s.append(s)
            merged_d.append(d)

    if not merged_s:
        return Layout(1, 0)
    return Layout(as_shape(merged_s), as_shape(merged_d))


def _coalesce_by_mode(layout: "Layout", profile: tuple) -> "Layout":
    """Coalesce a layout respecting mode boundaries defined by profile.

    If profile contains None, coalesce each original mode independently.
    Otherwise, partition the flattened layout by profile sizes and coalesce each partition.
    """
    profile_list = list(profile)

    # None-profile: coalesce each original mode independently
    if any(p is None for p in profile_list):
        result_s, result_d = [], []
        for i in range(len(profile_list)):
            if i >= rank(layout):
                result_s.append(1)
                result_d.append(0)
            else:
                coalesced = _coalesce_flat(Layout(mode(layout.shape, i), mode(layout.stride, i)))
                result_s.append(coalesced.shape)
                result_d.append(coalesced.stride)
        return Layout(as_shape(result_s), as_shape(result_d))

    # Int-profile: partition flattened layout by profile sizes
    flat = flatten(layout)
    flat_shapes, flat_strides = list(flat.shape), list(flat.stride)

    # Flatten profile to get target sizes
    target_sizes = list(flatten(profile))

    result_s, result_d = [], []
    idx = 0

    for target_size in target_sizes:
        # Consume modes until we reach target_size
        mode_s, mode_d = [], []
        accumulated = 1
        while accumulated < target_size and idx < len(flat_shapes):
            mode_s.append(flat_shapes[idx])
            mode_d.append(flat_strides[idx])
            accumulated *= flat_shapes[idx]
            idx += 1

        if not mode_s:
            result_s.append(1)
            result_d.append(0)
            continue

        # Sort by stride, filter size-1, merge contiguous (with nonzero stride check)
        paired = sorted(zip(mode_d, mode_s))
        merged_s, merged_d = [], []
        for d, s in paired:
            if s == 1:
                continue
            if merged_s and merged_d[-1] != 0 and d == merged_s[-1] * merged_d[-1]:
                merged_s[-1] *= s
            else:
                merged_s.append(s)
                merged_d.append(d)

        if not merged_s:
            result_s.append(1)
            result_d.append(0)
        else:
            result_s.append(as_shape(merged_s))
            result_d.append(as_shape(merged_d))

    return Layout(tuple(result_s), tuple(result_d))


# =============================================================================
# Complement, inverse, and slice operations
# =============================================================================
#
# The complement of a layout fills in the gaps. If layout L visits offsets
# {0, 2, 4, 6}, then complement(L) visits {0, 1} (the offsets within each
# stride gap). Together, make_ayout(L, complement(L)) covers every offset
# exactly once. This is the key building block for logical_divide.
#
# The right-inverse R of L satisfies L(R(i)) = i: it "undoes" L.
# The left-inverse R satisfies R(L(i)) = i: it recovers coordinates from offsets.
#
# Slicing fixes some coordinates and returns a sublayout over the remaining
# free dimensions, much like NumPy's array[3, :, :] syntax.
#


def complement(layout: "Layout", cosize_bound: Any = None) -> "Layout":
    """Compute the complement of a layout: a layout that fills in the gaps.

    If L visits offsets {0, 2, 4, 6} within a range of 8, then complement(L, 8)
    visits the in-between offsets {0, 1} (stride-1 within each stride-2 gap).
    Together, Layout(L, complement(L)) covers every offset exactly once.

    Why "complement"?  Think of L as selecting a subset of [0, cosize).
    The complement fills "the rest" — not by set subtraction, but by filling
    the stride gaps.  The bundled Layout(L, complement(L)) is a bijection
    onto [0, cosize), with L controlling position within each gap, and
    complement(L) controlling which gap.

    This is the key building block for logical_divide: dividing a layout by a
    tiler T is equivalent to composing with Layout(T, complement(T)).

    The algorithm sorts the layout's modes by stride, then folds _step_mode
    over them.  Each step checks for a gap between the current frontier and
    the next mode's stride; gaps become output modes.  A final step fills
    from the last mode's cosize up to cosize_bound.

    Args:
        layout: The layout to compute complement for
        cosize_bound: The target cosize. Defaults to cosize(layout).

    Examples:
        complement(Layout(4, 2), 16) -> Layout((2, 2), (1, 8))
        complement(Layout(4, 1), 16) -> Layout(4, 4)
        complement(Layout((2, 2), (1, 4)), 16) -> Layout((2, 2), (2, 8))
    """

    def _step_mode(current_stride, stride, shape):
        """Emit a gap-fill if there's a gap before this mode, then advance
        past it.  Returns (gap_size, next_current_stride)."""
        gap_size = stride // current_stride if stride > current_stride else 1
        return gap_size, stride * shape

    if cosize_bound is None:
        cosize_bound = cosize(layout)
    elif is_layout(cosize_bound):
        cosize_bound = cosize_bound.shape

    # ComposedLayout(outer, inner): the outer is an involution / permutation
    # in CuTe; only the inner controls the codomain image, so the complement
    # is the inner's complement. Matches CuTe C++ layout_composed.hpp:395-409.
    if isinstance(layout, ComposedLayout):
        _reject_swizzle_inner_composed(layout, "complement")
        return complement(layout.inner, cosize_bound)

    # Short-circuit unit layout AND zero-sized layouts (no elements to span)
    if is_empty(layout) or size(layout) == 0:
        return Layout(cosize_bound, 1) if cosize_bound > 1 else Layout()

    # Flatten, filter size-1 and stride-0 dims, sort by stride
    flat = flatten(layout)

    # Convert to lists for uniform processing (as_tuple promotes scalars).
    flat_shapes = as_list(flat.shape)
    flat_strides = as_list(flat.stride)

    modes = sorted(((d, s) for s, d in zip(flat_shapes, flat_strides) if s != 1 and d != 0))

    # Fold _step_mode over sorted modes, collecting gap-fills
    result_shapes = []
    result_strides = []
    current_stride = 1

    for stride, shape in modes:
        # CuTe/pycute asserts current_stride <= stride * shape (injectivity).
        # Negative strides or zero-sized shapes violate this invariant.
        if stride < 0:
            raise LayoutError(f"complement: negative stride {stride} is not supported")
        if shape == 0:
            raise LayoutError("complement: zero-sized shape is not supported")
        gap_size, next_stride = _step_mode(current_stride, stride, shape)
        if gap_size > 1:
            result_shapes.append(gap_size)
            result_strides.append(current_stride)
        current_stride = next_stride

    # Fill remaining space up to cosize_bound. Shape bounds stay hierarchical
    # instead of collapsing eagerly to size(...), matching CuTe C++.
    if is_tuple(cosize_bound):
        remaining = _coalesce_shape(_shape_ceil_div(cosize_bound, current_stride))
        remaining_stride = elem_scale(current_stride, compute_col_major_strides(remaining))
    else:
        remaining = _ceil_div(cosize_bound, current_stride)
        remaining_stride = current_stride

    # Always append (even if shape-1) to match CuTe/pycute; coalesce cleans up.
    result_shapes.append(remaining)
    result_strides.append(remaining_stride)

    # Coalesce the result (merges contiguous modes, removes size-1 modes)
    return coalesce(Layout(as_shape(result_shapes), as_shape(result_strides)))


def right_inverse(layout: Any) -> Any:
    """Compute the right-inverse of a layout.

    For a layout L, the right-inverse R satisfies: L(R(i)) == i
    for all i in range(size(R)).

    The algorithm sorts modes by stride and folds _step_mode over them,
    greedily building the longest contiguous prefix.  Each step checks
    whether the mode's stride matches the running frontier; if so the
    mode contributes to the inverse, otherwise iteration stops.

    Examples:
        right_inverse(Layout(4, 1)) -> Layout(4, 1)
        right_inverse(Layout(4, 2)) -> Layout((2, 2), (0, 1))
        right_inverse(Layout((8, 4), (1, 8))) -> Layout((8, 4), (1, 8))
        right_inverse(Layout((8, 4), (4, 1))) -> Layout((4, 8), (1, 4))
    """

    def _step_mode(current_idx, stride, shape):
        """Check if a mode is contiguous with the frontier.
        Returns (contiguous, next_current_idx)."""
        if shape == 1:
            return True, current_idx
        if current_idx != stride:
            return False, current_idx
        return True, shape * stride

    if layout is None:
        return None
    if isinstance(layout, Swizzle):
        return layout
    if isinstance(layout, ComposedLayout):
        if isinstance(layout.outer, Swizzle) and layout.offset == 0:
            return compose(right_inverse(layout.inner), layout.outer)
        return ComposedLayout(
            right_inverse(layout.inner),
            right_inverse(layout.outer),
            offset=-layout.offset,
        )
    # Path X: Layout is purely affine; the embedded-swizzle arm is gone.
    if isinstance(layout, int):
        return Layout(layout)

    flat = flatten(layout)

    # Convert to lists for uniform processing (as_tuple promotes scalars).
    flat_shapes = as_list(flat.shape)
    flat_strides = as_list(flat.stride)

    # Compute prefix products for inverse strides
    pp = prefix_product(flat.shape)
    if is_int(pp):
        pp = [pp]
    else:
        pp = list(pp)

    # Sort (stride, shape, prefix_prod) triples by stride
    triples = sorted(zip(flat_strides, flat_shapes, pp))

    result_shape = []
    result_stride = []
    current_idx = 1

    for stride, shape, rstride in triples:
        contiguous, current_idx = _step_mode(current_idx, stride, shape)
        if not contiguous:
            continue
        if shape != 1:
            result_shape.append(shape)
            result_stride.append(rstride)

    if not result_shape:
        return Layout(1, 0)

    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


def left_inverse(layout: Any) -> Any:
    """Compute the left-inverse of a layout.

    For an injective layout L, the left-inverse R satisfies:
        R(L(i)) == i for all i in range(size(L))

    For a general layout L, the weaker property holds:
        L(R(L(i))) == L(i) for all i in range(size(L))

    Algorithm matches CuTe C++ (layout.hpp:1324):
      1. Coalesce the layout
      2. Compute prefix product of shapes
      3. Sort modes by stride (ascending)
      4. Build inverse by filling gaps between strides

    Examples:
        left_inverse(Layout(4, 1)) -> Layout(4, 1)
        left_inverse(Layout(4, 2)) -> Layout((2, 4), (0, 1))
        left_inverse(Layout((8, 4), (1, 8))) -> Layout(32, 1)
        left_inverse(Layout((4, 8), (1, 5))) -> Layout((5, 8), (1, 4))
    """
    if layout is None:
        return None
    if isinstance(layout, Swizzle):
        return layout
    if isinstance(layout, ComposedLayout):
        if isinstance(layout.outer, Swizzle) and layout.offset == 0:
            return compose(left_inverse(layout.inner), layout.outer)
        return ComposedLayout(
            left_inverse(layout.inner),
            left_inverse(layout.outer),
            offset=-layout.offset,
        )
    # Path X: Layout is purely affine; the embedded-swizzle arm is gone.
    if isinstance(layout, int):
        return Layout(layout)

    flat = coalesce(layout)

    # Convert to lists for uniform processing (as_tuple promotes scalars).
    flat_shapes = as_list(flat.shape)
    flat_strides = as_list(flat.stride)

    R = len(flat_shapes)

    # Prefix product of shapes: [1, S0, S0*S1, ...]
    preprod = [1]
    for s in flat_shapes:
        preprod.append(preprod[-1] * s)

    # Sort mode indices by stride (ascending), filtering stride-0
    nonzero_indices = [(flat_strides[i], i) for i in range(R) if flat_strides[i] != 0]
    nonzero_indices.sort()  # sort by stride

    if not nonzero_indices:
        # All strides are 0: trivial inverse
        return Layout(size(layout), 0)

    # Build the inverse: CuTe C++ layout.hpp:1340-1360
    # For each mode (sorted by stride, skipping stride-0):
    #   new_shape = istride / size(result_shape_so_far)
    #   new_stride = prefix_product[original_mode_index]
    # Then append the shape of the last sorted mode.
    result_shapes = []
    result_strides = [0]  # initial stride-0 sentinel (matches C++ tuple<_0>)
    result_size = 1  # product of result_shapes so far

    for stride_val, idx in nonzero_indices:
        new_shape = stride_val // result_size
        result_shapes.append(new_shape)
        result_strides.append(preprod[idx])
        result_size *= new_shape

    # Append the shape of the last sorted mode
    _, last_idx = nonzero_indices[-1]
    result_shapes.append(flat_shapes[last_idx])

    return coalesce(Layout(as_shape(result_shapes), as_shape(result_strides)))


def nullspace(layout: "Layout") -> "Layout":
    """Compute the nullspace (kernel) of a layout.

    The nullspace contains all coordinates that map to offset 0. These are
    the stride-0 modes: dimensions along which movement in the logical domain
    produces no movement in memory (broadcast dimensions).

    The result is a layout whose domain enumerates all elements that map to 0:
        layout(nullspace(layout)(i)) == 0  for all i in range(size(result))

    The size of the nullspace is  size(layout) / size(filter(layout)),
    i.e., the total domain divided by the "effective" (non-broadcast) domain.

    Algorithm: flatten the layout, compute column-major strides for the full
    flat shape, then select the shapes and strides at stride-0 positions.
    The column-major strides ensure that nullspace coordinates, when mapped
    back through the layout via idx2crd, land on the broadcast dimensions.

    Examples:
        nullspace(Layout((2,2,2), (0,0,0))) -> (2,2,2):(1,2,4)
        nullspace(Layout((2,2,2), (1,0,2))) -> 2:2
        nullspace(Layout((4,8), (1,4)))      -> 1:0
    """
    flat = flatten(layout)

    # Column-major strides for the full flat shape: these are the strides
    # of a compact column-major layout with the same shape.
    col_major_strides = prefix_product(flat.shape)

    # Normalize to tuples so zip works on scalar (rank-0) layouts
    flat_shapes = as_tuple(flat.shape)
    flat_strides = as_tuple(flat.stride)
    col_strides = as_tuple(col_major_strides)

    # Select shapes and strides at stride-0 positions
    zero_shapes = []
    zero_strides = []
    for s, d, r in zip(flat_shapes, flat_strides, col_strides):
        if d == 0 and s != 1:
            zero_shapes.append(s)
            zero_strides.append(r)

    if not zero_shapes:
        return Layout(1, 0)

    return Layout(as_shape(zero_shapes), as_shape(zero_strides))


def max_common_layout(layout_a: "LayoutExpr", layout_b: "LayoutExpr") -> "LayoutExpr":
    """Return a layout expression for the maximum contiguous elements common to both.

    Two layouts "logically correspond" when indexing through one produces the
    same offsets as indexing through the other. max_common_layout finds the
    longest contiguous prefix where a(R(i)) == i and b(R(i)) == i.

    Algorithm: compose(a, right_inverse(b)), coalesce, then check if the
    leading mode has stride 1. If so, compose inv_b with that leading mode
    to get the common layout. Otherwise, return Layout(1, 0).

    Args:
        layout_a: First layout
        layout_b: Second layout

    Returns:
        A layout expression R such that a(R(i)) == i and b(R(i)) == i for all
        i < size(R). Non-affine cases may return an exact ComposedLayout.

    Examples:
        max_common_layout(Layout(8, 1), Layout(8, 1))       -> 8:1
        max_common_layout(Layout((4,2), (2,1)), Layout(8,1)) -> 1:0
        max_common_layout(Layout(8, 1), Layout((4,2), (1,4))) -> 4:1
    """
    if (
        split_outer_swizzle(layout_a) is not None
        or split_outer_swizzle(layout_b) is not None
    ):
        vec = max_common_vector(layout_a, layout_b)
        inv_b = right_inverse(layout_b)
        return coalesce(compose(inv_b, Layout(vec, 1)))

    if isinstance(layout_a, ComposedLayout) or isinstance(layout_b, ComposedLayout):
        inv_b = right_inverse(layout_b)
        common_size = 0
        for i in range(size(inv_b)):
            coord = inv_b(i)
            if layout_a(coord) != i or layout_b(coord) != i:
                break
            common_size += 1
        if common_size == 0:
            return Layout(0, 1)
        return coalesce(compose(inv_b, Layout(common_size, 1)))

    layout_a = as_affine_layout(layout_a)
    layout_b = as_affine_layout(layout_b)
    inv_b = right_inverse(layout_b)
    common = coalesce(compose(layout_a, inv_b))

    # Check if the leading mode has stride 1
    flat_common = flatten(common)
    flat_shape = flat_common.shape
    flat_stride = flat_common.stride

    # Handle scalar layouts (rank 0) - they are effectively rank 1
    if is_int(flat_shape):
        if flat_stride == 1:
            return coalesce(compose(inv_b, Layout(flat_shape, 1)))
        else:
            return Layout(1, 0)

    if rank(flat_common) > 0 and flat_stride[0] == 1:
        leading_shape = flat_shape[0]
        return coalesce(compose(inv_b, Layout(leading_shape, 1)))
    else:
        return Layout(1, 0)


def max_common_vector(layout_a: "LayoutExpr", layout_b: "LayoutExpr") -> int:
    """Return the number of contiguous elements that logically correspond in both layouts.

    This is the size of max_common_layout(a, b) — the length of the longest
    contiguous prefix where both layouts agree.

    Args:
        layout_a: First layout
        layout_b: Second layout

    Returns:
        An integer N >= 0 such that for all 0 <= i < N, both layouts map
        element i to offset i.

    Examples:
        max_common_vector(Layout(8, 1), Layout(8, 1))        -> 8
        max_common_vector(Layout((4,2), (2,1)), Layout(8,1)) -> 1
        max_common_vector(Layout(8, 1), Layout((4,2), (1,4))) -> 4
    """
    split_a = split_outer_swizzle(layout_a)
    split_b = split_outer_swizzle(layout_b)
    if split_a is not None:
        swizzle_a, inner_a = split_a
        if split_b is not None:
            swizzle_b, inner_b = split_b
            vec = max_common_vector(inner_a, inner_b)
            if swizzle_a == swizzle_b:
                return vec
            return min(vec, 1 << swizzle_a.base, 1 << swizzle_b.base)
        return min(max_common_vector(inner_a, layout_b), 1 << swizzle_a.base)
    if split_b is not None:
        swizzle_b, inner_b = split_b
        return min(max_common_vector(layout_a, inner_b), 1 << swizzle_b.base)
    return size(max_common_layout(layout_a, layout_b))


def _swizzle_bit_decomposition(swizzle: "Swizzle", yz_pre: int) -> "Layout":
    """Affine layout that encodes the swizzle's per-bit effect on its YZ window
    given a fixed YZ-portion of the offset.

    Builds the equivalent of CuTe C++'s "swizzle_layout" (cute/swizzle_layout.hpp
    lines 289-290): one size-2 mode per swizzlable bit position, with a stride
    equal to the swizzle's output difference when only that bit toggles. Bits
    outside the YZ window (the M low bits and the |S|-B gap between Y and Z)
    are emitted as single contiguous modes with their natural strides.

    The result has total size ``1 << (M + |S| + B)`` and is suitable as the
    LHS of a ``compose`` to project a sliced inner layout into the swizzle's
    address space. Pre-condition: the caller has already verified that the
    sliced inner's YZ-projection misses either Y or Z (so the swizzle is
    affine on the relevant subspace).
    """
    M = swizzle.base
    B = swizzle.bits
    abs_S = abs(swizzle.shift)
    base = swizzle(yz_pre)

    def _bit_stride(p: int) -> int:
        return swizzle(yz_pre + (1 << p)) - base

    shapes: list = []
    strides: list = []
    if M > 0:
        shapes.append(1 << M)
        strides.append(1)
    # Y bits (low side of the swizzle XOR)
    for i in range(B):
        shapes.append(2)
        strides.append(_bit_stride(M + i))
    if abs_S - B > 0:
        shapes.append(1 << (abs_S - B))
        strides.append(1 << (M + B))
    # Z bits (high side; stride encodes the Y-bit alias when relevant)
    for i in range(B):
        shapes.append(2)
        strides.append(_bit_stride(M + abs_S + i))
    return Layout(tuple(shapes), tuple(strides))


def _try_decay_swizzle_composed(composed: "ComposedLayout"):
    """If a ComposedLayout(Swizzle, affine_layout, offset) is reducible on
    its inner's image, decay it to a plain (Layout, offset) pair.

    Mirrors CuTe C++'s slice_and_offset decay path in cute/swizzle_layout.hpp
    (lines 263-294): if the inner's codomain doesn't hit BOTH Y and Z bits of
    the swizzle, the swizzle becomes affine on this subspace and we can build
    a per-bit affine encoding of the swizzle, then compose with the inner to
    fold the swizzle into normal strides plus a constant base offset. Returns
    None when the decay is unsafe (Y AND Z hit, or composition does not yield
    a stride pattern that reproduces the swizzled output) so the caller can
    keep the ComposedLayout wrapping.
    """
    if not isinstance(composed, ComposedLayout):
        return None
    swizzle = composed.outer
    inner = composed.inner
    if not isinstance(swizzle, Swizzle):
        return None
    if not isinstance(inner, Layout):
        return None  # only collapse when the inner is plain affine

    yz_mask = swizzle.yyy_msk | swizzle.zzz_msk
    yz_pre = composed.offset & yz_mask
    anti_yz_pre = composed.offset & ~yz_mask

    # Reducibility: OR the YZ-projection of the inner's image. If the swizzle
    # would flip any of those bits, both Y and Z are hit -> can't decay.
    n = size(inner)
    active_bits = 0
    for i in range(n):
        active_bits |= inner(i) & yz_mask
    if active_bits & (active_bits ^ swizzle(active_bits)) != 0:
        return None

    # Build the swizzle-as-affine-layout and compose with the inner. The
    # composition may fail when inner's strides don't divide cleanly into the
    # swizzle's bit-decomposition (e.g., non-power-of-2 strides over the YZ
    # window); treat that as a bail-out rather than a hard error.
    try:
        decomp = _swizzle_bit_decomposition(swizzle, yz_pre)
        decayed = _compose_layouts(decomp, inner)
    except (ValueError, TypeError):
        return None

    base_offset = swizzle(yz_pre) + anti_yz_pre

    # Sample-verify the result reproduces the swizzled output exactly. Cheap
    # at our typical sizes and protects against composition-edge-cases that
    # the bit-decomposition argument doesn't anticipate.
    for i in range(n):
        expected = swizzle(yz_pre + inner(i)) + anti_yz_pre
        actual = base_offset + decayed(i)
        if actual != expected:
            return None

    return (decayed, base_offset)


def slice_and_offset(crd, layout: "LayoutExpr"):
    """Slice a layout by a coordinate and return (sublayout, offset).

    Given a coordinate with None values marking sliced (free) dimensions
    and integer values marking fixed dimensions, returns:
    - sublayout: Layout over the free dimensions
    - offset: The linear offset from the fixed dimensions

    Args:
        crd: Coordinate tuple with None for sliced dims and ints for fixed dims
        layout: The layout to slice

    Returns:
        (sublayout, offset) tuple

    Examples:
        slice_and_offset((None, 3), Layout((4, 8), (1, 4)))
        -> (Layout((4,), (1,)), 12)  # sublayout over dim 0, offset = 3*4
    """
    if isinstance(layout, ComposedLayout):
        sublayout, offset = _slice_for_composition(crd, layout)
        # Try to decay the swizzled wrapper to a plain Layout when slicing has
        # restricted the inner's image enough that the swizzle is affine on it
        # (matches CuTe C++'s slice_and_offset decay; see _try_decay_*).
        # The full-slice case is intentionally kept composed by
        # _slice_for_composition and shouldn't be touched here.
        if isinstance(sublayout, ComposedLayout) and not coords_all_none(crd):
            decayed = _try_decay_swizzle_composed(sublayout)
            if decayed is not None:
                decayed_layout, base_offset = decayed
                return (decayed_layout, offset + base_offset)
        return (sublayout, offset)

    sliced_shape = slice_modes(crd, layout.shape)
    sliced_stride = slice_modes(crd, layout.stride)
    # When slicing drops some top-level modes, a single surviving hierarchical
    # mode can end up wrapped in a spurious outer tuple, e.g. ((3,2),).
    # Unwrap it so the result is (3,2) — but only when the single element is
    # itself a tuple (hierarchical); scalar modes like (4,) must stay wrapped.
    if len(sliced_shape) == 1 and is_tuple(sliced_shape[0]):
        sliced_shape = sliced_shape[0]
        sliced_stride = sliced_stride[0]
    sublayout = Layout(
        sliced_shape if sliced_shape else (),
        sliced_stride if sliced_stride else (),
    )
    offset = crd2offset(crd, layout.shape, layout.stride)

    # Path X: Layout is purely affine, so the legacy embedded-swizzle
    # Form-B promotion (slice contribution folded into a
    # ComposedLayout(Sw, sub_L, offset=delta)) is unreachable for in-tree
    # callers and is removed in C3.
    return (sublayout, offset)


def _slice_for_composition(crd, layout: "LayoutExpr"):
    """Slice a layout expression for use inside an outer composition.

    Returns (sublayout_expr, delta) such that the original sliced expression is
    equivalent to:

        delta + sublayout_expr(free_coord)

    for affine layouts, or to just ``sublayout_expr(free_coord)`` when the
    sliced contribution must remain inside a nonlinear inner expression.
    """
    if isinstance(layout, ComposedLayout):
        if isinstance(layout.inner, Swizzle):
            # Rank-1 integer domain — no further recursion possible. The caller
            # should slice on the outer if a sub-domain view is needed.
            if not coords_all_none(crd):
                raise UnsupportedComposedLayoutError(
                    "Slicing a ComposedLayout with a Swizzle in the inner slot is not supported"
                )
            return (layout, 0)
        inner_slice, delta = _slice_for_composition(crd, layout.inner)
        return (ComposedLayout(layout.outer, inner_slice, offset=layout.offset + delta), 0)

    sliced_shape = slice_modes(crd, layout.shape)
    sliced_stride = slice_modes(crd, layout.stride)
    if len(sliced_shape) == 1 and is_tuple(sliced_shape[0]):
        sliced_shape = sliced_shape[0]
        sliced_stride = sliced_stride[0]
    sublayout = Layout(
        sliced_shape if sliced_shape else (),
        sliced_stride if sliced_stride else (),
    )
    offset = crd2offset(crd, layout.shape, layout.stride)

    # Path X: Layout is purely affine; the legacy embedded-swizzle
    # Form-B promotion is unreachable for in-tree callers and is removed
    # in C3.
    return (sublayout, offset)


# =============================================================================
# Coordinate conversion: idx2crd, crd2flat, crd2offset, crd2idx, crd2crd
# =============================================================================
#
# These convert between the three coordinate representations in CuTe:
#   1D index   -- a single integer (the "flat" position in the domain)
#   nD coord   -- a tuple of per-mode coordinates, e.g. (row, col)
#   offset     -- the memory offset (what the layout computes)
#
# idx2crd:    1D index -> nD coordinate (decompose via shape)
# crd2flat:   nD coordinate -> 1D index (flatten via shape, inverse of idx2crd)
# crd2offset: nD coordinate -> offset (inner product with stride)
# crd2idx:    dispatches to crd2flat (2-arg) or crd2offset (3-arg), matching
#             C++ CuTe's overloaded crd2idx(coord, shape[, stride])
# crd2crd:    convert between two shapes' coordinate spaces
#


def idx2crd(coord: Any, shape: Any) -> Any:
    """Convert index into a hierarchical coordinate."""

    if isinstance(shape, Layout):
        shape = shape.shape

    if isinstance(shape, int):
        if isinstance(coord, int):
            return coord % shape
        return coord

    # Case: Input is a single integer index for this entire sub-hierarchy
    if isinstance(coord, int):
        res = []
        index = coord
        for s in shape:
            m_size = size(s)
            # Recurse: expand the index restricted to this mode's sub-shape
            res.append(idx2crd(index % m_size, s))
            index //= m_size
        return tuple(res)

    # Case: Input is a collection (Tuple/tuple)
    # We map the modes of the coordinate to the modes of the shape
    if is_tuple(coord):
        if len(coord) != len(shape):
            raise LayoutError(f"Coordinate rank {len(coord)} mismatch with Shape rank {len(shape)}")

        return zip_transform(coord, shape, idx2crd)

    raise TypeError(f"Cannot map {type(coord)} to shape {shape}")


def crd2flat(coord: Any, shape: Any = None) -> int:
    """Convert a hierarchical coordinate to a flat 1D index (inverse of idx2crd).

    Example: crd2flat((1, 1), (4, 4)) -> 5
    """

    if isinstance(shape, Layout):
        shape = shape.shape

    if isinstance(shape, int):
        if is_tuple(coord):
            raise LayoutError(f"Cannot map coordinate {coord} to scalar shape {shape}")
        return int(coord)

    if isinstance(coord, int):
        return coord

    if is_tuple(coord):
        if len(coord) != len(shape):
            raise LayoutError(f"Rank mismatch: coord {len(coord)} vs shape {len(shape)}")

        index = 0
        stride = 1
        for c, s in zip(coord, shape):
            tindex = crd2flat(c, s)
            index += tindex * stride
            stride *= size(s)
        return index

    raise TypeError(f"Unsupported coordinate type: {type(coord)}")


# crd2offset((1, 1), Layout((4,4),(1,100))) -> 101
def crd2offset(coord, shape, stride) -> int:
    """Convert coordinate to memory offset (inner product with stride).

    When coord is a 1D integer index and shape is a tuple, the index is
    decomposed across modes from left to right. Each mode (except the last)
    consumes its share via modular arithmetic. The last mode is NOT modded,
    allowing indices beyond the domain to extend through it. This matches
    CuTe's convention that the last mode is implicitly extensible.
    """
    # Case 0: None coordinate contributes 0 offset (used by slice operations)
    if coord is None:
        return 0

    # Case 1: Scalar shape - direct multiplication
    if is_int(shape):
        if is_tuple(coord):
            raise LayoutError(f"Cannot map coordinate {coord} to scalar shape {shape}")
        return coord * stride

    # Case 2: 1D index mapping (index -> nD -> offset)
    if isinstance(coord, int):
        offset = 0
        index = coord
        shape_list = list(shape)
        stride_list = list(stride)
        for i, (s, d) in enumerate(zip(shape_list, stride_list)):
            mode_size = size(s)
            if i < len(shape_list) - 1:
                # All modes except last: mod by mode size
                c = index % mode_size
                index //= mode_size
            else:
                # Last mode: do not mod — extend infinitely
                c = index

            # If s is a Tuple, d is also a Tuple. We must recurse.
            if is_tuple(s):
                offset += crd2offset(c, s, d)
            else:
                offset += c * d
        return offset

    # Case 3: nD coordinate mapping (coord tuple -> offset)
    if not is_tuple(coord):
        raise TypeError(f"Coordinate must be int or tuple, got {type(coord).__name__}")
    if len(coord) != len(shape):
        raise LayoutError(f"Coordinate rank {len(coord)} does not match layout rank {len(shape)}")
    offset = 0
    for c, s, d in zip(coord, shape, stride):
        if c is None:
            continue  # None coordinates contribute 0 (slice marker)
        if is_tuple(s):
            # If the shape element is nested, the coordinate part c
            # must also be nested (or be an int that we treat as a 1D index)
            offset += crd2offset(c, s, d)
        else:
            offset += c * d
    return offset


def crd2idx(coord, shape, stride=None):
    """Dispatch to crd2flat (2-arg) or crd2offset (3-arg).

    Matches C++ CuTe's overloaded crd2idx(coord, shape[, stride]):
      crd2idx(coord, shape)         -> 1D flat index (colexicographic)
      crd2idx(coord, shape, stride) -> memory offset (inner product with stride)
    """
    if stride is None:
        return crd2flat(coord, shape)
    return crd2offset(coord, shape, stride)


def crd2crd(crd: Any, dst_shape: Any, src_shape: Any = None) -> Any:
    """Transform a coordinate into a different shape's iteration space.

    If crd is a tuple and dst_shape is a tuple, recursively transform each mode.
    If crd is a tuple and dst_shape is an int, flatten the coordinate using src_shape.
    If crd is an int and dst_shape is a tuple, expand the index into dst_shape.
    If both are ints, return crd (identity).

    Args:
        crd: The coordinate to transform
        dst_shape: The target shape
        src_shape: The source shape (required when crd is tuple and dst_shape is scalar)

    Examples:
        crd2crd(3, (2, 4)) -> (1, 1)        # expand index 3 into (2,4)
        crd2crd((1, 0), 8, (2, 4)) -> 1     # flatten (1,0) from (2,4) space
        crd2crd((1, 2), (3, 4)) -> (1, 2)   # identity transform
        crd2crd(((0, 1), 0), (6, 2), ((2, 3), 2)) -> (2, 0)  # flatten per-mode
    """
    if is_tuple(crd):
        if is_tuple(dst_shape):
            if len(crd) != len(dst_shape):
                raise LayoutError(
                    f"Rank mismatch: crd has {len(crd)} elements, dst_shape has {len(dst_shape)}"
                )
            if src_shape is not None and is_tuple(src_shape):
                return tuple(crd2crd(c, d, s) for c, d, s in zip(crd, dst_shape, src_shape))
            return zip_transform(crd, dst_shape, crd2crd)
        else:
            # crd is tuple, dst_shape is scalar: flatten using src_shape
            if src_shape is None:
                raise LayoutError("src_shape required to flatten tuple coordinate to scalar")
            return crd2flat(crd, src_shape)
    else:
        if is_tuple(dst_shape):
            return idx2crd(crd, dst_shape)
        else:
            return crd


def slice_modes(crd, trg):
    """Filter trg according to crd: keep only elements paired with None.

    This implements CuTe's slice operator. Elements of trg that are paired
    with None in crd are kept (wrapped in a tuple); elements paired with
    concrete integers are dropped.

    Args:
        crd: A coordinate with None values indicating sliced dimensions
        trg: The target (shape or stride) to filter

    Returns:
        A tuple of the kept elements (flattened from nested results)

    Examples:
        slice_modes(None, 4) -> (4,)
        slice_modes(0, 4) -> ()
        slice_modes((None, 0), (3, 4)) -> (3,)
        slice_modes((0, None), (3, 4)) -> (4,)
        slice_modes((None, None), (3, 4)) -> (3, 4)
    """
    if is_tuple(crd):
        if is_tuple(trg):
            if len(crd) != len(trg):
                raise LayoutError(f"Rank mismatch: crd has {len(crd)} elements, trg has {len(trg)}")
            # Process each top-level mode independently, preserving hierarchy
            result = []
            for c, s in zip(crd, trg):
                sub = slice_modes(c, s)
                if sub:
                    # Unwrap single-element results to avoid extra nesting,
                    # but keep multi-element results as a nested tuple
                    result.append(sub[0] if len(sub) == 1 else sub)
            return tuple(result)
        else:
            raise LayoutError("Cannot slice scalar target with tuple coordinate")
    elif crd is None:
        return (trg,)
    else:
        return ()


def dice_modes(crd, layout):
    """Keep only the modes of a layout that are paired with integers in crd.

    Dice is the complement of slice: slice_modes keeps the None-marked modes
    (the free dimensions), while dice_modes keeps the integer-marked modes
    (the fixed dimensions).

    For layouts: returns a layout over only the "diced" modes.
    For tuples: returns a filtered tuple.

    Note the difference from the C++ entry point: when crd is a plain integer
    (not a tuple), dice_modes returns the target directly (unwrapped), matching
    CuTe's convention that dice(int, b) == b.

    Args:
        crd: A coordinate with None for modes to drop, integers for modes to keep
        layout: The Layout (or tuple) to filter

    Returns:
        A Layout (or value) over only the integer-marked modes

    Examples:
        dice_modes(0, Layout((3,4), (1,4)))       -> (3,4):(1,4)   # scalar crd: identity
        dice_modes((0, None), Layout((3,4),(1,4))) -> 3:1           # keep mode 0
        dice_modes((None, 0), Layout((3,4),(1,4))) -> 4:4           # keep mode 1
    """

    def dice_tuple(crd, trg):
        """Keep elements of trg paired with integers in crd."""
        if is_tuple(crd):
            if is_tuple(trg):
                if len(crd) != len(trg):
                    raise LayoutError(
                        f"Rank mismatch: crd has {len(crd)} elements, trg has {len(trg)}"
                    )
                result = []
                for c, s in zip(crd, trg):
                    result.extend(dice_tuple(c, s))
                return tuple(result)
            else:
                raise LayoutError("Cannot dice scalar target with tuple coordinate")
        elif crd is None:
            return ()
        else:
            return (trg,)

    if isinstance(layout, Layout):
        if is_tuple(crd):
            diced_shape = dice_tuple(crd, layout.shape)
            diced_stride = dice_tuple(crd, layout.stride)
            return Layout(as_shape(diced_shape), as_shape(diced_stride))
        elif crd is None:
            return Layout()
        else:
            return layout
    else:
        # Tuple-level dice
        if is_tuple(crd):
            return dice_tuple(crd, layout)
        elif crd is None:
            return ()
        else:
            return layout


# =============================================================================
# Tile and composition
# =============================================================================
#
# Composition is function composition: compose(A, B) produces a layout C where
# C(i) = A(B(i)). B selects which elements of A to visit, and in what order.
# This is the fundamental operation --- division, product, and tiling are all
# defined in terms of composition.
#
# A Tile is a tuple-of-Layouts used for mode-by-mode composition. When you
# compose a multi-mode layout with a Tile, each Tile element is composed with
# the corresponding mode independently.
#
# Shape arithmetic (shape_div, shape_mod) is the machinery that makes
# composition work on hierarchical shapes: it propagates a divisor through
# nested shape elements, consuming from the innermost (leftmost) modes first.
#


def safe_div(a: int, b: int) -> int:
    """Integer division where b must divide a evenly.

    In CuTe, this is used when we know the division is exact.
    Returns a // b, asserting that b divides a.
    """
    if b == 0:
        raise LayoutError("Division by zero")
    if a % b != 0:
        raise LayoutError(f"safe_div requires {b} to divide {a} evenly")
    return a // b


def shape_div(shape: Any, divisor: int) -> Any:
    """Divide a shape by a divisor, consuming from the innermost modes first.

    Intuition: shape_div and shape_mod together factor a shape into two
    pieces — the part consumed by the divisor (shape_mod) and the part
    that remains (shape_div). They are the hierarchical analog of
    integer divmod, respecting CuTe's column-major (leftmost-fastest)
    convention: divisors consume from the innermost (leftmost) modes
    first, then carry to outer modes.

    shape_div is to hierarchical shapes what integer division is to integers.
    It divides the shape element-by-element from left to right (innermost first
    in CuTe's column-major convention). When the divisor exceeds a mode's size,
    that mode becomes 1 and the remaining divisor carries to the next mode.

    For scalars, this implementation only supports the exact-factor cases
    needed by the complementary ``shape_mod`` algebra: either ``b`` divides
    ``a`` or ``a`` divides ``b``. In those supported cases,
    ``shape_div(a, b)`` equals ``ceil(a / b)`` (``a // b`` when ``b | a``,
    and ``1`` when ``a | b``). If neither divides the other,
    ``shape_div`` raises ``ValueError``.

    This is intentionally stricter than dynamic CuTe C++, which may return
    ``ceil_div(a, b)`` for non-divisible scalar pairs such as
    ``shape_div(6, 4) -> 2``.

    The key identity: size(shape_div(s, d)) * size(shape_mod(s, d)) == size(s)

    Examples:
        shape_div(12, 4) -> 3           # 12/4 = 3
        shape_div(12, 3) -> 4           # 12/3 = 4
        shape_div((4, 3), 2) -> (2, 3)  # Divide first mode: 4/2=2, rest untouched
        shape_div((4, 3), 4) -> (1, 3)  # First mode consumed: 4/4=1
        shape_div((4, 6), 8) -> (1, 3)  # Carries into second mode: 8/4=2, 6/2=3
        shape_div((4, 3), 12) -> (1, 1) # All consumed
        shape_div(6, 4) -> ValueError    # intentional strict policy
    """
    if divisor == 1:
        return shape

    def _scalar(s, d):
        if s % d != 0 and d % s != 0:
            raise LayoutError(
                f"shape_div({s}, {d}): one must divide the other for clean factorization"
            )
        return (s + d - 1) // d

    def _update(first, divisor):
        return shape_div(divisor, size(first))

    return fold_accumulate(shape, divisor, _scalar, _update)


def shape_mod(shape: Any, modulus: int) -> Any:
    """The complement of shape_div: returns the "kept" portion of a shape.

    If shape_div tells you what's left after dividing, shape_mod tells you
    what was consumed. The key identity:
        size(shape_div(s, d)) * size(shape_mod(s, d)) == size(s)

    For scalars: shape_mod(a, m) = min(a, m) when one divides the other (= gcd(a, m)).

    Examples:
        shape_mod(12, 4) -> 4           # gcd(12, 4) = 4
        shape_mod((4, 3), 2) -> (2, 1)  # 2 consumed from first mode, nothing from second
        shape_mod((4, 3), 12) -> (4, 3) # All kept (modulus >= size)
    """

    def _scalar(s, m):
        return s if m >= s else math.gcd(s, m)

    def _update(first, modulus):
        return shape_div(modulus, shape_mod(size(first), modulus))

    return fold_accumulate(shape, modulus, _scalar, _update)


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division: smallest integer >= a/b."""
    return (a + b - 1) // b


def _shape_ceil_div(shape: Any, divisor: int) -> Any:
    """CuTe-style ceil_div for shapes, preserving nested structure."""
    if divisor == 1:
        return shape

    def _scalar(s, d):
        return _ceil_div(s, d)

    def _update(first, d):
        return _ceil_div(d, size(first))

    return fold_accumulate(shape, divisor, _scalar, _update)


def _coalesce_shape(shape: Any) -> Any:
    """Coalesce a pure shape through its compact column-major layout."""
    return coalesce(Layout(shape, compute_col_major_strides(shape))).shape


def upcast(layout: "Layout", n: int) -> "Layout":
    """Reinterpret a layout from a finer to a coarser coordinate space.

    Mirrors CuTe's upcast<N>(layout).  Use case: GPU memory layouts are
    often defined in bits (to handle mixed-precision types uniformly),
    but you want to work with elements (fp16, int8, etc.).
    upcast(L, 16) converts a bit-addressed layout to an fp16-element layout.

    For the stride-1 mode the shape shrinks by n (the elements are now n×
    bigger, so there are fewer of them).  All strides are divided by n.

    Examples:
        # Bit layout → fp16 elements (÷16)
        upcast(Layout((32, 32), (32, 1)), 16)
        # => Layout((32, 2), (2, 1))

        # Hierarchical value mode
        upcast(Layout((32, (32, 4)), (32, (1, 1024))), 16)
        # => Layout((32, (2, 4)), (2, (1, 64)))

        # Transpose layout with sub-element innermost stride
        upcast(Layout(((4, 8), (16, 2)), ((256, 16), (1, 128))), 16)
        # => Layout(((4, 8), (1, 2)), ((16, 1), (1, 8)))
    """
    if n == 1:
        return layout

    def _upcast_leaf(s, d):
        if d == 0:
            return (s, d)
        shape_divisor = _ceil_div(n, abs(d))
        new_shape = _ceil_div(s, shape_divisor)
        new_stride = (1 if d > 0 else -1) * _ceil_div(abs(d), n)
        return (new_shape, new_stride)

    def _apply(shape, stride):
        if is_tuple(shape):
            if not is_tuple(stride) or len(shape) != len(stride):
                raise LayoutError(f"Shape/stride structure mismatch: {shape} vs {stride}")
            pairs = [_apply(s, d) for s, d in zip(shape, stride)]
            new_s = tuple(p[0] for p in pairs)
            new_d = tuple(p[1] for p in pairs)
            return (new_s, new_d)
        return _upcast_leaf(shape, stride)

    new_shape, new_stride = _apply(layout.shape, layout.stride)
    return Layout(new_shape, new_stride)


def downcast(layout: "Layout", n: int) -> "Layout":
    """Reinterpret a layout from a coarser to a finer coordinate space.

    Mirrors CuTe's downcast<N>(layout).  The inverse of upcast: converts
    element coordinates back to bit coordinates.  For the stride-1 mode
    the shape grows by n, and all other strides are multiplied by n.

    Examples:
        # Element layout → bit coordinates (×16)
        downcast(Layout((32, 2), (2, 1)), 16)
        # => Layout((32, 32), (32, 1))
    """
    if n == 1:
        return layout

    def _downcast_leaf(s, d):
        if abs(d) == 1:
            return (s * n, d)
        return (s, d * n)

    def _apply(shape, stride):
        if is_tuple(shape):
            if not is_tuple(stride) or len(shape) != len(stride):
                raise LayoutError(f"Shape/stride structure mismatch: {shape} vs {stride}")
            pairs = [_apply(s, d) for s, d in zip(shape, stride)]
            new_s = tuple(p[0] for p in pairs)
            new_d = tuple(p[1] for p in pairs)
            return (new_s, new_d)
        return _downcast_leaf(shape, stride)

    new_shape, new_stride = _apply(layout.shape, layout.stride)
    return Layout(new_shape, new_stride)


def _composition_1d(layout_a: "Layout", b_shape: int, b_stride: int) -> "Layout":
    """Compose layout A with a 1D layout (scalar shape and stride).

    This is the core composition algorithm. It answers: "if B selects
    b_shape elements from A with stride b_stride, what layout results?"

    Algorithm:
      1. Coalesce A to merge contiguous modes into a flat list.
      2. Fold over A's modes (except the last):
         - Compute how many of B's elements fit in this mode of A.
         - Emit a result mode with shape = elements consumed,
           stride = b_stride * a_mode_stride.
         - Carry remaining B shape/stride to the next mode.
      3. Last mode absorbs all remaining shape (CuTe's extensible-
         last-mode convention: the outermost mode is implicitly
         infinite).

    Example:
        A = Layout((4, 8), (1, 8))  # two modes with a stride gap
        B has shape=8, stride=1     # select 8 contiguous elements
        Result: Layout((4, 2), (1, 8))  # 4 from first mode, 2 from second
    """
    if b_stride == 0:
        return Layout(b_shape, 0)

    flat_a = coalesce(layout_a)
    flat_shapes = as_list(flat_a.shape)
    flat_strides = as_list(flat_a.stride)

    result_shape = []
    result_stride = []
    remaining_shape = b_shape
    remaining_stride = b_stride

    # Match pycute's post-coalesce truncation path for tuple-LHS / integral-
    # RHS composition. In particular:
    # - exact shape/stride divisibility still passes immediately
    # - otherwise a non-divisible stride may only continue when the remaining
    #   RHS has multiple points and all of them fit inside the current mode
    # - the chunk we consume from the RHS shape must divide the remaining
    #   shape exactly
    for curr_shape, curr_stride in zip(flat_shapes[:-1], flat_strides[:-1]):
        abs_stride = abs(remaining_stride)
        negative_stride = remaining_stride < 0

        divisible = curr_shape % abs_stride == 0 or abs_stride % curr_shape == 0
        fits_in_mode = remaining_shape > 1 and (remaining_shape - 1) * abs_stride < curr_shape

        if not divisible and not fits_in_mode:
            raise LayoutError(
                f"compose: shape {curr_shape} and stride {remaining_stride} are not divisible"
            )

        if fits_in_mode:
            result_shape.append(remaining_shape)
            result_stride.append(remaining_stride * curr_stride)
            remaining_shape = 1
            break

        next_shape = _ceil_div(curr_shape, abs_stride)
        next_stride = _ceil_div(abs_stride, curr_shape)
        if negative_stride:
            next_stride = -next_stride

        if next_shape == 1 or remaining_shape == 1:
            remaining_stride = next_stride
            continue

        new_shape = min(next_shape, remaining_shape)
        if remaining_shape % new_shape != 0:
            raise LayoutError(
                f"compose: shape {remaining_shape} and consumed extent {new_shape} are not divisible"
            )

        result_shape.append(new_shape)
        result_stride.append(remaining_stride * curr_stride)
        remaining_shape //= new_shape
        remaining_stride = next_stride

    # Last mode absorbs all remaining shape
    if remaining_shape != 1 or not result_shape:
        result_shape.append(remaining_shape)
        result_stride.append(remaining_stride * flat_strides[-1])

    return Layout(as_shape(result_shape), as_shape(result_stride))


def _compose_layouts(layout_a: "Layout", layout_b: "Layout") -> "Layout":
    """Compose two Layout objects."""
    if is_empty(layout_a) or size(layout_a) == 0:
        return Layout()
    if is_empty(layout_b) or size(layout_b) == 0:
        return Layout()

    def compose_element(b_shape, b_stride):
        """Recursively compose A with one element of B's shape/stride."""
        if is_tuple(b_shape):
            results = [compose_element(b_shape[i], b_stride[i]) for i in range(len(b_shape))]
            return Layout(tuple(r.shape for r in results), tuple(r.stride for r in results))
        return _composition_1d(layout_a, b_shape, b_stride)

    if is_tuple(layout_b.shape):
        results = [
            compose_element(layout_b.shape[i], layout_b.stride[i])
            for i in range(len(layout_b.shape))
        ]
        return Layout(tuple(r.shape for r in results), tuple(r.stride for r in results))

    return _composition_1d(layout_a, layout_b.shape, layout_b.stride)


def _compose_with_tiler(layout_a: "Layout", tiler) -> "Layout":
    """Compose a layout mode-by-mode with a tiler (Tile or tuple)."""
    # ComposedLayout inputs should already have been intercepted by
    # _forward_layout_domain() before we get here.  This helper rebuilds an
    # affine Layout by reading per-mode .shape/.stride, so a composed result
    # would mean a caller escaped the generic forwarding path.
    result_shapes = []
    result_strides = []

    for i, elem in enumerate(tiler):
        mode_layout = mode(layout_a, i)
        composed = compose(mode_layout, elem)
        if not isinstance(composed, Layout):
            raise TypeError(
                "_compose_with_tiler expects affine per-mode results; "
                "ComposedLayout inputs should be forwarded earlier"
            )
        result_shapes.append(unwrap(composed.shape))
        result_strides.append(unwrap(composed.stride))

    # Append remaining modes unchanged
    for i in range(len(tiler), rank(layout_a)):
        mode_layout = mode(layout_a, i)
        result_shapes.append(unwrap(mode_layout.shape))
        result_strides.append(unwrap(mode_layout.stride))

    return Layout(tuple(result_shapes), tuple(result_strides))


def _normalize_compose_tiler_element(elem):
    """Normalize a compose tiler element while preserving nested tuple structure.

    Leaf integers become stride-1 Layouts. Nested tuples stay nested so
    composition recurses mode-by-mode, matching CuTe's tuple-tiler semantics.
    """
    if isinstance(elem, Layout):
        return elem
    if isinstance(elem, ComposedLayout):
        raise TypeError(
            "ComposedLayout tiler elements are not supported in tuple composition; "
            "pass a single LayoutExpr as the second argument instead"
        )
    if isinstance(elem, int):
        return Layout(elem, 1)
    if is_tuple(elem):
        return tuple(_normalize_compose_tiler_element(e) for e in elem)
    raise TypeError(f"Invalid tiler element: {type(elem)}")


def _compose_into_composed_lhs(layout_a: "ComposedLayout", layout_b: Any) -> "ComposedLayout":
    """compose(ComposedLayout(outer, inner, offset), B).

    The outer/offset are external to the data domain, so composition
    pushes through to the inner: outer ∘ (inner ∘ B).
    """
    return ComposedLayout(
        layout_a.outer,
        compose(layout_a.inner, layout_b),
        offset=layout_a.offset,
    )


def _compose_swizzle_lhs(swizzle: "Swizzle", layout_b: Any) -> Any:
    """compose(Swizzle, layout_b): apply the swizzle to layout_b's outputs.

    Path X: a Swizzle composed with anything always lives as a
    ``ComposedLayout(swizzle, layout_b, offset=0)`` -- there is no
    embedded-swizzle Layout shortcut anymore. ``Layout`` is purely
    affine; ``ComposedLayout`` is the single home for swizzled forms.
    Identity swizzles still drop out entirely.
    """
    if not is_layout(layout_b):
        raise TypeError(
            "When composing with Swizzle, second argument must be a layout "
            f"expression, got {type(layout_b).__name__}"
        )
    if swizzle.bits == 0:
        return layout_b
    return ComposedLayout(swizzle, layout_b)


def _compose_with_swizzle_rhs(layout_a: "Layout", layout_b: "Swizzle") -> Any:
    """compose(Layout, Swizzle): push through when representable, else stay exact.

    Computes the swizzle masks under layout_a's mapping and keeps the old
    push-through fast path only when those active masks still define a valid
    CuTe swizzle. Otherwise we preserve exact semantics by swizzling the
    identity layout over layout_a's domain and composing layout_a outside it.
    """
    if not isinstance(layout_a, Layout):
        raise TypeError(
            "When composing with Swizzle, first argument must be Layout, got "
            f"{type(layout_a).__name__}"
        )
    active_Y = layout_a(layout_b.yyy_msk)
    active_Z = layout_a(layout_b.zzz_msk)
    try:
        active_swizzle = make_swizzle(active_Y, active_Z)
    except ValueError:
        return ComposedLayout(layout_a, compose(layout_b, Layout(layout_a.shape)))
    if active_swizzle is None:
        return layout_a
    return compose(active_swizzle, layout_a)


def _compose_layout_with_layout(layout_a: "Layout", layout_b: "Layout") -> Any:
    """compose(Layout, Layout): the affine-with-affine case (the textbook one).

    Path X: ``Layout`` is purely affine, so ``layout_b.swizzle`` is
    always None and the previous embedded-swizzle bypass is gone.
    """
    if not isinstance(layout_a, Layout):
        raise TypeError(
            "When composing with Layout, first argument must be Layout, Swizzle, "
            f"or ComposedLayout, got {type(layout_a).__name__}"
        )
    return _compose_layouts(layout_a, layout_b)


def _compose_with_composed_rhs(layout_a: "Layout", layout_b: "ComposedLayout") -> Any:
    """compose(Layout, ComposedLayout(outer, inner, offset)).

    Path X note: when the RHS is a canonical ``Sw o L`` (Swizzle outer,
    zero offset), associativity ``A o (Sw o L) = (A o Sw) o L`` would
    require ``A o Sw`` to push a swizzle through ``A``; that transfer
    only matches the pointwise function for swizzle-friendly affine
    layouts. Pre-Path-X, this path was hidden by the
    ``_compose_layout_with_layout`` short-circuit on
    ``Layout(.., swizzle=Sw)`` operands. Path X retires the embedded
    form, so we now keep the swizzled wrapper intact rather than
    risk an incorrect transfer. Pure-affine outer (non-Swizzle) still
    associates safely because layout composition is associative.
    """
    if not isinstance(layout_a, Layout):
        raise TypeError(
            "When composing with ComposedLayout, first argument must be Layout, "
            f"Swizzle, or ComposedLayout, got {type(layout_a).__name__}"
        )
    if layout_b.offset == 0 and is_layout(layout_b.outer) and not isinstance(
        layout_b.outer, Swizzle
    ):
        return compose(compose(layout_a, layout_b.outer), layout_b.inner)
    return ComposedLayout(layout_a, layout_b)


def _compose_with_tuple_tiler(layout_a: Any, layout_b: tuple) -> Any:
    """compose(A, (B0, B1, ...)): mode-by-mode composition.

    Each tuple element is normalized into a Tile-friendly Layout (an int
    becomes Layout(n, 1); a nested tuple stays as a tiler that recurses
    inside _compose_with_tiler).  When every element is a flat Layout we
    promote to Tile so it pretty-prints as one.
    """
    tiler = tuple(_normalize_compose_tiler_element(e) for e in layout_b)
    if all(isinstance(e, Layout) for e in tiler):
        tiler = Tile(*tiler)
    if len(tiler) > rank(layout_a):
        raise LayoutError(
            f"Tiler has {len(tiler)} elements but layout has only {rank(layout_a)} modes"
        )
    return _compose_with_tiler(layout_a, tiler)


def compose(layout_a: Any, layout_b: Any) -> Any:
    """The fundamental operation of CuTe layout algebra: function composition.

    compose(A, B) produces a layout C where C(i) = A(B(i)). B selects which
    elements of A to visit, and in what order.
    The resulting layout has B's shape, and maps indices through B then A:
        compose(A, B)(i) = A(B(i))

    When A is a Swizzle, the result is a Layout with an embedded swizzle that
    applies the underlying layout B first, then applies the swizzle function:
        compose(Swizzle, Layout)(i) = Swizzle(Layout(i))

    This is the fundamental composition operation in CuTe that allows
    building complex memory access patterns from simpler ones.

    A Tiler (the second argument) can be one of:
    1. A Layout - composition between two functions from integers to integers
    2. A tuple of Tilers - mode-by-mode composition until case (1) is found
    3. A Shape (tuple of ints) - interpreted as tuple of Layout(n, 1)
       at the leaves, while nested tuples recurse mode-by-mode

    When B is a tuple of Tilers, composition is done mode-by-mode:
        compose(A, (B0, B1, ...)) = Layout(compose(mode(A,0), B0),
                                           compose(mode(A,1), B1), ...)

    This recursive definition allows:
    - By-mode tiling: "Give me the 3x5x8 subblock of this MxNxL tensor"
    - 1-D reshaping: "Reorder this 8x16 block using this element order"

    Args:
        layout_a: The outer layout (the one being indexed into)
        layout_b: A Tiler - Layout, tuple of Tilers, or Shape

    Returns:
        A Layout (possibly with nested shape/stride, possibly with embedded swizzle)

    Examples:
        compose(Layout(8, 2), Layout(4, 1)) -> Layout(4, 2)
        compose(Layout((6,2), (8,2)), Layout((4,3), (3,1))) -> Layout(((2,2),3), ((24,2),8))

        # Mode-by-mode with explicit Tile:
        a = Layout((12, (4, 8)), (59, (13, 1)))
        tiler = Tile(Layout(3, 4), Layout(8, 2))
        compose(a, tiler) -> Layout((3, (2, 4)), (236, (26, 1)))

        # Shape as tiler (interpreted as tuple-of-layouts with stride 1):
        a = Layout((12, (4, 8)), (59, (13, 1)))
        tiler = (3, 8)  # Equivalent to Tile(Layout(3,1), Layout(8,1))
        compose(a, tiler) -> Layout((3, (4, 2)), (59, (13, 1)))

        # Mixed tuple of tilers:
        tiler = (Layout(3, 4), 8)  # Layout for mode 0, Shape for mode 1
        compose(a, tiler) -> Layout((3, (4, 2)), (236, (13, 1)))

        # Swizzle composition (returns Layout with embedded swizzle):
        compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1))) -> Layout with swizzle

    The implementation is a thin dispatcher: each (lhs, rhs) case lives in
    its own _compose_<case> helper just above so the algebra reads case by
    case.  Order matters -- the LHS cases peel off ComposedLayout and
    Swizzle first so subsequent branches can assume an affine LHS.
    """
    # LHS unwrapping: handle wrapped/swizzled forms before dispatching on rhs.
    if isinstance(layout_a, ComposedLayout):
        return _compose_into_composed_lhs(layout_a, layout_b)
    if isinstance(layout_a, Swizzle):
        return _compose_swizzle_lhs(layout_a, layout_b)

    # RHS dispatch: from here, layout_a is a (purely affine) Layout.
    if isinstance(layout_b, Swizzle):
        return _compose_with_swizzle_rhs(layout_a, layout_b)
    if isinstance(layout_b, Layout):
        return _compose_layout_with_layout(layout_a, layout_b)
    if isinstance(layout_b, ComposedLayout):
        return _compose_with_composed_rhs(layout_a, layout_b)
    if isinstance(layout_b, Tile):
        if len(layout_b) > rank(layout_a):
            raise LayoutError(
                f"Tiler has {len(layout_b)} elements but layout has only {rank(layout_a)} modes"
            )
        return _compose_with_tiler(layout_a, layout_b)
    if is_tuple(layout_b):
        return _compose_with_tuple_tiler(layout_a, layout_b)

    raise TypeError(f"Invalid tiler type: {type(layout_b)}")


# =============================================================================
# Division operations
# =============================================================================
#
# Division factors a layout into (tile, rest):
#   logical_divide(A, B) = compose(A, Layout(B, complement(B, shape(coalesce(A)))))
#
# The tile part tells you where within a tile, the rest part tells you
# which tile. Division answers: "how do I iterate in tiles of size T?"
#
# The zipped/tiled/flat variants control how the result modes are organized:
#   zipped_divide  -> ((tiles), (rests))          -- two modes
#   tiled_divide   -> ((tiles), rest0, rest1, ..) -- tiles grouped, rests flat
#   flat_divide    -> (tile0, tile1, rest0, ..)   -- everything flat
#


def logical_divide(layout: "LayoutExpr", tiler: Any) -> "LayoutExpr":
    """Divide a layout into (tile, rest) --- the core tiling operation.

    Division answers: "if I want to process this layout in tiles of size T,
    how do I organize the iteration?" The result has two parts:
    - Tile: coordinates *within* a tile (the inner loop)
    - Rest: coordinates *across* tiles (the outer loop)

    Formally: logical_divide(A, B) =
        compose(A, Layout(B, complement(B, shape(coalesce(A)))))

    Intuition: to tile A by B, we need two coordinates:
    - "which element within a tile?" -> B (the tiler itself)
    - "which tile?" -> complement(B) (fills the gaps between tiles)
    Layout(B, complement(B)) bundles these into (within-tile, across-tiles).
    Composing with A maps this coordinate space through A's pattern.

    The result has the structure: (Tile, Rest)

    For tuple tilers, each top-level mode is divided independently and nested
    tuple tilers recurse within that mode:
        ((TileM, RestM), (TileN, RestN), L, ...)

    The tiler can be:
    - An integer: simple 1D tile size (uses mode-by-mode division)
    - A tuple/Tuple of integers: tile sizes for each mode (mode-by-mode division)
    - A Layout: uses the CuTe formula with composition

    Args:
        layout: The layout to tile
        tiler: The tile specification (int, tuple, or Layout)

    Returns:
        A Layout with hierarchical (tile, rest) structure

    Examples:
        logical_divide(Layout(16), 4) -> Layout((4, 4), (1, 4))
        logical_divide(Layout((4,2,3), (2,1,8)), Layout(4, 2)) -> ((2,2),(2,3)):((4,1),(2,8))
    """
    forwarded = _forward_layout_domain(layout, lambda inner: logical_divide(inner, tiler))
    if forwarded is not _NO_FORWARD:
        return forwarded
    # logical_divide internally calls coalesce/complement on `layout`, both of
    # which reject the Swizzle-inner form. Guard explicitly so the error
    # surfaces against the user's actual call site rather than a callee.
    _reject_swizzle_inner_composed(layout, "logical_divide")
    if isinstance(tiler, Layout):
        # Layout tiler: use CuTe formula
        # logical_divide(A, B) =
        #   compose(A, Layout(B, complement(B, size(coalesce(A)))))
        tiler_complement = complement(tiler, size(coalesce(layout)))

        # Create bundled layout: (tiler, complement)
        combined = Layout(tiler, tiler_complement)

        # Compose layout with the combined tiler
        result = compose(layout, combined)
        return result
    elif isinstance(tiler, int):
        # Integer tiler: mode-by-mode division of first mode
        # If the tile doesn't evenly divide the first mode, use Layout path
        if is_int(layout.shape):
            # Scalar layout
            first_mode_size = layout.shape
        else:
            first_mode_size = size(layout.shape[0])
        if first_mode_size % tiler != 0:
            return logical_divide(layout, Layout(tiler, 1))
        return _logical_divide_by_shape(layout, (tiler,))
    elif is_tuple(tiler):
        # Tuple tiler: recurse mode-by-mode, preserving nested tuple structure
        return _logical_divide_with_tiler(layout, tiler)
    else:
        raise TypeError(f"Tiler must be int, tuple, or Layout, got {type(tiler)}")


def _logical_divide_with_tiler(layout: "Layout", tiler) -> "Layout":
    """Divide a layout mode-by-mode with a possibly nested tuple tiler."""
    # ComposedLayout inputs should already have been intercepted by
    # _forward_layout_domain() before we get here.  This helper rebuilds an
    # affine Layout from per-mode .shape/.stride pairs, so a composed result
    # would indicate that the earlier generic forwarding path was bypassed.
    if len(tiler) > rank(layout):
        raise LayoutError(
            f"logical_divide: tiler has more modes ({len(tiler)}) than layout ({rank(layout)})"
        )

    result_shapes = []
    result_strides = []

    for i, elem in enumerate(tiler):
        mode_layout = mode(layout, i)
        divided = logical_divide(mode_layout, elem)
        if not isinstance(divided, Layout):
            raise TypeError(
                "_logical_divide_with_tiler expects affine per-mode results; "
                "ComposedLayout inputs should be forwarded earlier"
            )
        result_shapes.append(unwrap(divided.shape))
        result_strides.append(unwrap(divided.stride))

    for i in range(len(tiler), rank(layout)):
        mode_layout = mode(layout, i)
        result_shapes.append(unwrap(mode_layout.shape))
        result_strides.append(unwrap(mode_layout.stride))

    return Layout(as_shape(result_shapes), as_shape(result_strides))


def _logical_divide_by_shape(layout: "Layout", tiler_shape: Any) -> "Layout":
    """Divide a layout mode-by-mode using a shape tuple.

    This is used when the tiler is a simple shape tuple (not a Layout).
    Each mode of the layout is divided by the corresponding tiler element.

    Result structure: ((TileM, RestM), (TileN, RestN), L, ...)
    """
    tiler_sizes = [tiler_shape] if isinstance(tiler_shape, int) else list(tiler_shape)

    # Convert scalar shapes to tuple for uniform processing
    layout_shapes = as_tuple(layout.shape)
    layout_strides = as_tuple(layout.stride)

    # CuTe C++ static_asserts: "logical_divide: Too many modes in tiler."
    if len(tiler_sizes) > len(layout_shapes):
        raise LayoutError(
            f"logical_divide: tiler has more modes ({len(tiler_sizes)}) "
            f"than layout ({len(layout_shapes)})"
        )

    result_shapes = []
    result_strides = []

    for i, (s, d) in enumerate(zip(layout_shapes, layout_strides)):
        if i >= len(tiler_sizes):
            result_shapes.append(s)
            result_strides.append(d)
            continue

        tile_size = tiler_sizes[i]

        # Layout tilers use the compose/complement path per mode,
        # matching CuTe C++ which treats tiler elements as Layouts.
        if isinstance(tile_size, Layout):
            mode_layout = Layout(s, d)
            divided = logical_divide(mode_layout, tile_size)
            result_shapes.append(divided.shape)
            result_strides.append(divided.stride)
            continue

        mode_size = size(s)

        # Hierarchical strides can't be handled by the simple shortcut.
        # Fall through to the formal compose/complement path, matching
        # CuTe C++ which always uses that path (layout.hpp:1576).
        if is_tuple(d):
            mode_layout = Layout(s, d)
            divided = logical_divide(mode_layout, Layout(tile_size, 1))
            result_shapes.append(divided.shape)
            result_strides.append(divided.stride)
        elif tile_size <= mode_size and mode_size % tile_size == 0:
            rest_size = mode_size // tile_size
            tile_stride = 0 if tile_size == 1 else d
            rest_stride = 0 if rest_size == 1 else elem_scale(d, tile_size)
            result_shapes.append((tile_size, rest_size))
            result_strides.append((tile_stride, rest_stride))
        elif tile_size <= mode_size:
            # Non-divisible: fall through to compose/complement path,
            # matching CuTe C++ which always uses that path for int tilers
            mode_layout = Layout(s, d)
            divided = logical_divide(mode_layout, Layout(tile_size, 1))
            result_shapes.append(divided.shape)
            result_strides.append(divided.stride)
        else:
            tile_part = compose(Layout(s, d), Layout(tile_size, 1))
            tile_s = unwrap(tile_part.shape) if is_tuple(tile_part.shape) else tile_part.shape
            tile_d = unwrap(tile_part.stride) if is_tuple(tile_part.stride) else tile_part.stride
            result_shapes.append((tile_s, 1))
            result_strides.append((tile_d, 0))

    # Use as_shape to unwrap single-element results back to scalar form
    return Layout(as_shape(result_shapes), as_shape(result_strides))


def _split_divided_modes(layout: "Layout", tiler: Any):
    """Split logical_divide() results for shape-like tilers.

    Performs logical_divide, then separates each divided mode into its tile
    portion and its remainder portion. Undivided modes go into the rest lists.
    This helper intentionally accepts only shape-like tilers (`int` or tuples);
    true `Layout` tilers follow CuTe's terminal `tile_unzip` path instead.

    Returns:
        (tile_shapes, tile_strides, rest_shapes, rest_strides) - four lists
    """
    if isinstance(tiler, int):
        tiler_shape = (tiler,)
    elif is_tuple(tiler):
        tiler_shape = tiler
    else:
        raise TypeError(f"_split_divided_modes expects an int or tuple tiler, got {type(tiler)}")

    divided = logical_divide(layout, tiler_shape)

    tile_shapes = []
    tile_strides = []
    rest_shapes = []
    rest_strides = []

    num_tiled = len(tiler_shape) if is_tuple(tiler_shape) else 1

    for i, (s, d) in enumerate(zip(divided.shape, divided.stride)):
        if i < num_tiled:
            if is_tuple(s) and len(s) == 2:
                tile_s, rest_s = s
                tile_d, rest_d = d
                tile_shapes.append(tile_s)
                tile_strides.append(tile_d)
                rest_shapes.append(rest_s)
                rest_strides.append(rest_d)
            else:
                tile_shapes.append(s)
                tile_strides.append(d)
        else:
            rest_shapes.append(s)
            rest_strides.append(d)

    return tile_shapes, tile_strides, rest_shapes, rest_strides


def _unpack_grouped_mode(grouped_mode: "Layout") -> list:
    """Return a grouped mode's members, or the mode itself if already scalar.

    CuTe's tiled/flat divide variants unpack the grouped modes of
    zipped_divide(). When the grouped mode is itself a scalar layout, it is
    left unchanged.
    """
    if is_tuple(grouped_mode.shape):
        return [mode(grouped_mode, i) for i in range(rank(grouped_mode))]
    return [grouped_mode]


def _layout_from_modes(modes: list) -> "Layout":
    """Build a layout from a sequence of mode layouts."""
    return Layout(
        as_shape([m.shape for m in modes]),
        as_shape([m.stride for m in modes]),
    )


def zipped_divide(layout: "LayoutExpr", tiler: Any) -> "LayoutExpr":
    """Divide a layout and zip the tile/rest modes together.

    Result structure: ((TileM, TileN), (RestM, RestN, L, ...))
    - Mode 0: all tile shapes zipped together
    - Mode 1: all rest shapes and undivided modes zipped together

    This is useful when you want to iterate over all tile coordinates together,
    then all rest/tile-index coordinates together.

    Args:
        layout: The layout to tile
        tiler: The tile shape (int, tuple, or Layout)

    Returns:
        A rank-2 Layout with ((tiles), (rests)) structure

    Examples:
        zipped_divide(Layout((4,8)), (2,4)) -> Layout(((2,4),(2,2)), ((1,4),(2,16)))
    """
    forwarded = _forward_layout_domain(layout, lambda inner: zipped_divide(inner, tiler))
    if forwarded is not _NO_FORWARD:
        return forwarded
    # True Layout tilers are terminals in CuTe's tile_unzip(). Preserve their
    # stride semantics instead of reducing them to tiler.shape.
    if isinstance(tiler, Layout):
        return logical_divide(layout, tiler)

    tile_shapes, tile_strides, rest_shapes, rest_strides = _split_divided_modes(layout, tiler)

    tiles_shape = as_shape(tile_shapes)
    tiles_stride = as_shape(tile_strides)

    if len(rest_shapes) == 0:
        rests_shape = 1
        rests_stride = 0
    else:
        rests_shape = as_shape(rest_shapes)
        rests_stride = as_shape(rest_strides)

    return Layout((tiles_shape, rests_shape), (tiles_stride, rests_stride))


def tiled_divide(layout: "LayoutExpr", tiler: Any) -> "LayoutExpr":
    """Divide a layout into tiles and tile indices.

    Result structure: ((TileM, TileN), RestM, RestN, L, ...)
    - Mode 0: all tile shapes grouped together
    - Modes 1+: individual rest shapes and undivided modes (flat)

    Args:
        layout: The layout to tile
        tiler: The tile shape (int, tuple, or Layout)

    Returns:
        A Layout with ((tiles), rest0, rest1, ...) structure

    Examples:
        tiled_divide(Layout((8,8)), (2,2)) -> Layout(((2,2), 4, 4), ...)
    """
    forwarded = _forward_layout_domain(layout, lambda inner: tiled_divide(inner, tiler))
    if forwarded is not _NO_FORWARD:
        return forwarded
    result = zipped_divide(layout, tiler)
    modes = [mode(result, 0)]
    modes.extend(_unpack_grouped_mode(mode(result, 1)))
    return _layout_from_modes(modes)


def flat_divide(layout: "LayoutExpr", tiler: Any) -> "LayoutExpr":
    """Divide a layout and flatten all modes.

    Result structure: (TileM, TileN, RestM, RestN, L, ...)
    - All tile shapes come first (flat)
    - Then all rest shapes (flat)
    - Then any undivided modes (flat)

    Args:
        layout: The layout to tile
        tiler: The tile shape (int, tuple, or Layout)

    Returns:
        A flat Layout with (tile0, tile1, ..., rest0, rest1, ..., L, ...) structure

    Examples:
        flat_divide(Layout((8,8)), (2,2)) -> Layout((2, 2, 4, 4), ...)
    """
    forwarded = _forward_layout_domain(layout, lambda inner: flat_divide(inner, tiler))
    if forwarded is not _NO_FORWARD:
        return forwarded
    result = zipped_divide(layout, tiler)
    modes = _unpack_grouped_mode(mode(result, 0))
    modes.extend(_unpack_grouped_mode(mode(result, 1)))
    return _layout_from_modes(modes)


# =============================================================================
# Product operations
# =============================================================================
#
# Product reproduces a layout across copies:
#   logical_product(A, B) = Layout(A, compose(complement(A), B))
#
# The result has A's pattern repeated at each position B describes.
# Product answers: "how do I replicate this pattern across B positions?"
#
# The zipped/tiled variants mirror their division counterparts:
#   zipped_product -> ((A-modes), (product-modes))
#   tiled_product  -> ((A-modes), rest0, rest1, ..)
#
# blocked_product interleaves modes: ((A0,B0), (A1,B1), ...) with B's strides
# scaled by cosize(A), placing each copy at a non-overlapping block.
#


def zipped_product(layout_a: "LayoutExpr", layout_b) -> "LayoutExpr":
    """Apply logical_product hierarchically and gather split modes into two modes.

    Like zipped_divide but uses logical_product instead of logical_divide.

    Args:
        layout_a: The layout to reproduce
        layout_b: The reproduction specification

    Returns:
        A rank-2 Layout with ((A-modes), (product-modes)) structure
    """
    forwarded = _forward_layout_domain(layout_a, lambda inner: zipped_product(inner, layout_b))
    if forwarded is not _NO_FORWARD:
        return forwarded
    return hier_unzip(logical_product, layout_a, layout_b)


def tiled_product(layout_a: "LayoutExpr", layout_b) -> "LayoutExpr":
    """Apply logical_product hierarchically and flatten the second mode.

    Like tiled_divide but uses logical_product instead of logical_divide.

    Args:
        layout_a: The layout to reproduce
        layout_b: The reproduction specification

    Returns:
        A Layout with ((A-modes), rest0, rest1, ...) structure
    """
    forwarded = _forward_layout_domain(layout_a, lambda inner: tiled_product(inner, layout_b))
    if forwarded is not _NO_FORWARD:
        return forwarded
    result = zipped_product(layout_a, layout_b)
    second = mode(result, 1)
    all_modes = [mode(result, 0)]
    if is_tuple(second.shape) and not is_scalar(second.shape):
        for i in range(rank(second)):
            all_modes.append(mode(second, i))
    else:
        all_modes.append(second)

    shapes = tuple(m.shape for m in all_modes)
    strides = tuple(m.stride for m in all_modes)
    return Layout(shapes, strides)


def hier_unzip(splitter, layout_a: "Layout", layout_b) -> "Layout":
    """Apply a splitter hierarchically and gather the split modes into two modes.

    This is the generic helper behind zipped_divide, zipped_product, etc.
    The splitter function (e.g., logical_divide or logical_product) is applied
    recursively through the modes of layout_b. The results are then gathered
    into a rank-2 layout: ((gathered-A-modes), (gathered-rest-modes)).

    Args:
        splitter: A function (layoutA, layoutB) -> rank-2 Layout
        layout_a: The layout to split
        layout_b: The splitting specification (Layout, tuple, int, or None)

    Returns:
        A rank-2 Layout with split modes gathered

    Examples:
        hier_unzip(logical_divide, Layout((4,8)), (2, 4))
        -> ((2,4),(2,2)):((1,4),(2,16))
    """
    if layout_b is None:
        return Layout(
            (1, layout_a.shape),
            (0, layout_a.stride),
        )

    if is_tuple(layout_b) and not isinstance(layout_b, Layout):
        if rank(layout_a) < len(layout_b):
            raise LayoutError(f"layout_a rank ({rank(layout_a)}) < tiler length ({len(layout_b)})")

        splits = [
            hier_unzip(splitter, mode(layout_a, i), layout_b[i]) for i in range(len(layout_b))
        ]

        first_shapes = [mode(s, 0).shape for s in splits]
        first_strides = [mode(s, 0).stride for s in splits]
        second_shapes = [mode(s, 1).shape for s in splits]
        second_strides = [mode(s, 1).stride for s in splits]

        for i in range(len(layout_b), rank(layout_a)):
            m = mode(layout_a, i)
            second_shapes.append(m.shape)
            second_strides.append(m.stride)

        return Layout(
            (as_shape(first_shapes), as_shape(second_shapes)),
            (as_shape(first_strides), as_shape(second_strides)),
        )

    if isinstance(layout_b, int):
        layout_b = Layout(layout_b)
    return splitter(layout_a, layout_b)


def logical_product(layout_a: "LayoutExpr", layout_b: "Layout") -> "LayoutExpr":
    """Reproduce layout A's pattern at each position B describes.

    Product is the reverse of division. If division splits A into tiles,
    product replicates A across B copies. The result has A's pattern repeated
    at non-overlapping memory locations determined by B.

    Formally: logical_product(A, B) = Layout(A, compose(complement(A, size(A)*cosize(B)), B))

    For multi-mode tilers (tuples), the operation is applied mode-by-mode.

    Args:
        layout_a: First layout
        layout_b: Second layout (or int or tuple of tilers)

    Returns:
        Layout combining both inputs

    Examples:
        logical_product(Layout(4,1), Layout(3,1)) -> Layout((4,3), (1,4))
    """
    forwarded = _forward_layout_domain(layout_a, lambda inner: logical_product(inner, layout_b))
    if forwarded is not _NO_FORWARD:
        return forwarded
    _reject_swizzle_inner_composed(layout_a, "logical_product")
    if layout_b is None:
        return layout_a
    if isinstance(layout_b, int):
        return logical_product(layout_a, Layout(layout_b))

    # For tuple tilers, apply mode-by-mode
    if is_tuple(layout_b) and not isinstance(layout_b, Layout):
        if rank(layout_a) < len(layout_b):
            raise LayoutError(f"layout_a rank ({rank(layout_a)}) < tiler length ({len(layout_b)})")
        result_modes = []
        for i in range(len(layout_b)):
            result_modes.append(logical_product(mode(layout_a, i), layout_b[i]))
        # Append remaining modes unchanged
        for i in range(len(layout_b), rank(layout_a)):
            result_modes.append(mode(layout_a, i))
        shapes = tuple(r.shape for r in result_modes)
        strides = tuple(r.stride for r in result_modes)
        return Layout(shapes, strides)

    # Swizzled-tile fast path: when layout_b is ComposedLayout(Swizzle, inner)
    # with zero offset, do the affine product on the inner first and then
    # transfer the swizzle to the new strides. The generic fallback below
    # would silently drop the swizzle (composing it through the complement
    # produces a Layout with embedded swizzle, but the final tuple-Layout
    # constructor doesn't carry it). Mirrors CuTe C++'s logical_product
    # specialization in cute/swizzle_layout.hpp:549-587.
    if (
        isinstance(layout_b, ComposedLayout)
        and isinstance(layout_b.outer, Swizzle)
        and layout_b.offset == 0
        and isinstance(layout_b.inner, Layout)
    ):
        return _logical_product_with_swizzled_tile(layout_a, layout_b)

    # CuTe definition:
    # logical_product(A, B) = Layout(A, compose(complement(A, size(A)*cosize(B)), B))
    comp = complement(layout_a, size(layout_a) * cosize(layout_b))
    composed = compose(comp, layout_b)

    # Path X: ``Layout`` is purely affine and ``compose`` no longer produces
    # an embedded swizzle. The legacy ``swizzle=embedded_swizzle``
    # reattachment is gone; ``_logical_product_with_swizzled_tile`` above
    # already returns the correct ComposedLayout for the swizzled-tile path.
    return Layout(
        (layout_a.shape, composed.shape),
        (layout_a.stride, composed.stride),
    )


def _logical_product_with_swizzled_tile(layout_a: "Layout", tile: "ComposedLayout") -> "LayoutExpr":
    """Compute logical_product(layout_a, ComposedLayout(Swizzle, inner)) by
    doing the affine product first and then transferring the swizzle to the
    new strides.

    The new swizzle's masks are derived by passing the original swizzle's
    active YZ bits through the inner tile and the new product layout, exactly
    as CuTe C++ does in cute/swizzle_layout.hpp:549-587. When the resulting
    masks don't form a representable Swizzle, fall back to wrapping the
    affine product in a ComposedLayout so the function still has the right
    semantics (matches the defensive fallback in _compose_with_swizzle_rhs).
    """
    swizzle = tile.outer
    tile_inner = tile.inner

    # Affine product on the inner tile.
    new_layout = logical_product(layout_a, tile_inner)

    # OR-walk the YZ projection of the inner tile's image to get the bits
    # that the swizzle would interact with. Same idea as the slice-decay
    # reducibility check; cheap at our typical sizes.
    yz_mask = swizzle.yyy_msk | swizzle.zzz_msk
    active_bits = 0
    n = size(tile_inner)
    for i in range(n):
        active_bits |= tile_inner(i) & yz_mask
    active_Y = active_bits & swizzle.yyy_msk
    active_Z = active_bits & swizzle.zzz_msk

    # Transfer the active bits through tile_inner and then through new_layout
    # at coord (0, *) so that the new active masks reflect the strides the
    # swizzle now needs to act on.
    new_active_Y = new_layout((0, tile_inner(active_Y)))
    new_active_Z = new_layout((0, tile_inner(active_Z)))

    try:
        new_swizzle = make_swizzle(new_active_Y, new_active_Z)
    except ValueError:
        return ComposedLayout(swizzle, new_layout)
    if new_swizzle is None:
        return new_layout
    return compose(new_swizzle, new_layout)


def _product_interleave(layout_a: "Layout", layout_b: "Layout") -> "Layout":
    """Interleave modes of two layouts, scaling B's strides by cosize(A).

    For each mode i: shape = (A_shape[i], B_shape[i]),
                      stride = (A_stride[i], B_stride[i] * cosize(A))

    Used by both logical_product (for rank > 1) and blocked_product.
    """
    a_cosize_val = cosize(layout_a)
    a_rank = rank(layout_a)
    b_rank = rank(layout_b)
    max_rank = max(a_rank, b_rank)

    result_shapes = []
    result_strides = []

    for i in range(max_rank):
        if i < a_rank and i < b_rank:
            a_s_val = mode(layout_a.shape, i)
            a_st_val = mode(layout_a.stride, i)
            b_s_val = mode(layout_b.shape, i)
            b_st_val = mode(layout_b.stride, i)
            b_st_scaled = transform_tuple(b_st_val, lambda s: s * a_cosize_val)
            result_shapes.append((a_s_val, b_s_val))
            result_strides.append((a_st_val, b_st_scaled))
        elif i < a_rank:
            a_s_val = mode(layout_a.shape, i)
            a_st_val = mode(layout_a.stride, i)
            result_shapes.append(a_s_val)
            result_strides.append(a_st_val)
        else:
            b_s_val = mode(layout_b.shape, i)
            b_st_val = mode(layout_b.stride, i)
            b_st_scaled = transform_tuple(b_st_val, lambda s: s * a_cosize_val)
            result_shapes.append(b_s_val)
            result_strides.append(b_st_scaled)

    return Layout(tuple(result_shapes), tuple(result_strides))


def blocked_product(layout_a: "LayoutExpr", layout_b: "Layout") -> "LayoutExpr":
    """Compute a blocked product of two layouts.

    Unlike logical_product which concatenates (A, B) for 1D, blocked_product
    always interleaves corresponding modes: ((A0, B0), (A1, B1), ...).

    A varies fastest (block-first): each block is contiguous, with blocks
    laid out according to B.  Think of A as the "block pattern" and B as
    the "grid of blocks."

    Compare with raked_product: both interleave, but raked has B vary
    fastest (rake-first), while blocked has A vary fastest (block-first).

    For each mode i:
        result_shape[i] = (A_shape[i], B_shape[i])
        result_stride[i] = (A_stride[i], B_stride[i] * cosize(A))

    Args:
        layout_a: First layout (the "inner" or "block" pattern)
        layout_b: Second layout (the "outer" or "tile count" pattern)

    Returns:
        Layout with blocked structure where modes are interleaved

    Examples:
        blocked_product((2,2):(1,2), (2,2):(1,2)) -> ((2,2),(2,2)):((1,4),(2,8))
    """
    forwarded = _forward_layout_domain(layout_a, lambda inner: blocked_product(inner, layout_b))
    if forwarded is not _NO_FORWARD:
        return forwarded
    a_cosize_val = cosize(layout_a)
    a_rank = rank(layout_a)
    b_rank = rank(layout_b)

    # Handle scalar layouts (rank 0)
    if a_rank == 0 and b_rank == 0:
        # Both scalar: create 2D result
        # Result should be (a_size, b_size) with proper strides
        new_shape = (size(layout_a), size(layout_b))
        new_stride = (layout_a.stride, layout_b.stride * a_cosize_val)
        return Layout(new_shape, new_stride)
    if a_rank == 0:
        # Scalar a with non-scalar b:
        # Pair scalar a with first mode of b, append remaining modes
        b_shapes = as_list(layout_b.shape)
        b_strides = as_list(layout_b.stride)

        # First mode: pair (a_size, b[0])
        result_shapes = [(size(layout_a), b_shapes[0])]
        result_strides = [(layout_a.stride, b_strides[0] * a_cosize_val)]

        # Remaining modes: scale strides by a_cosize
        for i in range(1, len(b_shapes)):
            result_shapes.append(b_shapes[i])
            result_strides.append(b_strides[i] * a_cosize_val)

        return Layout(tuple(result_shapes), tuple(result_strides))
    if b_rank == 0:
        # Non-scalar a with scalar b:
        # Pair b with first mode of a, keep remaining modes
        a_shapes = as_list(layout_a.shape)
        a_strides = as_list(layout_a.stride)

        # First mode: pair (a[0], b_size)
        result_shapes = [(a_shapes[0], size(layout_b))]
        result_strides = [(a_strides[0], layout_b.stride * a_cosize_val)]

        # Remaining modes: unchanged
        for i in range(1, len(a_shapes)):
            result_shapes.append(a_shapes[i])
            result_strides.append(a_strides[i])

        return Layout(tuple(result_shapes), tuple(result_strides))

    return _product_interleave(layout_a, layout_b)


def _pad_to_rank(layout: "Layout", target_rank: int) -> "Layout":
    """Pad a layout to a target rank by appending (1, 0) modes.

    Matches C++ CuTe's append<R>(layout) which pads with Layout<_1,_0>{}.
    """
    current_rank = rank(layout)
    if current_rank >= target_rank:
        return layout
    shapes = as_list(layout.shape)
    strides = as_list(layout.stride)
    for _ in range(target_rank - current_rank):
        shapes.append(1)
        strides.append(0)
    return Layout(tuple(shapes), tuple(strides))


def _zip_layouts(layout_a: "Layout", layout_b: "Layout") -> "Layout":
    """Zip two layouts mode-by-mode: ((a0,b0), (a1,b1), ...).

    Matches C++ CuTe's zip(layoutA, layoutB) which interleaves corresponding
    modes into paired tuples.

    Both layouts must have the same rank.
    """
    a_rank = rank(layout_a)
    b_rank = rank(layout_b)

    # Handle scalar layouts by treating them as rank-1
    if a_rank == 0 and b_rank == 0:
        # Both scalar: create a single mode with paired shapes/strides
        return Layout((layout_a.shape, layout_b.shape), (layout_a.stride, layout_b.stride))

    if a_rank != b_rank:
        raise LayoutError(f"Rank mismatch in zip: {a_rank} vs {b_rank}")
    r = a_rank
    result_shapes = []
    result_strides = []
    for i in range(r):
        a_s = mode(layout_a.shape, i)
        a_d = mode(layout_a.stride, i)
        b_s = mode(layout_b.shape, i)
        b_d = mode(layout_b.stride, i)
        result_shapes.append((a_s, b_s))
        result_strides.append((a_d, b_d))
    return Layout(tuple(result_shapes), tuple(result_strides))


def flat_product(block: "LayoutExpr", tiler) -> "LayoutExpr":
    """Compute a flat product: zipped_product with both modes unpacked.

    Like zipped_product, but flattens both the block modes and the product
    modes into a single flat layout: (BLK_0, BLK_1, ..., tiler_0, tiler_1, ...).

    Args:
        block: The block layout to reproduce
        tiler: The reproduction specification

    Returns:
        A flat Layout with all block modes followed by all product modes

    Examples:
        flat_product(Layout((2,4), (1,2)), Layout(3,1))
            -> Layout with shape (2, 4, 3, ...) and appropriate strides
    """
    forwarded = _forward_layout_domain(block, lambda inner: flat_product(inner, tiler))
    if forwarded is not _NO_FORWARD:
        return forwarded
    result = zipped_product(block, tiler)

    # Unpack both modes: result(repeat<R0>(_), repeat<R1>(_))
    # which is equivalent to flattening both the block mode and product mode
    m0 = mode(result, 0)
    m1 = mode(result, 1)

    shapes = []
    strides = []

    # Unpack mode 0 (block modes)
    if is_tuple(m0.shape):
        for i in range(rank(m0)):
            shapes.append(mode(m0.shape, i))
            strides.append(mode(m0.stride, i))
    else:
        shapes.append(m0.shape)
        strides.append(m0.stride)

    # Unpack mode 1 (product modes)
    if is_tuple(m1.shape):
        for i in range(rank(m1)):
            shapes.append(mode(m1.shape, i))
            strides.append(mode(m1.stride, i))
    else:
        shapes.append(m1.shape)
        strides.append(m1.stride)

    return Layout(tuple(shapes), tuple(strides))


def raked_product(block: "LayoutExpr", tiler: "Layout") -> "LayoutExpr":
    """Compute a raked product: block-interleaved reproduction.

    Like blocked_product, but with the tiler varying fastest within each mode.
    Where blocked_product zips as ((block, tiler), ...), raked_product zips as
    ((tiler, block), ...) — the tiler's elements are interleaved *within* each
    block, rather than the block appearing contiguously.

    This is useful for distributing work across threads where you want each
    thread to access interleaved (raked) elements rather than contiguous blocks.

    Algorithm: pad both layouts to the same rank, compute logical_product,
    then zip with reversed order: zip(product_modes, block_modes).

    Args:
        block: The block layout
        tiler: The tiler layout

    Returns:
        A Layout with interleaved (tiler, block) structure per mode

    Examples:
        raked_product(Layout((2,2), (1,2)), Layout((2,2), (1,2)))
            -> ((2,2),(2,2)):((4,1),(8,2))
        # Compare with blocked_product which gives:
            -> ((2,2),(2,2)):((1,4),(2,8))
    """
    forwarded = _forward_layout_domain(block, lambda inner: raked_product(inner, tiler))
    if forwarded is not _NO_FORWARD:
        return forwarded
    r = max(rank(block), rank(tiler))
    padded_block = _pad_to_rank(block, r)
    padded_tiler = _pad_to_rank(tiler, r)

    result = logical_product(padded_block, padded_tiler)

    # result is rank-2: (block_modes, product_modes)
    # For raked: zip(product_modes, block_modes) — reversed from blocked
    m0 = mode(result, 0)  # block modes
    m1 = mode(result, 1)  # product modes

    return _zip_layouts(m1, m0)
