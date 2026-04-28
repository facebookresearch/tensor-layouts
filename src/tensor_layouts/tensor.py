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

"""Tensor class: combines a Layout with a base offset and optional storage.

In CuTe, a Tensor is (Engine/Pointer, Layout). Here we represent the pointer
as an integer offset and optionally attach storage as a flat 1D backing buffer
such as a list, 1D numpy array, or 1D torch tensor.

When storage is present:
  - ``tensor[i, j]`` returns the data element at the computed offset
  - ``tensor[i, j] = val`` writes to that position
  - ``draw_layout(tensor)`` auto-labels cells with data values
  - Slicing produces sub-Tensors that share the same storage (view semantics)

When storage is absent, the Tensor is purely algebraic and ``tensor[i, j]``
returns the integer offset (the original behavior).

IMPORTANT: For swizzled layouts, the base offset is a *linear* offset that
gets added to the linear coordinate-to-offset result BEFORE swizzling.
This is critical for correct slicing behavior: slicing accumulates linear
offset, and the swizzle is applied at the final call.
"""

from .layouts import *


def _tensor_address(offset: int, layout: LayoutExpr, coords) -> int:
    """Resolve a tensor address from an external offset and a layout expression."""
    if isinstance(layout, Layout):
        linear_result = crd2offset(coords, layout.shape, layout.stride)
        total_linear = offset + linear_result
        if layout.swizzle is not None:
            return layout.swizzle(total_linear)
        return total_linear
    return offset + layout(coords)


def _linear_offset_bounds(shape, stride) -> tuple[int, int]:
    """Return the min/max linear offsets reachable by a shape/stride pair."""
    if is_tuple(shape):
        min_offset = 0
        max_offset = 0
        for sub_shape, sub_stride in zip(shape, stride):
            sub_min, sub_max = _linear_offset_bounds(sub_shape, sub_stride)
            min_offset += sub_min
            max_offset += sub_max
        return min_offset, max_offset

    extent = (shape - 1) * stride
    if extent < 0:
        return extent, 0
    return 0, extent


def _address_bounds(layout: LayoutExpr, offset: int) -> tuple[int, int]:
    """Return the min/max storage indices addressed by a Tensor."""
    if is_affine(layout) and layout.swizzle is None:
        min_linear, max_linear = _linear_offset_bounds(layout.shape, layout.stride)
        return offset + min_linear, offset + max_linear

    min_offset = None
    max_offset = None
    for flat_idx in range(size(layout)):
        actual = _tensor_address(offset, layout, flat_idx)
        if min_offset is None:
            min_offset = actual
            max_offset = actual
        else:
            min_offset = min(min_offset, actual)
            max_offset = max(max_offset, actual)

    if min_offset is None:
        return offset, offset
    return min_offset, max_offset


def _storage_rank(data) -> int | None:
    """Return the advertised rank of an array/tensor-like storage object."""
    ndim = getattr(data, "ndim", None)
    if ndim is not None:
        try:
            return int(ndim)
        except (TypeError, ValueError):
            pass

    shape = getattr(data, "shape", None)
    if shape is not None:
        try:
            return len(shape)
        except TypeError:
            pass

    return None


def _validate_storage(layout: LayoutExpr, offset: int, data) -> None:
    """Validate that storage covers every index addressed by (offset, layout)."""
    storage_rank = _storage_rank(data)
    if storage_rank is not None and storage_rank != 1:
        raise TypeError(
            "Tensor data must be flat 1D storage; "
            f"got {type(data).__name__} with rank {storage_rank}"
        )
    min_offset, max_offset = _address_bounds(layout, offset)
    if min_offset < 0 or max_offset >= len(data):
        raise ValueError(
            f"Storage length {len(data)} does not cover addressed range "
            f"[{min_offset}, {max_offset}] for offset={offset} and layout {layout}"
        )


class Tensor:
    """A Tensor combines a base offset and optional storage with a Layout.

    In CuTe C++, a Tensor is (Pointer, Layout) where the pointer holds the
    base address into memory.  Here we model the pointer as an integer offset
    and optionally attach a storage object so elements can be read and written
    through logical coordinates.

    Tensor supports slicing: fixing coordinates accumulates their offset
    contribution and returns a new Tensor with a reduced Layout.  Sub-Tensors
    produced by slicing share the parent's storage (view semantics).

    IMPORTANT: For swizzled layouts, the offset is a *linear* offset.
    The swizzle is applied to (linear_offset + linear_layout_result) at
    the final __call__. This ensures slicing works correctly:

        tensor(i, j) = swizzle(offset + crd2offset((i, j), shape, stride))
        tensor[i, :](j) = swizzle(offset + i*stride_i + j*stride_j)

    NOT: offset + swizzle(crd2offset(...)) which would be incorrect.

    Args:
        layout: The Layout describing the coordinate-to-offset mapping
        offset: The base offset in linear (pre-swizzle) space (default 0)
        data:   Optional flat 1D backing storage such as a list, 1D numpy
                array, or 1D torch tensor. Array/tensor-like objects that
                advertise rank via ``ndim`` or ``shape`` must be rank 1.
                It must cover every index addressed by ``offset + layout(coords)``
                (or the swizzled equivalent). For zero-offset, nonnegative-stride
                layouts this reduces to ``len(data) >= cosize(layout)``.
                Stored by reference (no copy).

    Examples:
        Algebraic (no storage)::

            t = Tensor(Layout((4, 8), (8, 1)))
            t(3, 5)   # 29  — memory offset
            t[3, :]   # Tensor with row 3's offset and column layout
            t[3, 5]   # 29  — all modes fixed, returns int offset

        With storage::

            t = Tensor(Layout((4, 8), (8, 1)), data=list(range(32)))
            t[2, 3]        # data[2*8 + 3] = data[19] → 19
            t[2, :][3]     # same result via slicing
            t[0, 0] = 99   # writes data[0] = 99
            t.data = list(range(100, 132))  # swap storage
    """

    def __init__(self, layout: LayoutExpr, offset: int = 0, data=None):
        self._layout = as_layout_expr(layout)
        self._offset = offset
        if data is not None:
            _validate_storage(self._layout, self._offset, data)
        self._data = data

    @property
    def layout(self) -> LayoutExpr:
        return self._layout

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def shape(self):
        return self._layout.shape

    @property
    def stride(self):
        """The stride of the underlying affine layout.

        Raises:
            TypeError: If the layout is a ComposedLayout (non-affine).
                       Check ``is_affine(tensor)`` first, or use the
                       ``layout`` property to access the full layout
                       expression.
        """
        return as_affine_layout(self._layout).stride

    @property
    def data(self):
        """The backing storage, or None if this is a purely algebraic Tensor.

        Assignable: ``tensor.data = new_array`` replaces the storage reference.
        The new storage must cover every index addressed by this Tensor.
        Array/tensor-like objects that advertise ``ndim`` or ``shape`` must
        be flat rank-1 storage.

        Note: sub-Tensors produced by slicing hold their own reference to the
        storage object.  Reassigning ``parent.data`` does *not* update existing
        sub-Tensors (they keep the old reference).  This matches numpy/torch
        view semantics.
        """
        return self._data

    @data.setter
    def data(self, value):
        if value is not None:
            _validate_storage(self._layout, self._offset, value)
        self._data = value

    def view(self, layout: LayoutExpr) -> "Tensor":
        """Return a new Tensor sharing this storage with a different layout.

        The new view preserves this Tensor's base offset and must satisfy
        the same addressed-range invariant as any data-backed Tensor.
        """
        if self._data is None:
            raise TypeError("Cannot create a view of a Tensor with no storage")
        return Tensor(layout, offset=self._offset, data=self._data)

    def __repr__(self) -> str:
        if self._offset:
            return f"Tensor({self._layout}, offset={self._offset})"
        return f"Tensor({self._layout})"

    def __str__(self) -> str:
        return f"{{{self._offset}}} ∘ {self._layout}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return False
        if self._layout != other._layout or self._offset != other._offset:
            return False
        if self._data is None and other._data is None:
            return True
        if self._data is None or other._data is None:
            return False
        if self._data is other._data:
            return True
        return len(self._data) == len(other._data) and all(
            a == b for a, b in zip(self._data, other._data)
        )

    def __hash__(self):
        return hash((self._layout, self._offset))

    def __call__(self, *args) -> int:
        """Map coordinates to a memory offset.

        The tensor offset is external pointer/base state. For affine layouts
        with an embedded swizzle, that external offset participates in the
        pre-swizzle linear address exactly as CuTe tensor slicing requires.
        For ComposedLayout, the layout's own preoffset remains internal and the
        tensor offset is added only after the layout expression evaluates.
        """
        coords = args[0] if len(args) == 1 else args
        return _tensor_address(self._offset, self._layout, coords)

    def __getitem__(self, key):
        """Access a data element or slice the tensor.

        A bare integer performs **flat 1D evaluation**: the index is
        decomposed via ``idx2crd`` into the natural coordinate, and the
        offset is computed.  This matches CuTe C++ ``Tensor::operator()(int)``
        and is consistent with ``__setitem__``, enabling the canonical
        copy loop ``dst[i] = src[i]`` on any-rank tensor.

        A bare full slice (``tensor[:]``) returns a view of the whole tensor.

        A tuple key performs **slicing**: fixed coordinates (ints) contribute
        to the offset; free coordinates (``:`` / ``None``) remain in the
        resulting sub-Tensor's layout.

        Examples:
            tensor[5]     -- flat 1D evaluation → data element or offset
            tensor[:]     -- full slice → Tensor view of the whole tensor
            tensor[2, 3]  -- fix all modes → data element or offset
            tensor[3, :]  -- fix mode 0 to 3, keep mode 1 → sub-Tensor
            tensor[:, 5]  -- keep mode 0, fix mode 1 to 5 → sub-Tensor
        """
        if isinstance(key, tuple):
            return self._slice_multi(key)
        elif isinstance(key, int):
            # Flat 1D evaluation — matches CuTe C++ and __setitem__
            offset = self(key)
            if self._data is not None:
                return self._data[offset]
            return offset
        elif isinstance(key, slice) and key == slice(None):
            return Tensor(self._layout, self._offset, data=self._data)
        else:
            raise TypeError(f"Invalid tensor index: {key!r}")

    def __setitem__(self, key, value):
        """Write a value to storage at the given coordinates.

        Only supports fully-fixed coordinates.  Free coordinates such as
        ``:`` or ``None`` are rejected; to write through a slice, first
        create the slice and then index that sub-Tensor.

        For hierarchical modes, a fully-fixed coordinate may be a tuple of
        integers for that mode.

        Examples:
            tensor[2, 3] = 42   # writes data[tensor(2, 3)] = 42
            tensor[5] = 99      # rank-1 tensor
            tensor[2, :][3] = 7  # write through a sliced sub-Tensor
        """
        if self._data is None:
            raise TypeError("Cannot assign to a Tensor with no storage")
        if self._contains_free_coordinates(key):
            raise TypeError(
                "Tensor assignment requires fully-fixed coordinates (no ':' or None in the key)"
            )
        if isinstance(key, tuple):
            if len(key) != rank(self._layout):
                raise IndexError(
                    f"Expected {rank(self._layout)} indices, got {len(key)}"
                )
            offset = _tensor_address(self._offset, self._layout, key)
        elif isinstance(key, int):
            offset = _tensor_address(self._offset, self._layout, key)
        else:
            raise TypeError(
                "Tensor assignment requires a fully-fixed coordinate: "
                "use an int for flat indexing or a tuple with one fixed "
                "coordinate per mode"
            )
        self._data[offset] = value

    def _slice_multi(self, keys: tuple) -> "Tensor | int":
        """Handle multi-dimensional slicing like tensor[i, :] or tensor[i, ((0, None), None)].

        Supports three kinds of per-mode keys:
          - int or tuple of ints: fix the mode to that coordinate
          - None or slice(None): keep the entire mode free
          - tuple containing None(s): partial hierarchical slice
        """
        if len(keys) != rank(self._layout):
            raise IndexError(f"Expected {rank(self._layout)} indices, got {len(keys)}")

        normalized = []
        has_free = False
        for i, key in enumerate(keys):
            if key is None or (isinstance(key, slice) and key == slice(None)):
                normalized.append(None)
                has_free = True
            elif isinstance(key, (int, tuple)):
                normalized.append(key)
                if self._has_nested_none(key):
                    has_free = True
            else:
                raise TypeError(f"Invalid slice key at position {i}: {key}")
        normalized = tuple(normalized)

        if not has_free:
            # All modes fixed - return data element or computed offset
            offset = _tensor_address(self._offset, self._layout, normalized)
            if self._data is not None:
                return self._data[offset]
            return offset

        sub, offset = slice_and_offset(normalized, self._layout)
        if isinstance(sub, Layout):
            sub = Layout(unwrap(sub.shape), unwrap(sub.stride), swizzle=sub.swizzle)
        return Tensor(sub, self._offset + offset, data=self._data)

    @staticmethod
    def _has_nested_none(key) -> bool:
        """Check if a key contains None inside a nested tuple."""
        if not isinstance(key, tuple):
            return False
        for item in key:
            if item is None:
                return True
            if isinstance(item, tuple) and Tensor._has_nested_none(item):
                return True
        return False

    @staticmethod
    def _contains_free_coordinates(key) -> bool:
        """Return True when a key contains slicing markers.

        Assignment only accepts fully-fixed coordinates, so ``None`` and any
        ``slice`` object are rejected wherever they appear, including inside
        hierarchical coordinate tuples.
        """
        if key is None or isinstance(key, slice):
            return True
        if isinstance(key, tuple):
            return any(Tensor._contains_free_coordinates(item) for item in key)
        return False
