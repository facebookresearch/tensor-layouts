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

"""Tests for the Tensor class.

Tensor combines a Layout with a base offset, analogous to CuTe's (Pointer, Layout).
Slicing a Tensor fixes coordinates, accumulating their offset contribution.

These tests are designed to validate behavior against NVIDIA CuTe C++.
In CuTe, a Tensor is (Engine/Pointer, Layout) where:
  - Pointer provides the base address
  - Layout maps coordinates to offsets

Python Tensor uses integer offset instead of pointer, but the algebra is identical.
The key invariant: tensor[fixed_coords, :](remaining_coords) == tensor(all_coords)
"""

import pytest

from tensor_layouts import (
    Layout,
    Swizzle,
    compose,
    complement,
    logical_divide,
    logical_product,
    rank,
    size,
    cosize,
    mode,
    flatten,
    coalesce,
)
from tensor_layouts import Tensor


# ============================================================================
# Tensor construction and properties (adapted from CuTe make_tensor patterns)
# ============================================================================


def test_rank2_column_major():
    """Column-major 4x8 matrix - most common pattern."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    assert tensor.layout == layout
    assert tensor.offset == 0
    assert tensor.shape == (4, 8)
    assert tensor.stride == (1, 4)
    assert rank(tensor.layout) == 2
    assert size(tensor.layout) == 32


def test_rank2_row_major():
    """Row-major 4x8 matrix."""
    layout = Layout((4, 8), (8, 1))
    tensor = Tensor(layout)

    assert tensor.shape == (4, 8)
    assert tensor.stride == (8, 1)
    # Row-major: adjacent columns are contiguous
    assert tensor(0, 0) == 0
    assert tensor(0, 1) == 1
    assert tensor(1, 0) == 8


def test_rank1_contiguous():
    """Simple 1D contiguous layout."""
    layout = Layout(32, 1)
    tensor = Tensor(layout)

    # Single-mode layout has rank 1 (one mode)
    assert rank(tensor.layout) == 1
    assert size(tensor.layout) == 32
    for i in range(32):
        assert tensor(i) == i


def test_rank1_strided():
    """1D layout with stride > 1 (e.g., every other element)."""
    layout = Layout(16, 2)
    tensor = Tensor(layout)

    for i in range(16):
        assert tensor(i) == 2 * i


def test_rank3_tensor():
    """3D tensor like batch x height x width."""
    layout = Layout((2, 4, 8), (32, 8, 1))
    tensor = Tensor(layout)

    assert rank(tensor.layout) == 3
    assert size(tensor.layout) == 64
    # Verify indexing
    assert tensor(0, 0, 0) == 0
    assert tensor(1, 0, 0) == 32
    assert tensor(0, 1, 0) == 8
    assert tensor(0, 0, 1) == 1
    assert tensor(1, 2, 3) == 32 + 16 + 3


def test_with_offset():
    """Tensor with non-zero base offset (pointer arithmetic)."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout, offset=100)

    assert tensor.offset == 100
    # All coordinates shifted by offset
    assert tensor(0, 0) == 100
    assert tensor(1, 0) == 101
    assert tensor(0, 1) == 104
    assert tensor(3, 7) == 100 + 3 + 7 * 4


def test_hierarchical_shape():
    """Nested/hierarchical shape - key CuTe feature."""
    # Mode 0 has shape (2, 4), mode 1 has shape 8
    layout = Layout(((2, 4), 8), ((1, 2), 8))
    tensor = Tensor(layout)

    assert rank(tensor.layout) == 2
    # Mode 0 has nested shape
    mode0 = mode(tensor.layout, 0)
    assert mode0.shape == (2, 4)
    # Total size is product of all leaf shapes
    assert size(tensor.layout) == 2 * 4 * 8


# ============================================================================
# Tensor Equality
# ============================================================================


def test_equal_tensors():
    layout = Layout((4, 8), (1, 4))
    t1 = Tensor(layout)
    t2 = Tensor(layout)

    assert t1 == t2
    assert hash(t1) == hash(t2)


def test_different_offset():
    layout = Layout((4, 8), (1, 4))
    t1 = Tensor(layout)
    t2 = Tensor(layout, offset=1)

    assert t1 != t2


def test_different_layout():
    t1 = Tensor(Layout((4, 8), (1, 4)))
    t2 = Tensor(Layout((4, 8), (8, 1)))

    assert t1 != t2


def test_repr_no_offset():
    tensor = Tensor(Layout((4, 8), (1, 4)))
    r = repr(tensor)
    assert "Tensor" in r
    assert "offset" not in r


def test_repr_with_offset():
    tensor = Tensor(Layout((4, 8), (1, 4)), offset=42)
    r = repr(tensor)
    assert "offset=42" in r


def test_is_affine():
    from tensor_layouts import ComposedLayout, Swizzle, is_affine

    # Affine layout
    t1 = Tensor(Layout((4, 8), (1, 4)))
    assert is_affine(t1) is True

    # ComposedLayout is not affine
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), preoffset=4)
    t2 = Tensor(composed)
    assert is_affine(t2) is False


def test_stride_on_composed_raises():
    from tensor_layouts import ComposedLayout, Swizzle

    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), preoffset=4)
    t = Tensor(composed)
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        _ = t.stride


# ============================================================================
# Tensor calling: coordinate -> offset
# ============================================================================


def test_column_major_indexing():
    """Column-major: offset = i + j * num_rows."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    for i in range(4):
        for j in range(8):
            expected = i + j * 4
            assert tensor(i, j) == expected


def test_row_major_indexing():
    """Row-major: offset = i * num_cols + j."""
    layout = Layout((4, 8), (8, 1))
    tensor = Tensor(layout)

    for i in range(4):
        for j in range(8):
            expected = i * 8 + j
            assert tensor(i, j) == expected


def test_with_offset_indexing():
    """Offset shifts all results."""
    layout = Layout((4, 8), (1, 4))
    base_tensor = Tensor(layout)
    offset_tensor = Tensor(layout, offset=1000)

    for i in range(4):
        for j in range(8):
            assert offset_tensor(i, j) == 1000 + base_tensor(i, j)


def test_hierarchical_indexing():
    """Hierarchical coordinates like ((i0, i1), j)."""
    layout = Layout(((2, 4), 8), ((1, 2), 8))
    tensor = Tensor(layout)

    # Test various hierarchical coordinates
    assert tensor((0, 0), 0) == 0
    assert tensor((1, 0), 0) == 1
    assert tensor((0, 1), 0) == 2
    assert tensor((0, 0), 1) == 8


# ============================================================================
# Non-swizzled tensor slicing -- the basics: tensor(i, _) or tensor(_, j)
# ============================================================================


def test_fix_mode0_keep_mode1():
    """tensor[i, :] - fix row, keep column."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    for i in range(4):
        row = tensor[i, :]
        assert isinstance(row, Tensor)
        assert row.offset == i  # row i contributes offset i
        assert row.shape == 8
        assert row.stride == 4

        # Verify: row(j) == tensor(i, j)
        for j in range(8):
            assert row(j) == tensor(i, j)


def test_fix_mode1_keep_mode0():
    """tensor[:, j] - keep row, fix column."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    for j in range(8):
        col = tensor[:, j]
        assert isinstance(col, Tensor)
        assert col.offset == j * 4  # column j contributes offset j*4
        assert col.shape == 4
        assert col.stride == 1

        # Verify: col(i) == tensor(i, j)
        for i in range(4):
            assert col(i) == tensor(i, j)


def test_fix_all_modes():
    """tensor[i, j] - all fixed returns int."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    for i in range(4):
        for j in range(8):
            result = tensor[i, j]
            assert isinstance(result, int)
            assert result == tensor(i, j)


def test_single_index_flat_eval():
    """tensor[i] — flat 1D evaluation on any-rank tensor.

    Matches CuTe C++ Tensor::operator()(int): the integer is decomposed
    via idx2crd into the natural coordinate, then the offset is computed.
    """
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    # tensor[2] decomposes: idx2crd(2, (4,8)) = (2, 0) → offset 2
    result = tensor[2]
    assert isinstance(result, int)
    assert result == layout(2)

    # Verify for all flat indices
    for i in range(size(layout)):
        assert tensor[i] == layout(i)


# ============================================================================
# Tensor Slicing Row Major
# ============================================================================


def test_row_major_fix_row():
    """Row-major: fixing row gives contiguous column slice."""
    layout = Layout((4, 8), (8, 1))
    tensor = Tensor(layout)

    row = tensor[2, :]
    assert row.offset == 2 * 8  # = 16
    assert row.stride == 1  # columns are contiguous

    for j in range(8):
        assert row(j) == tensor(2, j)


def test_row_major_fix_column():
    """Row-major: fixing column gives strided row slice."""
    layout = Layout((4, 8), (8, 1))
    tensor = Tensor(layout)

    col = tensor[:, 5]
    assert col.offset == 5
    assert col.stride == 8  # rows are strided

    for i in range(4):
        assert col(i) == tensor(i, 5)


# ============================================================================
# Tensor Slicing Rank3
# ============================================================================


def test_fix_mode0():
    """Fix batch dimension."""
    layout = Layout((2, 4, 8), (32, 8, 1))
    tensor = Tensor(layout)

    slice0 = tensor[1, :, :]
    assert slice0.offset == 32
    assert slice0.shape == (4, 8)
    assert slice0.stride == (8, 1)

    for i in range(4):
        for j in range(8):
            assert slice0(i, j) == tensor(1, i, j)


def test_fix_mode1():
    """Fix height dimension."""
    layout = Layout((2, 4, 8), (32, 8, 1))
    tensor = Tensor(layout)

    slice1 = tensor[:, 2, :]
    assert slice1.offset == 16
    assert slice1.shape == (2, 8)

    for b in range(2):
        for w in range(8):
            assert slice1(b, w) == tensor(b, 2, w)


def test_fix_mode2():
    """Fix width dimension."""
    layout = Layout((2, 4, 8), (32, 8, 1))
    tensor = Tensor(layout)

    slice2 = tensor[:, :, 5]
    assert slice2.offset == 5
    assert slice2.shape == (2, 4)

    for b in range(2):
        for h in range(4):
            assert slice2(b, h) == tensor(b, h, 5)


def test_fix_two_modes():
    """Fix two of three modes."""
    layout = Layout((2, 4, 8), (32, 8, 1))
    tensor = Tensor(layout)

    slice01 = tensor[1, 2, :]
    assert slice01.offset == 32 + 16
    assert slice01.shape == 8

    for w in range(8):
        assert slice01(w) == tensor(1, 2, w)


# ============================================================================
# Tensor Slicing Accumulation
# ============================================================================


def test_sequential_slices():
    """Multiple sequential slices accumulate offset."""
    layout = Layout((4, 8, 2), (1, 4, 32))
    tensor = Tensor(layout)

    # First slice
    s1 = tensor[2, :, :]
    assert s1.offset == 2

    # Second slice
    s2 = s1[3, :]
    assert s2.offset == 2 + 3 * 4  # = 14

    # Third slice
    s3 = s2[1]
    assert s3 == 2 + 3 * 4 + 1 * 32  # = 46


def test_slice_with_initial_offset():
    """Slicing tensor with non-zero offset accumulates correctly."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout, offset=100)

    row = tensor[2, :]
    assert row.offset == 102

    for j in range(8):
        assert row(j) == 100 + layout(2, j)


def test_double_slice_equivalence():
    """tensor[i, :][j] == tensor[i, j]."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    for i in range(4):
        row = tensor[i, :]
        for j in range(8):
            assert row[j] == tensor[i, j]


# ============================================================================
# Tensor Slicing Hierarchical
# ============================================================================


def test_hierarchical_mode0():
    """Slice mode 0 with hierarchical coordinate."""
    layout = Layout(((2, 4), 8), ((1, 2), 8))
    tensor = Tensor(layout)

    # Fix mode 0 to coordinate (1, 2)
    slice0 = tensor[(1, 2), :]
    assert isinstance(slice0, Tensor)

    for j in range(8):
        assert slice0(j) == tensor((1, 2), j)


def test_hierarchical_mode1():
    """Slice mode 1 when mode 0 is hierarchical."""
    layout = Layout(((2, 4), 8), ((1, 2), 8))
    tensor = Tensor(layout)

    slice1 = tensor[:, 5]
    assert isinstance(slice1, Tensor)
    assert slice1.shape == (2, 4)

    for i0 in range(2):
        for i1 in range(4):
            assert slice1((i0, i1)) == tensor((i0, i1), 5)


# =============================================================================
# Swizzled Tensor Operations
# =============================================================================
# Critical for GPU shared memory bank conflict avoidance


# ============================================================================
# Swizzled Tensor Basic
# ============================================================================


def test_swizzled_construction():
    """Create tensor with swizzled layout."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout)

    assert tensor.layout.swizzle is not None
    assert rank(tensor.layout) == 2


def test_swizzled_call_differs_from_linear():
    """Swizzled tensor gives different offsets than linear."""
    linear_layout = Layout((8, 8), (8, 1))
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))

    linear_tensor = Tensor(linear_layout)
    sw_tensor = Tensor(sw_layout)

    # At least some coordinates should differ
    differences = 0
    for i in range(8):
        for j in range(8):
            if linear_tensor(i, j) != sw_tensor(i, j):
                differences += 1

    assert differences > 0, "Swizzle should change some offsets"


def test_swizzle_xor_pattern():
    """Verify XOR pattern of Swizzle(3, 0, 3)."""
    sw = Swizzle(3, 0, 3)
    # Swizzle XORs bits [0:3] with bits [3:6]
    # At row 1 (binary 001 in bits 3-5 = offset 8), columns 0-7 XOR with 1
    assert sw(8) == 8 ^ 1  # = 9
    assert sw(9) == 9 ^ 1  # = 8
    assert sw(16) == 16 ^ 2  # = 18


# ============================================================================
# Swizzled Tensor Slicing
# ============================================================================


def test_swizzled_slice_row():
    """Slice row from swizzled tensor."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout)

    for i in range(8):
        row = tensor[i, :]
        assert isinstance(row, Tensor)
        # Offset is LINEAR (before swizzle)
        assert row.offset == i * 8
        # Swizzle preserved
        assert row.layout.swizzle is not None

        # Critical invariant: row(j) == tensor(i, j)
        for j in range(8):
            assert row(j) == tensor(i, j), f"Mismatch at [{i}, :]({j})"


def test_swizzled_slice_column():
    """Slice column from swizzled tensor."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout)

    for j in range(8):
        col = tensor[:, j]
        assert isinstance(col, Tensor)
        assert col.offset == j
        assert col.layout.swizzle is not None

        for i in range(8):
            assert col(i) == tensor(i, j), f"Mismatch at [:, {j}]({i})"


def test_swizzled_full_grid_exhaustive():
    """Exhaustive test: all slices match direct access."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout)

    # Row slices
    for i in range(8):
        row = tensor[i, :]
        for j in range(8):
            assert row(j) == tensor(i, j)

    # Column slices
    for j in range(8):
        col = tensor[:, j]
        for i in range(8):
            assert col(i) == tensor(i, j)


def test_swizzled_hierarchical_partial_slice():
    """Partial hierarchical slices keep swizzle semantics."""
    sw_layout = compose(Swizzle(2, 0, 2), Layout(((2, 2), 4), ((1, 2), 4)))
    tensor = Tensor(sw_layout)

    sub = tensor[((None, 1), None)]
    assert isinstance(sub, Tensor)
    assert sub.offset == 2
    assert sub.layout.shape == (2, 4)
    assert sub.layout.stride == (1, 4)
    assert sub.layout.swizzle == sw_layout.swizzle

    for i0 in range(2):
        for j in range(4):
            assert sub(i0, j) == tensor((i0, 1), j)


def test_swizzle_2_1_3():
    """Different swizzle parameters: Swizzle(2, 1, 3)."""
    sw_layout = compose(Swizzle(2, 1, 3), Layout((4, 8), (8, 1)))
    tensor = Tensor(sw_layout)

    for i in range(4):
        row = tensor[i, :]
        for j in range(8):
            assert row(j) == tensor(i, j)

    for j in range(8):
        col = tensor[:, j]
        for i in range(4):
            assert col(i) == tensor(i, j)


def test_swizzle_1_2_3():
    """Swizzle(1, 2, 3) - single bit XOR."""
    sw_layout = compose(Swizzle(1, 2, 3), Layout((2, 8), (8, 1)))
    tensor = Tensor(sw_layout)

    for i in range(2):
        row = tensor[i, :]
        for j in range(8):
            assert row(j) == tensor(i, j)


def test_swizzled_with_offset():
    """Swizzled tensor with non-zero base offset."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout, offset=100)

    for i in range(8):
        row = tensor[i, :]
        assert row.offset == 100 + i * 8

        for j in range(8):
            assert row(j) == tensor(i, j)


def test_swizzled_sequential_slices():
    """Sequential slicing of swizzled tensor."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8, 2), (8, 1, 64)))
    tensor = Tensor(sw_layout)

    # Slice mode 0
    s1 = tensor[3, :, :]
    assert s1.offset == 24

    # Slice mode 1 of result
    s2 = s1[5, :]
    assert s2.offset == 24 + 5

    # Final element
    for k in range(2):
        assert s2(k) == tensor(3, 5, k)


# ============================================================================
# Swizzled Tensor Column Major
# ============================================================================


def test_swizzled_column_major():
    """Swizzle applied to column-major layout."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (1, 8)))
    tensor = Tensor(sw_layout)

    for i in range(8):
        row = tensor[i, :]
        for j in range(8):
            assert row(j) == tensor(i, j)

    for j in range(8):
        col = tensor[:, j]
        for i in range(8):
            assert col(i) == tensor(i, j)


# =============================================================================
# Tensors from Layout Algebra Operations
# =============================================================================
# Test tensors created from compose, complement, logical_divide, logical_product


# ============================================================================
# Tensor From Compose
# ============================================================================


def test_composed_layout():
    """Tensor with compose(A, B) layout."""
    A = Layout((4, 8), (1, 4))
    B = Layout((2, 4), (1, 2))
    composed = compose(A, B)

    tensor = Tensor(composed)

    # Verify composition: tensor(i) == A(B(i))
    for i in range(size(composed)):
        expected = A(B(i))
        assert tensor(i) == expected


def test_composed_layout_slicing():
    """Slice tensor with composed layout."""
    A = Layout((4, 8), (1, 4))
    B = Layout((2, 4), (1, 2))
    composed = compose(A, B)

    tensor = Tensor(composed)

    if rank(tensor.layout) >= 2:
        # Can slice if rank >= 2
        for i in range(tensor.shape[0] if isinstance(tensor.shape, tuple) else 1):
            slice_i = tensor[i, :] if rank(tensor.layout) == 2 else tensor[i]
            # Verify slice correctness
            if isinstance(slice_i, Tensor):
                for j in range(size(slice_i.layout)):
                    if rank(tensor.layout) == 2:
                        assert slice_i(j) == tensor(i, j)


# ============================================================================
# Tensor From Complement
# ============================================================================


def test_complement_layout():
    """Tensor with complement layout."""
    base = Layout((4,), (2,))  # visits {0, 2, 4, 6}
    comp = complement(base, 8)  # visits {0, 1} interleaved

    tensor = Tensor(comp)

    # Complement visits the "gaps"
    for i in range(size(comp)):
        assert tensor(i) == comp(i)


def test_complement_tensor_slicing():
    """Slicing a tensor from complemented layout."""
    base = Layout((2, 2), (1, 4))
    comp = complement(base, 8)

    tensor = Tensor(comp)

    # Verify tensor call works
    for i in range(size(comp)):
        assert tensor(i) == comp(i)


# ============================================================================
# Tensor From Logical Divide
# ============================================================================


def test_logical_divide_simple():
    """Tensor from logical_divide - basic tiling."""
    layout = Layout(16, 1)
    tiler = Layout(4)
    divided = logical_divide(layout, tiler)

    tensor = Tensor(divided)

    # logical_divide creates (tile, rest) structure
    assert rank(tensor.layout) == 2

    # Verify all elements accessible
    for i in range(size(divided)):
        assert tensor(i) == layout(i)


def test_logical_divide_2d():
    """Tensor from 2D logical_divide."""
    layout = Layout((8, 8), (1, 8))
    tiler = Layout((4, 4))
    divided = logical_divide(layout, tiler)

    tensor = Tensor(divided)

    # Access through divided layout
    for i in range(size(divided)):
        offset = tensor(i)
        assert 0 <= offset < cosize(layout)


def test_logical_divide_slicing():
    """Slice tensor from logical_divide."""
    layout = Layout(16, 1)
    tiler = Layout(4)
    divided = logical_divide(layout, tiler)

    tensor = Tensor(divided)

    # Slice to get one tile
    if rank(tensor.layout) >= 2:
        tile0 = tensor[0, :]
        # Should have size 4 (tile size)
        if isinstance(tile0, Tensor):
            assert size(tile0.layout) == 4


# ============================================================================
# Tensor From Logical Product
# ============================================================================


def test_logical_product_simple():
    """Tensor from logical_product - basic replication."""
    A = Layout(4, 1)  # pattern to replicate
    B = Layout(2, 4)  # positions to replicate at

    product = logical_product(A, B)
    tensor = Tensor(product)

    # Product creates replicated pattern
    for i in range(size(product)):
        assert 0 <= tensor(i) < cosize(product)


def test_logical_product_slicing():
    """Slice tensor from logical_product."""
    A = Layout(4, 1)
    B = Layout(2, 4)
    product = logical_product(A, B)

    tensor = Tensor(product)

    if rank(tensor.layout) >= 2:
        for i in range(tensor.shape[0] if isinstance(tensor.shape, tuple) else 1):
            slice_i = tensor[i, :] if rank(tensor.layout) == 2 else tensor[i]
            if isinstance(slice_i, Tensor):
                for j in range(size(slice_i.layout)):
                    if rank(tensor.layout) == 2:
                        assert slice_i(j) == tensor(i, j)


# ============================================================================
# Tensor From Coalesce
# ============================================================================


def test_coalesced_layout():
    """Tensor from coalesced layout."""
    layout = Layout((2, 2, 2), (1, 2, 4))
    coal = coalesce(layout)

    tensor = Tensor(coal)

    # Coalesced layout should give same offsets
    for i in range(size(coal)):
        assert tensor(i) == layout(i)


def test_coalesced_slicing():
    """Slice tensor with coalesced layout."""
    layout = Layout((2, 2, 2), (1, 2, 4))
    coal = coalesce(layout)

    tensor = Tensor(coal)

    # Coalesced to rank-1, so slicing returns int
    if rank(tensor.layout) == 1:
        for i in range(size(coal)):
            assert tensor[i] == tensor(i)


# ============================================================================
# Tensor From Flatten
# ============================================================================


def test_flattened_layout():
    """Tensor from flattened hierarchical layout."""
    layout = Layout(((2, 4), (2, 2)), ((1, 2), (8, 16)))
    flat = flatten(layout)

    tensor = Tensor(flat)

    # Flattened layout should be rank-1
    # And give same offsets as original
    for i in range(size(flat)):
        assert tensor(i) == flat(i)


# =============================================================================
# Complex Swizzle + Algebra Combinations
# =============================================================================


# ============================================================================
# Swizzled Algebra Composition
# ============================================================================


def test_swizzled_after_divide():
    """Apply swizzle to divided layout."""
    base = Layout((8, 8), (1, 8))
    tiler = Layout((4, 4))
    divided = logical_divide(base, tiler)

    sw_divided = compose(Swizzle(2, 0, 2), divided)
    tensor = Tensor(sw_divided)

    # Verify we can still call through the tensor
    # The layout may be complex (hierarchical) after divide
    for i in range(size(tensor.layout)):
        # Just verify the call works - no exception
        offset = tensor(i)
        assert isinstance(offset, int)
        assert offset >= 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


# ============================================================================
# Tensor Edge Cases
# ============================================================================


def test_rank1_slicing():
    """Slicing rank-1 tensor returns int."""
    layout = Layout(16, 1)
    tensor = Tensor(layout)

    for i in range(16):
        result = tensor[i]
        assert isinstance(result, int)
        assert result == i


def test_rank1_swizzled():
    """Rank-1 swizzled tensor."""
    sw_layout = compose(Swizzle(2, 0, 2), Layout(8, 1))
    tensor = Tensor(sw_layout)

    for i in range(8):
        result = tensor[i]
        assert isinstance(result, int)
        # Swizzle applied
        assert result == tensor(i)


def test_zero_offset_preserved():
    """Zero offset tensors maintain zero offset after identity slices."""
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    row0 = tensor[0, :]
    assert row0.offset == 0

    col0 = tensor[:, 0]
    assert col0.offset == 0


# ============================================================================
# Tensor Errors
# ============================================================================


def test_wrong_rank_indices():
    """Too many indices raises IndexError."""
    tensor = Tensor(Layout((4, 8), (1, 4)))

    with pytest.raises(IndexError):
        tensor[1, 2, 3]


def test_invalid_key_type():
    """Invalid key type raises TypeError."""
    tensor = Tensor(Layout((4, 8), (1, 4)))

    with pytest.raises(TypeError):
        tensor["invalid", :]


def test_invalid_single_key():
    """Invalid single key raises TypeError."""
    tensor = Tensor(Layout((4, 8), (1, 4)))

    with pytest.raises(TypeError):
        tensor["bad"]


# =============================================================================
# CuTe C++ Compatibility Tests
# =============================================================================
# These tests are designed to match specific CuTe C++ behaviors


# ============================================================================
# Cu Te Compatibility
# ============================================================================


def test_cute_tensor_slice_pattern():
    """
    Match CuTe pattern: tensor(make_coord(i, _), make_coord(j, _))
    In Python: tensor[i, :][j] or tensor[(i, j), :]
    """
    layout = Layout((4, 8), (1, 4))
    tensor = Tensor(layout)

    # CuTe: tensor(i, _)(j) equivalent
    for i in range(4):
        row = tensor[i, :]
        for j in range(8):
            assert row(j) == i + j * 4


def test_cute_shared_memory_swizzle():
    """
    Common GPU shared memory swizzle pattern.
    Swizzle(3, 0, 3) avoids bank conflicts for 8x8 fp16 tiles.
    """
    # 8x8 tile, row-major, with swizzle
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout)

    # In CuTe, each thread accesses a row: tensor(thread_idx, _)
    for thread in range(8):
        row = tensor[thread, :]
        # All 8 accesses should go to different banks (no conflicts)
        bank_accesses = [row(col) % 32 for col in range(8)]
        # With swizzle, should have 8 unique banks
        assert len(set(bank_accesses)) == 8, f"Thread {thread} has bank conflicts"


def test_cute_tiled_tensor():
    """
    CuTe tiled tensor pattern: logical_divide then slice.
    """
    # 16 elements divided into tiles of 4
    layout = Layout(16, 1)
    tiler = Layout(4)
    divided = logical_divide(layout, tiler)

    tensor = Tensor(divided)

    # Access tile 0 elements: tensor(_, 0)
    if rank(tensor.layout) == 2:
        tile0 = tensor[:, 0]
        if isinstance(tile0, Tensor):
            for i in range(4):
                assert tile0(i) == i

        # Access tile 1 elements
        tile1 = tensor[:, 1]
        if isinstance(tile1, Tensor):
            for i in range(4):
                assert tile1(i) == 4 + i


def test_cute_mma_fragment_pattern():
    """
    Pattern used in MMA (Matrix Multiply Accumulate) fragments.
    Hierarchical layout representing warp-level tile distribution.
    """
    # Simulate warp tile: ((2 threads, 4 per thread), 8 columns)
    layout = Layout(((2, 4), 8), ((1, 2), 8))
    tensor = Tensor(layout)

    # Thread 0's elements: tensor((0, _), _)
    thread0_slice = tensor[(0, 0), :]
    assert isinstance(thread0_slice, Tensor)

    # Thread 1's elements: tensor((1, _), _)
    thread1_slice = tensor[(1, 0), :]
    assert isinstance(thread1_slice, Tensor)

    # Verify they access different starting positions
    assert thread0_slice.offset != thread1_slice.offset or thread0_slice(0) != thread1_slice(0)


def test_hierarchical_slice_preserves_structure():
    """Slicing with nested Nones preserves hierarchical mode boundaries.

    From arXiv:2603.02298, Figure 5: slicing the Cecka tensor.
    """
    A = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    T = Tensor(A)

    # (2, ((0, None), None)) — fix mode 0 and partial mode 1
    Ts = T[(2, ((0, None), None))]
    assert Ts.offset == 8
    assert Ts.layout.shape == (3, 2)
    for j1 in range(3):
        for j2 in range(2):
            assert A(2, ((0, j1), j2)) == Ts.offset + Ts.layout(j1, j2)

    # ((1, None), ((None, 0), None)) — partial slice on both modes
    Ts = T[((1, None), ((None, 0), None))]
    assert Ts.offset == 4
    assert Ts.layout.shape == (2, (2, 2))
    for i1 in range(2):
        for j0 in range(2):
            for j2 in range(2):
                assert A((1, i1), ((j0, 0), j2)) == Ts.offset + Ts.layout(i1, (j0, j2))


def test_tensor_slice_with_none():
    """Tensor.__getitem__ accepts None as equivalent to slice(None)."""
    layout = Layout((4, 8), (1, 4))
    T = Tensor(layout)

    # T[(2, None)] should equal T[2, :]
    Ts_none = T[(2, None)]
    Ts_colon = T[2, :]
    assert Ts_none.offset == Ts_colon.offset
    assert Ts_none.layout == Ts_colon.layout


def test_tensor_full_slice_matches_explicit_full_slice():
    """A bare full slice returns the whole tensor view."""
    layout = Layout((4, 8), (1, 4))
    buf = list(range(32))
    tensor = Tensor(layout, data=buf)

    full = tensor[:]
    explicit = tensor[:, :]

    assert isinstance(full, Tensor)
    assert full == explicit
    assert full.layout == layout
    assert full.offset == 0
    assert full.data is buf


def test_swizzled_tensor_full_slice_matches_explicit_full_slice():
    """A bare full slice preserves swizzled whole-tensor semantics."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout, offset=16)

    full = tensor[:]
    explicit = tensor[:, :]

    assert isinstance(full, Tensor)
    assert full == explicit
    assert full.layout.swizzle == sw_layout.swizzle
    assert full.offset == tensor.offset

    for i in range(8):
        for j in range(8):
            assert full(i, j) == tensor(i, j)


# =============================================================================
# Flat 1D Evaluation and Copy Algorithm
# =============================================================================


# ============================================================================
# Flat Evaluation
# ============================================================================


def test_flat_eval_rank2_row_major():
    """Row-major rank-2 tensor: tensor[i] == layout(i) for all i."""
    layout = Layout((4, 8), (8, 1))
    t = Tensor(layout)
    for i in range(size(layout)):
        assert t[i] == layout(i)


def test_flat_eval_rank2_col_major():
    """Column-major rank-2 tensor: tensor[i] == layout(i) for all i."""
    layout = Layout((4, 8), (1, 4))
    t = Tensor(layout)
    for i in range(size(layout)):
        assert t[i] == layout(i)


def test_flat_eval_rank3():
    """Rank-3 tensor: flat 1D evaluation through all elements."""
    layout = Layout((2, 4, 8), (32, 8, 1))
    t = Tensor(layout)
    for i in range(size(layout)):
        assert t[i] == layout(i)


def test_flat_eval_with_offset():
    """Flat 1D evaluation respects base offset."""
    layout = Layout((4, 8), (8, 1))
    t = Tensor(layout, offset=100)
    for i in range(size(layout)):
        assert t[i] == 100 + layout(i)


def test_flat_eval_with_data():
    """Flat 1D evaluation with storage returns data[offset]."""
    layout = Layout((4, 8), (8, 1))
    buf = list(range(32))
    t = Tensor(layout, data=buf)
    for i in range(size(layout)):
        assert t[i] == buf[layout(i)]


def test_flat_eval_consistent_with_setitem():
    """tensor[i] reads what tensor[i] = val wrote."""
    layout = Layout((4, 8), (8, 1))
    buf = [0] * 32
    t = Tensor(layout, data=buf)
    for i in range(size(layout)):
        t[i] = i * 10
    for i in range(size(layout)):
        assert t[i] == i * 10


def test_slice_mode0_still_works():
    """Mode-0 slicing uses tensor[i, :], not tensor[i]."""
    layout = Layout((4, 8), (1, 4))
    t = Tensor(layout)
    row = t[2, :]
    assert isinstance(row, Tensor)
    assert row.offset == 2
    for j in range(8):
        assert row(j) == t(2, j)


# ============================================================================
# Copy Algorithm
# ============================================================================


def test_copy_same_layout():
    """Copy between tensors with the same layout."""
    layout = Layout((4, 8), (8, 1))
    src_buf = list(range(32))
    dst_buf = [0] * 32
    src = Tensor(layout, data=src_buf)
    dst = Tensor(layout, data=dst_buf)

    for i in range(size(layout)):
        dst[i] = src[i]

    for i in range(32):
        assert dst_buf[i] == src_buf[i]


def test_copy_different_layouts():
    """Copy between row-major and column-major tensors.

    This is the motivating use case: different layouts over the same
    logical shape produce different physical orderings, and copy()
    must remap correctly.
    """
    row_major = Layout((4, 8), (8, 1))
    col_major = Layout((4, 8), (1, 4))

    src_buf = list(range(32))
    dst_buf = [0] * 32
    src = Tensor(row_major, data=src_buf)
    dst = Tensor(col_major, data=dst_buf)

    assert size(src.layout) == size(dst.layout)
    for i in range(size(src.layout)):
        dst[i] = src[i]

    # Verify: same logical element at every coordinate
    for i in range(4):
        for j in range(8):
            assert dst[i, j] == src[i, j]


def test_copy_rank3():
    """Copy works on rank-3 tensors."""
    layout_a = Layout((2, 4, 8), (32, 8, 1))
    layout_b = Layout((2, 4, 8), (1, 2, 8))

    src_buf = list(range(64))
    dst_buf = [0] * 64
    src = Tensor(layout_a, data=src_buf)
    dst = Tensor(layout_b, data=dst_buf)

    for i in range(size(layout_a)):
        dst[i] = src[i]

    for b in range(2):
        for h in range(4):
            for w in range(8):
                assert dst[b, h, w] == src[b, h, w]


# =============================================================================
# Tensor Storage
# =============================================================================

# ============================================================================
# Tensor Storage
# ============================================================================


def test_construction_with_data():
    """Tensor can be constructed with storage."""
    layout = Layout((4, 8), (8, 1))
    buf = list(range(32))
    t = Tensor(layout, data=buf)
    assert t.data is buf
    assert t.layout == layout
    assert t.offset == 0


def test_construction_with_numpy_1d_data():
    """A 1D numpy array is valid backing storage."""
    np = pytest.importorskip("numpy")

    layout = Layout((4, 8), (8, 1))
    buf = np.arange(32)
    t = Tensor(layout, data=buf)

    assert t.data is buf
    assert int(t[2, 3]) == 19


def test_construction_with_numpy_multidim_data_raises():
    """A multidimensional numpy array is rejected as implicit extra layout."""
    np = pytest.importorskip("numpy")

    layout = Layout((4, 8), (8, 1))
    buf = np.arange(32).reshape(4, 8)
    with pytest.raises(TypeError, match="flat 1D storage"):
        Tensor(layout, data=buf)


def test_construction_with_torch_1d_data():
    """A 1D torch tensor is valid backing storage."""
    torch = pytest.importorskip("torch")

    layout = Layout((4, 8), (8, 1))
    buf = torch.arange(32)
    t = Tensor(layout, data=buf)

    assert t.data is buf
    assert int(t[2, 3]) == 19


def test_construction_with_torch_multidim_data_raises():
    """A multidimensional torch tensor is rejected as implicit extra layout."""
    torch = pytest.importorskip("torch")

    layout = Layout((4, 8), (8, 1))
    buf = torch.arange(32).reshape(4, 8)
    with pytest.raises(TypeError, match="flat 1D storage"):
        Tensor(layout, data=buf)


def test_construction_with_data_and_offset():
    """Tensor can have both offset and data."""
    buf = list(range(64))
    t = Tensor(Layout((4, 8), (8, 1)), offset=5, data=buf)
    assert t.offset == 5
    assert t.data is buf


def test_data_too_small_raises():
    """Storage smaller than cosize raises ValueError."""
    layout = Layout((4, 8), (8, 1))  # cosize = 32
    with pytest.raises(ValueError, match="addressed range"):
        Tensor(layout, data=list(range(10)))


def test_data_larger_than_cosize():
    """Storage can be larger than cosize."""
    layout = Layout((4, 8), (8, 1))  # cosize = 32
    buf = list(range(100))
    t = Tensor(layout, data=buf)
    assert t.data is buf


def test_data_none_by_default():
    """Without data, tensor.data is None."""
    t = Tensor(Layout((4, 8), (8, 1)))
    assert t.data is None


def test_data_property_setter():
    """The data property is writable."""
    layout = Layout((4, 8), (8, 1))
    t = Tensor(layout)
    assert t.data is None

    buf = list(range(32))
    t.data = buf
    assert t.data is buf


def test_data_setter_validates_size():
    """Setting data too small raises ValueError."""
    layout = Layout((4, 8), (8, 1))  # cosize = 32
    t = Tensor(layout)
    with pytest.raises(ValueError, match="addressed range"):
        t.data = list(range(10))


def test_data_setter_rejects_multidim_numpy_storage():
    """Assigning multidimensional array storage must be explicit at the callsite."""
    np = pytest.importorskip("numpy")

    layout = Layout((4, 8), (8, 1))
    t = Tensor(layout)
    with pytest.raises(TypeError, match="flat 1D storage"):
        t.data = np.arange(32).reshape(4, 8)


def test_data_with_positive_offset_needs_full_address_range():
    """Construction validates the full addressed range, not just cosize(layout)."""
    buf = list(range(14))
    t = Tensor(Layout(4, 1), offset=10, data=buf)

    assert [t[i] for i in range(4)] == [10, 11, 12, 13]


def test_data_with_positive_offset_and_short_storage_raises():
    """A nonzero base offset must still fit in the backing storage."""
    with pytest.raises(ValueError, match="addressed range"):
        Tensor(Layout(4, 1), offset=10, data=[0] * 4)


def test_negative_stride_requires_nonnegative_address_range():
    """Negative strides without a compensating offset are rejected."""
    with pytest.raises(ValueError, match="addressed range"):
        Tensor(Layout(4, -1), data=list(range(10)))


def test_negative_stride_with_offset_reads_correct_data():
    """Negative strides work when the base offset keeps accesses in-bounds."""
    t = Tensor(Layout(4, -1), offset=3, data=list(range(8)))
    assert [t[i] for i in range(4)] == [3, 2, 1, 0]


def test_negative_stride_slice_preserves_valid_address_range():
    """Slicing a valid negative-stride tensor stays within the shared storage."""
    t = Tensor(Layout((2, 4), (4, -1)), offset=3, data=list(range(8)))
    row1 = t[1, :]

    assert row1.data is t.data
    assert [row1[j] for j in range(4)] == [7, 6, 5, 4]


def test_data_setter_validates_offset_and_signed_stride():
    """Assigning data uses the same addressed-range validation as construction."""
    t = Tensor(Layout(4, -1), offset=3)

    with pytest.raises(ValueError, match="addressed range"):
        t.data = list(range(3))

    buf = list(range(4))
    t.data = buf
    assert [t[i] for i in range(4)] == [3, 2, 1, 0]


def test_data_setter_accepts_none():
    """Setting data to None removes storage."""
    buf = list(range(32))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf)
    t.data = None
    assert t.data is None


def test_getitem_returns_data_element():
    """With data, tensor[i, j] returns data[offset]."""
    layout = Layout((4, 8), (8, 1))
    buf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    t = Tensor(layout, data=buf)

    assert t[0, 0] == "A"  # buf[0]
    assert t[0, 1] == "B"  # buf[1]
    assert t[1, 0] == "I"  # buf[8]
    assert t[3, 7] == "5"  # buf[31]


def test_getitem_without_data_returns_offset():
    """Without data, tensor[i, j] still returns the offset integer."""
    layout = Layout((4, 8), (8, 1))
    t = Tensor(layout)
    assert t[2, 3] == 19


def test_call_unaffected_by_data():
    """tensor(i, j) always returns the offset, even with data."""
    layout = Layout((4, 8), (8, 1))
    buf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    t = Tensor(layout, data=buf)
    assert t(2, 3) == 19  # offset, not 'T'


def test_getitem_single_mode_returns_data():
    """Rank-1 tensor: tensor[i] returns data element."""
    buf = list("ABCDEFGH")
    t = Tensor(Layout(8, 1), data=buf)
    assert t[0] == "A"
    assert t[7] == "H"


def test_getitem_flat_eval_returns_data():
    """Flat 1D evaluation with data returns data element, not sub-Tensor."""
    buf = list(range(32))
    t = Tensor(Layout((4, 8), (1, 4)), data=buf)
    # t[2] does flat 1D eval: layout(2) = 2, returns buf[2]
    result = t[2]
    assert isinstance(result, int)
    assert result == buf[t.layout(2)]


def test_setitem_writes_to_data():
    """tensor[i, j] = val writes data[offset]."""
    buf = list(range(32))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf)

    t[2, 3] = 999
    assert buf[19] == 999
    assert t[2, 3] == 999


def test_setitem_rank1():
    """Scalar write on rank-1 tensor."""
    buf = list(range(16))
    t = Tensor(Layout(8, 2), data=buf)
    t[3] = 42
    assert buf[6] == 42  # offset = 3*2 = 6


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), 1),
        (None, 1),
        ((0, None), 1),
        slice(None),
    ],
)
def test_setitem_rejects_free_coordinates(key):
    """Assignment must reject any key that leaves a mode free."""
    buf = list(range(32))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf)

    with pytest.raises(TypeError, match="fully-fixed coordinates"):
        t[key] = 999

    assert buf == list(range(32))


def test_setitem_hierarchical_fixed_coordinate_writes_to_data():
    """A fully-fixed hierarchical coordinate is still a valid assignment key."""
    layout = Layout(((2, 4), 8), ((1, 2), 8))
    buf = list(range(cosize(layout)))
    t = Tensor(layout, data=buf)

    t[(1, 3), 5] = 999

    assert buf[t((1, 3), 5)] == 999
    assert t[(1, 3), 5] == 999


def test_setitem_without_data_raises():
    """Writing to a tensor with no storage raises TypeError."""
    t = Tensor(Layout((4, 8), (8, 1)))
    with pytest.raises(TypeError, match="no storage"):
        t[2, 3] = 42


def test_slice_shares_data():
    """Sub-Tensors from slicing share the parent's data."""
    buf = list(range(32))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf)
    row2 = t[2, :]
    assert row2.data is buf


def test_slice_data_read():
    """Reading through a sliced sub-Tensor accesses correct data."""
    buf = list(range(32))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf)
    row2 = t[2, :]
    # row2[3] should be buf[tensor(2, 3)] = buf[19] = 19
    assert row2[3] == 19
    assert row2[3] == t[2, 3]


def test_slice_data_write():
    """Writing through a sliced sub-Tensor modifies shared data."""
    buf = list(range(32))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf)
    row2 = t[2, :]
    row2[3] = 999
    assert buf[19] == 999
    assert t[2, 3] == 999


def test_chained_slice_data():
    """tensor[i, :][j] returns the same data element as tensor[i, j]."""
    buf = list(range(32))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf)
    for i in range(4):
        for j in range(8):
            assert t[i, :][j] == t[i, j]


def test_hierarchical_slice_propagates_data():
    """Data is propagated through hierarchical partial slicing."""
    layout = Layout(((2, 4), 8), ((1, 2), 8))
    buf = list(range(cosize(layout)))
    t = Tensor(layout, data=buf)

    sub = t[((0, None), None)]
    assert sub.data is buf


def test_data_swap_does_not_affect_existing_slices():
    """Reassigning parent.data does not update existing sub-Tensors."""
    buf1 = list(range(32))
    buf2 = list(range(100, 132))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf1)
    row = t[2, :]

    t.data = buf2
    # row still references buf1
    assert row.data is buf1
    assert row[0] == 16  # buf1[16]
    # parent now reads from buf2
    assert t[2, 0] == 116  # buf2[16]


# =============================================================================
# Tensor Equality with Data
# =============================================================================

# ============================================================================
# Tensor Equality With Data
# ============================================================================


def test_same_contents_equal():
    """Tensors with same layout, offset, and data contents are equal."""
    layout = Layout(8, 1)
    a = Tensor(layout, data=[1, 2, 3, 4, 5, 6, 7, 8])
    b = Tensor(layout, data=[1, 2, 3, 4, 5, 6, 7, 8])
    assert a == b


def test_different_contents_not_equal():
    """Tensors with different data contents are not equal."""
    layout = Layout(8, 1)
    a = Tensor(layout, data=[1, 2, 3, 4, 5, 6, 7, 8])
    b = Tensor(layout, data=[8, 7, 6, 5, 4, 3, 2, 1])
    assert a != b


def test_one_has_data_other_does_not():
    """Tensor with data != Tensor without data."""
    layout = Layout(8, 1)
    a = Tensor(layout, data=[1, 2, 3, 4, 5, 6, 7, 8])
    b = Tensor(layout)
    assert a != b
    assert b != a


def test_both_none_equal():
    """Two algebraic Tensors with same layout/offset are equal (existing behavior)."""
    layout = Layout((4, 8), (8, 1))
    a = Tensor(layout)
    b = Tensor(layout)
    assert a == b


def test_same_data_identity_shortcut():
    """Tensors sharing the same data object are equal."""
    layout = Layout(8, 1)
    buf = [1, 2, 3, 4, 5, 6, 7, 8]
    a = Tensor(layout, data=buf)
    b = Tensor(layout, data=buf)
    assert a == b


def test_different_data_lengths_not_equal():
    """Tensors with different data lengths are not equal."""
    layout = Layout(4, 1)
    a = Tensor(layout, data=[1, 2, 3, 4])
    b = Tensor(layout, data=[1, 2, 3, 4, 5, 6, 7, 8])
    assert a != b


def test_hash_consistent_with_equality():
    """Equal Tensors must have equal hashes."""
    layout = Layout(8, 1)
    a = Tensor(layout, data=[1, 2, 3, 4, 5, 6, 7, 8])
    b = Tensor(layout, data=[1, 2, 3, 4, 5, 6, 7, 8])
    assert a == b
    assert hash(a) == hash(b)


def test_different_data_can_collide_hash():
    """Tensors with different data may have same hash (collision OK)."""
    layout = Layout(8, 1)
    a = Tensor(layout, data=[1, 2, 3, 4, 5, 6, 7, 8])
    b = Tensor(layout, data=[8, 7, 6, 5, 4, 3, 2, 1])
    # Same hash is allowed even though a != b
    assert hash(a) == hash(b)  # both have same (layout, offset)
    assert a != b


# ============================================================================
# View
# ============================================================================


def test_view_shares_data():
    """View shares the same backing storage."""
    buf = list("ABCDEFGH")
    t = Tensor(Layout((2, 4), (4, 1)), data=buf)
    flat = t.view(Layout(8, 1))
    assert flat.data is t.data


def test_view_reads_same_data():
    """View reads from the shared storage."""
    buf = list("ABCDEFGH")
    t = Tensor(Layout((2, 4), (4, 1)), data=buf)
    flat = t.view(Layout(8, 1))
    assert flat[0] == "A"
    assert flat[7] == "H"


def test_view_write_visible():
    """Writes through a view are visible in the original."""
    buf = list("ABCDEFGH")
    t = Tensor(Layout((2, 4), (4, 1)), data=buf)
    flat = t.view(Layout(8, 1))
    flat[0] = "Z"
    assert t[0, 0] == "Z"


def test_view_different_layout():
    """View can reshape to a different rank."""
    buf = list(range(12))
    t = Tensor(Layout(12, 1), data=buf)
    reshaped = t.view(Layout((3, 4), (4, 1)))
    assert reshaped[1, 2] == buf[1 * 4 + 2]


def test_view_preserves_offset():
    """A view keeps the parent Tensor's base offset."""
    buf = list(range(64))
    t = Tensor(Layout((4, 8), (8, 1)), offset=16, data=buf)

    flat = t.view(Layout(32, 1))

    assert flat.offset == 16
    assert flat[0] == 16
    assert flat[15] == 31


def test_view_negative_stride_uses_preserved_offset():
    """Preserving offset enables valid reverse views."""
    buf = list(range(8))
    t = Tensor(Layout(4, 1), offset=3, data=buf)

    rev = t.view(Layout(4, -1))

    assert rev.offset == 3
    assert [rev[i] for i in range(4)] == [3, 2, 1, 0]


def test_view_no_storage_raises():
    """View on algebraic Tensor raises TypeError."""
    t = Tensor(Layout(8, 1))
    with pytest.raises(TypeError):
        t.view(Layout(8, 1))


def test_view_cosize_too_large_raises():
    """View with cosize exceeding storage raises ValueError."""
    buf = list(range(8))
    t = Tensor(Layout(8, 1), data=buf)
    with pytest.raises(ValueError):
        t.view(Layout(16, 1))


def test_view_smaller_cosize():
    """View with smaller cosize is allowed."""
    buf = list(range(16))
    t = Tensor(Layout(16, 1), data=buf)
    small = t.view(Layout(4, 1))
    assert small[0] == 0
    assert small[3] == 3


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
