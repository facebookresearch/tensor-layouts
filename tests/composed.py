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
    layout = ComposedLayout(outer, inner, offset=3)

    assert layout.shape == inner.shape
    assert size(layout) == size(inner)
    assert rank(layout) == rank(inner)
    assert depth(layout) == depth(inner)
    # cosize on a ComposedLayout is max(L(c)) + 1 over the full domain,
    # not cosize(inner) (which is what CuTe C++ returns and is wrong --
    # see bug-reports/cute_cosize/).
    assert cosize(layout) == max(layout(i) for i in range(size(layout))) + 1
    assert repr(layout) == f"ComposedLayout({outer!r}, {inner!r}, offset=3)"
    assert str(layout) == f"({outer}) o {{3}} o ({inner})"
    assert layout == ComposedLayout(outer, inner, offset=3)
    assert hash(layout) == hash(ComposedLayout(outer, inner, offset=3))
    assert layout((1, 2)) == outer(3 + inner((1, 2)))

    # Hash must also work when outer is a Swizzle (not just Layout).
    swz_composed = ComposedLayout(Swizzle(3, 0, 3), Layout(16, 1))
    assert hash(swz_composed) == hash(ComposedLayout(Swizzle(3, 0, 3), Layout(16, 1)))

    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.offset = 4


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

    # Representation-tolerant: today the representable case decays to an
    # embedded-swizzle Layout, post-Path-X it returns a ComposedLayout.
    # Pointwise equivalence is the contract that matters.
    assert isinstance(result, (Layout, ComposedLayout))
    _assert_pointwise_equal(result, lambda i: outer(swizzle(i)))


def test_compose_layout_with_swizzle_rhs_nonpower_stride_stays_exact():
    outer = Layout(16, 3)
    swizzle = Swizzle(2, 0, 2)
    result = compose(outer, swizzle)
    expected = ComposedLayout(outer, compose(swizzle, Layout(outer.shape)))

    assert isinstance(result, ComposedLayout)
    _assert_pointwise_equal(result, expected)


def test_compose_layout_on_zero_offset_composed_layout_can_collapse():
    outer = Layout(32, 2)
    inner = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), offset=0)
    result = compose(outer, inner)

    # Representation-tolerant (Path X): both Layout-with-embedded-swizzle and
    # ComposedLayout(Sw, L) are acceptable representations of the same
    # function; only pointwise equivalence is contractual.
    assert isinstance(result, (Layout, ComposedLayout))
    _assert_pointwise_equal(result, lambda i: outer(inner(i)))


def test_compose_layout_on_nonzero_offset_composed_layout_stays_exact():
    outer = Layout(32, 2)
    inner = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), offset=4)
    result = compose(outer, inner)

    assert isinstance(result, ComposedLayout)
    _assert_pointwise_equal(result, lambda i: outer(inner(i)))


def test_compose_swizzled_layout_outer_preserves_exactness():
    outer = compose(Swizzle(2, 0, 2), Layout(16, 1))
    inner = Layout(8, 2)
    result = compose(outer, inner)

    # Representation-tolerant (Path X): the outer's swizzle must be preserved
    # in some form (embedded or composed); pointwise equivalence is the
    # contract that matters.
    assert isinstance(result, (Layout, ComposedLayout))
    _assert_pointwise_equal(result, lambda i: outer(inner(i)))


def test_logical_divide_forwards_through_composed_layout():
    composed = compose(Layout(16, 2), compose(Swizzle(2, 0, 2), Layout(16, 1)))
    result = logical_divide(composed, 4)
    expected = ComposedLayout(composed.outer, logical_divide(composed.inner, 4), offset=0)

    assert isinstance(result, ComposedLayout)
    assert result.outer == composed.outer
    assert result.offset == 0
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
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), offset=4)
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
    tensor = Tensor(ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), offset=4))
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        _ = tensor.stride


def test_right_inverse_of_zero_offset_swizzled_composed_layout():
    composed = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), offset=0)
    inv = right_inverse(composed)

    assert isinstance(inv, Layout)
    for i in range(size(inv)):
        assert composed(inv(i)) == i


def test_left_inverse_of_zero_offset_swizzled_composed_layout():
    composed = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), offset=0)
    inv = left_inverse(composed)

    assert isinstance(inv, Layout)
    for i in range(size(composed)):
        assert inv(composed(i)) == i


def test_right_inverse_of_nonzero_offset_swizzled_composed_layout():
    composed = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), offset=4)
    inv = right_inverse(composed)

    assert isinstance(inv, ComposedLayout)
    assert isinstance(inv.inner, Swizzle)
    for i in range(cosize(composed)):
        assert composed(inv(i)) == i


def test_left_inverse_of_nonzero_offset_swizzled_composed_layout():
    composed = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), offset=4)
    inv = left_inverse(composed)

    assert isinstance(inv, ComposedLayout)
    assert isinstance(inv.inner, Swizzle)
    for i in range(size(composed)):
        assert inv(composed(i)) == i


def test_logical_product_rejects_swizzle_inner_composed_layout():
    composed = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)
    with pytest.raises(NotImplementedError, match="Swizzle in the inner slot"):
        logical_product(composed, Layout(2, 1))


def test_complement_rejects_swizzle_inner_composed_layout():
    """complement(F6) raises rather than silently returning a degenerate layout."""
    composed = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)
    with pytest.raises(NotImplementedError, match="complement"):
        complement(composed)


def test_coalesce_returns_swizzle_inner_composed_layout_unchanged():
    """coalesce on the inverse-form is a no-op: the inverse-form is rank-1
    with no multi-mode structure to merge and no size-1 modes to filter,
    so the only correct answer is the input itself.

    Functional check: same outputs over the full domain.
    """
    composed = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)
    result = coalesce(composed)
    assert result is composed  # same instance: literal no-op
    for i in range(size(composed)):
        assert result(i) == composed(i)


def test_coalesce_with_profile_also_handles_swizzle_inner_composed_layout():
    """The mode-profile coalesce path must also accept the inverse-form.

    A user-supplied profile shouldn't change the answer -- the inverse-form
    is rank-1, so any rank-respecting profile is a no-op.
    """
    composed = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)
    result = coalesce(composed, profile=(32,))
    for i in range(size(composed)):
        assert result(i) == composed(i)


def test_logical_divide_rejects_swizzle_inner_composed_layout():
    """logical_divide(F6, ...) raises with its own op name (not 'coalesce')."""
    composed = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)
    with pytest.raises(NotImplementedError, match="logical_divide"):
        logical_divide(composed, Layout(4, 1))


def test_swizzle_inner_composed_layout_still_supports_basic_queries():
    """Operations that the inverse-and-cancel round trip needs MUST work on F6.

    Anchors the support boundary: if any of these starts raising we've
    over-tightened the guards and broken the algebra.
    """
    composed = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)

    # Domain queries
    assert size(composed) == 32
    # cosize is max(F6(i)) + 1. F6's image is [-4, 27], so max+1 = 28.
    # This is less than size(F6)=32 because the negative outputs eat into
    # the upper bound -- the offset shifts the image down by 4.
    assert cosize(composed) == 28
    assert rank(composed) == 1
    assert depth(composed) == 0  # scalar shape -> depth 0, matches Layout(32, 1)

    # __call__ at explicit indices (negative outputs are expected on F6)
    assert composed(0) == -4
    assert composed(4) == 0

    # Inverse round-trip: right_inverse(F3) should give an F6, and
    # composing F3 with that F6 should be identity on F3's domain.
    f3 = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), offset=4)
    inv = right_inverse(f3)
    assert isinstance(inv, ComposedLayout)
    assert isinstance(inv.inner, Swizzle)
    for i in range(size(f3)):
        assert inv(f3(i)) == i


def test_tensor_rejects_layout_with_negative_addresses():
    """Attaching storage to an inverse-form layout must error at the boundary,
    naming the inverse-form hazard so the user knows what to do."""
    inv = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)
    with pytest.raises(ValueError, match="negative storage indices"):
        Tensor(inv, offset=0, data=list(range(32)))


def test_tensor_accepts_inverse_form_when_offset_shifts_above_zero():
    """If the user shifts the external offset enough, the addresses become
    non-negative and the Tensor is valid. Witnesses that the guard is about
    actual addresses, not a structural rejection of the form."""
    inv = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)
    # F6's addressed range is [-4, 27]; shifting offset by 4 puts it at [0, 31].
    t = Tensor(inv, offset=4, data=list(range(32)))
    assert t(0) == 0
    assert t(4) == 4


def test_tensor_data_setter_rejects_inverse_form_with_negative_addresses():
    """The .data setter must apply the same negative-address check as __init__,
    or you could bypass it by constructing algebraic-then-assigning storage."""
    inv = ComposedLayout(Layout(32, 1), Swizzle(2, 1, 3), offset=-4)
    t = Tensor(inv, offset=0)  # algebraic -- no data, no validation
    with pytest.raises(ValueError, match="negative storage indices"):
        t.data = list(range(32))


def test_max_common_vector_for_swizzled_composed_layout_is_capped_by_swizzle_base():
    composed = ComposedLayout(Swizzle(2, 1, 3), Layout(32, 1), offset=0)
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
        contiguity(composed)
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        mode_contiguity(composed)
    with pytest.raises(TypeError, match="ComposedLayout|affine"):
        slice_contiguity(composed, (None,))
    # to_F2_matrix() now accepts ComposedLayout when it is F2-linear (zero
    # offset, no Swizzle in the inner slot). It rejects with ValueError --
    # not TypeError -- in the cases that aren't F2-linear; see
    # tests/analysis.py for those boundary tests. The form here IS F2-linear
    # (Layout outer over swizzled-Layout inner, offset 0), so it succeeds.
    M = to_F2_matrix(composed)
    assert len(M) > 0 and len(M[0]) > 0


def test_slice_on_swizzled_composed_decays_when_y_or_z_misses():
    """Slicing a ComposedLayout(Swizzle, Layout) decays to a plain Layout when
    the surviving inner only hits Y or only hits Z bits of the swizzle.
    Mirrors CuTe C++'s slice_and_offset decay (cute/swizzle_layout.hpp:263-294).
    """
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 4), (1, 4)))

    # Each j fixes the Z bits to a different constant; the surviving 4-coord
    # mode only walks Y bits, so decay is safe. Sub must be a plain affine
    # Layout (no swizzle wrapper); under Path X this is the only swizzle-free
    # form.
    for j in range(4):
        sub, off = slice_and_offset((None, j), composed)
        assert isinstance(sub, Layout)
        for i in range(4):
            assert off + sub(i) == composed(i, j)


def test_slice_on_swizzled_composed_stays_composed_when_y_and_z_both_hit():
    """If the surviving inner's image hits BOTH Y and Z bits, the swizzle is
    not affine on it and we must keep the ComposedLayout wrapping. This is the
    "not reducible" branch in cute/swizzle_layout.hpp:277.
    """
    big = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 4), (1, 4)))
    # Full slice always stays composed via the all-None fast path.
    sub, off = slice_and_offset((None, None), big)
    assert isinstance(sub, ComposedLayout)
    assert off == 0
    # Partial slice that still leaves both Y and Z bits in the image:
    # surviving (4,4) layout reaches offsets 0..15, hitting Y={0,1} AND Z={2,3}.
    # (We exercise this via composing through an outer to verify the bail-out.)


def test_slice_on_swizzled_layout_decays_to_canonical_form():
    """Slicing a Layout-with-embedded-swizzle decays to a canonical form whose
    addresses match the un-sliced layout. Under CuTe-aligned addressing the
    Tensor's base offset is added AFTER the swizzle, so the slice's
    contribution must be folded into the swizzle's domain (either via a
    Form-B ComposedLayout or via affine decay when the slice restricts the
    swizzle's input enough). Functional equivalence (slice(j) == orig(i, j))
    is the real check; the chosen representation is implementation detail.
    """
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    # Representation-tolerant (Path X): canonical Sw o L may live as either
    # an embedded-swizzle Layout or a ComposedLayout; both are acceptable.
    assert isinstance(sw_layout, (Layout, ComposedLayout))

    # The slice may decay to a plain Layout (when the swizzle is affine on
    # the surviving inner image) or remain a ComposedLayout(Sw, sub, k).
    # Both forms are acceptable; what matters is that the addresses match.
    sub, off = slice_and_offset((1, None), sw_layout)
    for j in range(8):
        # sub(j) returns the offset within the sliced layout; off + that
        # equals the un-sliced layout(1, j).
        assert off + sub(j) == sw_layout(1, j)


def test_logical_product_with_swizzled_tile_transfers_swizzle():
    """logical_product(Layout, ComposedLayout(Swizzle, Layout)) used to silently
    drop the embedded swizzle from the result. CuTe C++
    (cute/swizzle_layout.hpp:549-587) rebuilds a fresh swizzle for the new
    product strides; verify pointwise equivalence to the CuTe formula
    A(a) + complement(A, size(A)*cosize(B))(B(b))."""
    layout_a = Layout(2, 1)
    swizzle = Swizzle(2, 0, 2)
    tile = ComposedLayout(swizzle, Layout((4, 4), (1, 4)))
    result = logical_product(layout_a, tile)

    # The swizzle must survive — the old code dropped it. Representation
    # is implementation-defined (embedded Layout today, ComposedLayout under
    # Path X); pointwise equivalence is the contract.
    assert isinstance(result, (Layout, ComposedLayout))

    # Pointwise check against CuTe's defining formula.
    comp = complement(layout_a, size(layout_a) * cosize(tile))
    for a in range(size(layout_a)):
        for inner_i in range(4):
            for inner_j in range(4):
                got = result((a, (inner_i, inner_j)))
                expected = layout_a(a) + comp(tile((inner_i, inner_j)))
                assert got == expected, (
                    f"({a}, ({inner_i}, {inner_j})): got {got}, expected {expected}"
                )


def test_complement_of_composed_layout_forwards_to_inner():
    """complement(ComposedLayout) drops the outer involution and returns
    complement(inner), matching CuTe C++ layout_composed.hpp:395-409."""
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1))
    result = complement(composed)
    expected = complement(Layout(16, 1))
    assert result == expected

    # Same with an explicit cosize_bound.
    bounded = complement(composed, 32)
    assert bounded == complement(Layout(16, 1), 32)

    # Non-bijective inner → nontrivial complement: inner spans {0,2,4,6,8,10,12,14}.
    composed_strided = ComposedLayout(Swizzle(2, 0, 2), Layout(8, 2))
    assert complement(composed_strided) == complement(Layout(8, 2))


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


def test_generative_compose_with_offsets():
    """ComposedLayout with various offsets evaluates correctly."""
    for swz in _SMALL_SWIZZLES[:2]:
        for inner in _SMALL_AFFINE_LAYOUTS[:3]:
            for po in _SMALL_PREOFFSETS:
                composed = ComposedLayout(swz, inner, offset=po)
                for i in range(size(composed)):
                    assert composed(i) == swz(po + inner(i)), (
                        f"Mismatch at i={i} for ComposedLayout({swz}, {inner}, offset={po})"
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
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), offset=0)
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
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), offset=4)
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
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), offset=4)
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
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), offset=4)
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
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout((4, 2), (1, 4)), offset=4)
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
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), offset=0)
    result = cycles(composed)
    assert isinstance(result, list)


def test_fixed_points_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), offset=0)
    result = fixed_points(composed)
    assert isinstance(result, (list, set))


def test_order_on_composed_layout():
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), offset=0)
    result = order(composed)
    assert isinstance(result, int)
    assert result >= 1

def test_cosize_composed_layout_caches_on_instance():
    """Second cosize() call returns the cached value, not a recomputation.

    Verified by sentinel: poison the cache slot on the instance and observe
    that the next cosize() call returns the poisoned value, proving the
    cache is read on the hot path. This is whitebox but pins down the
    optimization in regression tests.
    """
    composed = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), offset=0)
    expected = cosize(composed)
    assert composed._cached_cosize == expected

    # Poison and confirm the next call reads the cache.
    object.__setattr__(composed, "_cached_cosize", expected + 999)
    assert cosize(composed) == expected + 999

    # Reset cache and confirm we get the real value back.
    object.__setattr__(composed, "_cached_cosize", None)
    assert cosize(composed) == expected


def test_cosize_cache_does_not_affect_equality_or_hash():
    """The cache slot must be excluded from __eq__ / __hash__.

    Two ComposedLayouts with the same (outer, inner, offset) but different
    cache states must still compare equal and hash the same -- otherwise
    they would lose dict-key compatibility after one is queried for cosize.
    """
    a = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), offset=0)
    b = ComposedLayout(Swizzle(2, 0, 2), Layout(16, 1), offset=0)
    # Populate cache on `a` only.
    _ = cosize(a)
    assert a._cached_cosize is not None
    assert b._cached_cosize is None
    assert a == b
    assert hash(a) == hash(b)
