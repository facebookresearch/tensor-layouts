# MIT License — see LICENSE file for details.

"""Structural tests for AMD RDNA WMMA atoms.

Validates the (thread, value) -> element mappings for all RDNA3 and RDNA4
WMMA atom definitions using algebraic invariants — no hardware required.
"""

import pytest

from tensor_layouts import Layout, size, rank, cosize
from tensor_layouts.atoms_amd import (
    MMA_ATOMS_RDNA3, MMA_ATOMS_RDNA4,
    RDNA3_16x16x16_F32F16F16_WMMA,
)


def _num_threads(layout):
    return size(layout.shape[0]) if isinstance(layout.shape, tuple) else size(layout.shape)


def _num_values(layout):
    return size(layout.shape[1]) if isinstance(layout.shape, tuple) else 1


ALL_ATOMS = MMA_ATOMS_RDNA3 + MMA_ATOMS_RDNA4


@pytest.mark.parametrize("atom", ALL_ATOMS, ids=lambda a: a.name)
class TestWMMAStructural:
    """Structural invariants that must hold for any valid WMMA atom."""

    def test_c_layout_covers_all_elements(self, atom):
        """Every element of the M × N output is touched exactly once."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        num_t = _num_threads(c)
        num_v = _num_values(c)

        seen = set()
        for t in range(num_t):
            for v in range(num_v):
                offset = c(t, v)
                assert 0 <= offset < m * n, \
                    f"{atom.name}: offset {offset} out of range [0, {m*n})"
                assert offset not in seen, \
                    f"{atom.name}: duplicate offset {offset} at t={t}, v={v}"
                seen.add(offset)

        assert len(seen) == m * n, \
            f"{atom.name}: covers {len(seen)} elements, expected {m*n}"

    def test_c_layout_thread_count(self, atom):
        """Thread dimension has exactly 32 elements (wave32)."""
        c = atom.c_layout
        assert _num_threads(c) == 32, \
            f"{atom.name}: {_num_threads(c)} threads, expected 32"

    def test_a_layout_thread_count(self, atom):
        a = atom.a_layout
        assert _num_threads(a) == 32

    def test_b_layout_thread_count(self, atom):
        b = atom.b_layout
        assert _num_threads(b) == 32

    def test_c_layout_cosize_equals_mn(self, atom):
        """C layout codomain spans exactly M × N."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        assert cosize(c) == m * n

    def test_a_layout_cosize_equals_mk(self, atom):
        """A layout codomain spans exactly M × K (no broadcast in WMMA)."""
        m, n, k = atom.shape_mnk
        a = atom.a_layout
        assert cosize(a) == m * k

    def test_b_layout_cosize_equals_nk(self, atom):
        """B layout codomain spans exactly N × K."""
        m, n, k = atom.shape_mnk
        b = atom.b_layout
        assert cosize(b) == n * k

    def test_thr_id_is_none(self, atom):
        """RDNA WMMA atoms use identity thread mapping."""
        assert atom.thr_id is None

    def test_c_layout_rank_is_2(self, atom):
        assert rank(atom.c_layout) == 2

    def test_a_layout_rank_is_2(self, atom):
        assert rank(atom.a_layout) == 2

    def test_b_layout_rank_is_2(self, atom):
        assert rank(atom.b_layout) == 2

    def test_layout_sizes_match_shape_mnk(self, atom):
        """Layout domain sizes are consistent with M, N, K."""
        m, n, k = atom.shape_mnk
        assert size(atom.c_layout) == m * n
        assert size(atom.a_layout) == 32 * _num_values(atom.a_layout)
        assert size(atom.b_layout) == 32 * _num_values(atom.b_layout)

    def test_a_covers_all_elements(self, atom):
        """A layout is a bijection from (T32, V) to M × K offsets."""
        m, n, k = atom.shape_mnk
        a = atom.a_layout
        num_t = _num_threads(a)
        num_v = _num_values(a)
        seen = set()
        for t in range(num_t):
            for v in range(num_v):
                offset = a(t, v)
                assert 0 <= offset < m * k
                seen.add(offset)
        assert len(seen) == m * k

    def test_b_covers_all_elements(self, atom):
        """B layout is a bijection from (T32, V) to N × K offsets."""
        m, n, k = atom.shape_mnk
        b = atom.b_layout
        num_t = _num_threads(b)
        num_v = _num_values(b)
        seen = set()
        for t in range(num_t):
            for v in range(num_v):
                offset = b(t, v)
                assert 0 <= offset < n * k
                seen.add(offset)
        assert len(seen) == n * k


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
