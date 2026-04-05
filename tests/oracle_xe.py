# MIT License — see LICENSE file for details.

"""Structural tests for Intel Xe GPU DPAS atoms.

Validates the (thread, value) -> element mappings for all Xe-HPC and
Xe-HPG DPAS atom definitions using algebraic invariants — no hardware required.
"""

import pytest

from tensor_layouts import Layout, size, rank, cosize
from tensor_layouts.atoms_xe import *


def _num_threads(layout):
    return size(layout.shape[0]) if isinstance(layout.shape, tuple) else size(layout.shape)


def _num_values(layout):
    return size(layout.shape[1]) if isinstance(layout.shape, tuple) else 1


ALL_ATOMS = MMA_ATOMS_XeHPC + MMA_ATOMS_XeHPG


@pytest.mark.parametrize("atom", ALL_ATOMS, ids=lambda a: a.name)
class TestDPASStructural:
    """Structural invariants that must hold for any valid DPAS atom."""

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

        assert len(seen) == m * n

    def test_c_layout_thread_count(self, atom):
        """Thread count matches N dimension (subgroup size)."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        assert _num_threads(c) == n

    def test_a_layout_thread_count(self, atom):
        m, n, k = atom.shape_mnk
        a = atom.a_layout
        assert _num_threads(a) == n

    def test_b_layout_thread_count(self, atom):
        m, n, k = atom.shape_mnk
        b = atom.b_layout
        assert _num_threads(b) == n

    def test_a_layout_broadcast(self, atom):
        """A layout broadcasts across subgroup (stride 0 on thread dim)."""
        a = atom.a_layout
        t_stride = a.stride[0] if isinstance(a.stride, tuple) else a.stride
        assert t_stride == 0, \
            f"{atom.name}: A thread stride is {t_stride}, expected 0 (broadcast)"

    def test_c_layout_cosize_equals_mn(self, atom):
        m, n, k = atom.shape_mnk
        assert cosize(atom.c_layout) == m * n

    def test_b_layout_cosize_equals_nk(self, atom):
        m, n, k = atom.shape_mnk
        assert cosize(atom.b_layout) == n * k

    def test_thr_id_is_none(self, atom):
        assert atom.thr_id is None

    def test_c_layout_rank_is_2(self, atom):
        assert rank(atom.c_layout) == 2

    def test_a_layout_rank_is_2(self, atom):
        assert rank(atom.a_layout) == 2

    def test_b_layout_rank_is_2(self, atom):
        assert rank(atom.b_layout) == 2

    def test_c_layout_column_ownership(self, atom):
        """Each thread owns exactly one column of C."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        num_t = _num_threads(c)
        num_v = _num_values(c)

        for t in range(num_t):
            cols = set()
            for v in range(num_v):
                offset = c(t, v)
                col = offset // m  # col-major: col = offset // M
                cols.add(col)
            assert len(cols) == 1, \
                f"{atom.name}: thread {t} touches columns {cols}, expected 1"
            assert cols.pop() == t, \
                f"{atom.name}: thread {t} owns wrong column"

    def test_b_layout_column_ownership(self, atom):
        """Each thread owns exactly one N-position of B."""
        m, n, k = atom.shape_mnk
        b = atom.b_layout
        num_t = _num_threads(b)
        num_v = _num_values(b)

        for t in range(num_t):
            ns = set()
            for v in range(num_v):
                offset = b(t, v)
                n_pos = offset % n  # col-major in N×K: n = offset % N
                ns.add(n_pos)
            assert len(ns) == 1
            assert ns.pop() == t


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
