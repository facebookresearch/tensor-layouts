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

"""Intel Xe GPU DPAS (Dot Product Accumulate Systolic) atom definitions.

Mirrors the lane-to-element mapping of Intel's DPAS instruction on Xe
architectures:
  - Xe-HPC (Ponte Vecchio / Data Center Max): subgroup_size = 8
  - Xe-HPG (Arc / DG2):                       subgroup_size = 16

Each DPAS atom maps (thread_idx, value_idx) -> element coordinate using
column-major encoding:
    A: (T, V) -> m + k*M   in the M × K matrix
    B: (T, V) -> n + k*N   in the N × K matrix
    C: (T, V) -> m + n*M   in the M × N matrix

DPAS Register Layout
====================

DPAS is a subgroup-cooperative systolic instruction:

    dpas.{systolic_depth}x{repeat_count} (exec_size) dst src0 src1 src2

    systolic_depth (sd):  8 (fixed on all Xe architectures) — the K dimension
    repeat_count (rc):    1–8 — the M dimension (typically 8 for peak throughput)
    exec_size:            N dimension = subgroup_size (8 for Xe-HPC, 16 for Xe-HPG)

C accumulator (M × N):
    Each work item (lane t) owns one column of C.
    Value v (0..rc-1) selects the row.

        col = t,  row = v
        offset = row + col * M = v + t * M

A operand (M × K):
    A is broadcast across the subgroup — all lanes see the same A data.
    Thread dimension has stride 0.

        offset = m + k * M    (col-major in M × K)

B operand (N × K, CuTe convention):
    Each lane t owns one column of B (one N-position).
    Value v selects the K-position.

        n = t,  k = v
        offset = n + k * N = t + v * N

References
----------

- Intel Graphics Compiler (IGC) VISA specification: DPAS.md
  https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
- Khronos OpenCL extension: cl_intel_subgroup_matrix_multiply_accumulate
- Intel XeTLA (Xe Templates for Linear Algebra):
  https://intel.github.io/xetla/
- CUTLASS experimental Xe backend (CuTe atom definitions for DPAS)

.. note:: Community feedback welcome

   These atom definitions were derived from public Intel ISA documentation
   and the CUTLASS experimental Xe backend.  If you find an incorrect
   mapping, please open an issue at
   https://github.com/facebookresearch/tensor-layouts/issues — both bug
   reports and confirmations that layouts match real hardware are valuable.

Usage::

    from tensor_layouts.atoms_xe import XeHPC_8x8x8_F32F16F16_DPAS
    print(XeHPC_8x8x8_F32F16F16_DPAS.c_layout)
"""

from .atoms import MMAAtom
from .layouts import Layout


# =============================================================================
# Helper: construct CuTe layouts from DPAS structural parameters
# =============================================================================

def _dpas_c_layout(m: int, n: int) -> Layout:
    """Build the (T_n, V_m) -> col-major(M, N) accumulator layout.

    Lane t owns column t, value v selects row v.
    offset = v + t * M
    """
    return Layout(
        (n, m),
        (m, 1),
    )


def _dpas_a_layout(m: int, k: int, n: int) -> Layout:
    """Build the (T_n, V_{m*k}) -> col-major(M, K) input layout for A.

    A is broadcast across the subgroup (stride 0 on thread dimension).
    All lanes see the same M × K tile.
    """
    return Layout(
        (n, (m, k)),
        (0, (1, m)),
    )


def _dpas_b_layout(n: int, k: int) -> Layout:
    """Build the (T_n, V_k) -> col-major(N, K) input layout for B.

    Lane t owns column t (N dimension), value v selects K position.
    offset = t + v * N
    """
    return Layout(
        (n, k),
        (1, n),
    )


def make_dpas_atom(
    name: str,
    inst: str,
    m: int,
    n: int,
    k: int,
) -> MMAAtom:
    """Create an Intel Xe DPAS atom.

    Args:
        m: repeat_count (rows of output, typically 8)
        n: exec_size = subgroup_size (8 for Xe-HPC, 16 for Xe-HPG)
        k: systolic_depth (always 8 on current Xe)
    """
    c_layout = _dpas_c_layout(m, n)
    a_layout = _dpas_a_layout(m, k, n)
    b_layout = _dpas_b_layout(n, k)

    return MMAAtom(
        name=name,
        ptx=inst,
        shape_mnk=(m, n, k),
        thr_id=None,   # identity: lane_id = thread_idx % subgroup_size
        a_layout=a_layout,
        b_layout=b_layout,
        c_layout=c_layout,
    )


# =============================================================================
# Xe-HPC (Ponte Vecchio / Data Center Max) — subgroup_size = 8
# DPAS shape: rc × 8 × sd = 8 × 8 × 8
# =============================================================================

# --- FP16 input, FP32 accumulator ---
XeHPC_8x8x8_F32F16F16_DPAS = make_dpas_atom(
    name="XeHPC_8x8x8_F32F16F16_DPAS",
    inst="dpas.8x8 (exec_size=8, FP16)",
    m=8, n=8, k=8,
)

# --- BF16 input, FP32 accumulator ---
XeHPC_8x8x8_F32BF16BF16_DPAS = make_dpas_atom(
    name="XeHPC_8x8x8_F32BF16BF16_DPAS",
    inst="dpas.8x8 (exec_size=8, BF16)",
    m=8, n=8, k=8,
)

# --- TF32 input, FP32 accumulator ---
XeHPC_8x8x8_F32TF32TF32_DPAS = make_dpas_atom(
    name="XeHPC_8x8x8_F32TF32TF32_DPAS",
    inst="dpas.8x8 (exec_size=8, TF32)",
    m=8, n=8, k=8,
)

# --- INT8 input, INT32 accumulator ---
XeHPC_8x8x8_I32I8I8_DPAS = make_dpas_atom(
    name="XeHPC_8x8x8_I32I8I8_DPAS",
    inst="dpas.8x8 (exec_size=8, INT8)",
    m=8, n=8, k=8,
)


# =============================================================================
# Xe-HPG (Arc / DG2) — subgroup_size = 16
# DPAS shape: rc × 16 × sd = 8 × 16 × 8
# =============================================================================

# --- FP16 input, FP32 accumulator ---
XeHPG_8x16x8_F32F16F16_DPAS = make_dpas_atom(
    name="XeHPG_8x16x8_F32F16F16_DPAS",
    inst="dpas.8x8 (exec_size=16, FP16)",
    m=8, n=16, k=8,
)

# --- BF16 input, FP32 accumulator ---
XeHPG_8x16x8_F32BF16BF16_DPAS = make_dpas_atom(
    name="XeHPG_8x16x8_F32BF16BF16_DPAS",
    inst="dpas.8x8 (exec_size=16, BF16)",
    m=8, n=16, k=8,
)

# --- INT8 input, INT32 accumulator ---
XeHPG_8x16x8_I32I8I8_DPAS = make_dpas_atom(
    name="XeHPG_8x16x8_I32I8I8_DPAS",
    inst="dpas.8x8 (exec_size=16, INT8)",
    m=8, n=16, k=8,
)


# =============================================================================
# Convenience lists
# =============================================================================

MMA_ATOMS_XeHPC = [
    XeHPC_8x8x8_F32F16F16_DPAS,
    XeHPC_8x8x8_F32BF16BF16_DPAS,
    XeHPC_8x8x8_F32TF32TF32_DPAS,
    XeHPC_8x8x8_I32I8I8_DPAS,
]

MMA_ATOMS_XeHPG = [
    XeHPG_8x16x8_F32F16F16_DPAS,
    XeHPG_8x16x8_F32BF16BF16_DPAS,
    XeHPG_8x16x8_I32I8I8_DPAS,
]
