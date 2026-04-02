#!/usr/bin/env python3
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

"""Tensor Storage Examples — data-backed tensors.

Demonstrates attaching storage to a Tensor so that coordinates return
actual data elements, writes go through the layout mapping, and
draw_layout auto-labels cells with data values.

Run:
    python tensor.py

See also:
    layouts.py   — pure algebraic layout examples (no dependencies)
    viz.py       — visualization examples (requires matplotlib)
    viz.ipynb    — Jupyter notebook gallery
"""

from pathlib import Path
import sys

# Prefer the local repo sources when running this script from a checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tensor_layouts import *
from tensor_layouts.viz import draw_layout, draw_composite


def setup_output_dir(name: str = "examples_output") -> Path:
    output = Path(name)
    output.mkdir(parents=True, exist_ok=True)
    return output


# =============================================================================
# Section 1: Attaching Storage
# =============================================================================

def example_storage():
    """Attach storage to a Tensor for element-level access.

    Without storage, tensor[i, j] returns the raw memory offset.
    With storage, tensor[i, j] returns data[offset] — the actual element.
    """
    print("\n" + "=" * 60)
    print("1. Attaching Storage")
    print("=" * 60)

    layout = Layout((4, 8), (8, 1))  # row-major 4x8

    # --- Algebraic (no storage) ---
    t_alg = Tensor(layout)
    print(f"  Algebraic: t[2, 3] = {t_alg[2, 3]}  (raw offset)")

    # --- Data-backed ---
    buf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    t = Tensor(layout, data=buf)
    print(f"  With data: t[2, 3] = {t[2, 3]!r}  (data[{t(2, 3)}])")
    print(f"  t(2, 3) still returns the offset: {t(2, 3)}")

    # Storage can be larger than the layout's codomain
    big_buf = list(range(100))
    t_big = Tensor(layout, data=big_buf)
    print(f"\n  Storage length {len(big_buf)} >= cosize {cosize(layout)}: OK")

    # Storage too small raises ValueError
    try:
        Tensor(layout, data=[0, 1, 2])
    except ValueError as e:
        print(f"  Too small: {e}")


# =============================================================================
# Section 2: Reading Through Coordinates
# =============================================================================

def example_reading():
    """Access elements using logical coordinates.

    The layout maps (row, col) to a flat offset; storage provides the
    value at that offset.  This is how CuTe's Tensor works: the layout
    defines the access pattern, the storage holds the data.
    """
    print("\n" + "=" * 60)
    print("2. Reading Through Coordinates")
    print("=" * 60)

    # Row-major: offset(i, j) = 8*i + j
    row_major = Layout((4, 8), (8, 1))
    buf = list(range(32))
    t = Tensor(row_major, data=buf)

    print(f"  Row-major layout: {row_major}")
    print(f"  Row 0: {[t[0, j] for j in range(8)]}")
    print(f"  Row 1: {[t[1, j] for j in range(8)]}")
    print(f"  Col 0: {[t[i, 0] for i in range(4)]}")

    # Column-major: offset(i, j) = i + 4*j
    col_major = Layout((4, 8), (1, 4))
    t2 = Tensor(col_major, data=buf)
    print(f"\n  Col-major layout: {col_major}")
    print(f"  Row 0: {[t2[0, j] for j in range(8)]}")
    print(f"  Col 0: {[t2[i, 0] for i in range(4)]}")

    # Same data, different arrangement — this is the power of layouts
    print(f"\n  Same data, different layouts:")
    print(f"  row_major[1, 2] = {t[1, 2]}  (data[{t(1, 2)}])")
    print(f"  col_major[1, 2] = {t2[1, 2]}  (data[{t2(1, 2)}])")


# =============================================================================
# Section 3: Writing Through Coordinates
# =============================================================================

def example_writing():
    """Write to storage through logical coordinates.

    tensor[i, j] = val writes to data[offset], where offset is computed
    by the layout.  This lets you populate storage using whichever
    coordinate system the layout defines.
    """
    print("\n" + "=" * 60)
    print("3. Writing Through Coordinates")
    print("=" * 60)

    layout = Layout((4, 8), (8, 1))
    buf = [0] * 32
    t = Tensor(layout, data=buf)

    # Write sequential values through row-major coordinates
    for i in range(4):
        for j in range(8):
            t[i, j] = i * 10 + j

    print(f"  After writing i*10+j through row-major:")
    for i in range(4):
        print(f"    Row {i}: {[t[i, j] for j in range(8)]}")

    print(f"\n  Underlying storage (first 16): {buf[:16]}")


# =============================================================================
# Section 4: View Semantics (Slicing Shares Data)
# =============================================================================

def example_views():
    """Slicing produces sub-Tensors that share storage.

    Just like numpy views, writing through a sub-Tensor modifies the
    same underlying buffer.  This is how CuTe's Tensor slicing works:
    the sub-Tensor gets a new layout and accumulated offset, but the
    pointer (data reference) stays the same.
    """
    print("\n" + "=" * 60)
    print("4. View Semantics")
    print("=" * 60)

    buf = list(range(32))
    t = Tensor(Layout((4, 8), (8, 1)), data=buf)

    # Slice row 2
    row2 = t[2, :]
    print(f"  t[2, :] = {row2}")
    print(f"  row2[3] = {row2[3]}")
    print(f"  t[2, 3] = {t[2, 3]}  (same value)")
    print(f"  row2.data is t.data: {row2.data is t.data}")

    # Write through the slice
    row2[3] = 999
    print(f"\n  After row2[3] = 999:")
    print(f"  t[2, 3] = {t[2, 3]}  (visible through parent)")
    print(f"  buf[{t(2, 3)}] = {buf[t(2, 3)]}  (visible in raw buffer)")

    # Chained slicing: tensor[i, :][j] == tensor[i, j]
    print(f"\n  Chained slice consistency:")
    for i in range(4):
        for j in range(8):
            assert t[i, :][j] == t[i, j]
    print(f"  t[i, :][j] == t[i, j] for all (i, j): OK")


# =============================================================================
# Section 5: Swapping Storage
# =============================================================================

def example_swap():
    """The data property is writable — swap storage without rebuilding.

    Existing sub-Tensors keep their reference to the old storage.
    This is the same tradeoff numpy makes with ndarray.data.
    """
    print("\n" + "=" * 60)
    print("5. Swapping Storage")
    print("=" * 60)

    layout = Layout((4, 8), (8, 1))
    buf1 = list(range(32))
    buf2 = list(range(100, 132))

    t = Tensor(layout, data=buf1)
    row = t[2, :]
    print(f"  Before swap: t[0, 0] = {t[0, 0]}, row[0] = {row[0]}")

    t.data = buf2
    print(f"  After swap:  t[0, 0] = {t[0, 0]}, row[0] = {row[0]}")
    print(f"  (row still references old storage)")

    # Remove storage entirely
    t.data = None
    print(f"  After t.data = None: t[0, 0] = {t[0, 0]}  (offset, no data)")


# =============================================================================
# Section 6: Two Layouts, One Buffer
# =============================================================================

def example_two_layouts():
    """Attach the same storage to two Tensors with different layouts.

    This demonstrates the core idea: the layout determines the mapping
    from coordinates to physical positions, but the data is shared.
    """
    print("\n" + "=" * 60)
    print("6. Two Layouts, One Buffer")
    print("=" * 60)

    buf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")

    row_major = Tensor(Layout((4, 8), (8, 1)), data=buf)
    col_major = Tensor(Layout((4, 8), (1, 4)), data=buf)

    print(f"  Same buffer viewed through two layouts:")
    print(f"  Row-major [0,:]: {''.join(str(row_major[0, j]) for j in range(8))}")
    print(f"  Col-major [0,:]: {''.join(str(col_major[0, j]) for j in range(8))}")
    print()
    print(f"  Row-major [1,0] = {row_major[1, 0]!r}  (data[8])")
    print(f"  Col-major [1,0] = {col_major[1, 0]!r}  (data[1])")


# =============================================================================
# Section 7: Auto-Labeling in draw_layout
# =============================================================================

def example_auto_label(output: Path):
    """draw_layout auto-labels cells with data values.

    When a Tensor with storage is passed to draw_layout, cells show
    data values instead of raw offsets.  Override with cell_labels="offset"
    or cell_labels=False.
    """
    print("\n" + "=" * 60)
    print("7. Auto-Labeling in draw_layout")
    print("=" * 60)

    layout = Layout((4, 8), (8, 1))
    buf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
    t = Tensor(layout, data=buf)

    # Auto-label: cells show A, B, C, ...
    draw_layout(t, output / "tensor_data_auto.svg",
                title="Auto-labeled from data")
    print(f"  tensor_data_auto.svg — cells show data values")

    # Override: show raw offsets
    draw_layout(t, output / "tensor_data_offsets.svg",
                title="cell_labels='offset'", cell_labels="offset")
    print(f"  tensor_data_offsets.svg — cells show offsets")

    # Override: no labels
    draw_layout(t, output / "tensor_data_nolabels.svg",
                title="cell_labels=False", colorize=True, cell_labels=False)
    print(f"  tensor_data_nolabels.svg — no cell text")

    # Side-by-side: same data, two layouts
    row_t = Tensor(Layout((4, 8), (8, 1)), data=buf)
    col_t = Tensor(Layout((4, 8), (1, 4)), data=buf)
    draw_composite([row_t, col_t],
                   output / "tensor_data_compare.svg",
                   titles=["Row-major", "Col-major"],
                   main_title="Same data, different layouts")
    print(f"  tensor_data_compare.svg — row vs col-major, same data")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all Tensor storage examples."""
    output = setup_output_dir()

    print("=" * 70)
    print("Tensor Storage Examples")
    print("=" * 70)

    example_storage()
    example_reading()
    example_writing()
    example_views()
    example_swap()
    example_two_layouts()
    example_auto_label(output)

    print("\n" + "=" * 70)
    print("All examples completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
