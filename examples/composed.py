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

"""Examples of exact composed-layout semantics.

Run:
    python examples/composed.py

Optional figures:
    python examples/composed.py --draw docs/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensor_layouts import *
from tensor_layouts.tensor import Tensor


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def example_fast_path() -> None:
    _banner("1. Canonical fast path: compose(Swizzle, Layout) stays a Layout")

    base = Layout((4, 4), (4, 1))
    swizzled = compose(Swizzle(2, 0, 2), base)

    print("base     :", base)
    print("swizzled :", swizzled)
    print("type     :", type(swizzled).__name__)
    print("is_layout:", is_layout(swizzled))
    print("affine   :", is_affine(swizzled))

    assert isinstance(swizzled, Layout)
    assert swizzled.swizzle == Swizzle(2, 0, 2)


def example_exact_fallback() -> LayoutExpr:
    _banner("2. Exact fallback: compositions that do not collapse return ComposedLayout")

    base = Layout((4, 4), (4, 1))
    swizzled = compose(Swizzle(2, 0, 2), base)
    exact = compose(Layout(16, 2), swizzled)

    print("inner :", swizzled)
    print("outer :", Layout(16, 2))
    print("exact :", exact)
    print("type  :", type(exact).__name__)
    print("shape :", exact.shape)
    print("size  :", size(exact))
    print("cosize (CuTe-style domain codomain):", cosize(exact))

    assert isinstance(exact, ComposedLayout)
    for i in range(size(exact)):
        assert exact(i) == Layout(16, 2)(swizzled(i))

    return exact


def example_double_swizzle() -> None:
    _banner("3. Double swizzle stays exact instead of overwriting the inner swizzle")

    base = Layout((8, 8), (8, 1))
    inner = compose(Swizzle(3, 0, 3), base)
    outer = Swizzle(1, 0, 3)
    result = compose(outer, inner)

    print("inner  :", inner)
    print("outer  :", outer)
    print("result :", result)
    print("type   :", type(result).__name__)

    assert isinstance(result, ComposedLayout)
    for i in range(8):
        for j in range(8):
            assert result(i, j) == outer(inner(i, j))


def example_slicing_and_tensor(exact: LayoutExpr) -> None:
    _banner("4. Slicing keeps composed preoffset inside the layout")

    sub, offset = slice_and_offset((2, None), exact)
    print("slice_and_offset((2, None), exact)")
    print("  sub    :", sub)
    print("  offset :", offset)

    assert isinstance(sub, ComposedLayout)
    assert offset == 0
    for j in range(4):
        assert sub(j) == exact(2, j)

    tensor = Tensor(exact, offset=100, data=list(range(512)))
    row = tensor[2, :]
    print("\nTensor(exact, offset=100)")
    print("  tensor :", tensor)
    print("  row    :", row)
    print("  row.offset:", row.offset)
    print("  row.layout:", row.layout)

    assert row.offset == 100
    for j in range(4):
        assert row(j) == tensor(2, j)
        assert row[j] == tensor[2, j]

    try:
        _ = tensor.stride
    except TypeError as exc:
        print("\nTensor.stride on ComposedLayout is affine-only:")
        print(" ", exc)


def maybe_draw(exact: LayoutExpr, outdir: Path | None) -> None:
    if outdir is None:
        return

    from tensor_layouts.viz import draw_layout, draw_slice

    outdir.mkdir(parents=True, exist_ok=True)
    draw_layout(exact, outdir / "composed_exact.png", title="Exact composed layout", colorize=True)
    draw_slice(
        exact,
        (None, 1),
        outdir / "composed_slice.png",
        title="Slice keeps composed offset internal",
        colorize=True,
    )
    print(f"\nWrote figures to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=Path, default=None, help="Optional directory for PNG output")
    args = parser.parse_args()

    example_fast_path()
    exact = example_exact_fallback()
    example_double_swizzle()
    example_slicing_and_tensor(exact)
    maybe_draw(exact, args.draw)


if __name__ == "__main__":
    main()
