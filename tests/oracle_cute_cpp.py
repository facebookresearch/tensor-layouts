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

"""Direct CuTe C++ oracle tests for regression cases.

These tests compile a tiny C++ program against locally installed CUTLASS/CuTe
headers and compare exact layout results for the regressions fixed in Python.
Unlike the pycute oracle, this validates behavior against the CuTe C++ source
of truth.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

from tensor_layouts import *


CPP_SOURCE = r"""
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/swizzle_layout.hpp>
#include <iostream>

using namespace cute;

template <class Layout>
void print_offsets(char const* name, Layout const& layout) {
  std::cout << name << "=";
  auto n = static_cast<int>(size(layout));
  for (int i = 0; i < n; ++i) {
    if (i) {
      std::cout << ",";
    }
    std::cout << layout(i);
  }
  std::cout << "\n";
}

int main() {
  auto compose_nested_tuple_tiler = composition(
      make_layout(make_shape(make_shape(_2{}, _3{}), _8{}),
                  make_stride(make_stride(_1{}, _2{}), _6{})),
      make_tile(make_shape(_2{}, _3{}), _4{}));
  std::cout << "compose_nested_tuple_tiler=" << compose_nested_tuple_tiler << "\n";

  auto logical_divide_unit_tile = logical_divide(make_layout(_2{}, _5{}), _1{});
  std::cout << "logical_divide_unit_tile=" << logical_divide_unit_tile << "\n";

  auto logical_divide_exact_division_unit_rest = logical_divide(make_layout(_4{}, _3{}), _4{});
  std::cout << "logical_divide_exact_division_unit_rest="
            << logical_divide_exact_division_unit_rest << "\n";

  auto logical_divide_oversize_tile = logical_divide(make_layout(_2{}, _5{}), _4{});
  std::cout << "logical_divide_oversize_tile=" << logical_divide_oversize_tile << "\n";

  auto logical_divide_exact_tuple = logical_divide(
      make_layout(make_shape(_2{}, _3{}), make_stride(_1{}, _2{})),
      make_shape(_2{}, _3{}));
  std::cout << "logical_divide_exact_tuple=" << logical_divide_exact_tuple << "\n";

  auto logical_divide_nested_tuple_tiler = logical_divide(
      make_layout(make_shape(make_shape(_2{}, _3{}), _8{}),
                  make_stride(make_stride(_1{}, _2{}), _6{})),
      make_shape(make_shape(_2{}, _3{}), _4{}));
  std::cout << "logical_divide_nested_tuple_tiler=" << logical_divide_nested_tuple_tiler
            << "\n";

  auto layout_tiler_1d = make_layout(_4{}, _2{});
  auto zipped_divide_layout_tiler_1d = zipped_divide(make_layout(_8{}, _1{}), layout_tiler_1d);
  std::cout << "zipped_divide_layout_tiler_1d=" << zipped_divide_layout_tiler_1d << "\n";

  auto layout_tiler_2d =
      make_layout(make_shape(_2{}, _2{}), make_stride(_1{}, _4{}));
  auto data_2d = make_layout(make_shape(_8{}, _8{}), make_stride(_1{}, _8{}));
  auto zipped_divide_layout_tiler_2d = zipped_divide(data_2d, layout_tiler_2d);
  auto tiled_divide_layout_tiler_2d = tiled_divide(data_2d, layout_tiler_2d);
  auto flat_divide_layout_tiler_2d = flat_divide(data_2d, layout_tiler_2d);
  std::cout << "zipped_divide_layout_tiler_2d=" << zipped_divide_layout_tiler_2d << "\n";
  std::cout << "tiled_divide_layout_tiler_2d=" << tiled_divide_layout_tiler_2d << "\n";
  std::cout << "flat_divide_layout_tiler_2d=" << flat_divide_layout_tiler_2d << "\n";

  auto compose_truncation = composition(
      make_layout(make_shape(_4{}, _2{}, _8{}), make_stride(_3{}, _12{}, Int<97>{})),
      make_layout(_3{}, _3{}));
  std::cout << "compose_truncation=" << compose_truncation << "\n";

  auto left_inverse_padded =
      left_inverse(make_layout(make_shape(_4{}, _8{}), make_stride(_1{}, _5{})));
  std::cout << "left_inverse_padded=" << left_inverse_padded << "\n";

  auto logical_product_layout_tiler = logical_product(
      make_layout(make_shape(_2{}, _2{}), make_stride(_1{}, _4{})),
      make_layout(make_shape(_2{}, _3{}), make_stride(_1{}, _4{})));
  std::cout << "logical_product_layout_tiler=" << logical_product_layout_tiler << "\n";

  auto slice_full_rank2 =
      slice(_, make_layout(make_shape(_4{}, _8{}), make_stride(_1{}, _4{})));
  std::cout << "slice_full_rank2=" << slice_full_rank2 << "\n";

  auto slice_full_scalar = slice(_, make_layout(_4{}, _1{}));
  std::cout << "slice_full_scalar=" << slice_full_scalar << "\n";

  auto double_swizzle_base =
      make_layout(make_shape(_8{}, _8{}), make_stride(_8{}, _1{}));
  auto double_swizzle_inner = composition(Swizzle<3,0,3>{}, double_swizzle_base);
  auto double_swizzle = composition(Swizzle<1,0,3>{}, double_swizzle_inner);
  print_offsets("compose_double_swizzle_offsets", double_swizzle);

  auto outer_layout = make_layout(make_shape(_4{}, _4{}), make_stride(_4{}, _1{}));
  auto swizzled_inner = composition(Swizzle<3,0,3>{}, make_layout(_16{}, _1{}));
  auto outer_on_swizzled = composition(outer_layout, swizzled_inner);
  print_offsets("compose_outer_layout_swizzled_offsets", outer_on_swizzled);
}
"""


PYTHON_CASES = {
    "compose_nested_tuple_tiler": lambda: compose(
        Layout(((2, 3), 8), ((1, 2), 6)),
        ((2, 3), 4),
    ),
    "logical_divide_unit_tile": lambda: logical_divide(Layout(2, 5), 1),
    "logical_divide_exact_division_unit_rest": lambda: logical_divide(Layout(4, 3), 4),
    "logical_divide_oversize_tile": lambda: logical_divide(Layout(2, 5), 4),
    "logical_divide_exact_tuple": lambda: logical_divide(Layout((2, 3), (1, 2)), (2, 3)),
    "logical_divide_nested_tuple_tiler": lambda: logical_divide(
        Layout(((2, 3), 8), ((1, 2), 6)),
        ((2, 3), 4),
    ),
    "zipped_divide_layout_tiler_1d": lambda: zipped_divide(Layout(8, 1), Layout(4, 2)),
    "zipped_divide_layout_tiler_2d": lambda: zipped_divide(
        Layout((8, 8), (1, 8)),
        Layout((2, 2), (1, 4)),
    ),
    "tiled_divide_layout_tiler_2d": lambda: tiled_divide(
        Layout((8, 8), (1, 8)),
        Layout((2, 2), (1, 4)),
    ),
    "flat_divide_layout_tiler_2d": lambda: flat_divide(
        Layout((8, 8), (1, 8)),
        Layout((2, 2), (1, 4)),
    ),
    "compose_truncation": lambda: compose(
        Layout((4, 2, 8), (3, 12, 97)),
        Layout(3, 3),
    ),
    "left_inverse_padded": lambda: left_inverse(Layout((4, 8), (1, 5))),
    "logical_product_layout_tiler": lambda: logical_product(
        Layout((2, 2), (1, 4)),
        Layout((2, 3), (1, 4)),
    ),
    "slice_full_rank2": lambda: Layout((4, 8), (1, 4))(None),
    "slice_full_scalar": lambda: Layout(4, 1)(None),
}


PYTHON_POINTWISE_CASES = {
    "compose_double_swizzle_offsets": lambda: ",".join(
        str(compose(Swizzle(1, 0, 3), compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1))))(i))
        for i in range(size(Layout((8, 8), (8, 1))))
    ),
    "compose_outer_layout_swizzled_offsets": lambda: ",".join(
        str(compose(Layout((4, 4), (4, 1)), compose(Swizzle(3, 0, 3), Layout(16, 1)))(i))
        for i in range(16)
    ),
}


def _module_root(module_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    if spec.origin is not None:
        return Path(spec.origin).parent
    return None


def _candidate_include_dirs() -> list[Path]:
    candidates = [
        (_module_root("cutlass_library"), "source/include"),
        (_module_root("nvidia.cuda_runtime"), "include"),
        (_module_root("triton.backends.nvidia"), "include"),
        (Path("/usr/local/cuda/include"), ""),
        (Path("/usr/local/cuda-12.8/targets/sbsa-linux/include"), ""),
    ]

    include_dirs = []
    seen = set()
    for root, relative in candidates:
        if root is None:
            continue
        path = (root / relative) if relative else root
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        include_dirs.append(resolved)
    return include_dirs


def _normalize_layout_repr(text: str) -> str:
    return text.replace(" ", "").replace("_", "")


@pytest.fixture(scope="session")
def cute_cpp_oracle() -> dict[str, str]:
    compiler = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("CuTe C++ oracle requires a C++ compiler")

    include_dirs = _candidate_include_dirs()
    if not any((path / "cute/layout.hpp").exists() for path in include_dirs):
        pytest.skip("CuTe headers not found; install CUTLASS headers to run this oracle")
    if not any((path / "cuda/std/utility").exists() for path in include_dirs):
        pytest.skip("CUDA C++ headers not found; CuTe C++ oracle cannot compile")

    with tempfile.TemporaryDirectory() as tmpdir:
        cpp_path = Path(tmpdir) / "oracle.cpp"
        exe_path = Path(tmpdir) / "oracle"
        cpp_path.write_text(CPP_SOURCE, encoding="ascii")

        command = [compiler, "-std=c++17", str(cpp_path), "-o", str(exe_path)]
        for include_dir in include_dirs:
            command.extend(["-I", str(include_dir)])

        compile_result = subprocess.run(command, capture_output=True, text=True, check=False)
        if compile_result.returncode != 0:
            stderr = compile_result.stderr.strip() or "unknown compiler error"
            pytest.skip(f"failed to compile CuTe oracle: {stderr}")

        run_result = subprocess.run([str(exe_path)], capture_output=True, text=True, check=False)
        if run_result.returncode != 0:
            stderr = run_result.stderr.strip() or "unknown runtime error"
            pytest.skip(f"failed to run CuTe oracle: {stderr}")

    outputs = {}
    for line in run_result.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        outputs[name.strip()] = value.strip()
    return outputs


@pytest.mark.parametrize("case_name", sorted(PYTHON_CASES))
def test_cute_cpp_oracle(case_name, cute_cpp_oracle):
    result = PYTHON_CASES[case_name]()
    assert _normalize_layout_repr(str(result)) == _normalize_layout_repr(cute_cpp_oracle[case_name])


@pytest.mark.parametrize("case_name", sorted(PYTHON_POINTWISE_CASES))
def test_cute_cpp_pointwise_oracle(case_name, cute_cpp_oracle):
    assert PYTHON_POINTWISE_CASES[case_name]() == cute_cpp_oracle[case_name]
