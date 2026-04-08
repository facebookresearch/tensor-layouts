<!--
MIT License

Copyright (c) 2026 Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Tensor API

A `Tensor` combines a `Layout` with a base offset and optional storage.
It is the Python equivalent of CuTe's `(Pointer, Layout)` pair.

For layout algebra see [`docs/layout_api.md`](layout_api.md).
For visualization see [`docs/viz_api.md`](viz_api.md).

## What is a Tensor?

A `Layout` is a pure function from coordinates to offsets.  A `Tensor`
adds two things:

1. **A base offset** — an integer that shifts every computed offset,
   modelling a pointer into a larger memory space.
2. **Optional storage** — any indexable object (list, numpy array,
   torch tensor, etc.) so that element access returns actual data
   values rather than raw offset integers.

When storage is absent the Tensor is purely algebraic and behaves
exactly like a `(offset, Layout)` pair.  When storage is present,
indexing reads and writes go through the layout mapping to reach
the correct position in the flat storage buffer.

## Construction

| Form | Description |
|------|-------------|
| `Tensor(layout)` | Algebraic, offset 0, no storage |
| `Tensor(layout, offset)` | Algebraic with explicit base offset |
| `Tensor(layout, data=buf)` | Data-backed, offset 0 |
| `Tensor(layout, offset, data=buf)` | Data-backed with explicit base offset |

The storage must cover every index addressed by the `(offset, layout)`
pair.  For the common zero-offset, nonnegative-stride case this reduces
to `len(data) >= cosize(layout)`.  It is stored by reference (no copy).
Storage that is too small raises `ValueError`.

```python
from tensor_layouts import Layout, Tensor

layout = Layout((4, 8), (8, 1))

# Algebraic
t = Tensor(layout)
t(2, 5)   # 21

# Data-backed
buf = list(range(32))
t = Tensor(layout, data=buf)
t[2, 5]   # buf[21] → 21
```

## Coordinate Mapping — `__call__`

`tensor(i, j)` always returns the **memory offset** (an integer),
regardless of whether storage is present.  For swizzled layouts the
swizzle is applied to the total linear offset:

```
tensor(i, j) = swizzle(base_offset + crd2offset((i, j), shape, stride))
```

This is unaffected by storage — use `__call__` when you need the raw
offset, and `__getitem__` when you want the data element.

```python
t = Tensor(Layout((4, 8), (8, 1)), data=list(range(32)))
t(2, 3)    # 19  — always the offset
t[2, 3]    # 19  — data[19] (happens to equal 19 here)
```

## Element Access — `__getitem__`

A bare integer performs **flat 1D evaluation** on any-rank tensor: the
index is decomposed via `idx2crd` into the natural coordinate and the
offset is computed.  This matches CuTe C++ `Tensor::operator()(int)`.

| Key | Returns |
|-----|---------|
| `tensor[i]` | Flat 1D evaluation — data element or offset |
| `tensor[i, j]` | All modes fixed — data element or offset |
| `tensor[i, :]` | Slicing — sub-Tensor (see [Slicing](#slicing)) |

When storage is present, fully-resolved accesses return `data[offset]`.
When absent, they return the raw offset integer.

```python
buf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
t = Tensor(Layout((4, 8), (8, 1)), data=buf)
t[0, 0]    # 'A'  — buf[0]
t[0, 1]    # 'B'  — buf[1]
t[1, 0]    # 'I'  — buf[8]
t[3, 7]    # '5'  — buf[31]
```

### Flat 1D evaluation

`tensor[i]` decomposes the flat index `i` into coordinates via `idx2crd`,
then computes the offset — even on rank-2+ tensors.  This is consistent
with `__setitem__` and enables the canonical copy loop:

```python
def copy(src: Tensor, dst: Tensor):
    assert size(src.layout) == size(dst.layout)
    for i in range(size(dst.layout)):
        dst[i] = src[i]
```

Because `src` and `dst` can have different layouts (e.g. row-major vs
column-major), `copy` automatically remaps elements through each tensor's
layout function:

```python
row_major = Layout((4, 8), (8, 1))
col_major = Layout((4, 8), (1, 4))

src = Tensor(row_major, data=list(range(32)))
dst = Tensor(col_major, data=[0] * 32)

for i in range(size(row_major)):
    dst[i] = src[i]

# Same logical element at every coordinate:
assert src[2, 5] == dst[2, 5]
```

To slice mode 0 (the old `tensor[i]` behavior), use `tensor[i, :]`
explicitly.

## Element Assignment — `__setitem__`

Scalar writes are supported when storage is present and mutable:

```python
t[2, 3] = 'X'     # writes buf[19] = 'X'
t[2, 3]            # 'X'
```

Only fully-fixed coordinates are supported (all modes must be
integers).  Attempting to write without storage raises `TypeError`.

## Slicing

Slicing fixes some coordinates and keeps others free.  The result
is a new Tensor with a reduced layout and accumulated base offset.

```python
t = Tensor(Layout((4, 8), (8, 1)))

t[2, :]    # Tensor with offset=16, layout=(8,):(1,) — row 2
t[:, 5]    # Tensor with offset=5,  layout=(4,):(8,) — column 5
t[2, 5]    # 21 — all modes fixed, returns int (or data element)
```

Sub-Tensors produced by slicing **share the parent's storage** (view
semantics).  Reading or writing through a sub-Tensor accesses the
same underlying buffer:

```python
buf = list(range(32))
t = Tensor(Layout((4, 8), (8, 1)), data=buf)

row2 = t[2, :]    # sub-Tensor viewing row 2
row2[3]            # buf[19] → 19
row2[3] = 999      # buf[19] = 999
t[2, 3]            # 999 — visible through parent too
```

### Hierarchical partial slicing

For layouts with nested (hierarchical) shapes, partial sub-coordinates
can be sliced using `None` as the free-dimension marker:

```python
layout = Layout(((2, 4), 8), ((1, 16), 2))
t = Tensor(layout)
t[(1, None), :]    # fix first sub-mode of mode 0 to 1, keep rest free
```

## Storage

### The `data` property

`tensor.data` returns the storage reference, or `None` for algebraic
Tensors.  It is assignable:

```python
t = Tensor(Layout((4, 8), (8, 1)), data=list(range(32)))
t.data                          # [0, 1, 2, ..., 31]
t.data = list(range(100, 132))  # swap to new storage
t[0, 0]                         # 100
```

The new storage must satisfy the same addressed-range requirement.
Assigning `None` removes storage and returns the Tensor to algebraic mode.

### View aliasing

Sub-Tensors hold their own reference to the storage object.
Reassigning `parent.data` does **not** update existing sub-Tensors —
they keep the old reference.  This matches numpy/torch view semantics:

```python
buf = list(range(32))
t = Tensor(Layout((4, 8), (8, 1)), data=buf)
row = t[2, :]          # row holds a reference to buf

t.data = [0] * 32      # parent now points to new storage
row[0]                  # still reads from buf → 16
```

### Storage can be larger than cosize

The storage may be larger than `cosize(layout)`.  Only offsets within
the layout's image are accessed; extra elements are simply unused.
This is useful when multiple Tensors with different layouts share the
same underlying buffer.

## Views — `view()`

`tensor.view(layout)` returns a new Tensor that shares the same backing
storage but uses a different layout.  The new layout's cosize must not
exceed the storage length.

```python
buf = list("ABCDEFGHIJKLMNOP")
t = Tensor(Layout((4, 4), (4, 1)), data=buf)   # row-major

# Flat 1D view of the same backing store
flat = t.view(Layout(16, 1))
flat[0]   # 'A'
flat[15]  # 'P'

# Column-major view
col = t.view(Layout((4, 4), (1, 4)))
col[0, 0]  # 'A'  — same element, different traversal order

# The view shares storage — writes are visible everywhere
flat[0] = 'Z'
t[0, 0]    # 'Z'
```

This is the Python equivalent of CuTe's `make_tensor(tensor.data(), new_layout)`.
It is useful for inspecting the physical storage order of a tensor:

```python
src = Tensor(Layout((4, 8), (8, 1)), data=list(range(32)))
physical = src.view(Layout(len(src.data), 1))  # flat view of backing store
```

Calling `view()` on a Tensor without storage raises `TypeError`.
A layout whose cosize exceeds the storage length raises `ValueError`.

## Visualization

When a data-backed Tensor is passed to `draw_layout`, cells are
automatically labeled with data values instead of raw offsets:

```python
from tensor_layouts import draw_layout

buf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
t = Tensor(Layout((4, 8), (8, 1)), data=buf)
draw_layout(t)                         # cells show A, B, C, ...
draw_layout(t, cell_labels="offset")   # override: show raw offsets
draw_layout(t, cell_labels=False)      # suppress all labels
```

## Equality and Hashing

Two Tensors are equal if and only if they have the same layout, the
same offset, **and** the same data contents (element-wise comparison):

```python
a = Tensor(Layout(8, 1), data=[1, 2, 3, 4, 5, 6, 7, 8])
b = Tensor(Layout(8, 1), data=[1, 2, 3, 4, 5, 6, 7, 8])
c = Tensor(Layout(8, 1), data=[8, 7, 6, 5, 4, 3, 2, 1])

a == b    # True  — same contents
a == c    # False — different contents

d = Tensor(Layout(8, 1))
a == d    # False — one has data, the other doesn't
```

`__hash__` is based on `(layout, offset)` only (data is not included).
This is correct per the hash contract: equal objects always have equal
hashes, and collisions when only data differs are harmless.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `layout` | `Layout` | The underlying layout |
| `offset` | `int` | Base offset in linear (pre-swizzle) space |
| `shape` | tuple | Shorthand for `layout.shape` |
| `stride` | tuple | Shorthand for `layout.stride` |
| `data` | indexable or `None` | Backing storage (read-write) |

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `view(layout)` | `Tensor` | New Tensor sharing storage with a different layout |
