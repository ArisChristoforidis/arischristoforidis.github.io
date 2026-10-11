---
layout: "post.njk"
title: "Tensor puzzles"
date: 2026-08-20
tags: ["posts", "machine-learning", "pytorch"]
summary: "Solving Sasha Rush's tensor puzzles with constrained PyTorch operations."
draft: false
---

## Introduction

[Sasha Rush's Tensor Puzzles](https://github.com/srush/tensor-puzzles) are a set of exercises that aim to help you practice tensor operations, mainly broadcasting. “While solving them, you must follow five rules:

1) These puzzles are about broadcasting.
2) Each puzzle needs to be solved in 1 line (<80 columns) of code.
3) You are allowed @, arithmetic, comparison, `shape`, any indexing (e.g. `a[:j], a[:, None], a[arange(10)]`), and previous puzzle functions.
4) You are _not allowed_ anything else. No `view`, `sum`, `take`, `squeeze`, `tensor`.
5) You can start with these two functions:

```python
def arange(i: int):
    "Use this function to replace a for-loop."
    return torch.tensor(range(i))

def where(q, a, b):
    "Use this function to replace an if-statement."
    return (q * a) + (~q) * b
```

I have three additional pieces of advice from solving them:
- Make sure you understand the fundamentals of tensor broadcasting.
- Try to think about the vector and matrix shapes. It especially helps to work backwards from the desired output shape and figure out how you can produce it with what you have.
- Use `print` to see what you are doing - pretty obvious but worth mentioning.

You can attempt to solve them in the [provided notebook](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb).

## Puzzle 1

**Goal**: Compute [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) - the vector of all ones.

**Example**

```text
i   = 4

out = [1, 1, 1, 1] # length i
```

The main challenge of this problem is finding a way to initialize a tensor array of the correct size. Since we don't have much to work with, I used `arange(i)`, subtracted it from itself, and added 1:

```python
def ones(i: int) -> TT["i"]:
    return arange(i) - arange(i) + 1
```

## Puzzle 2

**Goal**: Compute [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) - the sum of a vector.

**Example**

```text
a   = [ 1, 2, 3, 5] # shape (i,)

out = [11] # shape (1,)
```

This puzzle aims to introduce explicit broadcasting to us. In order to add the elements, we will need to perform matrix multiplication using `@` (between `a` acting as a row and a vector of 1s acting as a column) , but we can't do it right away as we are currently dealing with vectors. We want `out` to be of shape `(1,)` so we need to broadcast our input vectors to shapes `(1, i)` and `(i, 1)` - the inner dimension will collapse and leave us with the desired shape. Here is how this is done:

```python
def sum(a: TT["i"]) -> TT[1]:
    return a[None, :] @ ones(a.shape[0])
```

> **Note**: We didn't need to explicitly broadcast the `ones` vector, as it is done implicitly for us, since we broadcast `a` explicitly.

## Puzzle 3

**Goal**: Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.

**Example**

```text
a   = [a0, a1, a2] # shape (i,)
b   = [b0, b1] # shape (j,)

out = [a0 * b0, a0 * b1] # shape (i, j)
      [a1 * b0, a1 * b1]
      [a2 * b0, a2 * b1]
```

In practice, we want an `(i, j)` matrix multiplying every element of `a` with every element of `b`. To achieve this, we will be broadcasting `a` to a `(i, 1)` shape and `b` to a `(1, j)` shape - and multiplying them to get the correct output shape:

```python
def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    return a[:, None] @ b[None, :]
```

## Puzzle 4

**Goal**: Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.

**Example**

```text
a   = [1, 2, 3] # shape (i, i)
      [4, 5, 6]
      [7, 8, 9]

out = [1, 5, 9] # shape (i,)
```

For this one, all we have to do is directly index the diagonal elements, which we can do by utilizing `arange(i)`:

```python
def diag(a: TT["i", "i"]) -> TT["i"]:
    return a[arange(a.shape[0]), arange(a.shape[0])]
```

By doing this, `arange` will create 2 vectors, one for the rows and one for the columns, that look like this: `[0, 1, ..., i-1]` which correspond to a list of index pairs `(0,0), (1, 1), ..., (i-1, i-1)` and pick out the elements we care about.

## Puzzle 5

**Goal**: Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.

**Example**

```text
j   = 3

out = [1, 0, 0]
      [0, 1, 0]
      [0, 0, 1]
```

We want to create a 2D array with shape `(j, j)` where every element on the diagonal is 1. We can use `where`, comparing 2 `arange(j)`, one broadcast as a row and one as a column, to implicitly form a 2D matrix. The cells where the indexes of the 2 vectors match will be our diagonal, and may be set to 1. All other cells get 0.

Let's look at how the comparison works:

```python
arange(j)[:, None] == arange(j)[None, :]
```

This snippet creates our matrix. Here is what happens under the hood:

```text
row_vector = [0, 1, 2] # shape (1, j)

col_vector = [0] # shape (j, 1)
             [1]
             [2]
```

When we compare with `==` the two vectors get broadcast together, forming two `(j, j)` shaped matrices where we can compare values elementwise:

```text
row_matrix = [0, 1, 2]
             [0, 1, 2]
             [0, 1, 2]

col_matrix = [0, 0, 0]
             [1, 1, 1]
             [2, 2, 2]
```

All we have to do is place 1s wherever the values of the 2 matrices match:

```python
def eye(j: int) -> TT["j", "j"]:
    return where(arange(j)[:, None] == arange(j)[None, :], 1, 0)
```

In fact, we can just remove the `where` altogether. The boolean matrix can be converted into 1s and 0s by multiplying with 1.

```python
def eye(j: int) -> TT["j", "j"]:
    return (arange(j)[:, None] == arange(j)[None, :]) * 1
```

## Puzzle 6

**Goal**: Compute [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) - the upper triangular matrix.

**Example**

```text
j   = 3

out = [1, 1, 1]
      [0, 1, 1]
      [0, 0, 1]
```

This problem is almost identical to the previous one. We just need to change the comparison from `==` to `<=` to account for the extra cells we want to place 1s to. Those are basically the cells where the row index is smaller or equal to the column index:

```python
def triu(j: int) -> TT["j", "j"]:
    return (arange(j)[:, None] <= arange(j)[None, :]) * 1
```

## Puzzle 7

**Goal**: Compute [cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) - the cumulative sum.

**Example**

```text
a   = [-1, 3, 0, -5]

out = [-1, 2, 2, -3]
```

For this puzzle, we have to calculate the cumulative sum, so for `out[k]` we need to sum `a[0] + a[1] + ... + a[k]`. We are going to make use of the `triu` function we built in the previous puzzle. If we make a `(i, i)` shaped `triu` matrix, the element `[k, j]` will have the value of 1 if `out[j]` needs `a[k]` in its summary calculation, else it will be 0. If we do matrix multiplication between this table and `a`, we will have solved the puzzle:

```python
def cumsum(a: TT["i"]) -> TT["i"]:
    return a @ triu(a.shape[0])
```

You can visualize this for our example above as follows:

```text
a    = [-1, 3, 0, -5] # shape (i,)

triu = [1, 1, 1, 1] # shape (i, i)
       [0, 1, 1, 1]
       [0, 0, 1, 1]
       [0, 0, 0, 1]
```

When doing matrix multiplication, the inner dimensions must match, so a will be broadcast to `(1, i)`. Our output shape will therefore be `(1, i)` (which will be implicitly converted back to `(i,)`). The math will look like this:

```text
# General form
out[k] = a[0] * triu[0, k] + a[1] * triu[1, k] + ... + a[k] * triu[i-1, k]

out[0] = a[0] * triu[0, 0] + a[1] * triu[1, 0] + a[2] * triu[2, 0] + a[3] * triu[3, 0] = -1 * 1 + 3 * 0 + 0 * 0 + -5 * 0 = -1

out[1] = a[0] * triu[0, 1] + a[1] * triu[1, 1] + a[2] * triu[2, 1] + a[3] * triu[3, 1] = -1 * 1 + 3 * 1 + 0 * 0 + -5 * 0 = 2

out[2] = a[0] * triu[0, 2] + a[1] * triu[1, 2] + a[2] * triu[2, 2] + a[3] * triu[3, 2] = -1 * 1 + 3 * 1 + 0 * 1 + -5 * 0 = 2

out[3] = a[0] * triu[0, 3] + a[1] * triu[1, 3] + a[2] * triu[2, 3] + a[3] * triu[3, 3] = -1 * 1 + 3 * 1 + 0 * 1 + -5 * 1 = -3
```

> **Note**: There is a very subtle problem in this puzzle, if you decide to broadcast `a` explicitly: if you broadcast `a` to `(1, i)` manually, the `run_test` function will break because the final result from the matrix multiplication will not be converted back to shape `(i,)` and you will get a `RuntimeError`.

## Puzzle 8

**Goal**: Compute [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) - the running difference.

**Example**

```text
a   = [-5, 4, 2, -4]

out = [-5, 9, -2, -6]
```

For every element `a[i]`, `out[i]` will be equal to `a[i] - a[i-1]`. One special case is `out[0]` which is always equal to `a[0]` since there is no previous element. We can perform the base case by creating a vector of the `diff` results using `arange(i) - 1` to index `a` to "look back" one element:

```python
diff = a - a[arange(i) - 1] # this will do a[i] - a[i-1]
```

This will correctly calculate the differences between adjacent elements for all but the first element:

```python
a = [-5, 4, 2, -4]
a[arange(i) - 1] = a[[0, 1, 2, 3] - 1] = a[[-1, 0, 1, 2]] = [-4, -5, 4, 2]

diff = a - a[arange(i) - 1] = [-5, 4, 2, -4] - [-4, -5, 4, 2] = [-1, 9, -2, -6]
```

In that case, it will attempt to do `a[0] - a[-1]` which will fetch the last element. Thankfully, we have `where` to take care of the special case. We can use `arange` as a control vector and separate the first element:

```python
def diff(a: TT["i"], i: int) -> TT["i"]:
    return where(arange(i) > 0, a - a[arange(i) - 1], a[0])
```

## Puzzle 9

**Goal**: Compute [`vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors.

**Example**

```text
a   = [1, 2, 3] # shape (i, )
b   = [4, 5, 6] # shape (i, )

out = [1, 2, 3] # shape (2, i)
      [4, 5, 6]
```

The target for this puzzle is to place the 2 vectors together in an array with shape `(2, i)`.

Looking at the output spec, we need a `out` shape of `(2, i)`. One direct approach is to repeat each vector across two rows using `outer` and `ones`:

```python
outer(ones(2), a)
```

This duplicates the vector along two rows, exactly what we want. Doing this for both vectors gave me two `(2, i)` matrices. To combine them, a boolean mask using `arange(2)[:, None]` lets us select row 0 from `a` and row 1 from `b`:

```python
def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    return where(arange(2)[:, None] > 0, outer(ones(2), b), outer(ones(2), a))
```

This works, but it constructs two full matrices only to select one row from each. Since `arange(2)[:, None]` already has shape `(2, 1)`, passing simple 2D row vectors of shape `(1, i)` causes `where` to expand them along the vertical axis for us. Stripping out the `outer(ones(...))` calls leaves a much cleaner, more idiomatic solution:

```python
def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    return where(arange(2)[:, None] > 0, b[None, :], a[None, :])
```

## Puzzle 10

**Goal**: Compute [`roll`](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) - the vector shifted 1 circular position.

**Example**

```text
a   = [1, 2, 3, 4, 5]

out = [2, 3, 4, 5, 1]
```

We could build a conditional mask using `where` and `arange` (similar to the `diff` puzzle), but that feels overly complex and would quickly push against line length limits.

Using pure `arange` indexing, it's easy to shift elements, but my first attempt shifted them in the wrong direction:

```python
def roll(a: TT["i"], i: int) -> TT["i"]:
    return a[arange(i) - 1]

# Example Input:
# a =   [-2, -4, -2, 3, 1]
# Output:
# out = [1, -2, -4, -2, 3] # wrong, shifted right
# Desired Output:
# out = [-4, -2, 3, 1, -2] # correct, but not what we do above
```

To roll to the left, we need to add `1` to shift forward (`0` grabs index `1`, `1` grabs index `2`, etc.). The trick for cyclic shifting is the modulo operator (`%`), which cleanly wraps the final index back around to `0` without running out of bounds or needing conditional logic:

```python
def roll(a: TT["i"], i: int) -> TT["i"]:
    return a[(arange(i) + 1) % i]
```

## Puzzle 11

**Goal**: Compute [`flip`](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector.

**Example**

```text
a   = [1, 2, 3, 4, 5]

out = [5, 4, 3, 2, 1]
```

In standard NumPy, this would be a trivial `a[::-1]`. However, PyTorch doesn't allow negative step sizes in basic slicing within these puzzle constraints.

If we map out the index transformation we actually need, the pattern becomes clear:

```text
input index sequence:    0,   1, ..., i-2, i-1
output index sequence: i-1, i-2, ...,   1,   0
```

We can construct this reversed sequence directly using `arange(i)` arithmetic:

```python
def flip(a: TT["i"], i: int) -> TT["i"]:
    return a[i - arange(i) - 1]
```

Subtracting `arange(i)` from `i` flips the direction to decreasing (`i, i-1, ..., 1`), and subtracting `1` adjusts it back to zero-based indexing (`i-1, i-2, ..., 0`).

## Puzzle 12

**Goal**: Compute [`compress`](https://numpy.org/doc/stable/reference/generated/numpy.compress.html) - keep only masked entries (left-aligned).

**Example**

```text
v      = [   1,    2,     3,    4,     5]
g      = [True, True, False, True, False]
out    = [   1,    2,     4,    0,     0]
```

We basically want to take a vector `v` and a boolean mask `g`, keep only the values where `g` is `True`, pack them tightly to the left, and pad the rest with zeros. The difficult aspect of this problem is figuring out how to facilitate the "packing" procedure, which can be abstracted as "an operation that moves an element of the input array to a new index in the output array"

### Figuring out where items go

The items that have a `True` value at `g` should be "packed" to the left, i.e. placed at the start of the `out` vector. This means that each of these items will have sequential indexes in the output vector - which we need to find. This can be achieved with `cumsum`. Multiplying it by 1 can turn the `bool` vector into a vector of integers (`True` becomes 1, `False` becomes 0):

```text
g             = [ True, True, False, True, False]
g * 1         = [    1,    1,     0,    1,     0]
cumsum(g * 1) = [    1,    2,     2,    3,     3]
```

This works nicely, but it has a side effect: False elements also get a destination index value. We can clean these up by multiplying the `cumsum` by `g * 1` again:

```text
cumsum(g * 1)         = [    1,    2,     2,    3,     3]
cumsum(g * 1) * g * 1 = [    1,    2,     0,    3,     0]
```

The `False` values turn to 0s which cleans up the extra indexes. Of course the indexes are 1-based and 0 signifies that we mean to discard the element, but we will be able to easily deal with this later.

### Moving elements around

The previous section identifies the destination of each selected value. The remaining question is how to move those values into `out`. We can use a matrix A with shape `(i, i)` to move each value to its destination. To move `v[i]` to `out[j]` we need a 1 at `A[j, i]` and 0 everywhere else:

```text
A     = [1, 0, 0, 0, 0] # shape (i, i)
        [0, 1, 0, 0, 0]
        [0, 0, 0, 1, 0]
        [0, 0, 0, 0, 0]
        [0, 0, 0, 0, 0]

v     = [1, 2, 3, 4, 5] # shape (i,)

A @ v = [1, 2, 4, 0, 0] # shape (i,)
```

In other words, the core of the problem boils down to how we can construct `A` dynamically using allowed tensor operations. We already figured out what output indexes we want each element to end up in in the previous section. To convert this to matrix `A` with 1s and 0s, we can compare that against a broadcast array of indexes: wherever there is a match, we should get a 1, and 0 everywhere else. The array of indexes should be formed as a column by `arange(i)[:, None]` to allow us to broadcast it into a `(N, N)` matrix after the comparison with `cumsum(g * 1) * g * 1`. This is how this will look:

```python
(arange(i)[:, None] + 1 == cumsum(g * 1)) * g * 1
# This generates our matrix A. We are basically comparing a column vector
# with dimensions (N, 1) with a normal (row) vector with dimensions (1, N),
# which will force an auto broadcast: the output matrix will have shape of
# (N, N)
```

### Putting it all together

All we have to do to solve the problem now is take our `A` generator and multiply it with `v`, and we will have our solution:

```python
def compress(g: TT["i", bool], v: TT["i"], i: int) -> TT["i"]:
    return ((arange(i)[:, None) + 1 == cumsum(g * 1)) * g * 1) @ v
```

## Puzzle 13

**Goal**: Compute `pad_to` - eliminate or add 0s to change size of vector.

**Example**

```text

Case 1 - Trimming based on j
a      = [ 1, 2, 3, 4, 5]
i      = 5
j      = 3
out    = [ 1, 2, 3]

Case 2 - Padding based on j
a      = [ 1, 2, 3]
i      = 3
j      = 4
out    = [ 1, 2, 3, 0]
```

In this puzzle, `a` is our vector, `i` is the number of elements in `a`, and `j` is the number of elements in the `out` vector.

If you solved puzzle 12, this should be a piece of cake. We are going to reuse the same concept from the previous puzzle with matrix `A` to get our result. However, in this one, `A` will be either slicing or padding our vector. If we use `j` for our rows with `arange(j)[:, None]` and `i` for our columns with `arange(i)` we will create a `(j, i)` matrix that, when multiplied by our a vector, can support both cases as so:

- When `j > i` we are in the padding case: The matrix has more rows than a has elements. The extra rows at the bottom find no matching column indices, so they fill up the `out` vector with 0s. Matrix multiplication correctly pads the output up to length j
- When `j < i` we are in the trimming case: The matrix has fewer rows than `a` has elements. The excess input elements in a fall outside the row range of the matrix, meaning they are trimmed, leaving an output of exact length j.

The resulting solution is:

```python
def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    return (arange(j)[:, None] == arange(i)) * 1 @ a
```

## Puzzle 14

**Goal**: Compute [`sequence_mask`](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) - pad out to length per batch.

**Example**

```text
values = [ 1,  2,  1, -1,   2]
         [ 0,  2, -2, -3,  -5]
         [ 2,  0,  2,  1,  -5]
         [ 1,  2,  2,  4,   2]

length = [ 4,  1,  2,  2]

out =    [ 1,  2,  1, -1,   0] # Row 0 keeps 4 elements
         [ 0,  0,  0,  0,   0] # Row 1 keeps 1 element (happens to be 0)
         [ 2,  0,  0,  0,   0] # Row 2 keeps 2 elements (2nd element happens to be 0)
         [ 1,  2,  0,  0,   0] # Row 3 keeps 2 elements
```

We have a matrix `values` with a shape (i, j). We also get a vector `length` which has a shape (i,) and has integer values for the length of each row we want to mask (keep). All other values beyond the given length for each row should be turned to 0.

We can create a mask matrix with the same shape as `values`, placing 1s where we want to keep the value and 0s where we want to discard it. Then we could multiply this mask matrix with our given `values` matrix and solve the problem. Here is how our mask should look for the given example:

```text
mask = [ 1,  1,  1,  1,   0] # Row 0 has 4 1s
       [ 0,  0,  0,  0,   0] # Row 1 has 0 1s
       [ 1,  1,  0,  0,   0] # Row 2 has 2 1s
       [ 1,  1,  0,  0,   0] # Row 3 has 2 1s
```

To create this matrix I broadcast length as a (i, 1) vector and compared it to `arange(j)` to get the desired matrix shape of (i, j). This will result in a matrix of boolean values: the `greater than` operator will fill each row with `True` values where the column index is less than the one in `length` for that specific row, which is what we want.

```python
mask = length[:, None] > arange(values.shape[1])
```

We can multiply this by a scalar (or a matrix of scalars) to convert this to actual integers, which is evidently what we have in `values`. Doing the final multiplication solves the problem:

```python
def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    return (length[:, None] > arange(values.shape[1])) * values
```

## Puzzle 15

**Goal**: Compute [`bincount`](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) - count number of times an entry was seen.

**Example**

```text
a   = [0, 3, 4, 3]
j   = 5

out = [1, 0, 0, 2, 1]
```

Essentially, the `out` vector has `j` elements. The element at position `i` counts the frequency of `i` in vector `a`. So in the example above, the element at index 3 counts how many 3s there are in `a`. To construct `out`, create a matrix with shape `(a.shape[0], j)` containing a 1 at `[i, a[i]]` and 0 elsewhere. Summing each column then gives the required frequencies. Here is how this would look for the above example:

```text
A = [ 1, 0, 0, 0, 0]
    [ 0, 0, 0, 1, 0]
    [ 0, 0, 0, 0, 1]
    [ 0, 0, 0, 1, 0]
```

This matrix basically helps us build the `out` vector - if we sum each column, we have our result! Even though we can't use the common `sum` functions from Numpy/Pytorch we perform matrix multiplication with an array of 1s with the same shape as `a` to get the same result:

```python
def bincount(a: TT["i"], j: int) -> TT["j"]:
    return ones(a.shape[0]) @ ((arange(j)[None, :] == a[:, None]) * 1)
```

## Puzzle 16

**Goal**: Compute [`scatter_add`](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html) - add together values that link to the same location.

**Example**

```text
values = [-2, 5, 0, 1, 2]
link   = [ 2, 3, 0, 0, 2]
j      = 4

out    = [1, 0, 0, 5]
```

We want to sum all values from `values` into `out` based on the indices laid out in `link`. An element of the `values` vector with index `i` gets its value summed at position `link[i]` in the `out` vector.

As in the previous puzzle, the desired result can be obtained with matrix multiplication. Construct a matrix with a 1 at `[i, link[i]]` whenever `values[i]` should contribute to `out[link[i]]`. Then all we have to do is perform our matrix multiplication with `values` and we have solved the puzzle!

```python
def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    return values @ ((arange(j)[None, :] == link[:, None]) * 1)
```

## Puzzle 17

**Goal**: Compute [`flatten`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)

**Example**

```text
a   = [ 3, -2, -2]
      [-1, -2,  3]
      [-4, -5, -4]
      [ 3, -4, -4]

out = [ 3, -2, -2, -1, -2,  3, -4, -5, -4,  3, -4, -4]
```

In this puzzle, we are trying to convert a given 2D array to a 1D vector. This switches things up from the previous puzzles since it uses a different trick: index conversion from 2D to 1D. We can select specific elements from our `a` array using the following syntax:

```python
a[[0, 1], [2, 2]]
```

The first set of values indicates the row indexes and the second the column indexes. These get zipped together as pairs, so the above will return elements `[0, 2]` and `[1,2]` (elements -2, 3). Knowing this, the next step is to map the 1D indices of the final vector to the 2D indices of the input array. We can produce the 1D indexes by using `arange(i * j)`. To map each flattened 1D index to 2D, we can use integer division and modulo:

```python
row_indices = arange(i * j) // j
col_indices = arange(i * j) % j
```

And the problem is now solved.

```python
def flatten(a: TT["i", "j"], i: int, j: int) -> TT["i * j"]:
    return a[arange(i * j) // j, arange(i * j) % j]
```

## Puzzle 18

**Goal**: Compute [`linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

**Example**

```text
i   = 0
j   = 10
n   = 5

out = [0.0, 2.5, 5.0, 7.5, 10.0]
```

The `linspace` function must return a vector of evenly spaced samples over a specified interval. To find the value at any given step, we need to know our *step size*. If we have `n` points, we actually have `n - 1` gaps between them. So, the total distance `(j - i)` divided by the number of gaps `(n - 1)` gives us our step size. We can multiply this step size by our current index `arange(n)` and add it to our starting point `i`:

```python
i + arange(n) * (j - i) / (n - 1)
```

While this is theoretically correct, we have to account for the edge case where `n = 1`. In this case, the denominator will turn 0 and our result will be `nan` which is not the desired result.

To fix this, we need to safeguard the denominator before the division happens. We can use the `where` function to force the denominator to be `1` instead of `0` whenever `n = 1`. Because `arange(1)` evaluates to `[0]`, multiplying our step by `0` will simply leave us with our starting value `i`, neatly avoiding the `nan` error. Safeguarding the denominator gives:

```python
def linspace(i: TT[1], j: TT[1], n: int) -> TT["n", float]:
    return i + arange(n) * (j - i) / where(n > 1, n - 1, 1)
```

### Solving by reverse engineering the reference implementation

As an alternative way of solving it, we can arrive at the same solution by directly translating the `linspace_spec` into tensor operations without messing with the actual math:

```python
def linspace_spec(i, j, out):
    for k in range(len(out)):
        out[k] = float(i + (j - i) * k / max(1, len(out) - 1))
```

This `for` loop can be broken down and mapped to our allowed PyTorch operations:

1. `len(out)`: this is our target length, `n`
2. `k`: this is the loop index over `out` which can be replaced with a vector of indexes i.e. `arange(n)`
3. `max(1, len(out) - 1))`: As above, this safeguards against division-by-zero when `n = 1`. Since we don't have access to the `max` function, we have to rebuild this logic. We can use our allowed `where` function: if `n > 1`, we use `n - 1`, otherwise we default to `1`. This translates to `where(n > 1, n - 1, 1)`.

Everything else can stay as is. Substituting these tensor equivalents back into the original formula gives us our vectorized solution.

> **Note on Direction:** This formula works seamlessly whether `j > i` or `j < i`. If `j < i` (for example, stepping down from 10 to 0), `(j - i)` naturally evaluates to a negative number. Standard tensor arithmetic handles the sign automatically-multiplying a negative step size by `arange(n)` steadily decreases the value from `i` down to `j`.

## Puzzle 19

**Goal**: Compute [`heaviside`](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html)

**Example**

```text
a   = [ 2,  0, -2, -5,  0]
b   = [-5,  3,  1, -5, -5]

out = [ 1,  3,  0,  0, -5]
```

The heaviside function is defined as:

```text
                   0 if a <  0
heaviside(a, b) =  b if a == 0
                   1 if a >  0
```

The cleanest way I found to achieve this result is through the use of the `where` function. We want to sample from `b[i]` if `a[i] == 0` else we want to put either 1 or 0 in our `out[i]`, depending on whether the element in `a` is positive or not. It is very convenient to transform our `a` array to an array where instead of positive and negative elements, it has 1s (wherever the element is positive) and 0s (wherever the element is negative) - and sample from this. We don't care for elements that are already 0, since these will be handled by the other where clause.

```python
def heaviside(a: TT["i"], b: TT["i"]) -> TT["i"]:
    return where(a == 0, b, a > 0)
```

## Puzzle 20

**Goal**: Compute [`repeat`](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html)

**Example**

```text
a   = [-5, -2, -5, -1, 0]
d   = 3

out = [-5, -2, -5, -1, 0]
      [-5, -2, -5, -1, 0]
      [-5, -2, -5, -1, 0]
```

What we want is to repeat the input vector `a` `d` times, creating a 2D array. This problem requires nothing more than a simple matrix multiplication between an array of 1s and `a`. We just need to broadcast correctly. Since we know that matrix multiplication between a `(M, P)` shaped matrix and `(P, N)` shaped matrix results in a `(M, N)` shaped matrix, we can look at what shape we want our final matrix to be and infer the broadcast: Since `out` has a  `(d, a.shape[0])` shape, we understand that our first matrix should be an array of 1s broadcast to `(d, 1)` and our second matrix should be `a` broadcast to `(1, a.shape[0])`. This translates to the following solution:

```python
def repeat(a: TT["i"], d: TT[1]) -> TT["d", "i"]:
    return ones(d)[:, None] @ a[None, :]
```

### Attempt 2

What we have above is basically the outer product between `ones(d)` and `a`, so we can save some characters by reusing the `outer` function from Puzzle 3:

```python
def repeat(a: TT["i"], d: TT[1]) -> TT["d", "i"]:
    return outer(ones(d), a)
```

## Puzzle 21

**Goal**: Compute [`bucketize`](https://pytorch.org/docs/stable/generated/torch.bucketize.html)

**Example**

```text
v          = [-1, 2, 5, 1, -3]
boundaries = [ 1, 2, 3, 4]

out        = [ 0, 2, 4, 1, 0]
```

This is another function that I didn't exactly know how it works. In the reference PyTorch documentation, it is defined as:

>Returns the indices of the buckets to which each value in the `input` belongs, where the boundaries of the buckets are set by `boundaries`. Return a new tensor with the same size as `input`.

The boundaries array defines the following buckets:

- 0th bucket -> `(-inf, 1)`
- 1st bucket -> `[1, 2)`
- 2nd bucket -> `[2, 3)`
- 3rd bucket -> `[3, 4)`

So we want the index of the bucket that each item falls into as the answer for each element. Additionally, we can calculate the value in `out[i]` by counting how many elements `v[i]` is greater or equal to from the `boundaries` array. We can start with a simple comparison broadcast between `v` and `boundaries`:

```python
1 * (v[:, None] >= boundaries[None, :])
```

This will return a 2D array with shape (`v.shape[0]`, `boundaries.shape[0]`) where each row corresponds to an element in `v` and each column corresponds to an element in `boundaries`. If the element `[i,j]` is 1, that means that `v[i]` is larger or equal to `boundaries[j]`, otherwise it is 0. Here is the output of this in our example case:

```text
[0, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 1]
[1, 0, 0, 0]
[0, 0, 0, 0]
```

Now that we have this matrix, all we have to do is `sum` on the row axis - something we can achieve by multiplying with a column of 1s. We don't need to broadcast explicitly, PyTorch will do this for us automatically. Our final solution is the following:

```python
def bucketize(v: TT["i"], boundaries: TT["j"]) -> TT["i"]:
    return (1 * (v[:, None] >= boundaries)) @ ones(boundaries.shape[0])
```

> **Note**: I removed the explicit broadcast from `boundaries` to save some characters - it is also done implicitly since we run the comparison with the broadcast `v`.


## Benchmarks

Here are the benchmark results for my solutions:

| **puzzle**    | **chars** |
| ------------- | --------- |
| ones          | 36        |
| sum           | 41        |
| outer         | 34        |
| diag          | 52        |
| eye           | 57        |
| triu          | 57        |
| cumsum        | 31        |
| diff          | 31        |
| vstack        | 64        |
| roll          | 33        |
| flip          | 31        |
| compress      | 66        |
| pad_to        | 52        |
| sequence_mask | 63        |
| bincount      | 70        |
| scatter_add   | 63        |
| flatten       | 51        |
| linspace      | 51        |
| heaviside     | 34        |
| repeat        | 28        |
| bucketize     | 71        |
