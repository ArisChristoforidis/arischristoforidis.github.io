---
layout: "post.njk"
title: "Autodiff puzzles"
date: 2026-10-11
tags: ["posts", "machine-learning", "jax"]
summary: "Working through Sasha Rush's autodiff puzzles with manual Jacobians, broadcasting, and tensor indexing."
draft: false
---

## Introduction

[Sasha Rush's Autodiff Puzzles](https://github.com/srush/Autodiff-Puzzles) are a follow-up to the Tensor Puzzles, aimed at building an understanding of autodifferentiation. Each puzzle gives us a tensor-to-tensor function, and our goal is to calculate its Jacobian manually using basic tensor operations.

For a vector input of shape `(I,)` and an output of shape `(O,)`, the Jacobian has shape `(O, I)`. Each element contains the partial derivative of one output with respect to one input:

$$ J_{o,i} = \frac{\partial f_o}{\partial x_i} $$

For higher-dimensional tensors, the same idea applies: the Jacobian's shape consists of the output dimensions followed by the input dimensions. For functions with two inputs, we calculate one Jacobian for each input.

The notebook instructions give us a few rules and tips:

1. Aim for a single line of code within 80 columns.
2. Use basic operators rather than constructing answers with functions such as `np.array`.
3. Work out the derivatives mathematically first, then translate them into tensor operations.

I have a few additional pieces of advice from solving them:

- It is always a good idea to start by expanding one output and finding its partial derivatives. This makes it easier to see which inputs contribute and what coefficients remain.
- Keep track of the Jacobian's shape. Its rows correspond to outputs and its columns to inputs in the vector cases.
- Separate the indexing from the masking: first identify the values you need, then zero out the unrelated entries.
- Check what the actual function does, especially for convolution. Kernel flipping and boundary handling can change the indexing quite a bit.

The notebook uses `jax.numpy` under the alias `np`. Its `check` function compares our manually constructed Jacobians against JAX's autodifferentiation results and visualizes the input–output relationships.

You can attempt the puzzles in the [provided notebook](https://colab.research.google.com/github/srush/Autodiff-Puzzles/blob/main/autodiff_puzzlers.ipynb).

## Puzzle 1 - Id

**Function**
$$f(x_0) = [2x_0]$$
**Expected Jacobian Shape**: `(1, 1)`

> **Note:** The original notebook labels this as an identity function, but its supplied code uses `2 * x`. Here we follow the code.

The derivative is straightforward:

$$ \frac{\partial f_0}{\partial x_0} = 2 $$

We only have one input and one output, so our Jacobian is simply `[2]` with shape `(1, 1)`. Comparing `x[:, None]` with `x` produces a `(1, 1)` array containing `True`, which we multiply by 2:

```python
Is = np.arange(1)


def f(x: Shaped[Array, "1"]) -> Shaped[Array, "1"]:
    return 2 * x


def jac(x: Shaped[Array, "1"]) -> Shaped[Array, "1 1"]:
    return 2 * (x[:, None] == x)


check(f, jac, in_shape=Is.shape)
```


## Puzzle 2 - Cosine

**Function**

$$
f(x_0) = [\cos(x_0)]
$$

**Expected Jacobian Shape**: `(1, 1)`

From the derivatives table we know that:

$$
\frac{\partial f}{\partial x_0} = -\sin(x_0)
$$

Since the target dimensions are `(1, 1)`, we need to add a row dimension to the derivative. `np.sin(x)` has shape `(1,)`, so we can use `x[None, :]` style indexing to add a dimension at the beginning. In this case, `np.sin(x)[None, :]` changes the shape from `(1,)` to `(1, 1)`.

```python
def f(x: Shaped[Array, "1"]) -> Shaped[Array, "1"]:
    return np.cos(x)


def jac(x: Shaped[Array, "1"]) -> Shaped[Array, "1 1"]:
    return -np.sin(x)[None, :]

check(f, jac, in_shape=1)
```

This gives us our desired target shape.

## Puzzle 3 - Mean

**Function**

$$
f(x_0, x_1, \dots , x_{I-1}) = \left[\frac{x_0 + x_1 + \dots + x_{I-1}}{I} \right]
$$

**Expected Jacobian Shape**: `(1, I)`

We can start by calculating the derivatives for a few terms:

$$
\frac{\partial f}{\partial x_0} = \frac{\partial}{\partial x_0} \left( \frac{x_0 + x_1 + \dots + x_{I-1}}{I} \right) = \frac{\partial}{\partial x_0} \left( \frac{x_0}{I} + \frac{x_1}{I} + \dots + \frac{x_{I-1}}{I} \right)
$$

When taking the partial derivative with respect to $x_0$, all other $x_i$'s are treated as constants, so their derivatives are 0.

$$
\frac{\partial f}{\partial x_0} = \frac{\partial}{\partial x_0} \left( x_0 \cdot \frac{1}{I} \right) = \frac{1}{I}
$$

Similarly, for $x_1$:

$$
\frac{\partial f}{\partial x_1} = \frac{1}{I}
$$

And in general:

$$
\frac{\partial f}{\partial x_i} = \frac{1}{I} \qquad \text{for } i=0,\dots,I-1
$$

Therefore, the Jacobian is:

$$
J = \begin{bmatrix} \frac{1}{I} & \frac{1}{I} & \dots & \frac{1}{I} \end{bmatrix}
$$

`x` has shape `(I,)`. We need a Jacobian of shape `(1, I)`, so we add a dimension with `x[None, :]`, giving shape `(1, I)`. Comparing it with `x` broadcasts the `(I,)` array across the first dimension, producing a `(1, I)` array of `True` values. Since `True` behaves like `1`, multiplying by `1/I` gives the desired Jacobian.

```python
I = 10
Is = np.arange(I)


def f(x: Shaped[Array, "I"]) -> Shaped[Array, "1"]:
    return np.mean(x, axis=0, keepdims=True)


def jac(x: Shaped[Array, "I"]) -> Shaped[Array, "1 I"]:
    return 1 / I * (x[None, :] == x)


check(f, jac, in_shape=Is.shape)
```

## Puzzle 4 - Product

**Function**

$$
f(x_0, x_1, \ldots, x_{I-1}) = x_0 \times x_1 \times \ldots \times x_{I-1}
$$

**Expected Jacobian Shape**: `(1, I)`

Let's look at some derivatives to see how they are calculated:

$$
\frac{\partial f}{\partial x_0} = \frac{\partial}{\partial x_0} \left( x_0 \times x_1 \times \dots \times x_{I-1} \right) = x_1 \times x_2 \times \dots \times x_{I-1}
$$

So the partial derivative of $f$ for $x_0$ is the product without $x_0$.

For $x_1$, the partial derivative of $f$ is:

$$
\frac{\partial f}{\partial x_1} = \frac{\partial}{\partial x_1} \left( x_0 \times x_1 \times \dots \times x_{I-1} \right) = x_0 \times x_2 \times \dots \times x_{I-1}
$$

In its general form, this can be expressed as follows:

$$
\frac{\partial f}{\partial x_i} = \prod_{i\neq j}{x_j}
$$

Turns out that this generalizes so that for element $x_i$, the partial derivative of $f$ is the product of the function, without element $x_i$. To create our Jacobian, we can calculate the product for the entire function, and divide with element $x_i$ at each index position `i` in order to "remove" it from the product calculation:

> **Note:** This division-based solution assumes the input elements are nonzero. The product-excluding-one-element formula also applies at zero, but dividing the full product by that element does not.

```python
def f(x: Shaped[Array, "I"]) -> Shaped[Array, "1"]:
    return np.prod(x, keepdims=True)


def jac(x: Shaped[Array, "I"]) -> Shaped[Array, "1 I"]:
    return np.prod(x) / x[None, :]


check(f, jac, in_shape=Is.shape)
```

## Puzzle 5 - Repeat

**Function**

$$
f(x_0) = [x_0, x_0,  x_0, \ldots  x_0]
$$

**Expected Jacobian Shape**: `(O, 1)`

> **Note:** The original statement describes repeating `x_0`, but the supplied code repeats `x_0 + 1`. The added constant does not change the Jacobian. The notebook's scalar-`dx` hint refers to an individual partial derivative; our `jac` returns the full Jacobian.

This puzzle is not that hard, but it does introduce the concept of multiple outputs. If you look at the desired shape, we have 1 input and `O` outputs. The stated function repeats $x_0$ `O` times; the supplied implementation adds 1 to each repetition. The derivative itself is simple - the derivative of $x_0$ is 1:

$$
\frac{\partial f_o}{\partial x_0} = 1
$$

This is the first puzzle where we have multiple outputs. The Jacobian is still defined as:

$$
J_{o,i} = \frac{\partial f_o}{\partial x_i}
$$

where $o$ refers to an output and $i$ refers to an input. Since we only have one input $x_0$, but `O` outputs, the Jacobian has `O` rows and 1 column:

$$
J = \begin{bmatrix} \frac{\partial f_0}{\partial x_0} \\ \frac{\partial f_1}{\partial x_0} \\ \vdots \\ \frac{\partial f_{O-1}}{\partial x_0} \end{bmatrix}
$$

Every implemented output is $x_0 + 1$, so every partial derivative is still 1:

$$
J = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix}
$$

We can look at the definition of `Os` to see how we can take advantage of its shape. `Os` has shape `(O, 1)`. Multiplying it by 0 gives us an array of zeros with the desired shape, to which we can add 1:

```python
Is = np.arange(1)
O = 10
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "1"]) -> Shaped[Array, "O"]:
    return (x + (Os * 0 + 1))[:, 0]


def jac(x: Shaped[Array, "1"]) -> Shaped[Array, "O 1"]:
    return 1 + (Os * 0)


check(f, jac, in_shape=Is.shape)
```

## Puzzle 6 - Repeat and Scale

**Function**

$$
f(x_0) = [x_0 \times \frac{0}{N}, x_0 \times \frac{1}{N},  x_0 \times \frac{2}{N},  \ldots,  x_{0} \times \frac{N-1}{N}]
$$

**Expected Jacobian Shape**: `(O, 1)`

Since every output is just $x_0$ multiplied by a constant, differentiating with respect to $x_0$ leaves those constants:

$$
J = \begin{bmatrix} \frac{0}{N} \\ \frac{1}{N} \\ \vdots \\ \frac{N-1}{N} \end{bmatrix}
$$

This is straightforward. What we must do is divide `Os` with O, the number of output elements:

```python
Is = np.arange(1)
O = 10
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "1"]) -> Shaped[Array, "O"]:
    return x * (Os / O)[:, 0]


def jac(x: Shaped[Array, "1"]) -> Shaped[Array, "O 1"]:
    return Os / O


check(f, jac, in_shape=Is.shape)
```

**Note**: On this puzzle I used `N = O` for the formulas, because I felt it was confusing to use `I` which is associated with the input (of which we only have 1 in this problem).

## Puzzle 7 - Negation

**Function**

$$
f(x_0, x_1, \ldots) = [-x_0, -x_1, \ldots, -x_{I-1}]
$$

**Expected Jacobian Shape**: `(O, I)`

In this puzzle, we will have our first true 2D Jacobian matrix. Each element of the Jacobian is the partial derivative of one output with respect to one input:

$$
J_{o,i} = \frac{\partial f_o}{\partial x_i}
$$

Since each output only depends on its corresponding input, the derivative is -1 when $o=i$, and 0 otherwise:

$$
J_{o,i} = \begin{cases} -1 & o=i \\ 0 & o\ne i \end{cases}
$$

We can find where $o=i$ by comparing `Is` and `Os`. Their shapes are already compatible for broadcasting, resulting in a `(O, I)` shaped matrix. This matrix will have `True` values where $o=i$ and `False` everywhere else. Multiplying with -1 converts these to -1 and 0:

```python
I = 10
O = 10
Is = np.arange(I)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"]) -> Shaped[Array, "O"]:
    return -x


def jac(x: Shaped[Array, "I"]) -> Shaped[Array, "O I"]:
    return (Os == Is) * -1


check(f, jac, in_shape=Is.shape)
```

## Puzzle 8 - ReLU

**Function**

$$
f(x_0, x_1, \ldots) = [\text{relu}(x_0), \text{relu}(x_1), \ldots]
$$

Recall

$$
\text{relu}(x) = \begin{cases}
0 & x \lt 0 \\
x & x \ge 0
\end{cases}
$$

**Expected Jacobian Shape**: `(O, I)`

> **Note:** The notebook allows us to ignore non-differentiability at zero. The derivative below chooses 1 there, whereas JAX differentiates the supplied `x * (x > 0)` expression to 0 at zero. Away from zero, they agree.

Puzzle 8 follows the same pattern as the previous puzzle, but now, instead of having a constant to multiply our `True` values with, we will need to multiply it with the derivative of $\text{relu}(x)$. Deriving $\text{relu}(x)$ gives us:

$$
\text{relu}'(x) = \begin{cases}
0 & x \lt 0 \\
1 & x \ge 0
\end{cases}
$$

For each element of the Jacobian, we are taking the partial derivative of one output with respect to one input:

$$
\frac{\partial f_o}{\partial x_i} = \begin{cases} 0 & o \ne i \\ \text{relu}'(x_i) & o = i \end{cases}
$$

So the Jacobian is still diagonal, but the diagonal is now determined by the derivative of ReLU. We need to multiply with 1 if $x \ge 0$ and 0 otherwise. We can add this comparison directly into our code, and multiply with 1 to convert the `bool` vector to 1s and 0s:

```python
I = 10
O = 10
Is = np.arange(I)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"]) -> Shaped[Array, "O"]:
    return x * (x > 0)


def jac(x: Shaped[Array, "I"]) -> Shaped[Array, "O I"]:
    return (Os == Is) * ((x >= 0) * 1)


check(f, jac, in_shape=Is.shape)
```

## Puzzle 9 - Index

**Function**

$$
f(x_0, x_1, \ldots, x_{24}) = [x_{10}, x_{11}, \ldots, x_{24}]
$$

**Expected Jacobian Shape**: `(O, I)`

In this puzzle, we discard the first 10 input elements and return the remaining 15. Therefore, the first 10 input columns of the Jacobian will be 0. For the elements we keep, each output depends on exactly one input, so the corresponding partial derivative is 1. For example:

$$
\frac{\partial f_0}{\partial x_{10}} = 1 \qquad \frac{\partial f_0}{\partial x_i} = 0 \text{ for } i \ne 10
$$

This means we still have a diagonal of 1s, but it needs to be shifted by the 10 discarded inputs. Since $I-O=10$, we can offset the input indices by $I-O$:

```python
I = 25
O = 15
Is = np.arange(I)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"]) -> Shaped[Array, "O"]:
    return x[10:]


def jac(x: Shaped[Array, "I"]) -> Shaped[Array, "O I"]:
    return (Os == (Is - (I - O))) * 1


check(f, jac, in_shape=Is.shape)
```

We could just subtract 10 to save some characters, but using `I` and `O` gives us a more general solution.

## Puzzle 10 - Cumsum

**Function**

$$
f(x_0, x_1, \ldots) = [\sum^0_{i=0} x_{i}, \sum^1_{i=0} x_{i}, \sum^2_{i=0} x_{i}, \ldots, ] / 20
$$

**Expected Jacobian Shape**: `(O, I)`

Each output is the cumulative sum of the inputs up to that position. Therefore, an output $f_o$ depends on an input $x_i$ when $i \le o$. The derivative of each included $x_i$ is 1, while inputs after $x_o$ do not contribute to the output:

$$
\frac{\partial f_o}{\partial x_i} = \begin{cases} \frac{1}{20} & i \le o \\ 0 & i > o \end{cases}
$$

So the Jacobian contains $\frac{1}{20}$ on and below the diagonal, and 0 above it. We can use `Os >= Is` to find where $o \ge i$. This gives a `bool` matrix with the required shape, which we multiply by $\frac{1}{20} = 0.05$:

```python
I = 10
O = 10
Is = np.arange(I)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"]) -> Shaped[Array, "O"]:
    return np.cumsum(x) / 20


def jac(x: Shaped[Array, "I"]) -> Shaped[Array, "O I"]:
    return (Os >= Is) * 0.05


check(f, jac, in_shape=Is.shape)
```

## Puzzle 11 - Running Mean

**Function**

$$
f(x_0, x_1, \ldots)_o = \frac{\displaystyle \sum^{o+W-1}_{i=o} x_i}{W}
$$

**Expected Jacobian Shape**: `(O, I)`

Each output is the mean of a window of W consecutive inputs. For example, with `W = 3`, the first few outputs are:

$$
f_0 = \frac{x_0+x_1+x_2}{3}
$$

$$
f_1 = \frac{x_1+x_2+x_3}{3}
$$

So an input $x_i$ contributes to output $f_o$ when it is within that window:

$$
o \le i < o+W
$$

If $x_i$ is included in the sum, the output can be written as:

$$
f_o = \frac{x_i}{W} + \frac{\sum_{\substack{o \le j < o+W \\ j \ne i}} x_j}{W}
$$

Therefore:

$$
\frac{\partial f_o}{\partial x_i} = \frac{1}{W} + 0 = \frac{1}{W}
$$

If $x_i$ is not included in the window, changing it has no effect on $f_o$, so the derivative is 0.

$$
\frac{\partial f_o}{\partial x_i} = \begin{cases} \frac{1}{W} & o \le i < o+W \\ 0 & \text{otherwise} \end{cases}
$$

We can use two comparisons to identify the inputs within each window:

- `Os <= Is`: This checks the left side of the inequality, $o \le i$.

- `Is - Os < W`: This checks the right side, $i < o+W$.

Together, these give us a `(O, I)` matrix with `True` values exactly where an input belongs to the corresponding output's window. We then multiply by $\frac{1}{W}$ to get the required partial derivatives.

Our final code looks like this:

```python
I = 10
O = 8
Is = np.arange(I)
Os = np.arange(O)[:, None]
W = 3


def f(x: Shaped[Array, "I"]) -> Shaped[Array, "O"]:
    return np.convolve(x, np.ones(W) / W, mode="valid")


def jac(x: Shaped[Array, "I"]) -> Shaped[Array, "O I"]:
    return (Is - Os < W) * (Os <= Is) * 1 / W


check(f, jac, in_shape=Is.shape)
```

## Puzzle 12 - Sort

**Function**

$$
f(x_0, x_1, \ldots) = \text{x's in sorted order}
$$

**Expected Jacobian Shape**: `(O, I)`

> **Note:** The notebook permits `np.sort` here. We calculate how the sorted values change locally with the inputs; for distinct values, sorting acts as a fixed permutation until their order changes.

Sorting is a bit different from the previous puzzles. Each output is still just one of the original inputs, so each output has a derivative of 1 with respect to the input it came from, and 0 with respect to all the others. For example, if:

$$
x = [3, 1, 2]
$$

then:

$$
f(x) = [1, 2, 3]
$$

So:

$$
\frac{\partial f_0}{\partial x_1} = 1
$$

because $f_0 = x_1$, while the derivative with respect to the other inputs is 0. This means the Jacobian acts like a routing matrix: each row corresponds to a sorted output, and each column corresponds to an original input. We can find which input produced each output by comparing the sorted array with the original array:

```python
np.sort(x)[:, None] == x
```

This equality-based solution assumes distinct input values. With duplicates, an output can match multiple inputs, so the resulting matrix is no longer a permutation matrix.

Broadcasting gives us an `(O, I)` matrix where `True` means that the input in that column is the one routed to that output. Multiplying by 1 turns the `True`/`False` values into the required `1`/`0` derivatives.

```python
I = 10
O = 10
Is = np.arange(I)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"]) -> Shaped[Array, "O"]:
    return np.sort(x)


def jac(x: Shaped[Array, "I"]) -> Shaped[Array, "O I"]:
    return (np.sort(x)[:, None] == x) * 1


check(f, jac, in_shape=Is.shape)
```

> Next we move on to functions of two arguments. For these you will produce two derivatives:

$$
\frac{df(x, y)_o}{x_i}, \frac{df(x, y)_o}{y_j}
$$

Everything else is the same.

## Puzzle 13 - Elementwise Mean

**Function**

$$
f(x, y)_o = \frac{x_o + y_o}{2}
$$

**Expected Jacobian Shape**: `(O, I)`, `(O, I)`

Now we have two inputs, so we need two Jacobians: one for the derivatives with respect to $x$ and one for the derivatives with respect to $y$. Each output only depends on the corresponding elements $x_o$ and $y_o$. For the derivative with respect to $x_i$:

$$
\frac{\partial f_o}{\partial x_i} = \begin{cases} \frac{1}{2} & o=i \\ 0 & o\ne i \end{cases}
$$

The same applies to $y$:

$$
\frac{\partial f_o}{\partial y_i} = \begin{cases} \frac{1}{2} & o=i \\ 0 & o\ne i \end{cases}
$$

So both Jacobians are diagonal matrices with $\frac{1}{2}$ on the diagonal. We can use `Os == Is` to find the diagonal (as we did before) and multiply it by `1/2`:

```python
I = 10
O = 10
Is = np.arange(I)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"], y: Shaped[Array, "I"]) -> Shaped[Array, "O"]:
    return (x + y) / 2


def jac(
    x: Shaped[Array, "I"], y: Shaped[Array, "I"]
) -> tuple[Shaped[Array, "O I"], Shaped[Array, "O I"]]:
    return (Os == Is) * 1 / 2, (Os == Is) * 1 / 2


check(f, jac, in_shape=Is.shape, in_shape2=Is.shape)
```

The key thing I'd take away from this one is that **with multiple inputs, you get one Jacobian per input**. Here they're identical because the function treats `x` and `y` symmetrically.

## Puzzle 14 - Elementwise Mul

**Function**

$$
f(x, y)_o = x_o * y_o
$$

**Expected Jacobian Shape**: `(O, I)`, `(O, I)`

Each output only depends on the corresponding elements of $x$ and $y$. For the derivative with respect to $x_i$:

$$
\frac{\partial f_o}{\partial x_i} = \begin{cases} y_o & o=i \\ 0 & o\ne i \end{cases}
$$

Similarly, for $y_i$:

$$
\frac{\partial f_o}{\partial y_i} = \begin{cases} x_o & o=i \\ 0 & o\ne i \end{cases}
$$

So the two Jacobians are diagonal, with the values of $y$ and $x$ respectively on the diagonal:

```python
I = 10
O = 10
Is = np.arange(I)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"], y: Shaped[Array, "I"]) -> Shaped[Array, "O"]:
    return x * y


def jac(
    x: Shaped[Array, "I"], y: Shaped[Array, "I"]
) -> tuple[Shaped[Array, "O I"], Shaped[Array, "O I"]]:
    return (Os == Is) * y, (Os == Is) * x



check(f, jac, in_shape=Is.shape, in_shape2=Is.shape)
```

## Puzzle 15 - 1D Convolution

**Function**

$$
f(x, y)_o = \sum_{j=1}^W \frac{x_{o+j} * y_{j}}{W}
$$

**Expected Jacobian Shape**: `(O, I)`, `(O, W)`

This is the standard 1D convolution used in deep learning. **There is no wrap-around**.
**Note**: This is probably the hardest one. The answer is short but tricky.

This is definitely the hardest problem so far. Also, don't believe the note - it is not short (at least not the solution I found). We have to work out 2 Jacobians similarly to the other problems. However, there are a few things that we need to pay attention to. First we must address the fact that the function given above is misleading. What is described as 1D convolution above is actually cross correlation. In this notebook, `np` is an alias for `jax.numpy`. Its `np.convolve` performs convolution, so the kernel is reversed relative to the cross-correlation shown in the original statement. We follow the supplied code, which uses `mode="valid"`.

For these real-valued inputs, convolution is equivalent to cross-correlation with a reversed kernel when the same output mode is used. This explains the reversal; it does not imply that JAX uses NumPy's internal implementation.

This means that we can't actually use the function as described above. We have to adjust the indices of $y$ so it corresponds to the proper operation. Additionally, the function indices are 1-based, whereas we would be better off using 0-indexed notation to help us translate it into code later. With these in consideration, our function becomes:

$$
f_o =
\sum_{j=0}^{W-1}\frac{x_{o+j} * y_{W-1-j}}{W}
$$

With this change in mind, we can start thinking about the Jacobians.

### Jacobian w.r.t. **x**

For each output $f_o$, the $x$ values that contribute are:

$$
x_o, x_{o+1}, \ldots, x_{o+W-1}
$$

So an input $x_i$ contributes to $f_o$ only if:

$$
o \leq i < o+W
$$

We can see what the derivatives look like by expanding the first output. With $W=5$:

$$
f_0 = \frac{x_0y_4+x_1y_3+x_2y_2+x_3y_1+x_4y_0}{W}
$$

Therefore:

$$
\frac{\partial f_0}{\partial x_0} = \frac{y_4}{W}
$$

$$
\frac{\partial f_0}{\partial x_1} = \frac{y_3}{W}
$$

$$
\frac{\partial f_0}{\partial x_2} = \frac{y_2}{W}
$$

$$
\frac{\partial f_0}{\partial x_3} = \frac{y_1}{W}
$$

$$
\frac{\partial f_0}{\partial x_4} = \frac{y_0}{W}
$$

So for any $x_i$, its position inside the window is:

$$
j=i-o
$$

Now we need to turn this into an `(O, I)` matrix. `Is - Os` gives us $i-o$ for every combination of output $o$ and input $i$:

```text
[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9]
[-1,  0,  1,  2,  3,  4,  5,  6,  7,  8]
[-2, -1,  0,  1,  2,  3,  4,  5,  6,  7]
[-3, -2, -1,  0,  1,  2,  3,  4,  5,  6]
[-4, -3, -2, -1,  0,  1,  2,  3,  4,  5]
[-5, -4, -3, -2, -1,  0,  1,  2,  3,  4]
```

For example, the first row corresponds to $o=0$:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

The values `0, 1, 2, 3, 4` are exactly the positions of $x_0,\ldots,x_4$ inside the first window. The rest are outside the window. This is a nice start for an indexing matrix for $y$. If we could retain the values in `[0, W-1]` and put 0s everywhere else, we could use this to index $y$ and get our first Jacobian. However we are not quite there yet. We have to zero out the unrelated indexes, which we will be doing with 2 masks. Since valid positions must satisfy:

$$
0 \leq i-o < W
$$

We need a mask to zero out the negative elements, and a mask to zero out the elements larger than W-1. It is pretty straightforward to write these out:
- First mask: $0 \le i - o \Rightarrow o \le i$, or in code `Os <= Is`
- Second mask: $i - o \lt W$, or in code `Is - Os < W`

These two masks have both shape `(O, I)`. By combining them with multiplication, we are going to be performing a logical `AND` between them, and finding their intersection, which is exactly the indices that we care about. Therefore, for the first Jacobian, we arrive at the following result:

```python
(Is - Os < W) * (Is >= Os) * y[W - 1 - (Is - Os)] / W
```

> **Note:** Multiplying by a mask zeros out unwanted coefficients after indexing; it does not skip the lookup. This expression relies on JAX's indexing behavior for excluded pairs, and is not directly portable to NumPy. Clipping the kernel index first would make the lookup explicitly valid.

### Jacobian w.r.t. **y**

We can use the same expanded expression:

$$
f_0 = \frac{x_0y_4+x_1y_3+x_2y_2+x_3y_1+x_4y_0}{W}
$$

Now we differentiate with respect to each `y` value:

$$
\frac{\partial f_0}{\partial y_0} = \frac{x_4}{W}
$$

$$
\frac{\partial f_0}{\partial y_1} = \frac{x_3}{W}
$$

$$
\frac{\partial f_0}{\partial y_2} = \frac{x_2}{W}
$$

$$
\frac{\partial f_0}{\partial y_3} = \frac{x_1}{W}
$$

$$
\frac{\partial f_0}{\partial y_4} = \frac{x_0}{W}
$$

So for a given output `o` and kernel index `j`, the corresponding `x` index is:

$$
i = o + W - 1 - j
$$

The `o` tells us where the current window starts, while `W - 1 - j` accounts for the fact that `y` is reversed.

We can generate all these indices at once using `Os` and `Ws`:

```python
Os + W - 1 - Ws
```

For example, with `W=5`:

```text
[4, 3, 2, 1, 0]
[5, 4, 3, 2, 1]
[6, 5, 4, 3, 2]
```

The first row corresponds to `o=0`, so it selects:

$$
[x_4, x_3, x_2, x_1, x_0]
$$

which are exactly the `x` values paired with:

$$
[y_0, y_1, y_2, y_3, y_4]
$$

This already has the required `(O, W)` shape. Therefore the Jacobian with respect to $y$ is simply:

```python
x[Os + W - 1 - Ws] / W
```

Putting the two Jacobians together will give us the puzzle solution:

```python
I = 10
O = 6
W = 5
Is = np.arange(I)
Os = np.arange(O)[:, None]
Ws = np.arange(W)


def f(x: Shaped[Array, "I"], y: Shaped[Array, "W"]) -> Shaped[Array, "O"]:
    return np.convolve(x, y, mode="valid") / W


def jac(
    x: Shaped[Array, "I"], y: Shaped[Array, "W"]
) -> tuple[Shaped[Array, "O I"], Shaped[Array, "O W"]]:
    return (Is - Os < W) * (Is >= Os) * y[W - 1 - (Is - Os)] / W, x[Os + W - 1 - Ws] / W


check(f, jac, in_shape=Is.shape, in_shape2=Ws.shape)
```

## Puzzle 16 - View

**Function**

$$
f(X)_{iJ+j} = X_{i,j}
$$

Flatten the input matrix into a 1D vector.

**Expected Jacobian Shape**: `(O, I, J)`

In this puzzle, we want to calculate the Jacobian of a 2D array after it has been flattened into 1D. Our input $x$ has a shape `(I, J)`:

$$
X =
\begin{bmatrix}
x_{0,0} & x_{0,1} & x_{0,2} & x_{0,3} \\
x_{1,0} & x_{1,1} & x_{1,2} & x_{1,3} \\
x_{2,0} & x_{2,1} & x_{2,2} & x_{2,3} \\
x_{3,0} & x_{3,1} & x_{3,2} & x_{3,3}
\end{bmatrix}
$$

and `f` flattens this into a 1D array:

$$
f(X) =
\begin{bmatrix}
x_{0,0} \\
x_{0,1} \\
x_{0,2} \\
x_{0,3} \\
x_{1,0} \\
x_{1,1} \\
x_{1,2} \\
x_{1,3} \\
x_{2,0} \\
x_{2,1} \\
x_{2,2} \\
x_{2,3} \\
x_{3,0} \\
x_{3,1} \\
x_{3,2} \\
x_{3,3}
\end{bmatrix}
$$

So our problem is essentially finding a way to map the 2D index of $x$ into a 1D index. Thankfully, this is a very common pattern and can be performed by doing:

$$
\text{index}_{1D} = (\text{row}\cdot \text{width}) + \text{column}
$$

Or, in code:

```python
o = i * J + j
```

If we can construct the flattened view of $x$, the Jacobian itself is trivial. Since we are dealing with an identity function, for each $o$, we want it to contain 1 at the corresponding $(i, j)$ and 0 everywhere else. Our Jacobian has shape `(O,I,J)`, so each output gets its own `(I,J)` matrix. For example, the Jacobian slice for `o=5` should look like:

$$
J_5 =
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

because $f_5 = x_{1,1}$.

If we convert the above relationship to our vector arrays, we get the solution:

```python
I = 4
J = 4
O = 16
Is = np.arange(I)[:, None]
Js = np.arange(J)
Os = np.arange(O)[:, None, None]


def f(x: Shaped[Array, "I J"]) -> Shaped[Array, "O"]:
    return x.reshape((O,))


def jac(x: Shaped[Array, "I J"]) -> Shaped[Array, "O I J"]:
    return Os == (Is * J) + Js


check(f, jac, in_shape=(I, J))
```

`Os` has shape `(O,1,1)`, while `(Is * J) + Js` has shape `(I,J)`, so broadcasting produces the full `(O,I,J)` Jacobian. The result is `True` exactly where the output index corresponds to that input position. Since each output is simply a view of one input element, these `True` values represent the `1`s in the Jacobian.

## Puzzle 17 - Transpose

Swap the rows and columns of the input matrix.

**Function**

$$
f(X)_{o, p} = X_{p, o}
$$

**Expected Jacobian Shape**: `(O, P, I, J)`

The goal of this puzzle is to swap the rows and columns of a 2D array.

$$
X =
\begin{bmatrix}
x_{0,0} & x_{0,1} & x_{0,2} & x_{0,3} \\
x_{1,0} & x_{1,1} & x_{1,2} & x_{1,3} \\
x_{2,0} & x_{2,1} & x_{2,2} & x_{2,3} \\
x_{3,0} & x_{3,1} & x_{3,2} & x_{3,3}
\end{bmatrix}
$$

After transposing, the output is:

$$
f(X) =
\begin{bmatrix}
x_{0,0} & x_{1,0} & x_{2,0} & x_{3,0} \\
x_{0,1} & x_{1,1} & x_{2,1} & x_{3,1} \\
x_{0,2} & x_{1,2} & x_{2,2} & x_{3,2} \\
x_{0,3} & x_{1,3} & x_{2,3} & x_{3,3}
\end{bmatrix}
$$

So each output element $f(X)_{o,p}$ corresponds to the input element $X_{p,o}$. In other words, the output row `o` corresponds to the input column `o`, and the output column `p` corresponds to the input row `p`. Since each output is just one input value, the derivative is either `1` or `0`:

$$
\begin{cases}
1 & i=p \text{ and } j=o \\
0 & \text{otherwise}
\end{cases}
$$

So we need two comparisons:

- `Ps == Is` checks whether the output's column index matches the input's row index.

- `Os == Js` checks whether the output's row index matches the input's column index.

Together:

```python
(Ps == Is) * (Os == Js)
```

The two `bool` arrays broadcast together to produce the `(O,P,I,J)` Jacobian.

For example, $f(X)_{1,2} = X_{2,1}$, so its Jacobian slice should contain a 1 at `(i=2,j=1)` and 0 everywhere else.

```python
I = 4
J = 4
O = 4
P = 4
Is = np.arange(I)[:, None]
Js = np.arange(J)
Os = np.arange(O)[:, None, None, None]
Ps = np.arange(P)[:, None, None]


def f(x: Shaped[Array, "I J"]) -> Shaped[Array, "O P"]:
    return x.T


def jac(x: Shaped[Array, "I J"]) -> Shaped[Array, "O P I J"]:
    return (Ps == Is) * (Os == Js)


check(f, jac, in_shape=(I, J))
```

## Puzzle 18 - Mean Reduce

Compute the mean over rows.

**Function**

$$
f(X)_o = \sum_{i=0}^{I-1} \frac{X_{i,o}}{I}
$$

**Expected Jacobian Shape**: `(O, I, J)`

The goal here is pretty simple: we take the mean of each **column** of `X`.
For example, with `I = J = 4`:

$$
X =
\begin{bmatrix}
x_{0,0} & x_{0,1} & x_{0,2} & x_{0,3} \\
x_{1,0} & x_{1,1} & x_{1,2} & x_{1,3} \\
x_{2,0} & x_{2,1} & x_{2,2} & x_{2,3} \\
x_{3,0} & x_{3,1} & x_{3,2} & x_{3,3}
\end{bmatrix}
$$

we get:

$$
f(X) =
\begin{bmatrix}
\frac{x_{0,0}+x_{1,0}+x_{2,0}+x_{3,0}}{I} \\
\frac{x_{0,1}+x_{1,1}+x_{2,1}+x_{3,1}}{I} \\
\frac{x_{0,2}+x_{1,2}+x_{2,2}+x_{3,2}}{I} \\
\frac{x_{0,3}+x_{1,3}+x_{2,3}+x_{3,3}}{I}
\end{bmatrix}
$$

So each output depends on **all the inputs in the same column**, and anything from another column has derivative 0.

So all we really need to check is whether the output index `o` matches the input column `j`:

```python
Os == Js
```

The annoying part is that this only gives us the `(O,1,J)` part of the shape. We still need the `I` dimension because the same `1/I` derivative applies to **every row**.

The cleanest way I found to do this is to use `Is * 0` to introduce that missing dimension:

```python
I = 4
J = 4
O = 4
Is = np.arange(I)[:, None]
Js = np.arange(J)
Os = np.arange(O)[:, None, None]


def f(x: Shaped[Array, "I J"]) -> Shaped[Array, "O"]:
    return np.mean(x, axis=0)


def jac(x: Shaped[Array, "I J"]) -> Shaped[Array, "O I J"]:
    return (Os == Js) * (1 / I) + Is * 0


check(f, jac, in_shape=(I, J))
```

## Puzzle 19 - Matmul as vectors

Standard matrix multiplication

**Function**

$$
f(X, Y)_{o,p} = \sum_j X_{o, j} Y_{j,p}
$$

**Expected Jacobian Shape**: `(O, I)`, `(O, J)`

In this puzzle, even though we are working with 1D vectors, we have to be mindful of how indices are laid out and multiplied as 2D matrices. We can start tackling this problem by looking at $f$ as defined in code:

```python
def f(x: Shaped[Array, "I"], y: Shaped[Array, "J"]) -> Shaped[Array, "O "]:
    return (x.reshape(2, 2) @ y.reshape(2, 2)).reshape(O)
```

This basically translates to the following:

$$
X =
\begin{bmatrix}
x_{0} & x_{1}\\
x_{2} & x_{3}\\
\end{bmatrix}
\qquad
Y =
\begin{bmatrix}
y_{0} & y_{1}\\
y_{2} & y_{3}\\
\end{bmatrix}
$$

If we perform matrix multiplication, our $f$ will be:

$$
f = \begin{bmatrix}
x_0 \cdot y_0 + x_1 \cdot y_2 & x_0 \cdot y_1 + x_1 \cdot y_3\\
x_2 \cdot y_0 + x_3 \cdot y_2 & x_2 \cdot y_1 + x_3 \cdot y_3\\
\end{bmatrix}
$$

Or, alternatively:

$$
f_0 = x_0 \cdot y_0 + x_1 \cdot y_2
$$

$$
f_1 = x_0 \cdot y_1 + x_1 \cdot y_3
$$

$$
f_2 = x_2 \cdot y_0 + x_3 \cdot y_2
$$

$$
f_3 = x_2 \cdot y_1 + x_3 \cdot y_3
$$

### Jacobian w.r.t. **x**

If we take $f$ as shown in the 4 expressions above and derive it w.r.t $x_i$ to find each partial derivative, we will be zeroing out the unrelated term and replacing $x_i$ with 1, so essentially only the $y$. The indices $i$ of $x$ show which column the partial derivative should occupy. Putting all these together, we have the following:

$$
\frac{\partial f_0}{\partial x_0} = y_0 \qquad \text{(placed on row 0, column 0)}
$$

$$
\frac{\partial f_0}{\partial x_1} = y_2 \qquad \text{(placed on row 0, column 1)}
$$

$$
\frac{\partial f_1}{\partial x_0} = y_1 \qquad \text{(placed on row 1, column 0)}
$$

$$
\frac{\partial f_1}{\partial x_1} = y_3 \qquad \text{(placed on row 1, column 1)}
$$

$$
\frac{\partial f_2}{\partial x_2} = y_0 \qquad \text{(placed on row 2, column 2)}
$$

$$
\frac{\partial f_2}{\partial x_3} = y_2 \qquad \text{(placed on row 2, column 3)}
$$

$$
\frac{\partial f_3}{\partial x_2} = y_1 \qquad \text{(placed on row 3, column 2)}
$$

$$
\frac{\partial f_3}{\partial x_3} = y_3 \qquad \text{(placed on row 3, column 3)}
$$

So our Jacobian w.r.t $x$ will look like this:

$$
J_x =
\begin{bmatrix}
y_0 & y_2 & 0 & 0 \\
y_1 & y_3 & 0 & 0 \\
0 & 0 & y_0 & y_2 \\
0 & 0 & y_1 & y_3
\end{bmatrix}
$$

So we basically need to do some creative indexing on vector $y$ to get it to give us its values in the way that we want it to. Remember, $y$ is a 1D vector. Let's work on the first row. For each row, its elements are 2 positions apart, so we can figure out that something like `Is % 2` will move us forward in the general direction we want to go:

```text
Is % 2 = [0, 1, 0, 1]
```

These are kernel indices, not the final coefficients. We want to select `y_0` and `y_2` in the first two columns; the mask will zero out the other columns later. The first problem to solve, is that we need to multiply the expression above by 2 in order to get closer to the correct indices. This gives us:

```text
2 * (Is % 2) = [0, 2, 0, 2]
```

Let's now try to project this indexing downwards to create the rest of the rows. We can observe that we basically want to "add 1" to the index so `0, 2 -> 1, 3` for the second row, then we repeat the first two rows but on a different column. We can add 1 every other row by adding `Os % 2` to our expression. This has the added benefit that it will broadcast the indexes down to give us the desired shape:

```text
2 * (Is % 2) + Os % 2 =
    [0, 2, 0, 2]
    [1, 3, 1, 3]
    [0, 2, 0, 2]
    [1, 3, 1, 3]
```

We are now very close to the desired Jacobian, but we have some extra values. We can clear them by multiplying with a `bool` mask of the same shape. The mask should look something like this:

```text
mask =
    [1, 1, 0, 0]
    [1, 1, 0, 0]
    [0, 0, 1, 1]
    [0, 0, 1, 1]
```

This is an easy mask that can be formed by comparing `Os // 2` with `Is // 2`. On their own, they produce the array `[0, 0, 1, 1]`. Broadcast together, they give the desired mask:

```text
Os // 2 == Is // 2 =
    [ True,  True, False, False]
    [ True,  True, False, False]
    [False, False,  True,  True]
    [False, False,  True,  True]
```

Our final expression for the Jacobian w.r.t. $x$ is:

```python
(Os // 2 == Is // 2) * y[2 * (Is % 2) + Os % 2]
```

### Jacobian w.r.t. **y**

We basically want to repeat the above investigation, but for $y$. Let's get the partial derivatives and form our Jacobian. Everything we said for $x$ still applies, but for $y$ this time:

$$
\frac{\partial f_0}{\partial y_0} = x_0 \qquad \text{(placed on row 0, column 0)}
$$

$$
\frac{\partial f_0}{\partial y_2} = x_1 \qquad \text{(placed on row 0, column 2)}
$$

$$
\frac{\partial f_1}{\partial y_1} = x_0 \qquad \text{(placed on row 1, column 1)}
$$

$$
\frac{\partial f_1}{\partial y_3} = x_1 \qquad \text{(placed on row 1, column 3)}
$$

$$
\frac{\partial f_2}{\partial y_0} = x_2 \qquad \text{(placed on row 2, column 0)}
$$

$$
\frac{\partial f_2}{\partial y_2} = x_3 \qquad \text{(placed on row 2, column 2)}
$$

$$
\frac{\partial f_3}{\partial y_1} = x_2 \qquad \text{(placed on row 3, column 1)}
$$

$$
\frac{\partial f_3}{\partial y_3} = x_3 \qquad \text{(placed on row 3, column 3)}
$$

So our Jacobian w.r.t $y$ will look like this:

$$
J_y =
\begin{bmatrix}
x_0 & 0 & x_1 & 0 \\
0 & x_0 & 0 & x_1 \\
x_2 & 0 & x_3 & 0 \\
0 & x_2 & 0 & x_3
\end{bmatrix}
$$

We will take the same approach as above. We will get close to the desired Jacobian, then we will use a mask to zero out the undesirable elements. We can start by using `Js // 2` to get the base form of our first row set up:

```text
Js // 2 = [0, 0, 1, 1]
```

Our first 2 rows are basically ready to be masked if we broadcast this first row down. Moving on to the last two rows, we need to repeat the first 2 rows and add 2 to the index. We can do this with `2*(Os // 2)`, which essentially adds 2 to rows 2 and 3:

```text
2 * (Os // 2) = [[0],
                [0],
                [2],
                [2]]  # shape (O, 1)
```

Putting these together will form an index matrix that is very close to what we want:

```text
Js // 2 + 2*(Os // 2) =
    [0 0 1 1]
    [0 0 1 1]
    [2 2 3 3]
    [2 2 3 3]
```

Now, we need to mask out the extra values. The required comparison is similar to the previous mask, but it uses `mod` instead of integer division, to find the `True` values:

```text
Os % 2 == Js % 2 =
    [ True, False,  True, False]
    [False,  True, False,  True]
    [ True, False,  True, False]
    [False,  True, False,  True]
```

Our final expression for $J_y$ is:

```python
(Os % 2 == Js % 2) * x[Js // 2 + 2 * (Os // 2)]
```

and the solution is:

```python
I = 4
J = 4
O = 4
Is = np.arange(I)
Js = np.arange(J)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"], y: Shaped[Array, "J"]) -> Shaped[Array, "O "]:
    return (x.reshape(2, 2) @ y.reshape(2, 2)).reshape(O)


def jac(
    x: Shaped[Array, "I"], y: Shaped[Array, "J"]
) -> tuple[Shaped[Array, "O I"], Shaped[Array, "O J"]]:
    return (Os // 2 == Is // 2) * y[2 * (Is % 2) + Os % 2], (Os % 2 == Js % 2) * x[Js // 2 + 2 * (Os // 2)]


check(f, jac, in_shape=Is.shape, in_shape2=Js.shape)
```

> **Note:** This solution uses the fixed 2×2 matrix dimensions from the puzzle. A general version would need explicit integer matrix dimensions and matching reshape/index formulas. The final return line also exceeds the original 80-character limit.

## Puzzle 20 - 2D Conv

Two Dimensional Convolution

**Original notebook statement**

$$
f(X, Y)_{o} = \sum_{dh}\sum_{dw} X_{o+dh*W+dw} Y_{dh,dw}
$$

**Expected Jacobian Shape**: `(O, I)`, `(O, J)`

This puzzle follows a similar pattern to the 1D convolution, but now we have to keep track of both rows and columns. The derivatives themselves are pretty simple. The annoying part is figuring out which elements are multiplied together, while working with flattened arrays.

Let's start by looking at the function as defined in code:

```python
def f(x: Shaped[Array, "I"], y: Shaped[Array, "J"]) -> Shaped[Array, "O"]:
    X = x.reshape(4, 4)
    Y = y.reshape(2, 2)
    return convolve2d(X, Y, mode="same").reshape(-1) / (2 * 2)
```

Our inputs are 1D vectors, but the function treats them as the following matrices:

$$
X = \begin{bmatrix} x_0 & x_1 & x_2 & x_3 \\ x_4 & x_5 & x_6 & x_7 \\ x_8 & x_9 & x_{10} & x_{11} \\ x_{12} & x_{13} & x_{14} & x_{15} \end{bmatrix} \qquad Y = \begin{bmatrix} y_0 & y_1 \\ y_2 & y_3 \end{bmatrix}
$$

**Note**: Similarly to Puzzle 15, we need to be careful with the function given in the problem statement. `convolve2d` performs convolution, which flips the kernel along both axes. Additionally, we are using `mode="same"` rather than `"valid"`, and dividing the result by 4. These details affect both the indices and the derivatives.

The flipped kernel looks like this:

$$
Y_{\text{flipped}} = \begin{bmatrix} y_3 & y_2 \\ y_1 & y_0 \end{bmatrix}
$$

According to the [JAX docs](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.signal.convolve2d.html), `"same"` returns a centered portion of the full convolution, with the same shape as the first input. Positions outside the input are treated as zero.

With this particular 2×2 kernel, the bottom-right element of the flipped kernel aligns with the input position corresponding to the output. Therefore, each output uses that input and its neighbors to the left, above, and above-left.

For example, output $f_5$ corresponds to position `(1, 1)`. The input window and flipped kernel are:

$$
\begin{bmatrix} x_0 & x_1 \\ x_4 & x_5 \end{bmatrix} \qquad \begin{bmatrix} y_3 & y_2 \\ y_1 & y_0 \end{bmatrix}
$$

Multiplying the corresponding elements and summing gives us:

$$
f_5 = \frac{x_0y_3+x_1y_2+x_4y_1+x_5y_0}{4}
$$

At the boundaries, some of these input positions are outside the matrix. For example:

$$
f_0 = \frac{x_0y_0}{4}
$$

$$
f_1 = \frac{x_0y_1+x_1y_0}{4}
$$

$$
f_4 = \frac{x_0y_2+x_4y_0}{4}
$$

So we need to account for these boundaries when constructing our Jacobians.

To express the function in 2D coordinates, let $F_{r,c}$ refer to the output before flattening:

$$
F_{r,c} = \frac{1}{4} \sum_{a=0}^{1}\sum_{b=0}^{1} X_{r-a,c-b}Y_{a,b}
$$

Here, $X_{u,v}=0$ whenever the coordinates are outside the input matrix. This expression already accounts for the kernel flip through the subtraction in the input coordinates.

### Jacobian w.r.t. **x**

We can start with the expanded expression for $f_5$:

$$
f_5 = \frac{x_0y_3+x_1y_2+x_4y_1+x_5y_0}{4}
$$

Taking its partial derivatives gives us:

$$
\frac{\partial f_5}{\partial x_0} = \frac{y_3}{4} \qquad \frac{\partial f_5}{\partial x_1} = \frac{y_2}{4}
$$

$$
\frac{\partial f_5}{\partial x_4} = \frac{y_1}{4} \qquad \frac{\partial f_5}{\partial x_5} = \frac{y_0}{4}
$$

All the other inputs have derivative 0. So row 5 of our Jacobian should look like:

$$
(J_x)_{5,:} = \frac{1}{4}  \begin{bmatrix} y_3 & y_2 & 0 & 0 & y_1 & y_0 & 0 & \cdots & 0 \end{bmatrix}
$$

We will follow the same approach as before: construct a mask that identifies the contributing inputs, then select the corresponding values from $y$.

Even though our indices are flattened, we can recover their 2D coordinates using integer division and modulo:

$$
\text{row} = \text{index} \mathbin{//} 4 \qquad \text{column} = \text{index} \bmod 4
$$

For each output–input pair, we calculate the difference between their rows and columns:

```python
dr = Os // 4 - Is // 4
dc = Os % 4 - Is % 4
```

Since `Os` has shape `(O, 1)` and `Is` has shape `(I,)`, both expressions broadcast to shape `(O, I)`.

An input contributes only when both differences are either 0 or 1:

$$
0 \le dr \le 1 \qquad \text{and} \qquad 0 \le dc \le 1
$$

For example, for output 5:

|Input index|Input position|`dr`|`dc`|
|---|---|---|---|
|0|`(0, 0)`|1|1|
|1|`(0, 1)`|1|0|
|4|`(1, 0)`|0|1|
|5|`(1, 1)`|0|0|

We can form our mask with four comparisons:

```python
mask_x = (dr >= 0) * (dr <= 1) * (dc >= 0) * (dc <= 1)
```

This also handles the boundaries. We are comparing valid input positions, so there are no negative input coordinates to generate. Checking the rows and columns separately prevents an index from accidentally wrapping into another row.

Now we need to select the correct value from $y$. Conveniently, `dr` and `dc` already give us its kernel row and column. Since the kernel width is 2, its flattened index is:

$$
j = 2dr + dc
$$

For the four contributing inputs of output 5, this gives:

```text
input index:    0  1  4  5
kernel index:   3  2  1  0
```

So we select exactly the coefficients we wanted:

```python
y[2 * dr + dc]
```

There is no need to flip `y` again. The output-minus-input differences already produce the reversed spatial arrangement.

For excluded pairs, the index can fall outside the kernel. We can clip it to a valid index before the lookup; the mask will zero out those values anyway:

```python
dx = mask_x * y[np.clip(2 * dr + dc, 0, 3)] / 4
```

### Jacobian w.r.t. **y**

We can use the same expanded expression:

$$
f_5 = \frac{x_0y_3+x_1y_2+x_4y_1+x_5y_0}{4}
$$

This time, differentiating with respect to each kernel element leaves the corresponding input:

$$
\frac{\partial f_5}{\partial y_0} = \frac{x_5}{4} \qquad \frac{\partial f_5}{\partial y_1} = \frac{x_4}{4}
$$

$$
\frac{\partial f_5}{\partial y_2} = \frac{x_1}{4} \qquad \frac{\partial f_5}{\partial y_3} = \frac{x_0}{4}
$$

Therefore:

$$
(J_y)_{5,:} = \frac{1}{4} \begin{bmatrix} x_5 & x_4 & x_1 & x_0 \end{bmatrix}
$$

For each output and kernel element, we need to find the input multiplied by that kernel element.

The kernel coordinates are:

```python
Js // 2   # kernel row
Js % 2    # kernel column
```

Subtracting these offsets from the output coordinates gives us the corresponding input coordinates:

```python
r = Os // 4 - Js // 2
c = Os % 4 - Js % 2
```

These expressions broadcast to shape `(O, J)`. For output 5, they give:

```python
r = [1, 1, 0, 0]
c = [1, 0, 1, 0]
```

We can flatten these coordinates using the input width, which is 4:

$$
i = 4r+c
$$

```text
4*r + c = [5, 4, 1, 0]
```

This selects the exact input values needed for that Jacobian row.

The remaining issue is the boundary. Since we only move up and left from valid output positions, we only need to check whether the resulting row or column is negative:

```python
mask_y = (r >= 0) * (c >= 0)
```

Again, we can clip the flattened indices to keep the lookup valid, then apply the mask:

```python
dy = mask_y * x[np.clip(4 * r + c, 0, I - 1)] / 4
```

For example, at output 0, only the first kernel element overlaps the input, so this gives:

$$
(J_y)_{0,:} = \frac{1}{4} \begin{bmatrix} x_0 & 0 & 0 & 0 \end{bmatrix}
$$

Putting the two Jacobians together gives us the following expanded solution. The intermediate variables and clipped lookups above explain the logic; this version inlines the expressions and relies on JAX's indexing behavior for masked-out pairs. The masks do not prevent those lookups from being evaluated.

```python
from jax.scipy.signal import convolve2d

I = 16  # 4*4
J = 4  # 2*2
O = 16  # 4*4
Is = np.arange(I)
Js = np.arange(J)
Os = np.arange(O)[:, None]


def f(x: Shaped[Array, "I"], y: Shaped[Array, "J"]) -> Shaped[Array, "O"]:
    X = x.reshape(4, 4)
    Y = y.reshape(2, 2)

    return convolve2d(X, Y, mode="same").reshape(-1) / (2 * 2)


def jac(x: Shaped[Array, "I"], y: Shaped[Array, "J"]) -> tuple[Shaped[Array, "O I"], Shaped[Array, "O J"]]:
    # Sorry for this.
    return (Os // 4 - Is // 4 >= 0) * (Os // 4 - Is // 4 <= 1) * (Os % 4 - Is % 4 >= 0) * (Os % 4 - Is % 4 <= 1) * y[(Os // 4 - Is // 4) * 2 + (Os % 4 - Is % 4)] / 4, (Os // 4 - Js // 2 >= 0) * (Os % 4 - Js % 2 >= 0) * x[4 * (Os // 4 - Js // 2) + (Os % 4 - Js % 2)] / 4


check(f, jac, in_shape=Is.shape, in_shape2=Js.shape)
```

**Note**: I didn't really manage to keep this below 80 characters. Maybe a better solution exists?