---
author: Luca Lombardo
pubDatetime: 2025-09-16T00:00:00Z
title: Cache-Friendly, Low-Memory Lanczos Algorithm in Rust
slug: cache-friendly-low-memory-lanczos
featured: true
draft: false
tags:
  - Rust
  - numerical-analysis
description: Implementing a cache-friendly, low-memory two-pass Lanczos algorithm in Rust, focusing on efficient memory access patterns and minimal allocations.
---

The standard Lanczos method for computing matrix functions has a brutal memory requirement: storing an $n \times k$ basis matrix that grows with every iteration. For a 500,000-variable problem needing 1,000 iterations, that's ~4 GB just for the basis. In this post we will explore one of the most straightforward solutions to this problem: a two-pass variant of the Lanczos algorithm that only requires O(n) memory at the cost of doubling the number of matrix-vector products. The surprising part? When implemented carefully, the two-pass version isn't just memory-efficient, it can be faster for certain problems. In this post we will try to do some algorithm engineering to understand why.

- All code is available on GitHub: [two-pass-lanczos](https://github.com/lukefleed/two-pass-lanczos)
- The full technical report with proofs and additional experiments: [report.pdf](/two-pass-lanczos/tex/report.pdf)

---

## Table of Contents

# Computing Matrix Functions

Let's consider the problem of computing the action of matrix functions on a vector:

$$
\mathbf{x} = f(\mathbf{A})\mathbf{b}
$$

where $\mathbf{A}$ is a large sparse Hermitian matrix and $f$ is a matrix function defined on the spectrum of $\mathbf{A}$. This is a problem that appears pretty often in scientific computing: solving linear systems corresponds to $f(z) = z^{-1}$, exponential integrators for PDEs use $f(z) = \exp(tz)$, and many other problems require functions like $f(z) = z^{-1/2}$ or $f(z) = \text{sign}(z)$.

Indeed, there are a lot problems with computing $f(\mathbf{A})$ directly. First of all, even if $\mathbf{A}$ is sparse, $f(\mathbf{A})$ is generally dense. Storing it explicitly is out of the question for large problems. Even if we could store it, computing it directly would require algorithms like the Schur-Parlett method that scale as $O(n^3)$, which is impractical for large $n$.

However we know that given any matrix function $f$ defined on the spectrum of $\mathbf{A}$, we can express $f(\mathbf{A})$ as a polynomial in $\mathbf{A}$ of degree at most $n$ (the size of the matrix) such that $f(\mathbf{A}) = p_{n}(\mathbf{A})$ (this is a consequence of the Cayley-Hamilton theorem). This polynomial interpolates $f$ and its derivatives in the Hermitian sense at the eigenvalues of $\mathbf{A}$.

This gives a good and a bad news: the good news is that, well, we can express $f(\mathbf{A})$ as a polynomial in $\mathbf{A}$. The bad news is that the degree of this polynomial can be as high as $n$, which is huge for large problems. The idea is then to find a low-degree polynomial approximation to $f$ that is "good enough" for our purposes. If we can find a polynomial $p_k$ of degree $k \ll n$ such that $p_k(\mathbf{A}) \approx f(\mathbf{A})$, then we can approximate the solution as:

$$
f(\mathbf{A})\mathbf{b} \approx p_k(\mathbf{A})\mathbf{b} = \sum_{i=0}^k c_i \mathbf{A}^i \mathbf{b}
$$

## Krylov Projection

We notice that the polynomial approximation only involves vectors that lie in the following space, called the Krylov subspace of order $k$:

$$
\mathcal{K}_k(\mathbf{A}, \mathbf{b}) = \text{span}\{\mathbf{b}, \mathbf{Ab}, \mathbf{A}^2\mathbf{b}, \ldots, \mathbf{A}^{k-1}\mathbf{b}\}
$$

We can then search the approximate solution $\mathbf{x}_k$ in this subspace by just doing matrix-vector products with $\mathbf{A}$. This is great because it's the only operation we can do efficiently for large sparse matrices.

> Note that we don't need to construct and keep in memory $\mathbf{A}^j$ for any $j$. We can just compute it iteratively as $\mathbf{A}(\mathbf{A}^{j-1}\mathbf{b})$.

### Finding a basis

Technically, we already have a basis for this space. The raw vectors $\{\mathbf{A}^j\mathbf{b}\}$ form a basis, but it's a terrible one for computation since they quickly become nearly parallel. We need a stable, orthonormal basis. The standard way to build one is the Arnoldi process. It's essentially a Gram-Schmidt procedure tailored for Krylov subspaces.

We start by normalizing our initial vector, $\mathbf{v}_1 = \mathbf{b} / \|\mathbf{b}\|_2$. The process is iterative. At each step $j$, we first create a new candidate vector by applying the operator, $\mathbf{w}_j = \mathbf{A}\mathbf{v}_j$. Next, we make it orthogonal to our existing basis vectors $\mathbf{v}_1, \ldots, \mathbf{v}_j$ using the Gram-Schmidt procedure. This involves subtracting the projections:

$$
\tilde{\mathbf{v}}_j = \mathbf{w}_j - \sum_{i=1}^j (\mathbf{v}_i^H \mathbf{w}_j) \mathbf{v}_i
$$

The coefficients we compute here, $h_{ij} = \mathbf{v}_i^H \mathbf{w}_j = \mathbf{v}_i^H \mathbf{A} \mathbf{v}_j$, become the entries of our projected matrix. Finally, we normalize the new vector to get the next basis vector, where $h_{j+1, j} = \|\tilde{\mathbf{v}}_j\|_2$ and $\mathbf{v}_{j+1} = \tilde{\mathbf{v}}_j / h_{j+1, j}$.

After $k$ steps, we end up with

- $\mathbf{V}_k \in \mathbb{C}^{n \times k} = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$, an orthonormal basis for $\mathcal{K}_k(\mathbf{A}, \mathbf{b})$.
- $\mathbf{H}_k \in \mathbb{C}^{k \times k}$, an upper Hessenberg matrix with entries $h_{ij}$. This matrix represents the projection of $\mathbf{A}$ onto the Krylov subspace.

> Why $\mathbf{H}_k$ is upper Hessenberg? Because in the Arnoldi process, each new vector $\mathbf{v}_{j+1}$ is orthogonalized against all previous vectors $\mathbf{v}_1, \ldots, \mathbf{v}_j$. This means that when we compute the projections $\mathbf{v}_i^H \mathbf{A} \mathbf{v}_j$, only the first $j+1$ basis vectors contribute to the new vector. Therefore, $h_{ij} = 0$ for all $i > j + 1$, resulting in an upper Hessenberg structure.

We can summarize the whole process with the matrix formulation known as the Arnoldi decomposition:

$$
\mathbf{A}\mathbf{V}_k = \mathbf{V}_k \mathbf{H}_k + h_{k+1,k} \mathbf{v}_{k+1} \mathbf{e}_k^T
$$

How do we use this to approximate our original problem? Well it depends on what we consider a "good" approximation. For this blog post we'll just consider Full Orthogonal Method (FOM), which enforce that the residual $\mathbf{r}_k = f(\mathbf{A})\mathbf{b} - \mathbf{x}_k$ is orthogonal to the Krylov subspace. Therefore, we can express the approximate solution as:

$$
\mathbf{x}_k = \mathbf{V}_k \mathbf{y}_k
$$

where $\mathbf{y}_k$ is a small $k$-dimensional vector that we need to compute as

$$
\mathbf{y}_k = f(\mathbf{H}_k) \mathbf{e}_1 \|\mathbf{b}\|_2
$$

> The term $\mathbf{e}_1 \|\mathbf{b}\|_2$ comes from the fact that our initial vector $\mathbf{v}_1$ is $\mathbf{b}$ normalized. In the Krylov subspace, this corresponds to the first basis vector scaled by the norm of $\mathbf{b}$. This is just the projection of $\mathbf{b}$ onto the Krylov subspace.

This problems is now reduced to computing $f(\mathbf{H}_k)$, which is a small $k \times k$ matrix. This is relatively cheap since $k \ll n$ and we can use direct methods like the Schur-Parlett method that require $O(k^3)$ operations.

> If we choose $f(z) = z^{-1}$, then is reduces to solving the linear system $\mathbf{H}_k \mathbf{y}_k = \mathbf{e}_1 \|\mathbf{b}\|_2$, which can be done efficiently with direct methods like LU decomposition.

<!--
If we left-multiply by $\mathbf{V}_k^H$, we see that $\mathbf{H}_k = \mathbf{V}_k^H \mathbf{A} \mathbf{V}_k$. This is the key: $\mathbf{H}_k$ is the projection of the huge operator $\mathbf{A}$ onto our small Krylov subspace. We can now approximate our original problem by solving it in this small space:

$$
\mathbf{x}_k = \|\mathbf{b}\|_2 \mathbf{V}_k f(\mathbf{H}_k) \mathbf{e}_1
$$

This is a huge win. We've turned a problem of dimension $n$ into one of dimension $k$, which is cheap to solve.

But we can do even better. Our problem states that $\mathbf{A}$ is Hermitian. This imposes a special structure on the projection. Since $\mathbf{A} = \mathbf{A}^H$, the projection $\mathbf{H}_k$ must also be Hermitian. A matrix that is both upper Hessenberg and Hermitian has to be real, symmetric, and tridiagonal. This simplification is what turns the general Arnoldi process into the much more efficient symmetric Lanczos process. -->

# Lanczos Iterations

At the beginning of this post we said that $\mathbf{A}$ is Hermitian. This means that, in the real case, $\mathbf{A}$ is symmetric: $\mathbf{A} = \mathbf{A}^T$. In the complex case, it means $\mathbf{A} = \mathbf{A}^H$, where $\mathbf{A}^H$ is the conjugate transpose. This special property has huge implications for the Krylov projection.

We can prove that $\mathbf{H}_k = \mathbf{V}_k^H \mathbf{A} \mathbf{V}_k$. This means that if $\mathbf{A}$ is Hermitian, then $\mathbf{H}_k$ must also be Hermitian. A matrix that is both upper Hessenberg and Hermitian has to be real, symmetric, and tridiagonal.

In the literature we refer to this projected matrix as $\mathbf{T}_k$ instead of $\mathbf{H}_k$ to highlight its special tridiagonal structure. Where $\alpha_j \in \mathbb{R}$ (reals since $h_{ij} = \overline{h_{ji}}$) are the diagonal elements and $\beta_j = h_{j+1,j} \in \mathbb{R}$ are the off-diagonal elements.

Let's re-write the Arnoldi decomposition for this case:

$$
\mathbf{A}\mathbf{V}_k = \mathbf{V}_k \mathbf{T}_k + \beta_k \mathbf{v}_{k+1} \mathbf{e}_k^T
$$

We can extract the $j$-th column from this matrix equation. On the left, we have $\mathbf{A}\mathbf{v}_j$. On the right, we need the $j$-th column of the whole expression. The second term, $\beta_k \mathbf{v}_{k+1} \mathbf{e}_k^T$, only affects the final column ($j=k$), so for now let's assume $j < k$. For the right-end-side for the equation we can exploit the tridiagonal structure of $\mathbf{T}_k$ to simplify the expression:

$$
\mathbf{V}_k (\mathbf{T}_k \mathbf{e}_j) = \beta_{j-1} \mathbf{v}_{j-1} + \alpha_j \mathbf{v}_j + \beta_j \mathbf{v}_{j+1}
$$

Thus, we get the famous **Lanczos three-term recurrence**:

$$
\mathbf{A}\mathbf{v}_j = \beta_{j-1}\mathbf{v}_{j-1} + \alpha_j \mathbf{v}_j + \beta_j \mathbf{v}_{j+1}
$$

The crucial detail to notice here is that to generate the next basis vector $\mathbf{v}_{j+1}$, we only need the two previous vectors, $\mathbf{v}_j$ and $\mathbf{v}_{j-1}$. We don't need to orthogonalize against the entire history of the basis, because the Hermitian nature of $\mathbf{A}$ guarantees that the new vector is already orthogonal to all vectors $\mathbf{v}_1, \ldots, \mathbf{v}_{j-2}$.

So as before, we can determine the coefficients directly from this recurrence. By rearranging it, we get an expression for the unnormalized next vector:

$$
w_{j+1} = \mathbf{A}\mathbf{v}_j
$$

Then we compute the first component along $v_j$:

$$
\alpha_j = \mathbf{v}_j^H w_{j+1}
$$

This will be the diagonal element of the matrix. Now we subtract the component along $v_j$ and $v_{j-1}$ (that we know from the previous step):

$$
\tilde{\mathbf{v}}_{j+1} = w_{j+1} - \alpha_j \mathbf{v}_j - \beta_{j-1}\mathbf{v}_{j-1}
$$

This vector is orthogonal to both $\mathbf{v}_j$ and $\mathbf{v}_{j-1}$ for construction, and the tridiagonal property guarantees it's orthogonal to all previous vectors. Finally, we normalize it to get the next basis vector:

$$
\beta_j = \|\tilde{\mathbf{v}}_{j+1}\|_2 \qquad \mathbf{v}_{j+1} = \frac{\tilde{\mathbf{v}}_{j+1}}{\beta_j}
$$

## The Memory Bottleneck

After $k$ iterations of this process, we end up with as before with two matrices: $\mathbf{V}_k \in \mathbb{C}^{n \times k}$ and the tridiagonal matrix $\mathbf{T}_k \in \mathbb{R}^{k \times k}$. The approximate solution is still given by:

$$
\mathbf{x}_k = \mathbf{V}_k \mathbf{y}_k = \sum_{j=1}^k (\mathbf{y}_k)_j \mathbf{v}_j
$$

where

$$
\mathbf{y}_k = f(\mathbf{T}_k) \mathbf{e}_1 \|\mathbf{b}\|_2
$$

This sum requires that all basis vectors $\mathbf{v}_j$ are stored in memory. There is a clear timing problem here: the coefficients $\mathbf{y}_k$ depends on the full $\mathbf{T}_k$, which only becomes available after $k$ iterations. Therefore, we cannot accumulate the solution $\mathbf{x}_k$ as we go. We have to wait until the end, when all basis vectors are already computed and stored.

The memory complexity is $O(nk)$, which can be prohibitive for large $n$ and moderate $k$. For example, with $n = 500,000$ and $k = 1,000$, storing $\mathbf{V}_k$ alone requires about 4 GB of memory (assuming double-precision complex numbers). This can be a serious limitation in practice, this is why there is a lot of interest in reducing the memory footprint of Krylov methods.

Let's see how we can do better in terms of memory usage.

# Two-Pass Algorithm

To avoid storing the full basis $\mathbf{V}_k$, we can divide the Lanczos process into two separate passes over the data. The idea is to first run the Lanczos iterations to compute and store only the tridiagonal matrix $\mathbf{T}_k$ (i.e., the scalars $\alpha_j$ and $\beta_j$), and then in a second pass, regenerate the basis vectors $\mathbf{v}_j$ one at a time to accumulate the final solution $\mathbf{x}_k$.

This takes the memory complexity down to $O(n)$, since at any point in time we only need to keep a few vectors in memory. However, it comes at the cost of doubling the number of matrix-vector products with $\mathbf{A}$, since we need to apply $\mathbf{A}$ once in each pass.

## First pass

We begin with $\beta_0 = 0$ and $\mathbf{v}_0 = \mathbf{0}$. As before, we initialize $\mathbf{v}_1 = \mathbf{b} / \|\mathbf{b}\|_2$. The Lanczos recurrence proceeds as usual for $k$ steps, computing the scalars $\alpha_j$ and $\beta_j$ at each iteration. However, instead of storing the basis vectors $\mathbf{v}_j$, we discard them after use. This means that at any point in time, we only need to keep in memory the two most recent basis vectors, $\mathbf{v}_j$ and $\mathbf{v}_{j-1}$, along with a work vector for the matrix-vector product.

At the end of the first pass, we have the tridiagonal matrix $\mathbf{T}_k$ defined by the stored scalars $\{\alpha_j, \beta_j\}$. We can then compute the small problem:

$$
\mathbf{y}_k = f(\mathbf{T}_k) \mathbf{e}_1 \|\mathbf{b}\|_2
$$

## Second pass

With the coefficients $\mathbf{y}_k$ computed, we proceed to the second pass. Here, we regenerate the basis vectors $\mathbf{v}_j$ one at a time using the stored scalars $\{\alpha_j, \beta_j\}$ and the Lanczos recurrence relation that we derived earlier. While regenerating each basis vector, we immediately accumulate its contribution to the solution $\mathbf{x}_k$:

$$
\mathbf{x}_k \gets \mathbf{x}_k + (\mathbf{y}_k)_j \mathbf{v}_j
$$

Then, we discard $\mathbf{v}_j$ and move on to the next one. This way, we never need to store the full basis in memory. There are two important details to note:

- The basis vectors are generated in the same order as in the first pass, since floating-point arithmetic is deterministic, this ensures that the basis is bitwise identical to the one that would have been computed in a single pass.
- The solution, however, will change slightly due to the different order of operations in floating-point arithmetic. In the standard Lanczos, the solution is built as matrix-vector product $\mathbf{V}_k \mathbf{y}_k$, while in the two-pass method, it's built as a sum of scaled vectors. This is compiles to two different BLAS calls (`gemv` vs `axpy`), which have different rounding errors. However, in practice, the difference is in the order of machine epsilon and doesn't affect convergence.

Of course. Here is the content as a Markdown snippet.

# Implementation

Let's build this in Rust. The performance of the two-pass algorithm depends entirely on controlling memory access patterns, and Rust's ownership model and zero-cost abstractions give us the tools to reason about where data lives and how it moves through the cache hierarchy.

For the linear algebra primitives, we'll use [`faer`](https://github.com/sarah-ek/faer-rs), a modern pure-Rust numerical library. Three features are particularly important for what we're about to build:

- **Stack allocation:** `MemStack` provides a pre-allocated scratch buffer that we can reuse, making the hot path of the algorithm allocation-free.
- **Matrix-free operators:** The `LinOp` trait lets us define operators by their action (`apply`) without needing to materialize a matrix. This is essential for large-scale problems.
- **SIMD-optimized kernels:** The `zip!` macro generates vectorized loops that compile down to packed SIMD instructions for core vector operations.

## Building the Core Recurrence

The foundation of our algorithm is the Lanczos three-term recurrence:

$\beta_j \mathbf{v}_{j+1} = \mathbf{A}\mathbf{v}_j - \alpha_j \mathbf{v}_j - \beta_{j-1}\mathbf{v}_{j-1}$

Let's start by translating this directly into a Rust function. The signature should look something like this:

```rust
fn lanczos_recurrence_step<T: ComplexField, O: LinOp<T>>(
    operator: &O,
    mut w: MatMut<'_, T>,
    v_curr: MatRef<'_, T>,
    v_prev: MatRef<'_, T>,
    beta_prev: T::Real,
    stack: &mut MemStack,
) -> (T::Real, Option<T::Real>)
```

This way, the function is generic over the field type `T` (`f64`, `c64`, etc.) and the operator type `O`. It operates on `faer`'s matrix views (`MatMut` and `MatRef`) to avoid unnecessary data copies. The return type `(T::Real, Option<T::Real>)` gives us the diagonal element $\alpha_j$ and, if the process hasn't broken down, the off-diagonal $\beta_j$.

Now let's implement the body by following the math step-by-step. First, we handle the operator application, which is the most computationally expensive part:

```rust
// 1. Apply operator: w = A * v_curr
operator.apply(w.rb_mut(), v_curr, Par::Seq, stack);
```

Next, we orthogonalize against the previous vector $\mathbf{v}_{j-1}$. This is a perfect use case for `faer`'s `zip!` macro, which the compiler can turn into efficient SIMD instructions.

```rust
// 2. Orthogonalize against v_{j-1}: w -= β_{j-1} * v_{j-1}
let beta_prev_scaled = T::from_real_impl(&beta_prev);
zip!(w.rb_mut(), v_prev).for_each(|unzip!(w_i, v_prev_i)| {
    *w_i = sub(w_i, &mul(&beta_prev_scaled, v_prev_i));
});
```

With the vector partially orthogonalized, we can compute the diagonal coefficient $\alpha_j$ via an inner product. Since $\mathbf{A}$ is Hermitian, this value is guaranteed to be real.

```rust
// 3. Compute α_j = v_j^H * w
let alpha = T::real_part_impl(&(v_curr.adjoint() * w.rb())[(0, 0)]);
```

Then we complete the orthogonalization against the current vector $\mathbf{v}_j$.

```rust
// 4. Orthogonalize against v_j: w -= α_j * v_j
let alpha_scaled = T::from_real_impl(&alpha);
zip!(w.rb_mut(), v_curr).for_each(|unzip!(w_i, v_curr_i)| {
    *w_i = sub(w_i, &mul(&alpha_scaled, v_curr_i));
});
```

The vector `w` now holds the unnormalized next Lanczos vector. We can compute its norm to get our off-diagonal coefficient, $\beta_j$. If this norm is numerically zero, it signals that the Krylov subspace is invariant, and the iteration must stop. This is known as breakdown.

```rust
// 5. Compute β_j = ||w||_2 and check for breakdown
let beta = w.rb().norm_l2();
let tolerance = T::Real::from_f64_impl(f64::EPSILON * 1000.0);

if beta <= tolerance {
    (alpha, None)
} else {
    (alpha, Some(beta))
}
```

## An Iterator for the Recurrence

The recurrence step is a pure function. If we just call it in a simple loop, it would be correct but inefficient. We need a stateful object to manage the vectors between steps, which is a classic use case for the iterator pattern. Let's create a struct, `LanczosIteration`, to encapsulate the state:

```rust
struct LanczosIteration<'a, T: ComplexField, O: LinOp<T>> {
    operator: &'a O,
    v_prev: Mat<T>,       // v_{j-1}
    v_curr: Mat<T>,       // v_j
    work: Mat<T>,         // Workspace for Av_j
    beta_prev: T::Real,   // β_{j-1}
    // ... iteration counters
}
```

A key insight here is that the vectors are _owned_ (`Mat<T>`), not borrowed. This allows for a critical optimization inside the `next_step` method. After computing the next vector and normalizing it into the `work` buffer, we can cycle the state for the next iteration without any memory copies.

```rust
// Inside next_step, after normalization...
// Cycle vectors: v_prev <- v_curr <- work
core::mem::swap(&mut self.v_prev, &mut self.v_curr);
core::mem::swap(&mut self.v_curr, &mut self.work);
```

These `mem::swap` calls are zero-cost. On x86-64, swapping two `Mat<T>` structures (which are fat pointers) compiles to just three `mov` instructions. No data is moved. The logical flow is: `v_prev` gets `v_curr`'s data, `v_curr` gets the new vector from `work`, and `work` gets the old `v_prev`'s allocation to be reused.

This keeps a fixed working set of three n-dimensional vectors in play. The same memory gets reused, which is great for cache locality.

## First Pass: Collecting the Scalars

Now let's build the first pass. Its job is to run the iteration and compute the coefficients $\{\alpha_j, \beta_j\}$ while discarding the basis vectors. We can implement this pass to directly consume our `LanczosIteration`. The function signature will look like this:

```rust
pub fn lanczos_pass_one<T: ComplexField>(
    operator: &impl LinOp<T>,
    b: MatRef<'_, T>,
    k: usize,
    stack: &mut MemStack,
) -> Result<LanczosDecomposition<T::Real>, LanczosError> {
    // ...
}
```

We start by initializing storage for the scalars, using a capacity hint to prevent reallocations.

```rust
let mut alphas = Vec::with_capacity(k);
let mut betas = Vec::with_capacity(k - 1);
```

Then, we create the iterator, which allocates its three internal work vectors. After this point, the hot path is allocation-free. The main loop just drives the iterator and collects the results.

```rust
let mut lanczos_iter = LanczosIteration::new(operator, b, k, b_norm)?;

for i in 0..k {
    if let Some(step) = lanczos_iter.next_step(stack) {
        alphas.push(step.alpha);

        if step.beta <= tolerance { break; }

        if i < k - 1 {
            betas.push(step.beta);
        }
    } else {
        break;
    }
}
```

At the end, we move the scalar vectors into a `LanczosDecomposition` struct. The memory high-water mark for this whole pass is constant: three n-dimensional vectors plus the two small, growing scalar arrays. This gives us the desired $O(n)$ memory complexity.

## Second Pass: Reconstructing the Solution

Now for the second pass. We need to take the computed coefficients and reconstruct the solution $\mathbf{x}_k = \sum_{j=1}^k (\mathbf{y}_k)_j \mathbf{v}_j$ without storing the full basis. The core of this pass is a slightly different recurrence step. Since we already have the coefficients, we don't need to compute inner products or norms.

```rust
fn lanczos_reconstruction_step<T: ComplexField, O: LinOp<T>>(
    // ... same arguments as before, plus alpha_j and beta_prev
) {
    // Apply operator
    operator.apply(w.rb_mut(), v_curr, Par::Seq, stack);

    // Orthogonalize using the pre-computed coefficients
    // ... (same zip! calls as before, no inner products)
}
```

This reconstruction is slightly cheaper than the generation step. The main function `lanczos_pass_two` orchestrates the process. Here, we initialize the same three work vectors (`v_prev`, `v_curr`, `work`) and an accumulator for the final solution, `x_k`.

We can start building the solution with the first term of the sum:

```rust
// v_curr is initialized to v_1
let mut x_k = &v_curr * Scale(T::copy_impl(&y_k[(0, 0)]));
```

Our main loop then regenerates each subsequent basis vector, normalizes it using the stored $\beta_j$, and immediately accumulates its contribution to the solution.

```rust
// Inside the main loop...

// 1. Regenerate the unnormalized next vector into `work`
lanczos_reconstruction_step(...);

// 2. Normalize it using the stored β_j
let inv_beta = T::from_real_impl(&T::Real::recip_impl(&beta_j));
zip!(work.as_mut()).for_each(|unzip!(w_i)| {
    *w_i = mul(w_i, &inv_beta);
});

// 3. Accumulate: x_k += y_{j+1} * v_{j+1}
let coeff = T::copy_impl(&y_k[(j + 1, 0)]);
zip!(x_k.as_mut(), work.as_ref()).for_each(|unzip!(x_i, v_i)| {
    *x_i = add(x_i, &mul(&coeff, v_i));
});

// 4. Cycle the vectors for the next iteration
core::mem::swap(&mut v_prev, &mut v_curr);
core::mem::swap(&mut v_curr, &mut work);
```

The accumulation step `x_k += ...` uses another `zip!` loop that compiles to a fused multiply-add on hardware that supports it. We cycle the same three vectors (`v_prev`, `v_curr`, `work`), keeping the working set small and hot in the cache. This is a stark contrast to the standard method's final step, which involves a large matrix-vector product that streams through the entire $n \times k$ basis, likely from main memory.

## A Clean Public API

Finally, let's wrap these low-level components in a high-level solver function to present a clean interface to the user.

```rust
pub fn lanczos_two_pass<T, O, F>(
    operator: &O,
    b: MatRef<'_, T>,
    k: usize,
    stack: &mut MemStack,
    mut f_tk_solver: F,
) -> Result<Mat<T>, LanczosError>
where
    // ... generic bounds
    F: FnMut(&[T::Real], &[T::Real]) -> Result<Mat<T>, anyhow::Error>,
{
    // First pass: compute T_k coefficients
    let decomposition = lanczos_pass_one(operator, b, k, stack)?;

    if decomposition.steps_taken == 0 {
        return Ok(Mat::zeros(b.nrows(), 1));
    }

    // Solve projected problem: y_k' = f(T_k) * e_1
    let y_k_prime = f_tk_solver(&decomposition.alphas, &decomposition.betas)?;

    // Scale by ||b||
    let y_k = &y_k_prime * Scale(T::from_real_impl(&decomposition.b_norm));

    // Second pass: reconstruct solution
    lanczos_pass_two(operator, b, &decomposition, y_k.as_ref(), stack)
}
```

The key piece of this API is the `f_tk_solver` closure. We designed it this way to decouple the Lanczos iteration from the specific matrix function $f$. We require the user to provide a solver for the small $k \times k$ projected problem, and our main function orchestrates the two passes around it. This approach makes our implementation flexible enough to handle linear solves, matrix exponentials, or any other function without changing the core algorithm.

### Example: The Solver for a Linear System

Let's see what this solver closure looks like in a real-world scenario. If we want to solve a linear system $\mathbf{Ax} = \mathbf{b}$, we are computing the action of the inverse function, $f(z) = z^{-1}$. This means our `f_tk_solver` needs to solve the small, tridiagonal linear system $\mathbf{T}_k \mathbf{y}' = \mathbf{e}_1$.

Since $\mathbf{T}_k$ is tridiagonal, we can solve this system very efficiently. A sparse LU decomposition is an excellent choice, as its complexity for a tridiagonal system is only $O(k)$. Here's how we can implement the solver:

```rust
let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
    let steps = alphas.len();
    if steps == 0 {
        return Ok(Mat::zeros(0, 1));
    }

    // 1. Assemble the sparse tridiagonal matrix T_k from coefficients.
    // We use a triplet format for efficient construction.
    let mut triplets = Vec::with_capacity(3 * steps - 2);
    for (i, &alpha) in alphas.iter().enumerate() {
        triplets.push(Triplet { row: i, col: i, val: alpha });
    }
    for (i, &beta) in betas.iter().enumerate() {
        triplets.push(Triplet { row: i, col: i + 1, val: beta });
        triplets.push(Triplet { row: i + 1, col: i, val: beta });
    }
    let t_k_sparse = SparseColMat::try_new_from_triplets(steps, steps, &triplets)?;

    // 2. Create the right-hand side vector, e_1.
    let mut e1 = Mat::zeros(steps, 1);
    e1.as_mut()[(0, 0)] = 1.0;

    // 3. Solve the system using a sparse LU factorization.
    Ok(t_k_sparse.as_ref().sp_lu()?.solve(e1.as_ref()))
};
```

This closure takes the raw `alphas` and `betas` from the Lanczos process, constructs the `faer` sparse matrix, and then uses its built-in LU solver. This is the exact approach we used in our experimental binaries.

# Some interesting results

Now that we have a working implementation we can run some tests. The core idea of what we have done is simple: trade flops for better memory access. But does this trade actually pay off on real hardware? To find out, we need a reliable way to benchmark it.

For the data, we know that the performance of any Krylov method is tied to the operator's spectral properties. We need a way to generate a family of test problems where we can precisely control the size, sparsity, and numerical difficulty. A great way to do this is with Karush-Kuhn-Tucker (KKT) systems, which are sparse, symmetric, and have a specific block structure.

$$
A =
\begin{pmatrix}
    D & E^T \\
    E & 0
\end{pmatrix}
$$

This structure gives us two critical knobs to turn. First, with the [netgen](https://commalab.di.unipi.it/files/Data/MCF/netgen.tgz) utility, we can control the E matrix, which lets us dial in the problem dimension, $n$. Second, we build the diagonal block D with random entries from a range $[1, C_D]$. This parameter, $C_D$, gives us direct control over the numerical difficulty of the problem.Let's look at how. For a symmetric matrix like $D$, the 2-norm condition number, $\kappa_2(D)$, is the ratio of its largest to its smallest eigenvalue: $\kappa_2(D) = \lambda_{\max}(D) / \lambda_{\min}(D)$. Since $D$ is diagonal, its eigenvalues are simply its diagonal entries. We are drawing these entries from a uniform distribution $U[1, C_D]$, so we have $\lambda_{\max}(D) \approx C_D$ and $\lambda_{\min}(D) \approx 1$. This means we get direct control: $\kappa_2(D) \approx C_D$.The spectral properties of this block heavily influence the spectrum of the entire matrix $A$. A large condition number in $D$ leads to a more ill-conditioned system for $A$. The convergence rate of Krylov methods like Lanczos is fundamentally governed by the distribution of the operator's eigenvalues. An ill-conditioned matrix, with a wide spread of eigenvalues, will require more iterations, $k$, to reach the desired accuracy. By simply adjusting the $C_D$ parameter, we can generate everything from well-conditioned problems that converge quickly to ill-conditioned ones that force us to run a large number of iterations. This is exactly what we need to rigorously test our implementation.

## The Memory and Computation Trade-off

Let's put this to the test. We'll start by fixing the problem size to a large $n=500,000$ and then vary the number of iterations, $k$. We have two clear hypotheses based on a standard complexity analysis.

First, regarding memory, the one-pass method has a complexity of $O(nk)$ due to storing the basis $\mathbf{V}_k$. We expect its memory usage to grow linearly with $k$. Our two-pass method, with its $O(n)$ complexity, should have a flat memory footprint.

Second, for wall-clock time, the analysis seems just as simple. The two-pass method performs twice the number of matrix-vector products, which is the most expensive operation. Therefore, we should expect it to be about twice as slow.

Let's see what the data says. First, memory usage.

![Memory vs Iterations](/public/assets/lanczos/tradeoff_arcs500k_rho3_memory.png)

The results confirm our memory model perfectly. The standard algorithm's memory footprint grows in a perfectly straight line, a direct consequence of storing the basis $\mathbf{V}_k$. Our two-pass method is a flat line, using only the constant $O(n)$ memory for its small working set of vectors. No surprises here.

Now for the wall-clock time. This is where the simple model starts to fall apart.

![Runtime vs Iterations](/public/assets/lanczos/tradeoff_arcs500k_rho3_time.png)

The two-pass method is always slower, as expected, but the difference is nowhere near a factor of two. For a small number of iterations, the gap is very narrow, and as we increase $k$, the two-pass method's runtime starts becoming _slightly_ slower than the one-pass method.

So what's going on? This behavior is governed by the memory access patterns during the solution reconstruction phase. We can model the total execution time as the sum for matrix-vector products and vector operations. For the standard method, the final solution is computed via a single dense matrix-vector product, $\mathbf{x}_k = \mathbf{V}_k \mathbf{y}_k$. When $n$ and $k$ are large, the basis matrix $\mathbf{V}_k$ becomes too large to fit in any level of the CPU cache. Consequently, this operation becomes _memory-bandwidth-bound_: its performance is limited by the speed at which data can be streamed from main memory, leading to high latency. In contrast, the two-pass method reconstructs the solution incrementally, operating on a small working set of vectors at each step ($\mathbf{v}_{\text{prev}}$, $\mathbf{v}_{\text{curr}}$, and $\mathbf{x}_k$).

This allows the processor's cache hierarchy to manage the data effectively and maintain a high cache-hit rate. This high data locality means the processor is constantly fed data. The cost of re-computing the basis vectors turns out to be less than the cost of the memory latency paid by the standard method.

This effect isn't just for large-scale problems. On a medium-scale instance ($n=50,000$), we find an equilibrium point where the performance is nearly identical. Here, the penalty the standard method pays for increasing cache misses almost perfectly balances the cost of the extra matrix-vector products in our two-pass implementation.

![Medium Scale Runtime vs Iterations](/public/assets/lanczos/tradeoff_arcs50k_rho3_time.png)
![Medium Scale Memory Usage vs Iterations](/public/assets/lanczos/tradeoff_arcs50k_rho3_memory.png)

### The Final Proof: A Dense Matrix

To confirm our hypothesis that this is all about memory access, we can run one final test. What happens if we use a dense matrix? Here, the matrix-vector product is an $O(n^2)$ operation, making it so computationally expensive that it should dominate all other costs, including memory latency. In this compute-bound regime, our cache efficiency advantage should become negligible.

We can run an experiment with a dense random matrix of size $n=10,000$. Let's see the results.

![Dense Matrix Runtime vs Iterations](/public/assets/lanczos/dense-tradeoff.png)

The runtime of the two-pass method is almost exactly twice that of the standard one. The slopes of the lines confirm it. This proves our analysis: the surprisingly strong performance of our two-pass method on sparse problems is a direct result of its superior cache efficiency. It's a textbook example of how modern hardware architecture can turn a seemingly worse algorithm into a winner.

## Scalbility

Finally, let's look at how our implementation scales with increasing problem size. We'll fix the number of iterations to $k=500$ and vary $n$ from $50,000$ to $500,000$.

> Using netget allows us to generate a variety of problem sizes while keeping the sparsity pattern realistic.

Based on what we've seen so far, we expect the two-pass method memory usage to scale linearly with $n$, but with a very small slope. The standard method should also scale linearly, but with a much steeper slope due to the $O(nk)$ memory for storing the basis. We can see that this is exactly what happens:

![Scalability Memory Usage](/public/assets/lanczos/scalability_k500_rho3_memory.png)

> I had to use a logarithmic scale on the y-axis in order to make visible the very small memory usage increase of the two-pass method.

The runtime scaling also scales linearly with $n$ for both methods, as expected. For dimensions that are smaller then $n=150,000$, the two algorithms have very similar performance. As we increase $n$ further, the cost of the matrix-vector products starts to dominate, and the two-pass method's runtime becomes to diverge, becoming progressively slower then the standard method.

![Scalability Runtime](/public/assets/lanczos/scalability_k500_rho3_time.png)
