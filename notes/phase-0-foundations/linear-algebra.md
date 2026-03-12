# Linear Algebra for AI Engineering

> Core reference: Mathematics for Machine Learning (Deisenroth et al.) — https://mml-book.github.io/

---

## Vectors

A **vector** is an ordered list of numbers. In ML, vectors represent:
- A single data point (feature vector)
- A word embedding
- A model's weights in one layer

```
x = [x₁, x₂, ..., xₙ]  ∈ ℝⁿ
```

Geometrically: an arrow in n-dimensional space, with magnitude (length) and direction.

**Magnitude (L2 norm):**
```
||x|| = sqrt(x₁² + x₂² + ... + xₙ²)
```

---

## Dot Product

The dot product of two vectors is a scalar:
```
x · y = x₁y₁ + x₂y₂ + ... + xₙyₙ = Σᵢ xᵢyᵢ
```

Geometric interpretation:
```
x · y = ||x|| · ||y|| · cos(θ)
```

Where θ is the angle between the vectors.

**Why it matters for AI:**
- When θ = 0° (same direction): dot product is maximized → vectors are similar
- When θ = 90° (orthogonal): dot product is 0 → vectors are unrelated
- When θ = 180° (opposite): dot product is negative

This is the foundation of the **attention score** in transformers: `Q · Kᵀ` measures similarity between queries and keys.

---

## Matrix Multiplication

Matrix multiplication combines two matrices to produce a third.

For A ∈ ℝᵐˣⁿ and B ∈ ℝⁿˣᵖ, the result C = AB ∈ ℝᵐˣᵖ where:
```
Cᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ
```

Each element of C is the dot product of row i of A with column j of B.

**Why it matters for AI:**
- A linear layer `y = Wx + b` is a matrix-vector multiplication
- Batched inference does `Y = XW` where X is a batch of inputs

---

## Eigenvalues & Eigenvectors

For a square matrix A, vector **v** is an eigenvector if:
```
Av = λv
```

Where λ (a scalar) is the corresponding eigenvalue.

Geometric interpretation: eigenvectors are directions that A only stretches (does not rotate). λ tells you by how much.

**Why it matters for AI:**
- PCA finds the eigenvectors of the covariance matrix
- Understanding optimization landscapes (Hessian eigenvectors → curvature)

---

## SVD (Singular Value Decomposition)

Any matrix M ∈ ℝᵐˣⁿ can be decomposed as:
```
M = UΣVᵀ
```

Where:
- U ∈ ℝᵐˣᵐ — left singular vectors (orthogonal)
- Σ ∈ ℝᵐˣⁿ — diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V ∈ ℝⁿˣⁿ — right singular vectors (orthogonal)

**Why it matters for AI:**
- **LoRA** (Low-Rank Adaptation): approximates weight update matrices as low-rank products — this is directly inspired by SVD
- **PCA**: the principal components are the left singular vectors of the centered data matrix
- **Data compression**: keep only the top-k singular values/vectors

---

## Key Takeaways

| Concept | Where it shows up in AI |
|---------|------------------------|
| Dot product | Attention scores, cosine similarity |
| Matrix multiply | Every linear layer, batched forward pass |
| Eigenvalues | PCA, understanding optimization curvature |
| SVD | LoRA, compression, PCA |
| Norms | Regularization (L1, L2), gradient clipping |
