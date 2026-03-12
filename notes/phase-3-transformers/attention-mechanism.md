# The Attention Mechanism

> Based on: "Attention Is All You Need" — Vaswani et al., 2017
> arXiv: https://arxiv.org/abs/1706.03762

---

## The Problem Attention Solves

Before attention, sequence models (RNNs/LSTMs) processed tokens one at a time and compressed the entire history into a fixed-size hidden state vector. This created two problems:

1. **Bottleneck**: all information about the entire sequence had to pass through one vector
2. **Long-range dependencies**: information from early in the sequence would fade away by the time the model processed later tokens

Attention solves this by allowing every token to **directly look at every other token** in the sequence, regardless of distance.

---

## The Core Intuition: Soft Search

Imagine you're translating a sentence. When generating the word "bank" in French, you want to know: does the English sentence say "river bank" or "money bank"? You look back at the English words and focus on the relevant context.

Attention formalizes this as a **soft, differentiable lookup**:

- **Query (Q):** what I'm looking for ("I am currently generating word X — what context is relevant?")
- **Keys (K):** what each input token offers ("I am token Y, and this is my identity")
- **Values (V):** the actual content to retrieve ("if you look at me, here's what I contribute")

The attention score between a query and a key determines how much of that key's value to retrieve.

---

## Scaled Dot-Product Attention

Given:
- Queries Q ∈ ℝⁿˣᵈₖ (n tokens, each a d_k-dim vector)
- Keys K ∈ ℝᵐˣᵈₖ (m tokens to attend over)
- Values V ∈ ℝᵐˣᵈᵥ

The attention output is:
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

Step by step:

**Step 1: Compute similarity scores**
```
scores = QKᵀ  ∈ ℝⁿˣᵐ
```
Each entry `scores[i][j]` is the dot product of query i with key j.
Higher score = query i and key j are more similar = token i should attend more to token j.

**Step 2: Scale by √d_k**
```
scores = scores / √d_k
```
Why? In high dimensions (large d_k), dot products grow large in magnitude. Large values push the softmax into saturation regions where gradients are near zero. Scaling by √d_k counteracts this.

Derivation: if Q and K have zero-mean unit-variance entries, then QKᵀ has variance d_k. Dividing by √d_k gives variance 1.

**Step 3: Apply softmax (row-wise)**
```
weights = softmax(scores)  ∈ ℝⁿˣᵐ
```
Each row sums to 1. These are the attention weights — how much each query should attend to each key.

**Step 4: Weighted sum of Values**
```
output = weights · V  ∈ ℝⁿˣᵈᵥ
```
For each query token, compute a weighted combination of the value vectors.

---

## Why Softmax?

Softmax converts raw scores into a probability distribution:
```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

Properties:
- All outputs are positive and sum to 1 (valid probability distribution)
- Differentiable — gradients can flow back through it
- Amplifies the largest score (relatively) — focuses attention

---

## Multi-Head Attention

Instead of doing attention once, we do it h times in parallel with different linear projections:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ

where headᵢ = Attention(Q·Wᵢᴼ, K·Wᵢᴷ, V·Wᵢᵛ)
```

**Why multiple heads?**

Each head learns to attend to different types of relationships:
- One head might track syntactic dependencies (subject → verb)
- Another might track semantic relationships (pronoun → antecedent)
- Another might track positional patterns

With d_model = 512 and h = 8 heads, each head operates in 64-dimensional space (d_k = d_model / h = 64). We project down for each head, compute attention, then concatenate and project back up.

---

## Causal (Masked) Self-Attention

In a decoder (like GPT), the model generates tokens left to right. Token at position t should only attend to positions ≤ t (not future tokens it hasn't generated yet).

This is enforced by adding a mask before softmax:
```
masked_scores[i][j] = scores[i][j]   if j ≤ i
                     = -∞             if j > i
```

After softmax, -∞ becomes 0. Future tokens contribute nothing to the attention output.

This is called **causal** or **autoregressive** attention.

---

## Self-Attention vs. Cross-Attention

**Self-attention:** Q, K, V all come from the same sequence
- Used in encoder layers (each token attends to all other tokens in the same input)
- Used in decoder layers for the first attention sublayer (causal, attends to prior outputs)

**Cross-attention:** Q comes from one sequence, K and V from another
- Used in encoder-decoder models (decoder attends to encoder output)
- Q = decoder hidden state, K = V = encoder output

---

## Computational Complexity

For a sequence of length n:
- Attention is O(n²·d) — quadratic in sequence length
- This is the key limitation — long sequences are expensive
- Modern variants (Flash Attention, linear attention) address this

---

## What the Paper Found

From "Attention Is All You Need":
- Transformers trained **much faster** than RNNs due to parallelism (no sequential dependency between positions during training)
- Achieved SOTA on translation with less compute than previous models
- Showed attention heads learn interpretable linguistic patterns

---

## The Key Equations to Memorize

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ
headᵢ = Attention(QWᵢᴼ, KWᵢᴷ, VWᵢᵛ)
```

These two equations are the heart of the transformer.
