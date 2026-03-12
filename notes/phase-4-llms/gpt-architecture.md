# GPT Architecture & Language Modeling

> Based on:
> - "Improving Language Understanding by Generative Pre-Training" (GPT-1) — Radford et al., 2018
> - "Language Models are Few-Shot Learners" (GPT-3) — Brown et al., 2020

---

## What is a Language Model?

A language model assigns a probability to a sequence of tokens:
```
P(x₁, x₂, ..., xₙ)
```

By the chain rule of probability, this factorizes as:
```
P(x₁, x₂, ..., xₙ) = Π P(xₜ | x₁, ..., xₜ₋₁)
```

The model predicts each token conditioned on all previous tokens. This is **autoregressive** language modeling.

Training objective: maximize the log-likelihood of the training data:
```
L = Σₜ log P(xₜ | x₁, ..., xₜ₋₁; θ)
```

This is equivalent to minimizing cross-entropy loss (or NLL) over the training corpus.

---

## Tokenization

Before the model sees text, it must be converted to integers.

**Byte-Pair Encoding (BPE):**
- Start with character-level vocabulary
- Iteratively merge the most frequent pairs of tokens into a new token
- Result: a vocabulary of subword units (words, word pieces, characters)

GPT-2 vocabulary size: 50,257 tokens
GPT-3 vocabulary size: 50,257 tokens (same tokenizer)

Why subwords? They handle:
- Rare and out-of-vocabulary words (split into known pieces)
- Multiple languages
- Code and special characters

---

## GPT Architecture (Decoder-Only Transformer)

GPT uses a stack of **transformer decoder blocks**, but without the cross-attention sublayer (since there's no encoder). It's a pure language model.

```
Input Text
    ↓
Tokenizer (text → token IDs)
    ↓
Token Embedding (vocab_size → d_model)
    +
Positional Embedding (position → d_model)
    ↓
[Transformer Block] × N layers
    ↓
Layer Norm
    ↓
Linear (d_model → vocab_size)    ← same weights as token embedding (weight tying)
    ↓
Softmax → Probability distribution over vocabulary
```

**Transformer Block (GPT-style):**
```
x → LayerNorm → Causal Self-Attention → + residual
  → LayerNorm → Feed-Forward MLP    → + residual
```

Note: GPT uses **pre-norm** (LayerNorm before the sublayer), while the original Transformer paper used post-norm. Pre-norm trains more stably.

---

## Key Hyperparameters

| Model | Layers | d_model | Heads | d_ff | Params |
|-------|--------|---------|-------|------|--------|
| GPT-1 | 12 | 768 | 12 | 3072 | 117M |
| GPT-2 small | 12 | 768 | 12 | 3072 | 124M |
| GPT-2 medium | 24 | 1024 | 16 | 4096 | 355M |
| GPT-2 large | 36 | 1280 | 20 | 5120 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 6400 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 49152 | 175B |

d_ff (feed-forward dimension) is typically 4× d_model.
Each head has dimension d_k = d_model / num_heads.

---

## The Feed-Forward Sublayer

After attention, each position passes through an MLP applied independently:
```
FFN(x) = activation(xW₁ + b₁)W₂ + b₂
```

- W₁ ∈ ℝᵈᵐᵒᵈᵉˡˣᵈᶠᶠ (expands to d_ff = 4 × d_model)
- W₂ ∈ ℝᵈᶠᶠˣᵈᵐᵒᵈᵉˡ (projects back down)
- Activation: ReLU in GPT-1/2, GELU in GPT-3

The FFN is thought to store "factual knowledge" about the world (key-value memory interpretation by Geva et al., 2020).

---

## Positional Encodings

Attention is permutation-invariant — it doesn't know the order of tokens. We must explicitly inject position information.

**GPT-1 / GPT-2: Learned positional embeddings**
- A learnable embedding matrix P ∈ ℝˢᵉᵠˡᵉⁿˣᵈᵐᵒᵈᵉˡ
- Position i gets embedding P[i]
- Added to the token embedding: `x = token_embed + position_embed`

**Original Transformer: Sinusoidal (fixed) positional encoding**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Modern models (LLaMA, GPT-4) use **RoPE** (Rotary Position Embedding) — applied inside the attention computation rather than added to inputs.

---

## Weight Tying

The output linear layer (d_model → vocab_size) shares weights with the token embedding matrix (vocab_size → d_model).

Why?
- Reduces parameters significantly (especially for large vocabularies)
- Embeds a useful inductive bias: tokens that are similar in the embedding space should receive similar output probabilities

---

## GPT-3: Scale Changes Everything

GPT-3's key finding was that **scale unlocks emergent capabilities**:

**Few-shot learning without fine-tuning:** by providing a few examples in the prompt, GPT-3 could perform tasks it was never explicitly trained on.

From the paper:
- Zero-shot: no examples, just a task description
- One-shot: one example in the prompt
- Few-shot: K examples (GPT-3 used 10-100 depending on task)

No gradient update. The model learned from examples purely from context.

---

## Scaling Laws (Kaplan et al., 2020)

Model performance (loss) follows power laws in:
- Number of parameters (N)
- Dataset size (D)
- Compute budget (C)

```
L ∝ N^(-αN)
L ∝ D^(-αD)
L ∝ C^(-αC)
```

Key implication: given a fixed compute budget, there is an **optimal model size and data size** tradeoff. Training a smaller model on more data can outperform a larger model on less data (Chinchilla scaling laws, 2022).

---

## Inference: Autoregressive Generation

At inference time, the model generates one token at a time:

```python
# Pseudocode
input_ids = tokenize(prompt)
for _ in range(max_new_tokens):
    logits = model(input_ids)          # forward pass
    next_token_logits = logits[-1]     # logits for the last position
    probs = softmax(next_token_logits / temperature)
    next_token = sample(probs)         # or argmax for greedy
    input_ids = append(input_ids, next_token)
    if next_token == EOS: break
```

**Temperature:** controls randomness
- T → 0: greedy (always pick highest probability token)
- T = 1: sample from the model's distribution
- T > 1: more random, more creative

**KV Cache:** during generation, key and value matrices are cached from prior positions so they don't need to be recomputed each step.
