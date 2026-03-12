# Probability & Information Theory for AI Engineering

---

## Probability Basics

**Sample space (Ω):** the set of all possible outcomes
**Event:** a subset of the sample space
**Probability P(A):** a number in [0, 1] assigned to each event

**Key rules:**
```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)   # union
P(A ∩ B) = P(A) · P(B|A)              # chain rule
P(A|B) = P(A ∩ B) / P(B)              # conditional probability
```

---

## Probability Distributions

A distribution assigns probabilities to outcomes.

**Discrete (Categorical):** used for token predictions
```
P(X = k) = pₖ,  where Σₖ pₖ = 1
```

**Continuous (Gaussian):** used in latent spaces, noise modeling
```
p(x) = (1/√(2πσ²)) · exp(-(x-μ)²/2σ²)
```

Parameters: μ (mean), σ² (variance)

**Why it matters:** the output of an LLM is a **categorical distribution** over the vocabulary. The model outputs logits, we apply softmax to get a valid probability distribution, then sample from it.

---

## Maximum Likelihood Estimation (MLE)

Given observed data X = {x₁, x₂, ..., xₙ}, MLE finds parameters θ that maximize:
```
L(θ) = P(X | θ) = Π P(xᵢ | θ)    # likelihood
```

In practice, we maximize the **log-likelihood** (product → sum, easier to optimize):
```
log L(θ) = Σ log P(xᵢ | θ)
```

Equivalently, we **minimize the negative log-likelihood**:
```
NLL = -Σ log P(xᵢ | θ)
```

**This is exactly what we do when training a language model.** The training objective is:
```
minimize  -Σₜ log P(xₜ | x₁, ..., xₜ₋₁; θ)
```

---

## Cross-Entropy Loss

Given:
- True distribution: p (the one-hot target, e.g., the actual next token)
- Predicted distribution: q (the model's softmax output)

Cross-entropy:
```
H(p, q) = -Σₓ p(x) · log q(x)
```

For one-hot p (classification), this simplifies to:
```
H(p, q) = -log q(correct class)
```

**Cross-entropy loss IS negative log-likelihood.** They are the same thing expressed differently. Training an LLM with cross-entropy loss is performing MLE.

---

## KL Divergence

KL Divergence measures how much distribution q differs from distribution p:
```
KL(p || q) = Σₓ p(x) · log(p(x) / q(x))
```

Properties:
- KL(p || q) ≥ 0 always
- KL(p || q) = 0 iff p = q
- Not symmetric: KL(p || q) ≠ KL(q || p)

Relationship to cross-entropy:
```
H(p, q) = H(p) + KL(p || q)
```

Minimizing cross-entropy = minimizing KL divergence from true distribution.

**Why it matters in AI:**
- **RLHF**: KL penalty between the RL policy and the SFT model prevents the model from exploiting the reward model
- **VAEs**: the loss has a KL term regularizing the latent space
- **Knowledge distillation**: KL loss between teacher and student distributions

---

## Entropy

Shannon entropy measures the uncertainty/information content of a distribution:
```
H(p) = -Σₓ p(x) · log₂ p(x)
```

- High entropy = high uncertainty = spread-out distribution
- Low entropy = low uncertainty = peaked distribution

**Perplexity** (used to evaluate language models):
```
Perplexity = 2^H(p) = exp(H(p)) [in nats]
```

Or equivalently, for a model with cross-entropy loss L:
```
Perplexity = exp(L)
```

Interpretation: a perplexity of K means the model is as confused as if it had to choose uniformly between K words. Lower is better.

---

## Bayes' Theorem

```
P(θ | X) = P(X | θ) · P(θ) / P(X)
```

- P(θ | X) — posterior: what we believe about θ after seeing data
- P(X | θ) — likelihood: how probable the data is given θ
- P(θ) — prior: what we believed before seeing data
- P(X) — marginal likelihood / evidence (normalizing constant)

**Why it matters:** Bayesian thinking pervades probabilistic ML. Even if we don't explicitly use Bayes' theorem in training, understanding it helps with:
- Understanding regularization as priors on weights
- Understanding uncertainty estimation
- Understanding how models should update beliefs given evidence

---

## Summary: The Chain from Math to LLM Training

```
Probability distribution over tokens
    ↓
Maximum Likelihood Estimation (find θ that maximizes P(data | θ))
    ↓
Minimize Negative Log-Likelihood
    ↓
Minimize Cross-Entropy Loss
    ↓
Stochastic Gradient Descent updates θ
    ↓
Trained Language Model
```
