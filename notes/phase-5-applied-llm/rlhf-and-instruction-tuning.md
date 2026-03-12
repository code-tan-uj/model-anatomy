# RLHF & Instruction Tuning

> Based on: "Training language models to follow instructions with human feedback" — Ouyang et al., 2022 (InstructGPT)
> arXiv: https://arxiv.org/abs/2203.02155

---

## The Alignment Problem

A pre-trained language model (e.g., GPT-3) is trained to predict the next token on internet text. This means it learned to:
- Imitate all kinds of text — including harmful, biased, or misleading content
- Continue text, not necessarily *answer* questions helpfully
- Optimize for likelihood, not for being *helpful, harmless, and honest*

The model is not "aligned" with what we actually want from it as an assistant.

**Example:**
- Prompt: "How do I make bread?"
- GPT-3 might generate: "How do I make bread? Here are 10 questions about bread-making:" (continuing internet-style text)
- InstructGPT would generate: an actual helpful recipe

---

## The Three-Phase RLHF Pipeline

### Phase 1: Supervised Fine-Tuning (SFT)

Collect human-written demonstrations:
```
(prompt, ideal response) pairs — written by human contractors
```

Fine-tune the pre-trained model (GPT-3) on these pairs using standard cross-entropy loss.

This gives a model that:
- Follows instruction format
- Produces helpful responses
- ...but may still have subtle misalignments

### Phase 2: Reward Model Training

Collect human preference data:
```
For prompt p, show raters responses (rₐ, r_b) and ask which is better
```

Train a **reward model** RM(prompt, response) → scalar score to predict human preferences.

The reward model is trained with a ranking loss:
```
loss = -log(σ(RM(prompt, rₐ) - RM(prompt, r_b)))
```

Where rₐ is the preferred response. This pushes the RM to score preferred responses higher.

The reward model starts from the SFT model with the final layer replaced by a scalar head.

### Phase 3: Reinforcement Learning (PPO)

Use the reward model as a reward signal to further fine-tune the SFT model using **PPO** (Proximal Policy Optimization):

```
Maximize:  E[RM(prompt, response)]
Subject to: KL(π_RL || π_SFT) ≤ δ     (stay close to SFT model)
```

The full objective (per token):

```
reward(xₜ) = RM(prompt, response) - β · log(π_RL(xₜ|x<t) / π_SFT(xₜ|x<t))
```

Where:
- `RM(prompt, response)` — the reward from the reward model (applied at the end of the sequence)
- `β · KL(...)` — penalty for diverging too much from the SFT policy

**Why the KL penalty?**
Without it, the RL model would learn to "hack" the reward model — generate gibberish that the reward model happens to rate highly (reward hacking / mode collapse). The KL penalty keeps the model generating coherent text.

---

## Key Terms

**Policy (π):** in RL, the policy is the model itself — a mapping from state (the prompt so far) to action (the next token distribution)

**PPO (Proximal Policy Optimization):** a policy gradient algorithm that updates the policy by taking the largest possible step without going too far from the previous policy

**KL Divergence (as a penalty):** measures how different two distributions are. By penalizing KL, we prevent the RL policy from drifting too far from the SFT policy.

**Reward Hacking:** when an RL agent finds ways to maximize the reward signal that don't correspond to actually solving the intended task

---

## Results from the Paper

- InstructGPT (1.3B parameters + RLHF) was preferred over GPT-3 (175B) by human raters 71% of the time
- RLHF significantly reduced harmful and untruthful outputs
- The model showed better ability to follow explicit instructions

Key finding: **scale alone doesn't solve alignment**. A much smaller model trained with RLHF beats a much larger model without it.

---

## Modern Variations

**DPO (Direct Preference Optimization)** — 2023
- Eliminates the separate reward model
- Directly fine-tunes the LLM on preference pairs using a closed-form loss
- Simpler, more stable training
- Loss: `L = -E[log σ(β · log(π(rw|x)/πref(rw|x)) - β · log(π(rl|x)/πref(rl|x)))]`
- Where rw = preferred response, rl = rejected response

**Constitutional AI (CAI)** — Anthropic
- The model critiques its own responses according to a "constitution" of principles
- Uses AI feedback (RLAIF) instead of human feedback at scale
- Basis for how Claude is trained

---

## Instruction Tuning Without RL

**FLAN (Fine-tuned Language Net)** — Google, 2022
- Fine-tune on a large collection of NLP tasks formatted as instructions
- Shows that instruction tuning generalizes to unseen tasks

**Alpaca** — Stanford, 2023
- Generates instruction-following data using GPT-3.5 (the "self-instruct" approach)
- Fine-tunes LLaMA-7B on 52K synthetic instruction pairs
- Produced a GPT-3.5-comparable model at very low cost

Key insight: you don't always need human preference data — high-quality instruction datasets can be synthetically generated using stronger models.

---

## Summary: Pre-training → SFT → RLHF

```
Pre-trained LLM
(predicts text, not aligned)
        ↓
Supervised Fine-Tuning (SFT)
(learns instruction-following format from human demos)
        ↓
Reward Model Training
(learns to score responses by human preferences)
        ↓
PPO with KL Penalty
(maximizes reward while staying close to SFT model)
        ↓
InstructGPT / ChatGPT / Claude-style assistant
```
