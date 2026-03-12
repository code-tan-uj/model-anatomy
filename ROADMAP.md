# AI Engineering Roadmap — Ground Up

> From first principles to production-grade AI systems.
> Each phase builds directly on the last. Do not skip phases.

---

## How to use this roadmap

Each phase has:
- **Core concepts** — what you must understand deeply
- **Key papers** — landmark research to read and reproduce
- **Implementations** — what to build from scratch
- **Free resources** — where to learn it (no paywalls where possible)

Papers are referenced with arXiv or open-access links. Local PDFs are gitignored.

---

## Phase 0 — Mathematical Foundations

> Everything in deep learning is linear algebra, calculus, and probability. Skipping this means you will hit walls constantly. Build this bedrock first.

### Core Concepts

**Linear Algebra**
- Vectors, matrices, tensors — what they are and how to think about them geometrically
- Matrix multiplication — the core operation underlying every neural network layer
- Eigenvalues and eigenvectors — appear in PCA, attention, and optimization
- SVD (Singular Value Decomposition) — used in LoRA, compression, and data analysis
- Dot product as similarity — the geometric intuition behind attention scores

**Calculus**
- Derivatives and partial derivatives — the foundation of gradient descent
- Chain rule — this IS backpropagation. Understand it cold.
- Gradients, Jacobians, Hessians — multivariate differentiation
- Computational graphs — how frameworks like PyTorch compute gradients automatically

**Probability & Statistics**
- Probability distributions — Gaussian, Bernoulli, Categorical, etc.
- Expectation, variance, covariance
- Bayes' theorem — appears everywhere in probabilistic modeling
- Maximum Likelihood Estimation (MLE) — the theoretical basis for training loss functions
- KL Divergence — used in VAEs, RLHF, and any distributional comparison
- Cross-entropy loss — why it works, derived from MLE

**Information Theory**
- Entropy — a measure of uncertainty / information content
- Cross-entropy — measures how well a distribution approximates another
- KL Divergence — the "distance" between two distributions
- Perplexity — how we evaluate language models (derived from entropy)

### Free Resources
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (YouTube series — the best visual intuition)
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [StatQuest with Josh Starmer](https://www.youtube.com/c/joshstarmer) — probability and statistics, visually explained
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/) — Marc Peter Deisenroth et al. (openly licensed)
- CS188 AI Textbook (local) — chapters on probability and search

### Milestone
- [ ] Implement matrix multiplication from scratch in NumPy (no `np.matmul`)
- [ ] Derive the chain rule and implement scalar autograd from scratch
- [ ] Prove that cross-entropy loss is MLE under a categorical distribution

---

## Phase 1 — Classical Machine Learning

> Before neural nets, understand what we were doing before them and why they have limitations. Also learn the engineering patterns (train/val/test split, overfitting, regularization) that apply to everything.

### Core Concepts

**Supervised Learning**
- Linear regression — the simplest model; derive the closed-form solution
- Logistic regression — classification as probability estimation
- Decision trees and random forests — non-parametric models; understand variance vs. bias
- SVMs — maximum margin classifiers; kernel trick

**Unsupervised Learning**
- K-Means clustering
- PCA (Principal Component Analysis) — dimensionality reduction via SVD
- GMMs (Gaussian Mixture Models) — soft clustering with EM algorithm

**ML Engineering Fundamentals**
- The bias-variance tradeoff — the core tension in all of ML
- Overfitting, underfitting, and regularization (L1, L2, dropout)
- Cross-validation
- Evaluation metrics — accuracy, precision, recall, F1, AUC-ROC
- The ML development loop — data → model → evaluate → iterate

### Free Resources
- [fast.ai Practical ML](https://course.fast.ai/) — pragmatic, code-first approach
- [Stanford CS229 (Andrew Ng lectures)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) — rigorous mathematical treatment
- [Scikit-learn documentation](https://scikit-learn.org/stable/user_guide.html) — excellent conceptual explanations

### Milestone
- [ ] Implement linear regression with gradient descent from scratch
- [ ] Implement logistic regression from scratch
- [ ] Build a decision tree from scratch without sklearn

---

## Phase 2 — Deep Learning & Neural Networks

> The computational backbone of everything that follows. Understand how neural networks learn — not just that they do.

### Core Concepts

**The Neuron & Activation Functions**
- What a neuron computes: `output = activation(W·x + b)`
- Why we need non-linearity — without it, a deep network collapses to linear regression
- ReLU, Sigmoid, Tanh, GELU — when and why to use each
- Dead neurons, vanishing gradients — the problems that motivated modern choices

**Feedforward Networks (MLPs)**
- Layer-by-layer computation
- Universal approximation theorem — what it says and what it *doesn't* say
- Weight initialization — why it matters (Xavier, He initialization)

**Backpropagation**
- Forward pass, backward pass
- Gradient flow through the computational graph
- Implementing backprop by hand — the most important exercise in this phase

**Training Dynamics**
- Gradient descent variants: SGD, SGD+Momentum, Adam, AdamW
- Learning rate schedules: warmup, cosine decay
- Batch size and its effect on generalization
- Loss surfaces, local minima, saddle points

**Regularization in Deep Learning**
- Dropout — regularization as random circuit disruption
- Batch Normalization — accelerates training, reduces internal covariate shift
- Layer Normalization — preferred in transformers (works on sequence data)
- Weight decay — L2 regularization in the optimizer

**Convolutional Neural Networks (CNNs)**
- Convolution operation — weight sharing, translation invariance
- Pooling layers
- Classic architectures: LeNet → AlexNet → ResNet (understand residual connections)

**Recurrent Neural Networks (RNNs)**
- Sequence modeling with shared weights across time
- Vanishing gradient problem in long sequences
- LSTMs and GRUs — gating mechanisms as solutions
- Why transformers replaced RNNs for language

### Free Resources
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — the single best hands-on series
- [Deep Learning book (Goodfellow et al.) — free online](https://www.deeplearningbook.org/)
- [fast.ai Deep Learning Part 1](https://course.fast.ai/)
- [CS231n (CNNs for Visual Recognition)](http://cs231n.stanford.edu/)

### Milestone
- [ ] Implement a scalar autograd engine from scratch (like micrograd)
- [ ] Train an MLP on MNIST from scratch (no PyTorch autograd — use your own)
- [ ] Implement backprop through an LSTM by hand

---

## Phase 3 — The Transformer Revolution

> The attention mechanism is the engine of modern AI. Every LLM, vision model, and multimodal system builds on this. Understand it at the mathematical level.

### Core Concepts

**Sequence-to-Sequence Problems**
- Encoder-decoder architecture
- The bottleneck problem in early seq2seq with RNNs

**The Attention Mechanism**
- Intuition: allow every token to "look at" every other token
- Query, Key, Value (Q, K, V) — the search engine analogy
- Scaled dot-product attention: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`
- Why we scale by `sqrt(d_k)` — prevents softmax saturation in high dimensions
- Attention as soft retrieval — weighted sum of values

**Multi-Head Attention**
- Running multiple attention heads in parallel
- Each head can learn to attend to different aspects of the sequence
- Concatenating and projecting heads back to model dimension

**The Transformer Architecture (Encoder-Decoder)**
- Positional encodings — injecting sequence order information
- Residual connections — allow gradients to flow through deep stacks
- Layer normalization — stabilize training in deep networks
- Feed-forward sublayer — pointwise MLP applied after attention
- Encoder stack vs. decoder stack (masked attention)

**Variants**
- Encoder-only (BERT-style) — good for understanding/classification tasks
- Decoder-only (GPT-style) — good for generation
- Encoder-decoder (T5-style) — good for translation, summarization

### Key Paper

**"Attention Is All You Need"** (Vaswani et al., 2017)
- arXiv: https://arxiv.org/abs/1706.03762
- This paper introduced the Transformer. Read it after building up the intuition.
- Local copy: `papers/1.Attention Is All You Need.pdf`

What to extract from the paper:
- The full architecture diagram (Figure 1) — reproduce it from memory
- The exact attention formula — derive it from scratch
- Why they used sinusoidal positional encodings
- Training setup, optimizer, warmup schedule

### Free Resources
- [Andrej Karpathy — Let's build GPT (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — best hands-on transformer walkthrough
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) — visual intuition
- [Annotated Transformer — Harvard NLP](http://nlp.seas.harvard.edu/annotated-transformer/) — paper + code side-by-side

### Milestone
- [ ] Implement scaled dot-product attention from scratch in PyTorch
- [ ] Implement multi-head attention from scratch
- [ ] Build a complete Transformer encoder from scratch
- [ ] Train a small character-level language model using a decoder-only transformer

---

## Phase 4 — Large Language Models

> How do we go from "a transformer" to GPT? This phase covers pre-training paradigms, the GPT lineage, and the emergent capabilities of scale.

### Core Concepts

**Language Modeling as Next-Token Prediction**
- Autoregressive generation — predict one token at a time, conditioned on all prior tokens
- The training objective: minimize negative log-likelihood of the next token
- Tokenization — BPE, WordPiece, SentencePiece
- The vocabulary and embedding matrix

**Pre-training vs. Fine-tuning**
- Pre-training: unsupervised learning on massive text corpora
- Fine-tuning: supervised adaptation to specific tasks
- The pre-train/fine-tune paradigm changed NLP forever

**The GPT Lineage**
- GPT-1 → GPT-2 → GPT-3: same architecture, radically different scale
- Scaling laws: loss decreases predictably with compute, data, and parameters
- Emergent abilities — capabilities that appear at scale but not at small size

**In-Context Learning (Few-Shot Prompting)**
- No gradient update — the model learns from examples in the prompt
- Zero-shot, one-shot, few-shot prompting
- Why this works is still not fully understood — a key research area

**Key Architectures**
- GPT-2 architecture walkthrough (the open-source reference)
- Key differences from the original Transformer: decoder-only, pre-norm, no encoder

### Key Papers

**"Improving Language Understanding by Generative Pre-Training"** (Radford et al., 2018) — GPT-1
- OpenAI blog: https://openai.com/research/language-unsupervised
- Local copy: `papers/7.Improving Language Understanding by Generative Pre-Training.pdf`

**"Language Models are Few-Shot Learners"** (Brown et al., 2020) — GPT-3
- arXiv: https://arxiv.org/abs/2005.14165
- Local copy: `papers/2.llm-as-few-shot-learners.pdf`
- Key reading: Section 2 (model architecture), Section 3 (tasks and results)

**Key Book**

**"Build a Large Language Model (From Scratch)"** — Sebastian Raschka (2024, Manning)
- Publisher: https://www.manning.com/books/build-a-large-language-model-from-scratch
- Local copy: `Books/Sebastian Raschka - Build a Large Language Model (From Scratch).pdf`
- This is the implementation companion for this phase. Work through every chapter.

### Free Resources
- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) — the cleanest GPT implementation
- [Lilian Weng — The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Stanford CS324 — Large Language Models](https://stanford-cs324.github.io/winter2022/)

### Milestone
- [ ] Implement GPT-2 (small, 124M) from scratch using PyTorch
- [ ] Pre-train it on a small dataset (OpenWebText subset or TinyStories)
- [ ] Reproduce GPT-2's text generation behavior

---

## Phase 5 — Applied LLM Engineering

> Pre-trained LLMs are powerful but raw. This phase covers how to make them useful, steerable, and reliable in practice.

### 5A — Instruction Tuning & RLHF

**Core Concepts**
- The alignment problem — pre-trained LLMs predict text, not "helpful responses"
- Instruction tuning — supervised fine-tuning on (instruction, response) pairs
- RLHF (Reinforcement Learning from Human Feedback):
  1. Supervised Fine-Tuning (SFT) on demonstration data
  2. Reward Model training on human preference comparisons
  3. PPO optimization against the reward model
- KL penalty — prevents the model from drifting too far from the pre-trained base

**Key Paper**

**"Training language models to follow instructions with human feedback"** (Ouyang et al., 2022) — InstructGPT
- arXiv: https://arxiv.org/abs/2203.02155
- Local copy: `papers/1.1Training language models to follow instructions with human feedback.pdf`
- This paper describes how ChatGPT-style behavior is trained. Read it carefully.

### 5B — Retrieval-Augmented Generation (RAG)

**Core Concepts**
- The hallucination problem — LLMs generate plausible-sounding but false text
- RAG pipeline: Retrieve relevant documents → Augment the prompt → Generate
- Dense retrieval: embedding-based semantic search (vs. sparse BM25 keyword search)
- Chunking strategies — how to split documents for retrieval
- Vector databases — FAISS, Chroma, Pinecone, Weaviate
- Reranking — improving retrieval precision after initial retrieval

**Key Papers**

**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., 2020)
- arXiv: https://arxiv.org/abs/2005.11401
- Local copy: `papers/4.RAG.pdf`

**MasteringRAG** (local book)
- Local copy: `papers/4.1MasteringRAG.pdf`
- Covers advanced RAG patterns: HyDE, reranking, query decomposition, RAPTOR

### 5C — Fine-tuning at Scale

**Core Concepts**
- Full fine-tuning vs. parameter-efficient fine-tuning (PEFT)
- LoRA (Low-Rank Adaptation) — adapting large models with tiny adapter matrices
- QLoRA — LoRA + 4-bit quantization for fine-tuning on consumer hardware
- When to fine-tune vs. when to use RAG or prompting
- Synthetic data generation — using LLMs to generate training data for other LLMs

**Key Paper**

**"Best Practices for Synthetic Data Generation"**
- Local copy: `papers/3.best_practices_synthetic_data.pdf`

### Free Resources
- [Hugging Face PEFT library](https://huggingface.co/docs/peft)
- [LlamaIndex documentation](https://docs.llamaindex.ai/) — RAG framework
- [LangChain documentation](https://python.langchain.com/) — orchestration framework
- [Lilian Weng — Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)

### Milestone
- [ ] Build an end-to-end RAG pipeline on a document set from scratch
- [ ] Fine-tune a small model using LoRA with Hugging Face PEFT
- [ ] Implement a reward model and simulate the RLHF training loop

---

## Phase 6 — Agentic AI & Tool Use

> LLMs that can reason, plan, and take actions in the world. This is where AI engineering meets software engineering.

### Core Concepts

**What Makes an Agent**
- Perception: the agent observes state (a prompt, tool results, memory)
- Reasoning: the agent decides what to do next (chain-of-thought, ReAct)
- Action: the agent executes (tool call, code execution, API call)
- Memory: short-term (context window), long-term (retrieval), episodic

**ReAct Pattern**
- Reason + Act interleaved: Thought → Action → Observation → Thought → ...
- Enables multi-step problem solving with external tools

**Tool Use & Function Calling**
- Structured output — forcing LLMs to output JSON for tool calls
- Tool definitions — telling the model what tools exist and how to use them
- Parsing and executing tool calls in a loop

**Model Context Protocol (MCP)**
- A standardized protocol for connecting LLMs to external tools and data sources
- Tool servers, resource servers, and prompts
- Why a protocol matters for building composable agentic systems

**Multi-Agent Systems**
- Orchestrator + worker agents
- Parallel vs. sequential task execution
- Shared memory and state between agents

**Key Paper**

**"Model Context Protocol"** — Anthropic (2024)
- Official docs: https://modelcontextprotocol.io/
- Local copy: `papers/5.mcp.pdf`

### Free Resources
- [Anthropic MCP documentation](https://modelcontextprotocol.io/)
- [Lilian Weng — LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [OpenAI Swarm](https://github.com/openai/swarm) — lightweight multi-agent framework

### Milestone
- [ ] Build a ReAct agent from scratch (no LangChain — just raw API calls)
- [ ] Build an MCP server that exposes a custom tool
- [ ] Build a multi-agent system where agents coordinate to complete a task

---

## Phase 7 — Production, Safety & LLMOps

> Getting AI systems from your laptop to the real world, safely and reliably.

### Core Concepts

**Evaluation**
- Why evaluation is the hardest problem in LLMs
- Benchmarks: MMLU, HumanEval, HellaSwag, TruthfulQA — what they measure and their limits
- LLM-as-judge evaluation
- Building custom evaluation suites for your specific use case

**Observability & Monitoring**
- Tracing LLM calls (inputs, outputs, latency, cost)
- Detecting hallucinations and failures in production
- A/B testing prompts and models

**Safety & Alignment**
- Constitutional AI — using AI to critique and revise AI outputs
- Red-teaming — systematically probing for failures
- Guardrails — input/output filtering
- Responsible disclosure and model cards

**Deployment & Optimization**
- Quantization — reducing model precision (INT8, INT4) for faster inference
- Speculative decoding — using a small model to propose tokens for a large model
- KV cache — why attention in inference is different from training
- Batching strategies for throughput

**LLMOps**
- Prompt version control
- Dataset management and lineage
- CI/CD for ML models

### Free Resources
- [Chip Huyen — Designing ML Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [Eugene Yan — Applied ML](https://eugeneyan.com/)
- [Hugging Face Inference Endpoints docs](https://huggingface.co/docs/inference-endpoints/)

### Milestone
- [ ] Deploy a fine-tuned model with a REST API endpoint
- [ ] Build an LLM evaluation harness for a specific task
- [ ] Red-team your own model and document findings

---

## Paper Reading Order

For someone starting fresh, read the papers in this order:

| # | Paper | Phase | Why Now |
|---|-------|-------|---------|
| 1 | Attention Is All You Need (2017) | 3 | Foundation of everything |
| 2 | Improving Language Understanding by Generative Pre-Training — GPT-1 (2018) | 4 | First GPT |
| 3 | Language Models are Few-Shot Learners — GPT-3 (2020) | 4 | Scale + emergent abilities |
| 4 | Training language models to follow instructions — InstructGPT (2022) | 5A | How ChatGPT works |
| 5 | RAG (2020) | 5B | Grounding LLMs in facts |
| 6 | Best Practices for Synthetic Data (2024) | 5C | Data for fine-tuning |
| 7 | Model Context Protocol (2024) | 6 | Agentic tool use standard |

---

## Implementation Projects (in order)

| Project | Phase | What you'll learn |
|---------|-------|-------------------|
| Scalar autograd engine | 2 | Backpropagation from scratch |
| Character-level language model | 2/3 | RNN → Transformer transition |
| Transformer encoder from scratch | 3 | Attention mechanics |
| GPT-2 from scratch | 4 | Full decoder-only LLM |
| RAG pipeline from scratch | 5B | Retrieval + generation |
| ReAct agent from scratch | 6 | Agentic reasoning loops |
| MCP server | 6 | Tool use protocol |

---

## Guiding Principles

1. **Read the paper, then read it again.** First pass for intuition. Second pass to extract every equation.
2. **Implement before moving on.** Understanding is not the same as being able to build it.
3. **Use the simplest dataset possible.** Don't fight data problems while learning architecture.
4. **Cite your sources.** Every implementation notes which paper it is based on.
5. **Ask "why" constantly.** Not just what the equation is, but why that form, why that choice.
