# Retrieval-Augmented Generation (RAG)

> Based on: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — Lewis et al., 2020
> arXiv: https://arxiv.org/abs/2005.11401
> Also: MasteringRAG (local)

---

## The Hallucination Problem

LLMs have a fundamental weakness: they generate text based on patterns learned during training, not from verified facts. They will confidently generate false information that "sounds right."

This is called **hallucination** and it happens because:
- The model's "knowledge" is baked into its weights during training
- Training data has a cutoff date (no recent information)
- The model has no access to private/enterprise data
- For rare facts, the model may have seen insufficient examples

**RAG's solution:** rather than asking the model to recall facts from memory, *retrieve* the relevant facts first and give them to the model as context.

---

## The RAG Pipeline

```
User Query
    ↓
[Retrieval Step]
    Query → Embedding Model → Query Vector
    Query Vector → Vector Database → Top-K Document Chunks
    ↓
[Augmentation Step]
    Prompt = System Prompt + Retrieved Chunks + User Query
    ↓
[Generation Step]
    Augmented Prompt → LLM → Final Response
```

---

## Step 1: Indexing (Offline, Done Once)

Before any queries, you must index your documents:

1. **Load documents** — PDFs, web pages, databases, etc.
2. **Chunk documents** — split into smaller pieces (e.g., 512-token chunks with overlap)
3. **Embed chunks** — convert each chunk to a dense vector using an embedding model
4. **Store in vector database** — index vectors for fast similarity search

**Why chunk?** Embedding models have limited context length, and smaller chunks allow more precise retrieval.

**Chunking strategies:**
- Fixed-size with overlap (simple, widely used)
- Sentence-level splitting (more semantically coherent)
- Paragraph / section splitting (document-structure-aware)
- Recursive character splitting (LangChain default)

---

## Step 2: Retrieval

Given a user query:
1. Embed the query using the **same embedding model** used for indexing
2. Compute similarity between query embedding and all document embeddings
3. Return the top-K most similar chunks

**Similarity metric:** cosine similarity or dot product (for normalized vectors, these are equivalent)
```
similarity(q, d) = (q · d) / (||q|| · ||d||)
```

**Sparse vs. Dense Retrieval:**

| Type | Method | Example | Strengths |
|------|--------|---------|-----------|
| Sparse | TF-IDF / BM25 | Elasticsearch | Exact keyword matching, fast |
| Dense | Embedding-based | FAISS, Chroma | Semantic similarity, handles paraphrase |
| Hybrid | Sparse + Dense | RRF fusion | Best of both worlds |

---

## Step 3: Generation

The retrieved chunks are injected into the prompt:

```
System: You are a helpful assistant. Answer using only the provided context.
        If you cannot find the answer in the context, say so.

Context:
[Chunk 1: ...]
[Chunk 2: ...]
[Chunk 3: ...]

User: {user_query}