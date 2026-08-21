# Single-Source RAG Assistant — Phase 1

A single-document Research Paper RAG assistant for the paper **Attention Is All You Need**.
It implements the Phase 1 pipeline required by the assignment:

**PDF upload → page-aware chunking → local embeddings → FAISS retrieval → query routing/decomposition → cross-encoder re-ranking → LCEL RAG generation → source attribution → RAGAS evaluation**

## Why this matches the rubric

| Requirement | Implementation |
|---|---|
| PDF upload | FastAPI `/ingest` + Streamlit upload |
| Chunking | `PyPDFLoader` + `RecursiveCharacterTextSplitter` |
| Local embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| FAISS | Persistent LangChain FAISS index |
| LCEL RAG chain | `backend/chains/rag_chain.py` |
| Query routing | single-fact / multi-part / summarization |
| Query decomposition | multi-part questions create separate retrieval queries |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| RAGAS | faithfulness, answer relevancy, context precision |
| Citations | chunk ID + source filename + page + snippet |
| Evaluation logging | `data/evaluation/ragas_results.jsonl` |

The assignment specifically requires retrieval to change with different questions, a visible before/after re-ranking example, plausible RAGAS metrics, and verifiable page citations. The provided problem statement lists those as evaluation criteria. fileciteturn0file0L59-L63

## Architecture

```text
                    ┌──────────────────────┐
                    │   Streamlit Frontend │
                    └──────────┬───────────┘
                               │ HTTP
                    ┌──────────▼───────────┐
                    │      FastAPI API     │
                    └───────┬───────┬──────┘
                            │       │
                  /ingest   │       │ /query
                            │       │
             ┌──────────────▼┐   ┌──▼─────────────────┐
             │ PDF + Chunking│   │ Query Router       │
             │ page metadata │   │ + decomposition    │
             └───────┬───────┘   └─────────┬───────────┘
                     │                     │
             ┌───────▼────────┐            │
             │ Local Embedding│            │
             └───────┬────────┘            │
                     ▼                     ▼
             ┌───────────────┐     ┌───────────────────┐
             │ FAISS top-N    │────► Cross-Encoder     │
             │ retrieval      │     │ re-ranking top-K  │
             └───────────────┘     └─────────┬─────────┘
                                             │
                                      ┌──────▼──────┐
                                      │ LCEL + Gemini│
                                      └──────┬──────┘
                                             │
                                      answer + sources
                                             │
                                      ┌──────▼──────┐
                                      │ RAGAS + JSONL│
                                      └─────────────┘
```

## Fastest local setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Put your Gemini key in `.env` as `GOOGLE_API_KEY=...`.

### Start backend

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Health check:

```text
http://localhost:8000/health
```

### Start frontend in another terminal

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL, upload `data/papers/attention-is-all-you-need.pdf`, click **Build / Replace Index**, then ask questions.

## Demo questions

1. **Single fact:** What architecture is introduced in the paper?
2. **Multi-part:** What is the Transformer architecture and how does it compare with recurrent neural networks?
3. **Summary:** Summarize the main contributions of the paper.
4. **Out of scope:** What is the capital of France?

The second question demonstrates query decomposition; the final question demonstrates the no-hallucination behavior.

## Re-ranking evidence

Every reranking call appends a record to:

```text
data/evaluation/reranking_comparisons.jsonl
```

Each record contains the candidate order before reranking and the order after the cross-encoder. This directly supports the assignment's requested before/after evidence. fileciteturn0file1L62-L67

## RAGAS evaluation

The evaluation dataset contains 10 questions covering architecture, self-attention, positional encoding, encoder/decoder components, comparison with recurrent models, and contributions.

Run:

```bash
python -m backend.evals.run_ragas_eval
```

Output:

```text
data/evaluation/ragas_results.jsonl
```

The required metrics are faithfulness, answer relevancy, and context precision. The assignment asks for an 8–15 question labeled evaluation set and JSONL logging. fileciteturn0file1L68-L74

## API

### GET `/health`
Returns `{ "status": "ok" }`.

### POST `/ingest`
Multipart upload of one PDF. Extracts pages, creates chunks, embeds locally, and persists FAISS.

### POST `/query`
Body:

```json
{"question": "What is self-attention?"}
```

Response contains:

```json
{
  "answer": "...",
  "sources": [
    {
      "id": "chunk_0007",
      "page": 3,
      "source_file": "attention-is-all-you-need.pdf",
      "snippet": "..."
    }
  ]
}
```

## Important implementation notes

- Page numbers are normalized to human-readable 1-based numbers during ingestion.
- FAISS uses normalized embeddings, so inner-product similarity corresponds to cosine similarity.
- Retrieval is deliberately separated from generation so retrieval quality can be tested independently.
- The router has a deterministic fallback when no Gemini key is configured; this keeps unit tests and local debugging usable. With a key configured, Gemini performs structured routing/decomposition.
- A deterministic extractive fallback is also available when no Gemini key is configured. With a key configured, the normal LCEL Gemini generation path is used.
- No hardcoded answer is returned for normal questions; answers are based on retrieved document chunks.

## Phase 1 scope

This project intentionally stays within single-source RAG. Multi-source hybrid search, GraphRAG, Text-to-SQL, multi-agent orchestration, and production observability/caching are not implemented because they are outside Phase 1 scope.
