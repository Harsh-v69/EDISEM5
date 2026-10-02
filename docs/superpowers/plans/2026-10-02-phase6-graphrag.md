# Phase 6 Plan: Ayurveda GraphRAG

**Goal:** A question-answering system over classical Ayurveda texts and the project knowledge graph that answers with checkable citations, compared against plain RAG, with an evaluation that is honest about what it can and cannot show without an expert.

**Spec:** `docs/superpowers/specs/2026-10-01-ayurveda-kg-hdi-design.md` section 6.4b.

## Approved inputs (logged in `progress.md` section 9)
- Corpus: Kaviratna *Charaka-Samhita* English translation (1890-1908, archive.org item `BIUSante_47357`, 5.0 MB text) and Bhishagratna *Sushruta Samhita* vols 1-3 (1907-1916, archive.org `in.ernet.dli.2015.43171/39322/39323`, 3.2 MB). Public domain; the later 1949 Gulabkunverba Charaka translation is deliberately **not** used (copyright unclear).
- Embedding model all-MiniLM-L6-v2 (~90 MB). Generator: the already-installed `qwen3:8b` via Ollama. AyurParam GGUF is deferred by the project owner.

## Design decisions
1. **Index:** brute-force cosine over a NumPy matrix (about 10-15k chunks x 384 dims). The spec's FAISS/Chroma is unnecessary at this size; upgrade path noted.
2. **Citations:** every chunk has a stable id and a source label (text, volume). Context items sent to the model carry ids: `P` (text passage), `G` (fact extracted from the texts, with its source passage), `K` (fact from the project KG, e.g. herb-drug risk score).
3. **Text graph:** (a) a lexicon linker for scoped herbs (names, aliases, Sanskrit names), scoped drugs, doshas and disease terms, giving co-occurrence edges with passage provenance; (b) LLM triple extraction with `qwen3:8b` on a budgeted subset, accepted only if both entities literally appear in the source chunk (grounding check). Extraction precision is a pending expert check.
4. **Hybrid retrieval:** dense top-k passages + graph neighbours of entities linked in the question + KG facts (herb-drug risk, compounds, CYP links), merged by reciprocal-rank fusion.
5. **Generation:** temperature 0, must cite ids, must say so when the sources do not contain the answer, always appends the research-use disclaimer.
6. **Baselines:** plain RAG (dense only) and no-retrieval LLM; AyurParam comparison deferred.
7. **Evaluation (three layers, each labelled for what it proves):**
   - *Retrieval benchmark (synthetic):* questions generated from sampled passages, gold = that passage; recall@k and MRR, dense vs hybrid. Caveat: synthetic questions are lexically close to their passage.
   - *KG-grounded multi-hop questions:* gold answers computed from the KG and silver labels (for example drugs that are CYP3A4 substrates and that a herb's compounds are predicted to inhibit). Advantage of graph retrieval here is by construction; reported as such.
   - *Expert set (50-100 questions):* a CSV template and a scoring harness (citation validity, sentence-level lexical support as a faithfulness proxy, hallucination proxy, optional LLM judge) ready for a domain advisor. **Not claimed until an expert fills it.**
8. **Safety:** answers about interactions are framed as research risk scores, never advice.

## Tasks (tests first, validate, then progress.md)
- 6.0 `rag/corpus.py`: fetch (manifest-tracked), clean OCR text, chunk with ids. Real-run stats.
- 6.1 `rag/index.py`: embeddings and dense retrieval; synthetic retrieval benchmark.
- 6.2 `rag/lexicon.py`: entity lexicon and linker.
- 6.3 `rag/textgraph.py`: co-occurrence graph and grounded LLM triple extraction (resumable, background).
- 6.4 `rag/retrieve.py`: hybrid retrieval with KG facts and RRF.
- 6.5 `rag/generate.py`: Ollama client, citation-constrained prompt, baselines.
- 6.6 `rag/evaluate.py`: metrics, question sets, expert template.
- 6.7 Run, report `docs/phase6_results.md`, update `progress.md` and `context.md`.
