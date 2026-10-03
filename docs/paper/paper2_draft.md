# Alias-aware GraphRAG over classical Ayurvedic texts and a herb–drug knowledge graph: the vocabulary gap, over-reading of graph facts, and failure modes of a small local model

> **DRAFT for internal and mentor review. Not submitted.** Every number is generated from the project's result files by `python -m ayurveda_kg.paper`; do not edit numbers by hand. Items marked **[VERIFY]** must be checked against the primary source. **There is no expert-verified question set yet**, so this draft makes no claim about answer correctness on real user questions; that is the main thing to add before submission.

## Abstract

Classical Ayurvedic texts hold centuries of documented medical knowledge that is hard to search, and a general language model answers health questions fluently but without checkable sources. We build a retrieval-augmented question-answering system over two public-domain English translations (Kaviratna's *Charaka-Samhita* and Bhishagratna's *Sushruta Samhita*; 6,493 passages) and a herb–drug knowledge graph, using a fully local 8-billion-parameter model. The system links entities with an alias-aware lexicon, extracts a text graph with a grounding check (532 accepted triples, 710 rejected), retrieves passages, text-graph facts and knowledge-graph facts, and answers with citations and an appended research-use disclaimer. Three results stand out. First, the *vocabulary gap* between modern names and Sanskrit names in the texts is severe: for questions naming a herb in English, recall@5 of plain dense retrieval is 0.017, rising to 0.167 with alias expansion and graph facts. Second, graph facts invite *over-reading*: a model turned "co-mentioned with" into "balances" until the wording and prompt were revised, and an early automatic check wrongly reported the fix as harmful because it ignored negation. Third, a small local model refuses or blends irrelevant passages in ways that automatic faithfulness proxies cannot see. We report what the evaluation can and cannot show, and release code and identifiers.

## 1. Introduction

A student, practitioner or patient cannot easily ask a question such as "what does classical Ayurveda say about treating cough?" and receive a sourced answer. Plain question answering with a language model is unsafe in a health domain because it can state confident, unsupported claims, so we ground answers in retrieved passages and in an explicit graph of entities and relations. Grounded graph retrieval has been demonstrated for Traditional Chinese Medicine (OpenTCM, 2025) **[VERIFY: citation details]**, and an open Ayurveda-domain language model (AyurParam, arXiv:2511.02374) has been released. We found no equivalent system for Ayurveda **[VERIFY: confirm with a systematic search]**. We deliberately use a small local model so that the system is reproducible without external services, and we study its failure modes rather than hiding them.

## 2. System

**Corpus.** The Kaviratna *Charaka-Samhita* translation (1890–1908) and the Bhishagratna *Sushruta Samhita* translation (1907–1916), both US public domain, as OCR text from archive.org. The text is cleaned (running headers, page numbers, line-break hyphenation, OCR garbage lines), and footnotes are kept because they carry Latin plant names. Passages are packed to about 190 words with one-paragraph overlap, giving 6,493 passages with stable ids. There are no page numbers, so citations use passage ids.

**Entities.** A hand-written lexicon links the scoped herbs (canonical, Latin, common, Sanskrit and OCR-era spellings), the scoped drugs, doshas and a compact list of conditions. It finds 16 of the 20 scoped herbs in the texts. A surface form claimed by two entities is dropped rather than guessed.

**Text graph.** Herb–condition and herb–dosha co-occurrence gives 211 edges with provenance. A local model extracts explicit triples from 476 herb-and-condition passages; a triple is accepted only if its subject and object literally appear in the passage and resolve to lexicon entities, and the model's verbs ("allays", "destroys") are mapped transparently to *treats*, *pacifies* or *aggravates* with the raw verb kept. Precision of accepted triples is unverified until an expert checks a sample.

**Retrieval.** Dense retrieval (a sentence-embedding model) is combined with alias expansion of herbs named in the question, text-graph facts, and knowledge-graph facts (herb–drug research risk scores from our companion work), merged by reciprocal-rank fusion. Each context item has an id (`P` passage, `G` text-graph fact, `K` knowledge-graph fact).

**Generation.** A local 8B model (temperature 0) must answer only from the numbered sources, cite ids, say when the sources are insufficient, treat "co-mentioned" as co-occurrence only, and report a research risk score (never declare a combination safe) for interaction questions. A research-use disclaimer is appended by code, never left to the model.

## 3. Evaluation design

We use three layers and label what each proves. (1) *Synthetic retrieval questions* (60 written by the local model from sampled passages; gold = the source passage): lexically easy, so only the comparison between configurations is meaningful. (2) *Vocabulary-gap questions* (60) built from extracted triples that name the herb in English while every supporting passage uses a Sanskrit name. (3) *Knowledge-graph-grounded questions* with programmatically computed gold answers (12 risk-score, 20 top-drugs and 35 held-out top-herbs questions): the graph's advantage here is by construction, so it measures what the graph adds, not general answer quality. Automatic faithfulness is approximated by citation validity and sentence-level lexical support, which are proxies. An expert-verified question set is not yet available.

## 4. Results

**Retrieval.** On the synthetic questions, recall@5 is 0.450 for plain retrieval and 0.533 with graph facts (MRR 0.316 and 0.371). Alias expansion fired on almost none of these questions because they rarely name a herb in English, which is why the vocabulary-gap set is needed.

**The vocabulary gap.** On questions that say "turmeric" where the passages say "Haridra", recall@5 is 0.017 (plain), 0.067 (alias expansion), 0.083 (graph facts) and 0.167 (both). The relative gain is large, but the absolute level shows the problem is far from solved.

**Knowledge-graph questions.** The full system reaches a correctness of 83% on risk-score questions, 75% on top-drugs questions (mean recall of the three gold drugs) and 97% on held-out top-herbs questions. Plain retrieval correctly refuses these (0% correct on top-drugs), and a model with no retrieval reaches 30% on top-drugs only by naming commonly interacting drugs. Even with the answer in its context, the full system refused 25% of top-drugs questions: a reliability limit of a small model. Placing knowledge-graph facts first in the prompt did not help (20% on top-drugs and 71% on held-out top-herbs), so we keep the original order.

**Over-reading graph facts.** Reading the answers showed the model converting "X is co-mentioned with Y" into "X balances Y". We revised the graph-fact wording and the prompt (after seeing these outputs, so this is a fix, not a blind test) and counted sentences that cite only co-occurrence facts but state a therapeutic role: 1 of 16 before the fix and 0 of 17 after, on the same 30 questions (small samples). A first version of this automatic check reported the fix as harmful (6% to 35%) because it flagged sentences such as "they do not state that neem treats skin disease"; making it negation-aware and recounting the stored answers corrected this, a reminder that proxy metrics need the same scrutiny as models.

## 5. Failure modes and limitations

- **Undetected blending.** An irrelevant retrieved passage can be blended into an answer (an answer about ginger and digestion cited a passage about another plant), which lexical support scores as fully supported.
- **No expert verification.** Correctness on real questions is untested; the expert question template is ready.
- **Triple precision is unverified**, and object linking is by containment, so a phrase containing the word "wind" can link to the vata dosha.
- **OCR and coverage.** Two translations only, noisy OCR, no page numbers, front-matter retrieved as content, and four scoped herbs not found in the texts under our spellings.
- **A single small local model** and small question sets mean differences of a few answers are within noise; prompt rules were revised after reading outputs.
- Interaction facts are research risk scores with known false alarms, never advice. The AyurParam comparison from our plan is not done.

## 6. Conclusion

Alias-aware, graph-assisted retrieval is a large relative improvement where modern and Sanskrit names diverge, and a small local model can answer from a knowledge graph with checkable citations, but its failure modes (refusal despite evidence, over-reading co-occurrence, blending irrelevant passages) are not captured by lexical faithfulness proxies. An expert-verified evaluation is the necessary next step.

## References (verified in this project unless marked)

- AyurParam: a state-of-the-art bilingual language model for Ayurveda. arXiv:2511.02374.
- Vivek-Ananth RP et al. IMPPAT 2.0. ACS Omega 2023. doi:10.1021/acsomega.3c00156 **[VERIFY authors]**
- Kaviratna AC (tr.). *Charaka-Samhita*, Calcutta, 1890–1908 (archive.org item BIUSante_47357); Bhishagratna KL (tr.). *The Sushruta Samhita*, vols 1–3, 1907–1916 (archive.org in.ernet.dli.2015.43171, .39322, .39323).
- **[VERIFY]** OpenTCM (GraphRAG for Traditional Chinese Medicine, 2025); sentence-embedding model (all-MiniLM-L6-v2) and Qwen3 model citations; retrieval-augmented generation and GraphRAG surveys.
- Companion work: the herb–drug knowledge graph and risk scores (Paper 1 draft, this repository).

## Appendix: reproducibility

`python -m ayurveda_kg.rag.build_index`, `python -m ayurveda_kg.rag.run_extract` and `python -m ayurveda_kg.phase6` (requires a running local model server), then `python -m ayurveda_kg.paper`. Corpus text and processed data are not committed.
