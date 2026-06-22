# Multilingual Health Question Answering in Low-Resource African Languages

**ALU ML Techniques I — Final Project**
**Author:** Edwin Bayingana
**Email:** e.bayingana@alustudent.com

---

## Overview

This project addresses the Zindi competition *Multilingual Health Question Answering in Low-Resource African Languages*. The task is to build a retrieval system that answers health questions in five African languages — Amharic, Akan, Luganda, Swahili, and English — by retrieving the most semantically similar answer from a provided training corpus. No text generation or fine-tuning is involved; the system returns an existing answer verbatim.

Ten experiments are conducted, progressing from a TF-IDF sparse retrieval baseline through dense semantic retrieval using three multilingual sentence transformers, hybrid scoring with per-language weight tuning, two-stage reranking, and MBR ensemble decoding.

**Best result (Experiment 8):** Zindi ROUGE-1 = 0.5763, LLM Judge = 0.7401 — a 20.8% improvement over the TF-IDF baseline.

---

## Repository Structure

```
.
├── Notebook/
│   └── health_qa_final.ipynb       # Main Kaggle notebook — all 10 experiments
├── Submissions/                    # Zindi submission CSV files per experiment
├── Edwin_Bayingana_FinalProject.docx  # Academic report (Word format)
├── Edwin_Bayingana_FinalProject.pdf   # Academic report (PDF format)
└── README.md
```

---

## Experiments Summary

| # | Approach | Val ROUGE-1 | Zindi ROUGE-1 | LLM Judge |
|---|----------|-------------|---------------|-----------|
| 1 | TF-IDF Baseline | 0.3927 | 0.4771 | 0.6469 |
| 2 | mT5-base Zero-Shot | 0.0093 | 0.4771 | 0.6469 |
| 3 | MPNet Semantic | 0.4354 | 0.5080 | 0.6884 |
| 4 | MPNet Hybrid (tuned) | 0.4774 | 0.5404 | 0.7136 |
| 5 | LaBSE Semantic | 0.4467 | 0.5340 | — |
| 6 | LaBSE Hybrid (tuned) | 0.4840 | 0.5701 | 0.7063 |
| 7 | E5-Large Semantic (buggy prefix) | 0.4250 | 0.4785 | 0.6721 |
| 8 | **E5-Large Hybrid (tuned)** | **0.5068** | **0.5763** | **0.7401** |
| 9 | LaBSE + TF-IDF Reranking | 0.4364 | 0.5044 | 0.6628 |
| 10 | MBR Ensemble (LaBSE + E5) | 0.4414 | 0.4790 | 0.6724 |

---

## Key Technical Design Decisions

**Per-language hybrid weight tuning**
A grid search over TF-IDF weight alpha ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} is run independently per language subset on the validation set. The final retrieval score is:

```
score(q, d) = alpha * sim_tfidf(q, d) + (1 - alpha) * sim_semantic(q, d)
```

This reveals that E5-Large converges to pure semantic retrieval (alpha = 0.0) for 7 of 8 subsets — its asymmetric training objective already produces retrieval-optimised cosine similarities.

**E5-Large asymmetric encoding**
E5 requires different prefixes for queries and corpus documents:
- Corpus documents: `"passage: " + text`
- Queries: `"query: " + text`

Applying the wrong prefix (as in Experiment 7) places embeddings in the wrong region of the vector space, causing significant performance degradation. This was identified and corrected in Experiment 8.

**Embedding caching**
All corpus embeddings are cached as `.npy` files to `/kaggle/working/` to avoid re-encoding across sessions. A `load_or_encode()` helper checks for cached files before running the encoder.

**ROUGE evaluation**
ROUGE-1 and ROUGE-L F1 are computed with a whitespace tokeniser (no stemming) to ensure fair, script-agnostic scoring across Ethiopic (Amharic) and Latin-script languages.

---

## Running the Notebook

The notebook is designed to run on Kaggle with GPU acceleration enabled. All dependencies are installed within the notebook.

**Required GPU memory:** approximately 8 GB for E5-Large encoding of the full corpus.

**Estimated runtime:** 3 to 5 hours for all 10 experiments end-to-end, depending on Kaggle GPU allocation.

**To reproduce:**
1. Upload `health_qa_final.ipynb` to Kaggle.
2. Attach the Zindi competition dataset.
3. Enable GPU accelerator.
4. Run all cells in order. Embedding caches will be written to `/kaggle/working/`.

---

## Dependencies

All installed within the notebook via pip. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| sentence-transformers | latest | MPNet, LaBSE, E5-Large encoding |
| scikit-learn | latest | TF-IDF vectorisation |
| rouge-score | latest | ROUGE-1 and ROUGE-L evaluation |
| transformers | latest | mT5-base generation (Experiment 2) |
| torch | latest | GPU acceleration |
| pandas, numpy | latest | Data handling and embedding storage |

---

## Links

- **Demo Video:** [add link here]
- **Zindi Competition:** [Multilingual Health QA](https://zindi.africa)
- **Report:** See `Edwin_Bayingana_FinalProject.pdf` in this repository

---

## Acknowledgements

Competition dataset provided by Zindi Africa. Pre-trained models sourced from the Hugging Face model hub. This project was completed as the final assessment for ML Techniques I at the African Leadership University.
