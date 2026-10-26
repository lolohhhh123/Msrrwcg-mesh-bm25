# Multi-signal ranking reveals weak co-occurrence gaps for Interpretable Biomedical Literature Retrieval with MeSH Concept-Level Gap Discovery

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

> Interpretable biomedical literature retrieval that **fuses lexical (BM25), concept-level (MeSH NPMI), and discovery structural (hypergraph GNN)** signals, and uses **concept-level MeSH gaps** to discover research topics that existing retrieval methods systematically miss.

---

## ✨ Highlights

- **Three-signal retrieval**: BM25 (lexical) + MeSH co-occurrence / NPMI (concept) + hypergraph GNN (structural), with both weighted fusion and reciprocal rank fusion (RRF).
- **Hypergraph representation**: sentences as hyperedges; MeSH terms and keywords as nodes. Node features combine SVD-reduced MeSH embeddings with character n-gram hashing vectors.
- **Multiple training objectives**: Triplet, ListNet, and Ordinal losses, each paired with its own document scoring head.
- **Interpretable retrieval**: node-ablation explains keyword / MeSH contributions for case studies.
- **Concept-level gap discovery**: mines MeSH pairs from BM25-only documents, keeps candidates with **high expected × low observed** co-occurrence, and compares them against the 25 pairs from *Mine the Gap!* with bootstrap CIs and threshold sensitivity analysis.
- **Publication-grade statistics**: Spearman / Kendall τ, Top-k Jaccard, UpSet intersections, bootstrap CIs, paired bootstrap significance tests.

---

## 🧭 Method Overview

```
┌───────────────────────────────────────────────────────────────┐
│   Query (title)  ─────────────┬────────────────────────────── │
│                               │                               │
│   ┌──────────────┐   ┌────────▼────────┐    ┌──────────────┐  │
│   │    BM25      │   │ MeSH cooc/NPMI  │    │ Hypergraph   │  │
│   │  (lexical)   │   │   (concept)     │    │ GNN (struct) │  │
│   └──────┬───────┘   └────────┬────────┘    └──────┬───────┘  │
│          │                    │                    │          │
│          ▼                    ▼                    ▼          │
│      Top-k docs          Top-k docs           Top-k docs      │
│          │                    │                    │          │
│          └───────────────┬────┴────────────────────┘          │
│                          ▼                                    │
│             Weighted fusion / RRF → final ranking             │
│                          │                                    │
│                          ▼                                    │
│        Explainability (node ablation) + concept-level gaps    │
└───────────────────────────────────────────────────────────────┘
```

Hypergraph definition: `H[node, hyperedge] = 1`, where each sentence is a hyperedge and nodes are MeSH IDs / keywords. Node features are 200-d vectors obtained from `TruncatedSVD(mesh_cooccurrence)` or character n-gram hashing.

---

## 📁 Repository Structure

```text
hgmesh/
├── README.md
├── requirements.txt
├── requirements-optional.txt
├── .gitignore
│
├── src/
│   ├── hypergraph_builder.py        # Hypergraph construction (supports doc_cache)
│   ├── gn_scorer_paralle.py         # Main model: HypergraphGNNScorer (doc embeddings)
│   ├── relish_dataset.py            # RELISH dataset loader
│   ├── mesh_translator.py           # MeSH ID <-> name translation
│   ├── build_mesh_mapping.py        # MeSH XML -> {ID: name} mapping
│   ├── evaluate_baselines.py        # BM25 / Dense / Cross-Encoder baselines
│   └── evaluate_full_corpus.py      # Full-corpus evaluation (no negative sampling)
│
├── train/
│   ├── train_triplet_query_doc.py   # Main training script (Triplet + parallel precompute + resume)
│   └── train_mesh_cooccurrence.py   # Build MeSH co-occurrence matrix
│
├── analysis/
│   ├── analyze_multi_signal_ranking_bst.py  # Ranking correlation / Jaccard / sparsity
│   ├── bootstrap_ci.py                      # Bootstrap / paired bootstrap CIs
│   ├── extract_concept_level_gaps.py        # Concept-level MeSH gap mining
│   ├── sensitivity_analysis.py              # Threshold sensitivity analysis
│   ├── translate_and_compare.py             # Compare with Mine the Gap!
│   └── export_divergence_cases.py           # Export divergence cases for paper tables
│
├── scripts/
│   ├── precompute_ai_ad_features.py # Precompute AI+AD corpus features
│   └── fetch_pubmed_demo.py         # PubMed retrieval demo (sanitized)
│
├── examples/
│   └── sample.jsonl                 # Tiny sample data (optional)
│
├── fix_mesh_prefix.py               # When MeSH terms incorrectly read, run it
└── hybrid_retriever.py              # A combine search demo
```

---

## ⚙️ Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA 11.8+ recommended)
- spaCy + scispaCy (`en_core_sci_scibert`)
- Optional: GPU (recommended for training / full-corpus evaluation)

> spaCy and scispaCy are version-sensitive. If you hit a `spacy` version conflict, please follow the [scispaCy documentation](https://github.com/allenai/scispacy) and pick a matching version pair. The tested combination for this project is:
> `spacy==3.7.5` + `scispacy==0.5.5`.

---

## 🛠 Installation

```bash
git clone https://github.com/<your-name>/hgmesh.git
cd hgmesh

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 1) Core dependencies
pip install -r requirements.txt

# 2) scispaCy model (large, must be installed via URL; cannot go in requirements.txt)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz

# 3) (Optional) dense / cross-encoder baseline dependencies
pip install -r requirements-optional.txt

# 4) (Optional) nltk data
python -c "import nltk; nltk.download('punkt')"
```

---

## 📦 Data Preparation

Place the following files under `./data/` (or any directory of your choice) and pass them via CLI arguments:

| Data | Description | Safe to commit? |
|---|---|---|
| `desc2026.xml` | MeSH descriptor XML ([NCBI download](https://www.nlm.nih.gov/databases/download/mesh.html)) | ❌ too large, `.gitignore` |
| `train_annotated.npy` | Preprocessed RELISH dataset | ❌ |
| `relevance_matrix.tsv` | RELISH relevance labels | ❌ |
| `ai_ad_corpus_*.jsonl` | Custom AI+AD corpus (`pmid/title/abstract/mesh`) | ❌ |
| `mesh_cooccurrence.pkl` | Generated by `train/train_mesh_cooccurrence.py` | ❌ |
| `doc_features.pkl` | Generated by `scripts/precompute_ai_ad_features.py` or the training script | ❌ |
| `*.pth` | Model weights | ❌ |

---

## ⚠️ Important: Fix the Paths First

Some scripts in `src/` and `analysis/` still contain absolute paths. Before running, replace them with your local paths, or better, pass them via CLI args:

- `src/hypergraph_builder.py`: `cooccurrence_path`, `svd_cache_path`
- `scripts/precompute_ai_ad_features.py`: `corpus_path`, `out_path`, `cooc_path`
- `analysis/*.py`: `CORPUS_PATH`, `COOC_PATH`, `CACHE_PATH`
- `train/train_triplet_query_doc.py`: `--npy_path` / `--relevance_path` / `--mesh_xml`

---

## 🚀 Quick Start

### 1) Build the MeSH co-occurrence matrix

```bash
python train/train_mesh_cooccurrence.py \
    --npy_path data/train_annotated.npy \
    --relevance_path data/relevance_matrix.tsv \
    --mesh_xml data/desc2026.xml
```

Output: `mesh_cooccurrence.pkl`

### 2) Train the Triplet model (main entry point)

```bash
python train/train_triplet_query_doc.py \
    --npy_path data/train_annotated.npy \
    --relevance_path data/relevance_matrix.tsv \
    --mesh_xml data/desc2026.xml \
    --num_negatives 5 \
    --epochs 20 \
    --batch_size 16 \
    --device cuda
```

Output: `model/triplet_querydoc_model.pth`

> The training script supports **parallel precomputation + resumable training** (`./precomputed_triplets/`). If training is interrupted, re-running it will resume from the last completed batch.

### 3) Full-corpus evaluation

```bash
python src/evaluate_full_corpus.py \
    --npy_path data/train_annotated.npy \
    --relevance_path data/relevance_matrix.tsv \
    --mesh_xml data/desc2026.xml \
    --model_dir model/ \
    --split test \
    --num_queries 30 \
    --k 10
```

### 4) Core analyses (paper pipelines)

**(a) Precompute AI+AD features**

```bash
python scripts/precompute_ai_ad_features.py
```

**(b) Multi-signal ranking analysis**

```bash
python analysis/analyze_multi_signal_ranking_bst.py \
    --corpus data/ai_ad_corpus_fixed.jsonl \
    --doc_features data/ai_ad_doc_features.pkl \
    --model_path model/triplet_querydoc_model.pth \
    --num_queries 50 \
    --top_k 10 \
    --output_dir multi_signal_analysis/
```

**(c) Bootstrap confidence intervals**

```bash
python analysis/bootstrap_ci.py \
    --per_query_stats multi_signal_analysis/per_query_stats.json
```

**(d) Concept-level gaps**

```bash
python analysis/extract_concept_level_gaps.py
python analysis/translate_and_compare.py
python analysis/sensitivity_analysis.py
```

**(e) Divergence cases (paper case tables)**

```bash
python analysis/export_divergence_cases.py \
    --corpus data/ai_ad_corpus_fixed.jsonl \
    --doc_features data/ai_ad_doc_features.pkl \
    --model_path model/triplet_querydoc_model.pth \
    --output divergence_cases.json
```

---

## 🧪 Script Sheet

| Script | Purpose | Key Output |
|---|---|---|
| `train/train_mesh_cooccurrence.py` | Build MeSH co-occurrence matrix | `mesh_cooccurrence.pkl` |
| `train/train_triplet_query_doc.py` | Train the main Triplet model | `*.pth` |
| `scripts/precompute_ai_ad_features.py` | Precompute AI+AD corpus hypergraph features | `*_doc_features.pkl` |
| `src/evaluate_baselines.py` | BM25 / Dense / Cross-Encoder baselines | `keyword_retrieval_results.csv` |
| `src/evaluate_full_corpus.py` | Full-corpus evaluation (BM25 + Triplet / ListNet / Ordinal) | Console metrics + tables |
| `analysis/analyze_multi_signal_ranking_bst.py` | Spearman/Kendall, Jaccard, sparsity, scatter, UpSet | `multi_signal_analysis/*` |
| `analysis/bootstrap_ci.py` | Confidence intervals and significance tests | `bootstrap_ci.{json,txt,tex}` |
| `analysis/extract_concept_level_gaps.py` | Concept-level gap candidates | `concept_level_gaps.json` |
| `analysis/translate_and_compare.py` | Compare with the 25 pairs from Mine the Gap! | `concept_level_gaps_final.json` + CSV |
| `analysis/sensitivity_analysis.py` | Threshold grid (4×4×4 = 64) sensitivity | `sensitivity_analysis.csv` |
| `analysis/export_divergence_cases.py` | Top-3 divergence cases per query | `divergence_cases.json` |

---

## 🖼 Example Outputs (placeholders)

Running the analysis pipeline populates `multi_signal_analysis/` with:

- `fig_ranking_correlation.png` — Spearman correlation heatmap among BM25 / Cooc / GNN
- `fig_jaccard.png` — Top-k overlap curves
- `fig_sparsity.png` — within-document MeSH density and mean NPMI distributions
- `fig_upset_top10.png` — intersection of the three Top-10 sets
- `fig_scatter.png` — pairwise score scatter plots
- `ranking_stats.json` / `per_query_stats.json`

---

## 🔐 PubMed API Key Handling

`scripts/fetch_pubmed_demo.py` reads credentials from environment variables and **does not hard-code them**:

```bash
export NCBI_EMAIL="you@example.com"
export NCBI_API_KEY="your_api_key"

python scripts/fetch_pubmed_demo.py --query "Alzheimer disease" --max_records 200
```
---

## 📄 License

MIT License. Underlying data (MeSH, RELISH, PubMed-fetched content) is subject to the original providers' terms of use.

---

## 🙏 Acknowledgements

- [scispaCy](https://github.com/allenai/scispacy) — biomedical entity recognition and MeSH linking
- [RELISH](https://github.com/zbmed-semtec/RELISH) — literature relevance dataset
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) — batched hypergraph inference
- [Mine the Gap!](https://github.com/zbmed-semtec/mine-the-gap) — concept-level gap benchmark
