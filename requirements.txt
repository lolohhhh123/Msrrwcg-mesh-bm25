# Msrrwcg-mesh-bm25
# ============================================================
# Multi-signal ranking reveals weak co-occurrence gaps - Core dependencies
# Tested with Python >= 3.9
# ============================================================

# ---- Deep learning & graph ----
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0,<2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0

# ---- Data processing ----
pandas>=2.0.0

# ---- NLP ----
spacy>=3.6.0,<3.8.0
scispacy>=0.5.4,<0.6.0
nltk>=3.8.1
ftfy>=6.1.1

# ---- Retrieval / ranking ----
rank-bm25>=0.2.2

# ---- PubMed fetching ----
biopython>=1.81

# ---- Visualization & progress ----
matplotlib>=3.7.0
seaborn>=0.12.2
tqdm>=4.65.0

# ============================================================
# Optional: only needed for dense / cross-encoder baselines
#   pip install -r requirements-optional.txt
# ============================================================
# sentence-transformers>=2.2.2
