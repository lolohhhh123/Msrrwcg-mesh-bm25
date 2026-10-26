"""
analyze_multi_signal_ranking.py

Purpose
-------
Given an AI+AD corpus and a set of queries, compute:
  (1) BM25 ranking           (lexical signal)
  (2) MeSH co-occurrence ranking with NPMI  (concept signal)
  (3) GNN/ hypergraph ranking               (structural signal)

Then produce:
  - Ranking correlation matrices (Spearman, Kendall tau)
  - Top-k Jaccard overlap curves
  - Co-occurrence sparsity distributions
  - Three-signal scatter plots
  - UpSet-style intersection bar chart
  - Divergence case table

Outputs are saved to ./multi_signal_analysis/
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
from rank_bm25 import BM25Okapi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score

# ----- project modules -----
from hypergraph_builder import HypergraphBuilder
from gn_scorer_paralle import HypergraphGNNScorer
from evaluate_baselines import clean_text

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

AI_MESH = {"meshd000465", "meshd001185", "meshd016108", "meshd016109"}  # placeholder
AD_MESH = {"meshd000544", "meshd000755", "meshd001158"}                  # placeholder

# ============================================================
# 1. Data loading
# ============================================================
def load_ai_ad_corpus(corpus_path):
    """
    Expected format: JSONL, one doc per line with fields:
      {"pmid": ..., "title": ..., "abstract": ..., "mesh": [...]}
    Returns a list of dicts.
    """
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(d)
    print(f"Loaded {len(docs)} AI+AD documents.")
    return docs


def build_queries(docs, num_queries=50, min_mesh=3):
    """
    Use the titles of the first N documents that have >= min_mesh MeSH terms
    as queries. This mimics a title-based retrieval scenario.
    """
    queries = []
    for d in docs:
        if len(d.get("mesh", [])) >= min_mesh:
            queries.append({
                "qid": d["pmid"],
                "text": d["title"],
                "mesh": d["mesh"],
            })
        if len(queries) >= num_queries:
            break
    print(f"Built {len(queries)} queries.")
    return queries


# ============================================================
# 2. Signal 1: BM25
# ============================================================
def build_bm25(docs):
    tokenized = [clean_text(d["title"] + " " + d["abstract"]).lower().split()
                 for d in docs]
    return BM25Okapi(tokenized)


def bm25_scores(bm25, query):
    return np.array(bm25.get_scores(clean_text(query).lower().split()))


# ============================================================
# 3. Signal 2: MeSH co-occurrence + NPMI
# ============================================================
def build_cooccurrence(docs):
    """
    Build MeSH-MeSH co-occurrence counts and compute NPMI.
    Returns: mesh_to_idx, cooc_counts (|M| x |M|), npmi (|M| x |M|)
    """
    mesh_sets = [set(d["mesh"]) for d in docs]
    vocab = sorted({m for s in mesh_sets for m in s})
    mesh_to_idx = {m: i for i, m in enumerate(vocab)}
    V = len(vocab)

    cooc = np.zeros((V, V), dtype=np.float32)
    freq = np.zeros(V, dtype=np.float32)
    for s in mesh_sets:
        idxs = [mesh_to_idx[m] for m in s]
        for i in idxs:
            freq[i] += 1
            for j in idxs:
                cooc[i, j] += 1
    N = len(docs)
    p_i = freq / N
    # NPMI
    npmi = np.zeros_like(cooc)
    for i in range(V):
        for j in range(V):
            if cooc[i, j] == 0 or p_i[i] == 0 or p_i[j] == 0:
                npmi[i, j] = -1.0
                continue
            p_ij = cooc[i, j] / N
            pmi = np.log(p_ij / (p_i[i] * p_i[j] + 1e-12) + 1e-12)
            denom = -np.log(p_ij + 1e-12)
            npmi[i, j] = pmi / denom if denom > 0 else -1.0
    return mesh_to_idx, cooc, npmi, freq


def cooccurrence_scores(query_mesh, doc_mesh_list, mesh_to_idx, npmi, freq,
                        use_npmi=True, alpha=1.0):
    """
    Score each document by the mean NPMI between its MeSH terms and the query MeSH terms.
    Documents with no overlapping MeSH get score 0.
    """
    q_idx = [mesh_to_idx[m] for m in query_mesh if m in mesh_to_idx]
    if not q_idx:
        return np.zeros(len(doc_mesh_list), dtype=np.float32)
    scores = np.zeros(len(doc_mesh_list), dtype=np.float32)
    for k, d_mesh in enumerate(doc_mesh_list):
        d_idx = [mesh_to_idx[m] for m in d_mesh if m in mesh_to_idx]
        if not d_idx:
            continue
        vals = []
        for i in q_idx:
            for j in d_idx:
                if i == j:
                    continue
                v = npmi[i, j] if use_npmi else (npmi[i, j] + 1.0) / 2.0
                if v > 0:
                    vals.append(v)
        if vals:
            scores[k] = float(np.mean(vals))
    return scores


# ============================================================
# 4. Signal 3: GNN over hypergraph
# ============================================================
class DocFeatureDataset(Dataset):
    def __init__(self, doc_features, pmids):
        self.pmids = pmids
        self.doc_features = doc_features

    def __len__(self):
        return len(self.pmids)

    def __getitem__(self, idx):
        pid = self.pmids[idx]
        feat = self.doc_features[pid]
        return Data(x=feat["node_features"], edge_index=feat["edge_index"])


def precompute_doc_embeddings(model, doc_features, pmids, device, batch_size=256):
    loader = DataLoader(
        DocFeatureDataset(doc_features, pmids),
        batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=Batch.from_data_list
    )
    embs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding docs"):
            batch = batch.to(device)
            emb = model.get_document_embedding(
                batch.x, batch.edge_index, batch=batch.batch)
            embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


def gnn_scores(model, query_emb_raw, doc_embs, device, hidden_dim):
    """
    query_emb_raw: [200] spaCy mean vector
    doc_embs: [N, hidden_dim]
    Returns dot-product scores [N]
    """
    with torch.no_grad():
        q = torch.tensor(query_emb_raw, dtype=torch.float32).unsqueeze(0).to(device)
        q = model.query_proj(q)                      # [1, hidden_dim]
        d = doc_embs.to(device)                      # [N, hidden_dim]
        s = torch.mm(q, d.t()).cpu().numpy().flatten()
    return s


def get_query_embedding(builder, query_text, query_mesh_ids=None, dim=200):
    if query_mesh_ids and builder.mesh_embeddings is not None:
        vecs = []
        for m in query_mesh_ids:
            if m in builder.mesh_to_idx:
                vecs.append(builder.mesh_embeddings[builder.mesh_to_idx[m]])
        if vecs:
            return np.mean(vecs, axis=0).astype(np.float32)
    return builder._hash_text_to_vector(query_text.lower(), dim)


# ============================================================
# 5. Ranking statistics
# ============================================================
def ranking_stats(scores_dict, k_values=(5, 10, 20, 50)):
    """
    scores_dict: {'BM25': arr, 'Cooc': arr, 'GNN': arr}
    Returns dict of pairwise Spearman, Kendall, Jaccard@k.
    """
    names = list(scores_dict.keys())
    stats = {}

    # pairwise Spearman / Kendall
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = scores_dict[names[i]], scores_dict[names[j]]
            rho, _ = spearmanr(a, b)
            tau, _ = kendalltau(a, b)
            stats[f"Spearman_{names[i]}_{names[j]}"] = rho
            stats[f"Kendall_{names[i]}_{names[j]}"] = tau

    # Jaccard@k
    for k in k_values:
        top_sets = {n: set(np.argsort(scores_dict[n])[::-1][:k]) for n in names}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = top_sets[names[i]], top_sets[names[j]]
                inter = len(A & B)
                union = len(A | B)
                stats[f"Jaccard@{k}_{names[i]}_{names[j]}"] = inter / union if union else 0.0
    return stats


def ndcg_at_k(scores, truth, k=10):
    if len(truth) < k:
        return 0.0
    return ndcg_score([truth], [scores], k=k)


# ============================================================
# 6. Co-occurrence sparsity
# ============================================================
def sparsity_metrics(doc_mesh_list, mesh_to_idx, npmi, cooc,
                     ai_mesh_set=None, ad_mesh_set=None):
    """
    For each document, compute:
      - density: observed co-occurring MeSH pairs / possible pairs
      - mean_npmi: mean NPMI of within-document MeSH pairs
      - ai_ad_bridge: number of pairs crossing AI-MeSH and AD-MeSH
    Requires a set of AI-related MeSH and AD-related MeSH (see below).
    """

    if ai_mesh_set is None: ai_mesh_set = set()
    if ad_mesh_set is None: ad_mesh_set = set()
    mesh_list = list(mesh_to_idx.keys())
    rows = []
    for d_mesh in doc_mesh_list:
        idxs = [mesh_to_idx[m] for m in d_mesh if m in mesh_to_idx]
        n = len(idxs)
        if n < 2:
            rows.append({"density": 0.0, "mean_npmi": 0.0, "ai_ad_bridge": 0})
            continue
        pairs = [(idxs[i], idxs[j]) for i in range(n) for j in range(i+1, n)]
        observed = sum(1 for i, j in pairs if cooc[i, j] > 0)
        density = observed / len(pairs)
        vals = [npmi[i, j] for i, j in pairs if npmi[i, j] > -1.0]
        mean_npmi = float(np.mean(vals)) if vals else -1.0
        bridge = 0
        for i, j in pairs:
            mi, mj = mesh_list[i], mesh_list[j]
            if (mi in ai_mesh_set and mj in ad_mesh_set) or \
               (mj in ai_mesh_set and mi in ad_mesh_set):
                bridge += 1
        rows.append({"density": density, "mean_npmi": mean_npmi, "ai_ad_bridge": bridge})
    return pd.DataFrame(rows)


# ============================================================
# 7. Plotting
# ============================================================
def plot_ranking_correlation(stats, out_dir):
    names = ["BM25", "Cooc", "GNN"]
    mat = np.ones((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            key = f"Spearman_{names[i]}_{names[j]}"
            mat[i, j] = mat[j, i] = stats.get(key, 0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=names, yticklabels=names, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman rank correlation between signals")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ranking_correlation.png", dpi=300)
    plt.close()


def plot_jaccard(stats, out_dir, k_values=(5, 10, 20, 50)):
    names = ["BM25", "Cooc", "GNN"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (i, j) in pairs:
        ys = [stats.get(f"Jaccard@{k}_{names[i]}_{names[j]}", 0.0) for k in k_values]
        ax.plot(k_values, ys, marker="o", label=f"{names[i]} vs {names[j]}")
    ax.set_xlabel("k")
    ax.set_ylabel("Jaccard@k")
    ax.set_title("Top-k overlap between signals")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_jaccard.png", dpi=300)
    plt.close()


def plot_scatter(scores_bm25, scores_cooc, scores_gnn, out_dir, sample_size=3000):
    if len(scores_bm25) > sample_size:
        idx = np.random.choice(len(scores_bm25), sample_size, replace=False)
        scores_bm25 = scores_bm25[idx]
        scores_cooc = scores_cooc[idx]
        scores_gnn = scores_gnn[idx]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(scores_bm25, scores_cooc, s=4, alpha=0.4)
    axes[0].set_xlabel("BM25"); axes[0].set_ylabel("Co-occurrence NPMI")
    axes[1].scatter(scores_bm25, scores_gnn, s=4, alpha=0.4, color="orange")
    axes[1].set_xlabel("BM25"); axes[1].set_ylabel("GNN score")
    axes[2].scatter(scores_cooc, scores_gnn, s=4, alpha=0.4, color="green")
    axes[2].set_xlabel("Co-occurrence NPMI"); axes[2].set_ylabel("GNN score")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_scatter.png", dpi=300)
    plt.close()


def plot_sparsity(df_sparsity, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_sparsity["density"], bins=30, color="steelblue", edgecolor="k")
    axes[0].set_xlabel("MeSH co-occurrence density")
    axes[0].set_ylabel("Number of documents")
    axes[0].set_title("Within-document MeSH density")
    axes[1].hist(df_sparsity["mean_npmi"], bins=30, color="coral", edgecolor="k")
    axes[1].set_xlabel("Mean NPMI of within-document MeSH pairs")
    axes[1].set_ylabel("Number of documents")
    axes[1].set_title("Within-document mean NPMI")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_sparsity.png", dpi=300)
    plt.close()


def plot_upset(all_top_sets, out_dir, k=10):
    """
    Simple bar chart of intersection counts among the three top-k sets.
    """
    names = list(all_top_sets.keys())
    combos = []
    counts = []
    for mask in range(1, 8):
        selected = [names[i] for i in range(3) if mask & (1 << i)]
        if len(selected) < 2:
            continue
        inter = set.intersection(*[all_top_sets[n] for n in selected])
        combos.append(" & ".join(selected))
        counts.append(len(inter))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(combos, counts, color="teal")
    ax.set_ylabel(f"|intersection| of top-{k}")
    ax.set_title(f"Top-{k} overlap among BM25, Co-occurrence, GNN")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_upset_top{k}.png", dpi=300)
    plt.close()


# ============================================================
# 8. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True,
                        help="AI+AD JSONL corpus with pmid/title/abstract/mesh")
    parser.add_argument("--doc_features", required=True,
                        help="Pickle with {pmid: {'node_features':..., 'edge_index':...}}")
    parser.add_argument("--model_path", required=True,
                        help="Trained triplet GNN .pth")
    parser.add_argument("--output_dir", default="./multi_signal_analysis")
    parser.add_argument("--num_queries", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---------- Load corpus ----------
    docs = load_ai_ad_corpus(args.corpus)
    pmids = [d["pmid"] for d in docs]
    doc_mesh_list = [d["mesh"] for d in docs]
    doc_texts = [clean_text(d["title"] + " " + d["abstract"]) for d in docs]

    # ---------- Load cached doc features ----------
    with open(args.doc_features, "rb") as f:
        doc_features = pickle.load(f)
    doc_features = {p: doc_features[p] for p in pmids if p in doc_features}
    print(f"Loaded features for {len(doc_features)} / {len(pmids)} documents.")

    # ---------- Build signals ----------
    print("Building BM25 index...")
    bm25 = build_bm25(docs)

    print("Building MeSH co-occurrence / NPMI...")
    mesh_to_idx, cooc, npmi, freq = build_cooccurrence(docs)

    print("Loading GNN model...")
    model = HypergraphGNNScorer(node_feat_dim=200, hidden_dim=args.hidden_dim, num_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    builder = HypergraphBuilder(
    cooccurrence_path="./mesh_cooccurrence.pkl"  # change to your actual path
    )
    nlp = builder.nlp   # keep the nlp reference to avoid errors elsewhere

    print("Precomputing GNN document embeddings...")
    doc_embs = precompute_doc_embeddings(
        model, doc_features, list(doc_features.keys()), device, args.batch_size)
    print(f"doc_embs shape: {doc_embs.shape}")
    print(f"doc_embs mean: {doc_embs.mean().item():.4f}")
    print(f"doc_embs std: {doc_embs.std().item():.4f}")
    print(f"doc_embs is all zero: {(doc_embs == 0).all().item()}")
    # ---------- Query loop ----------
    queries = build_queries(docs, args.num_queries)
    all_stats = defaultdict(list)
    all_sparsity = []
    example_rows = []

    for q in tqdm(queries, desc="Queries"):
        q_text = q["text"]
        q_mesh = q["mesh"]

        # BM25
        s_bm25 = bm25_scores(bm25, q_text)

        # Co-occurrence NPMI
        s_cooc = cooccurrence_scores(
            q_mesh, doc_mesh_list, mesh_to_idx, npmi, freq, use_npmi=True)

        # GNN
        q_emb_raw = get_query_embedding(builder, q_text, q_mesh)
        s_gnn = gnn_scores(model, q_emb_raw, doc_embs, device, args.hidden_dim)
        print(f"GNN scores (first query): min={s_gnn.min():.4f}, max={s_gnn.max():.4f}, "
            f"mean={s_gnn.mean():.4f}, std={s_gnn.std():.4f}")
        print(f"Non-zero GNN scores: {(np.abs(s_gnn) > 1e-6).sum()} / {len(s_gnn)}")
        # align GNN scores with the full pmid list
        s_gnn_full = np.zeros(len(pmids), dtype=np.float32)
        for i, p in enumerate(doc_features.keys()):
            s_gnn_full[pmids.index(p)] = s_gnn[i]

        # Ranking statistics for this query
        stats = ranking_stats({"BM25": s_bm25, "Cooc": s_cooc, "GNN": s_gnn_full},
                              k_values=(5, 10, 20, 50))
        for k, v in stats.items():
            all_stats[k].append(v)

        # Top-k sets for the global UpSet plot
        if q is queries[0]:
            all_top_sets = {}
        for name, s in [("BM25", s_bm25), ("Cooc", s_cooc), ("GNN", s_gnn_full)]:
            top = set(np.argsort(s)[::-1][:args.top_k])
            all_top_sets.setdefault(name, set()).update(top)

        # Sparsity per query
        sp = sparsity_metrics(
            doc_mesh_list, mesh_to_idx, npmi, cooc,
            ai_mesh_set=AI_MESH, ad_mesh_set=AD_MESH
        )
        sp["query"] = q["qid"]
        all_sparsity.append(sp)

        # Example divergence cases: BM25 high but Cooc low, etc.
        top_bm25 = set(np.argsort(s_bm25)[::-1][:10])
        top_cooc = set(np.argsort(s_cooc)[::-1][:10])
        top_gnn = set(np.argsort(s_gnn_full)[::-1][:10])
        example_rows.append({
            "query": q["qid"],
            "|BM25 ∩ Cooc|": len(top_bm25 & top_cooc),
            "|BM25 ∩ GNN|": len(top_bm25 & top_gnn),
            "|Cooc ∩ GNN|": len(top_cooc & top_gnn),
            "|BM25 ∩ Cooc ∩ GNN|": len(top_bm25 & top_cooc & top_gnn),
        })

    # ---------- Aggregate stats ----------
    avg_stats = {k: float(np.mean(v)) for k, v in all_stats.items()}
    with open(out_dir / "ranking_stats.json", "w") as f:
        json.dump(avg_stats, f, indent=2)

    # Save per-query stats for bootstrap CI
    with open(out_dir / "per_query_stats.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in all_stats.items()}, f)
        
    sparsity_df = pd.concat(all_sparsity, ignore_index=True)
    sparsity_df.to_csv(out_dir / "sparsity.csv", index=False)

    pd.DataFrame(example_rows).to_csv(out_dir / "topk_overlap_per_query.csv", index=False)

    # ---------- Plots ----------
    print("Generating figures...")
    plot_ranking_correlation(avg_stats, out_dir)
    plot_jaccard(avg_stats, out_dir)
    plot_sparsity(sparsity_df, out_dir)
    plot_upset(all_top_sets, out_dir, k=args.top_k)

    # Scatter for the first query (illustrative)
    q0 = queries[0]
    s_bm25 = bm25_scores(bm25, q0["text"])
    s_cooc = cooccurrence_scores(q0["mesh"], doc_mesh_list,
                                 mesh_to_idx, npmi, freq, use_npmi=True)
    q_emb_raw = get_query_embedding(builder, q0["text"], q0["mesh"])
    s_gnn = gnn_scores(model, q_emb_raw, doc_embs, device, args.hidden_dim)
    s_gnn_full = np.zeros(len(pmids), dtype=np.float32)
    for i, p in enumerate(doc_features.keys()):
        s_gnn_full[pmids.index(p)] = s_gnn[i]
    plot_scatter(s_bm25, s_cooc, s_gnn_full, out_dir)

    # ---------- Summary print ----------
    print("\n=== Average ranking statistics ===")
    for k, v in avg_stats.items():
        print(f"{k}: {v:.4f}")
    print("\n=== Mean sparsity ===")
    print(sparsity_df[["density", "mean_npmi"]].mean())
    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
