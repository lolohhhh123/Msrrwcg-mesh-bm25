"""
export_divergence_cases.py
Exports the top-3 documents from each of the three signals for the specified queries,
for use in the case study table of Section 3.6.5.
"""
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from scipy.stats import spearmanr

from hypergraph_builder import HypergraphBuilder
from gn_scorer_paralle import HypergraphGNNScorer
from evaluate_baselines import clean_text


def load_corpus(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def build_bm25(docs):
    tokenized = [clean_text(d["title"] + " " + d["abstract"]).lower().split()
                 for d in docs]
    return BM25Okapi(tokenized)


def build_cooc(docs):
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


def cooc_scores(q_mesh, doc_mesh_list, mesh_to_idx, npmi):
    q_idx = [mesh_to_idx[m] for m in q_mesh if m in mesh_to_idx]
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
                v = npmi[i, j]
                if v > 0:
                    vals.append(v)
        if vals:
            scores[k] = float(np.mean(vals))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--doc_features", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="./divergence_cases.json")
    parser.add_argument("--target_pmids", nargs="+", default=[
        "38335558", "37654029", "36257926", "35681440", "42536022"
    ])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading corpus...")
    docs = load_corpus(args.corpus)
    pmids = [d["pmid"] for d in docs]
    pmid_to_doc = {d["pmid"]: d for d in docs}
    doc_mesh_list = [d["mesh"] for d in docs]

    print("Building BM25...")
    bm25 = build_bm25(docs)

    print("Building MeSH co-occurrence / NPMI...")
    mesh_to_idx, cooc, npmi, freq = build_cooc(docs)

    print("Loading GNN...")
    model = HypergraphGNNScorer(node_feat_dim=200, hidden_dim=256, num_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(args.doc_features, "rb") as f:
        doc_features = pickle.load(f)

    print("Precomputing GNN doc embeddings...")
    from torch_geometric.data import Batch, Data
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __init__(self, features, pmids):
            self.features = features
            self.pmids = pmids

        def __len__(self):
            return len(self.pmids)

        def __getitem__(self, idx):
            p = self.pmids[idx]
            f = self.features[p]
            return Data(x=f["node_features"], edge_index=f["edge_index"])

    valid_pmids = [p for p in pmids if p in doc_features]
    loader = DataLoader(DS(doc_features, valid_pmids),
                        batch_size=256, shuffle=False,
                        collate_fn=Batch.from_data_list)
    embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            e = model.get_document_embedding(batch.x, batch.edge_index, batch=batch.batch)
            embs.append(e.cpu())
    doc_embs = torch.cat(embs, dim=0)
    pmid_to_emb_idx = {p: i for i, p in enumerate(valid_pmids)}

    # builder for hash/SVD query
    print("Loading HypergraphBuilder...")
    builder = HypergraphBuilder(cooccurrence_path="./mesh_cooccurrence.pkl")

    def get_query_emb(query_text, query_mesh):
        if query_mesh and builder.mesh_embeddings is not None:
            vecs = [builder.mesh_embeddings[builder.mesh_to_idx[m]]
                    for m in query_mesh if m in builder.mesh_to_idx]
            if vecs:
                return np.mean(vecs, axis=0).astype(np.float32)
        return builder._hash_text_to_vector(query_text.lower(), 200)

    # Iterate over target queries
    results = {}
    for target_pmid in args.target_pmids:
        if target_pmid not in pmid_to_doc:
            print(f"WARNING: {target_pmid} not in corpus")
            continue

        q_doc = pmid_to_doc[target_pmid]
        q_text = q_doc["title"]
        q_mesh = q_doc["mesh"]

        # Exclude the query itself
        valid_idx = [i for i, p in enumerate(pmids) if p != target_pmid]

        # BM25
        s_bm25 = np.array(bm25.get_scores(clean_text(q_text).lower().split()))
        # Cooc
        s_cooc = cooc_scores(q_mesh, doc_mesh_list, mesh_to_idx, npmi)
        # GNN
        q_emb_np = get_query_emb(q_text, q_mesh)
        with torch.no_grad():
            q_tensor = torch.tensor(q_emb_np).unsqueeze(0).to(device)
            q_proj = model.query_proj(q_tensor)
            s_gnn = torch.mm(q_proj, doc_embs.to(device).t()).cpu().numpy().flatten()
        # Align to the pmids order
        s_gnn_full = np.zeros(len(pmids), dtype=np.float32)
        for i, p in enumerate(valid_pmids):
            s_gnn_full[pmids.index(p)] = s_gnn[i]

        # Take top-3
        def get_top3(scores):
            order = np.argsort(scores)[::-1]
            top = []
            for i in order:
                if pmids[i] == target_pmid:
                    continue
                d = docs[i]
                top.append({
                    "pmid": d["pmid"],
                    "title": d["title"],
                    "score": float(scores[i]),
                    "mesh": d["mesh"],
                })
                if len(top) == 3:
                    break
            return top

        results[target_pmid] = {
            "query_pmid": target_pmid,
            "query_title": q_doc["title"],
            "query_mesh": q_mesh,
            "BM25_top3": get_top3(s_bm25),
            "Cooc_top3": get_top3(s_cooc),
            "GNN_top3": get_top3(s_gnn_full),
        }
        print(f"Done: {target_pmid} - {q_doc['title'][:60]}...")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
