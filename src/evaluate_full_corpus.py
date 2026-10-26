"""
evaluate_full_corpus.py
Full-corpus document retrieval evaluation (no negative sampling) + extreme acceleration
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from pathlib import Path
import argparse
import time
from multiprocessing import Pool, cpu_count
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import unicodedata
import re
import warnings
warnings.filterwarnings('ignore')

from relish_dataset import RELISHDataset
from hypergraph_builder import HypergraphBuilder
from gn_scorer_paralle import HypergraphGNNScorer
from evaluate_baselines import clean_text

nltk.download('punkt', quiet=True)

# ==================== BM25 full evaluation (fast version) ====================
def evaluate_bm25_full(dataset, num_queries=30):
    """BM25 full retrieval (compute all document scores at once)"""
    print("Building BM25 index...")
    pmids = list(dataset.pmid_to_doc.keys())
    doc_texts = [clean_text(dataset.pmid_to_doc[pid]['full_readable']) for pid in pmids]
    tokenized_docs = [word_tokenize(doc.lower()) for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)
    
    # Build queries
    query_pmids = pmids[:num_queries]
    query_texts = [dataset.pmid_to_doc[pid]['title_readable'] for pid in query_pmids]
    
    # Batch query BM25
    print(f"Running BM25 retrieval for {len(query_pmids)} queries...")
    all_scores = []
    for q_text in tqdm(query_texts, desc="BM25 queries"):
        q_tokens = word_tokenize(clean_text(q_text).lower())
        scores = bm25.get_scores(q_tokens)
        all_scores.append(scores)
    
    all_scores = np.array(all_scores)  # [num_queries, total_docs]
    
    # Build ground truth
    ground_truth = build_ground_truth(dataset, query_pmids)
    
    return all_scores, ground_truth, query_pmids


# ==================== GNN full evaluation (GPU-accelerated version) ====================
class DocumentFeatureDataset(Dataset):
    """Preload all document features into memory"""
    def __init__(self, doc_features, pmids):
        self.pmids = pmids
        self.doc_features = doc_features
    
    def __len__(self):
        return len(self.pmids)
    
    def __getitem__(self, idx):
        pid = self.pmids[idx]
        feat = self.doc_features[pid]
        return Data(x=feat['node_features'], edge_index=feat['edge_index'])

def evaluate_gnn_full(model_path, dataset, builder, device, num_queries=30, batch_size=512, num_classes=1):
    """GNN full retrieval (GPU batch inference)"""
    print("Loading GNN model...")
    model = HypergraphGNNScorer(node_feat_dim=200, hidden_dim=128, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    nlp = builder.nlp
    
    pmids = list(dataset.pmid_to_doc.keys())
    query_pmids = pmids[:num_queries]
    
    # Preload all document features into memory (already cached in builder)
    doc_features = builder.doc_features if hasattr(builder, 'doc_features') else {}
    if not doc_features:
        # If builder has no doc_features, load from cache
        with open("./hypergraph_cache/doc_features.pkl", "rb") as f:
            doc_features = pickle.load(f)
    
    # Create DataLoader to batch-load documents
    doc_dataset = DocumentFeatureDataset(doc_features, pmids)
    doc_loader = DataLoader(doc_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=24, pin_memory=True, collate_fn=Batch.from_data_list)
    
    # Precompute embeddings for all documents
    print(f"Precomputing embeddings for {len(pmids)} documents...")
    all_doc_embs = []
    with torch.no_grad():
        for batch in tqdm(doc_loader, desc="Encoding documents"):
            batch = batch.to(device)
            doc_emb = model.get_document_embedding(batch.x, batch.edge_index, batch=batch.batch)
            all_doc_embs.append(doc_emb.cpu())
    all_doc_embs = torch.cat(all_doc_embs, dim=0)  # [total_docs, hidden_dim]
    ground_truth = build_ground_truth(dataset, query_pmids)
    # Score each query
    print(f"Running GNN retrieval for {len(query_pmids)} queries...")
    all_scores = []
    get_query_emb = get_query_embedding_func(nlp, model)
    
    for idx, q_pmid in enumerate(tqdm(query_pmids, desc="GNN queries")):
        doc_info = dataset.pmid_to_doc[q_pmid]
        q_text = ' '.join(builder.extract_mesh_ids(doc_info['title_readable'] + ' ' + doc_info['abstract_readable']))
        q_emb = get_query_emb(q_text).to(device)
        q_emb = model.query_proj(q_emb)  # [1, hidden_dim]
        with torch.no_grad():
            scores = torch.mm(q_emb, all_doc_embs.to(device).t()).cpu().numpy().flatten()
        if idx == 0:
            print(f"q_emb sample (first 5): {q_emb[0, :5].detach().cpu().numpy()}")
            print(f"doc_embs sample (first 5): {all_doc_embs[0, :5].detach().cpu().numpy()}")
            print(f"scores sample (first 5): {scores[:5]}")
            # Debug: check the ranking of relevant documents
        truth = ground_truth[q_pmid]
        rel_indices = np.where(truth >= 1)[0]
        if len(rel_indices) > 0:
            # Get the scores of these documents
            rel_scores = scores[rel_indices]
            # Get the ranking of all document scores
            sorted_indices = np.argsort(scores)[::-1]
            # For each relevant document, find its rank
            for idx in rel_indices:
                rank = np.where(sorted_indices == idx)[0][0] + 1
                print(f"Query {q_pmid}: relevant doc {idx} rank {rank}, score {scores[idx]}")
        all_scores.append(scores)
    
    all_scores = np.array(all_scores)  # [num_queries, total_docs]
    
    return all_scores, ground_truth, query_pmids


# ==================== Ordinal model full evaluation ====================
def evaluate_ordinal_full(model_path, dataset, builder, device, num_queries=30, batch_size=512):
    """Ordinal model full retrieval (take class-2 probability)"""
    print("Loading Ordinal model...")
    model = HypergraphGNNScorer(node_feat_dim=200, hidden_dim=128, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    nlp = builder.nlp
    
    pmids = list(dataset.pmid_to_doc.keys())
    query_pmids = pmids[:num_queries]
    
    # Load document features
    with open("./hypergraph_cache/doc_features.pkl", "rb") as f:
        doc_features = pickle.load(f)
    
    # Precompute embeddings and logits for all documents
    print(f"Precomputing Ordinal logits for {len(pmids)} documents...")
    all_logits = []
    
    doc_dataset = DocumentFeatureDataset(doc_features, pmids)
    doc_loader = DataLoader(doc_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=24, pin_memory=True, collate_fn=Batch.from_data_list)
    
    # Since Ordinal needs query embeddings, we first compute doc_emb for all documents, then concatenate with the query
    # A better approach: precompute doc_emb and batch-compute at query time
    all_doc_embs = []
    with torch.no_grad():
        for batch in tqdm(doc_loader, desc="Encoding documents"):
            batch = batch.to(device)
            doc_emb = model.get_document_embedding(batch.x, batch.edge_index, batch=batch.batch)
            all_doc_embs.append(doc_emb.cpu())
    all_doc_embs = torch.cat(all_doc_embs, dim=0)  # [total_docs, hidden_dim]
    
    # Score each query (take class-2 probability)
    get_query_emb = get_query_embedding_func(nlp, model)
    all_scores = []
    
    for q_pmid in tqdm(query_pmids, desc="Ordinal queries"):
        q_text = ' '.join(builder.extract_mesh_ids(dataset.pmid_to_doc[q_pmid]['title_readable']))
        q_emb = get_query_emb(q_text).to(device)
        q_emb = model.query_proj(q_emb)  # [1, hidden_dim]
        
        # Batch-compute logits
        q_expanded = q_emb.repeat(all_doc_embs.size(0), 1)  # [total_docs, hidden_dim]
        combined = torch.cat([q_expanded, all_doc_embs.to(device)], dim=-1)  # [total_docs, 2*hidden_dim]
        logits = model.score_mlp(combined)  # [total_docs, 3]
        prob_class2 = F.softmax(logits, dim=-1)[:, 2].detach().cpu().numpy()
        all_scores.append(prob_class2)
    
    all_scores = np.array(all_scores)  # [num_queries, total_docs]
    ground_truth = build_ground_truth(dataset, query_pmids)
    
    return all_scores, ground_truth, query_pmids


# ==================== Helper functions ====================
def build_ground_truth(dataset, query_pmids):
    """Build the true relevance vector for each query (full corpus)"""
    pmids = list(dataset.pmid_to_doc.keys())
    pmid_to_idx = {pid: i for i, pid in enumerate(pmids)}
    total_docs = len(pmids)
    
    ground_truth = {}
    rel_df = dataset.pairs
    for q_pmid in query_pmids:
        truth = np.zeros(total_docs)
        pairs = rel_df[(rel_df['PMID1'] == q_pmid) | (rel_df['PMID2'] == q_pmid)]
        for _, row in pairs.iterrows():
            if row['PMID1'] == q_pmid:
                other = row['PMID2']
            else:
                other = row['PMID1']
            if other in pmid_to_idx:
                truth[pmid_to_idx[other]] = row['Relevance']
        ground_truth[q_pmid] = truth
    return ground_truth


def get_query_embedding_func(nlp, model):
    def func(query_text):
        doc = nlp(query_text)
        vecs = [t.vector for t in doc if t.has_vector]
        if vecs:
            emb = torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32)
        else:
            emb = torch.zeros(200, dtype=torch.float32)
        return emb.unsqueeze(0)  # Ensure the shape is [1, 200]
    return func


def compute_metrics(all_scores, ground_truth, query_pmids, k=10):
    """Compute MRR, nDCG@k, Precision@k, Recall@k"""
    metrics_list = []
    for i, q_pmid in enumerate(query_pmids):
        scores = all_scores[i]
        truth = ground_truth[q_pmid]
        # Exclude itself (the query document itself)
        # But in full evaluation, we keep itself because the real scenario would not exclude it
        # But here we exclude itself to avoid bias
        # Need to know the index of the query document
        # Simplification: do not exclude, because the query document in the training set may not exist in the test pairs
        # But we can ignore itself when computing metrics
        # For now, do not exclude; keep it simple
        
        # Compute metrics
        true = np.array(truth)
        pred = np.array(scores)
        
        # MRR
        relevant_indices = np.where(true >= 1)[0]
        if len(relevant_indices) == 0:
            mrr = 0.0
        else:
            sorted_indices = np.argsort(pred)[::-1]
            first_rel_rank = None
            for rank, idx in enumerate(sorted_indices):
                if true[idx] >= 1:
                    first_rel_rank = rank + 1
                    break
            mrr = 1.0 / first_rel_rank if first_rel_rank else 0.0
        
        # nDCG@k
        binary_true = (true >= 1).astype(int)
        if len(true) > 0:
            top_k_indices = np.argsort(pred)[-k:][::-1]
            top_k_true = binary_true[top_k_indices]
            top_k_pred = pred[top_k_indices]
            if np.sum(top_k_true) > 0:
                from sklearn.metrics import ndcg_score
                ndcg = ndcg_score([top_k_true], [top_k_pred], k=k)
            else:
                ndcg = 0.0
        else:
            ndcg = 0.0
        
        # Precision@k, Recall@k
        if len(true) >= k:
            top_k_indices = np.argsort(pred)[-k:][::-1]
            top_k_true = binary_true[top_k_indices]
            precision = np.sum(top_k_true) / k
            recall = np.sum(top_k_true) / max(np.sum(binary_true), 1)
        else:
            precision = 0.0
            recall = 0.0
        
        metrics_list.append({'MRR': mrr, f'nDCG@{k}': ndcg, f'Precision@{k}': precision, f'Recall@{k}': recall})
    
    avg_metrics = {m: np.mean([d[m] for d in metrics_list]) for m in metrics_list[0].keys()}
    return avg_metrics


# ==================== Main program ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", default="./train_annotated.npy")
    parser.add_argument("--relevance_path", default="./relevance_matrix.tsv")
    parser.add_argument("--mesh_xml", default="./desc2026.xml")
    parser.add_argument("--model_dir", default="./model", help="model directory")
    parser.add_argument("--num_queries", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Full-corpus document retrieval evaluation (no negative sampling)")
    print("=" * 80)
    
    # Load dataset
    print("Loading dataset...")
    dataset = RELISHDataset(
        npy_path=args.npy_path,
        relevance_path=args.relevance_path,
        mesh_xml_path=Path(args.mesh_xml),
        split='test',
        normalize_labels=False
    )
    print(f"Total documents: {len(dataset.pmid_to_doc)}")
    print(f"Relevance pairs: {len(dataset.pairs)}")
    # After loading the dataset
    pmids = list(dataset.pmid_to_doc.keys())

    # Build ground truth for the first 500 queries
    print("Building ground truth and filtering valid queries...")
    all_gt = build_ground_truth(dataset, pmids[:500])
    valid_query_pmids = []
    for q in pmids[:500]:
        if np.sum(all_gt[q] >= 1) > 0:
            valid_query_pmids.append(q)
    print(f"Number of valid queries: {len(valid_query_pmids)}")

    if len(valid_query_pmids) == 0:
        print("Error: no valid queries in the test set; please use split='train' or check the data.")
        return

    query_pmids = valid_query_pmids[:args.num_queries]

    # 1. BM25 full evaluation
    print("\n" + "=" * 40)
    print("BM25 full evaluation")
    print("=" * 40)
    bm25_scores, bm25_gt, query_pmids = evaluate_bm25_full(dataset, args.num_queries)
    bm25_metrics = compute_metrics(bm25_scores, bm25_gt, query_pmids, args.k)
    print(f"BM25 results:")
    for k, v in bm25_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 2. Triplet model
    print("\n" + "=" * 40)
    print("Triplet model full evaluation")
    print("=" * 40)
    builder = HypergraphBuilder()
    triplet_path = Path(args.model_dir) / "full_Triplet_model.pth"
    if triplet_path.exists():
        triplet_scores, triplet_gt, _ = evaluate_gnn_full(
            str(triplet_path), dataset, builder, args.device, args.num_queries, args.batch_size, num_classes=1
        )
        triplet_metrics = compute_metrics(triplet_scores, bm25_gt, query_pmids, args.k)
        print(f"Triplet results:")
        for k, v in triplet_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Triplet model not found; skipping")
    
    # 3. ListNet model
    print("\n" + "=" * 40)
    print("ListNet model full evaluation")
    print("=" * 40)
    listnet_path = Path(args.model_dir) / "full_ListNet_model.pth"
    if listnet_path.exists():
        listnet_scores, listnet_gt, _ = evaluate_gnn_full(
            str(listnet_path), dataset, builder, args.device, args.num_queries, args.batch_size
        )
        listnet_metrics = compute_metrics(listnet_scores, bm25_gt, query_pmids, args.k)
        print(f"ListNet results:")
        for k, v in listnet_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("ListNet model not found; skipping")
    
    # 4. Ordinal model
    print("\n" + "=" * 40)
    print("Ordinal model full evaluation")
    print("=" * 40)
    ordinal_path = Path(args.model_dir) / "full_Ordinal_model.pth"
    if ordinal_path.exists():
        ordinal_scores, ordinal_gt, _ = evaluate_ordinal_full(
            str(ordinal_path), dataset, builder, args.device, args.num_queries, args.batch_size
        )
        ordinal_metrics = compute_metrics(ordinal_scores, bm25_gt, query_pmids, args.k)
        print(f"Ordinal results:")
        for k, v in ordinal_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Ordinal model not found; skipping")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("Full-corpus document retrieval results summary (k=10, 30 queries)")
    print("=" * 80)
    print(f"{'Model':<15} {'MRR':<10} {'nDCG@10':<10} {'P@10':<10} {'R@10':<10}")
    print("-" * 60)
    print(f"{'BM25':<15} {bm25_metrics['MRR']:.4f}     {bm25_metrics['nDCG@10']:.4f}     {bm25_metrics['Precision@10']:.4f}     {bm25_metrics['Recall@10']:.4f}")
    if triplet_path.exists():
        print(f"{'Triplet':<15} {triplet_metrics['MRR']:.4f}     {triplet_metrics['nDCG@10']:.4f}     {triplet_metrics['Precision@10']:.4f}     {triplet_metrics['Recall@10']:.4f}")
    if listnet_path.exists():
        print(f"{'ListNet':<15} {listnet_metrics['MRR']:.4f}     {listnet_metrics['nDCG@10']:.4f}     {listnet_metrics['Precision@10']:.4f}     {listnet_metrics['Recall@10']:.4f}")
    if ordinal_path.exists():
        print(f"{'Ordinal':<15} {ordinal_metrics['MRR']:.4f}     {ordinal_metrics['nDCG@10']:.4f}     {ordinal_metrics['Precision@10']:.4f}     {ordinal_metrics['Recall@10']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
