"""
hybrid_retriever.py
Implements three-way fusion retrieval: BM25 + Co-occurrence + Triplet.
Supports weighted sum and reciprocal rank fusion (RRF), and outputs the final ranking
and the contributing MeSH terms.
"""

import torch
import numpy as np
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from torch_geometric.data import Batch, Data

from hypergraph_builder import HypergraphBuilder
from gn_scorer_paralle import HypergraphGNNScorer
from evaluate_baselines import clean_text

class HybridRetriever:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.builder = HypergraphBuilder(cooccurrence_path="./mesh_cooccurrence.pkl")
        self.nlp = self.builder.nlp
        
        # Load the Triplet model (upgraded version)
        self.model = HypergraphGNNScorer(node_feat_dim=200, hidden_dim=256, num_classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load document feature cache (assuming it already exists)
        with open("./hypergraph_cache/doc_features.pkl", "rb") as f:
            self.doc_cache = pickle.load(f)
        
        # Prepare BM25 index (requires document texts)
        # Note: here we assume corpus is preloaded; in actual use, it needs to be built
        # from the document library. We use the R2MED or PubMed document collection,
        # but since this script is generic, we will build or preload it dynamically
        # at retrieval time.
        self.bm25 = None
        self.doc_ids = None
        self.doc_texts = None
    
    def build_bm25_index(self, doc_texts, doc_ids):
        """Build BM25 index"""
        tokenized_docs = [clean_text(doc).lower().split() for doc in doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.doc_ids = doc_ids
        self.doc_texts = doc_texts
    
    def _get_query_embedding(self, query):
        doc = self.nlp(query)
        vecs = [t.vector for t in doc if t.has_vector]
        if vecs:
            return torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32)
        else:
            return torch.zeros(200, dtype=torch.float32)
    
    def retrieve(self, query_text, top_k=10, k_retrieve=200, weights=(0.2, 0.3, 0.5), fusion='weighted'):
        """
        Perform hybrid retrieval
        query_text: user query string
        top_k: number of documents to return finally
        k_retrieve: number of BM25 recall candidates
        weights: (bm25_weight, cooc_weight, triplet_weight)
        fusion: 'weighted' or 'rrf' (reciprocal rank fusion)
        """
        # 1. BM25 recall
        if self.bm25 is None:
            raise ValueError("Please call build_bm25_index first to build the index")
        bm25_scores = self.bm25.get_scores(clean_text(query_text).lower().split())
        # Get Top-k_retrieve candidates
        top_indices = np.argsort(bm25_scores)[-k_retrieve:][::-1]
        candidate_ids = [self.doc_ids[i] for i in top_indices]
        candidate_texts = [self.doc_texts[i] for i in top_indices]
        
        # Extract MeSH terms of the query
        q_mesh = set(self.builder.extract_mesh_ids(query_text))
        
        # 2. Co-occurrence score
        cooc_scores = []
        for doc_text in candidate_texts:
            doc_mesh = set(self.builder.extract_mesh_ids(doc_text))
            common = q_mesh.intersection(doc_mesh)
            if not common:
                cooc_scores.append(0.0)
            else:
                # Simple co-occurrence count
                cooc_scores.append(len(common) / (len(q_mesh) + 1e-8))
        cooc_scores = np.array(cooc_scores)
        
        # 3. Triplet score
        triplet_scores = []
        query_emb = self._get_query_embedding(query_text).to(self.device)
        query_emb = self.model.query_proj(query_emb)  # [1, hidden_dim]
        for doc_id in candidate_ids:
            feat = self.doc_cache[doc_id]
            node_feat = feat['node_features'].unsqueeze(0).to(self.device)
            edge_idx = feat['edge_index'].to(self.device)
            with torch.no_grad():
                logit = self.model(node_feat, edge_idx, query_emb, batch=None)
                score = logit.item()
            triplet_scores.append(score)
        triplet_scores = np.array(triplet_scores)
        
        # 4. Score fusion
        # Normalization (Min-Max or Z-score; Min-Max is used here)
        def min_max_norm(arr):
            if np.max(arr) == np.min(arr):
                return np.ones_like(arr) / len(arr)
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        
        bm25_norm = min_max_norm(bm25_scores[top_indices])
        cooc_norm = min_max_norm(cooc_scores)
        triplet_norm = min_max_norm(triplet_scores)
        
        if fusion == 'weighted':
            final_scores = weights[0]*bm25_norm + weights[1]*cooc_norm + weights[2]*triplet_norm
        elif fusion == 'rrf':
            # Reciprocal rank fusion
            k_rrf = 60
            rrf_scores = {}
            for rank, (bm25_score, cooc_score, triplet_score) in enumerate(zip(bm25_scores[top_indices], cooc_scores, triplet_scores)):
                # The RRF score of each document is the sum of reciprocals of the three rankings
                # Sort first to get ranks
                bm25_rank = np.argsort(bm25_scores[top_indices])[::-1].tolist().index(rank) + 1
                cooc_rank = np.argsort(cooc_scores)[::-1].tolist().index(rank) + 1
                triplet_rank = np.argsort(triplet_scores)[::-1].tolist().index(rank) + 1
                rrf_score = 1/(k_rrf + bm25_rank) + 1/(k_rrf + cooc_rank) + 1/(k_rrf + triplet_rank)
                rrf_scores[candidate_ids[rank]] = rrf_score
            # Sort
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            final_scores = sorted_docs
        else:
            # Weighted sum
            final_scores = sorted(zip(candidate_ids, final_scores), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 5. Extract contributing MeSH terms (for the top-1 document)
        if final_scores:
            top_doc_id = final_scores[0][0]
            top_doc_text = self.doc_texts[self.doc_ids.index(top_doc_id)]
            doc_mesh = set(self.builder.extract_mesh_ids(top_doc_text))
            mesh_contributions = {}
            for m in q_mesh.intersection(doc_mesh):
                # Compute co-occurrence weight (using the co-occurrence matrix)
                if m in self.builder.mesh_to_idx:
                    idx = self.builder.mesh_to_idx[m]
                    # Accumulate the co-occurrences of this MeSH with all MeSH terms in the document
                    total = 0
                    for n in doc_mesh:
                        if n in self.builder.mesh_to_idx:
                            total += self.builder.cooccurrence[idx, self.builder.mesh_to_idx[n]]
                    mesh_contributions[m] = total
            sorted_mesh = sorted(mesh_contributions.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            sorted_mesh = []
        
        return {
            'results': final_scores,
            'top_contributing_mesh': sorted_mesh
        }

# ==================== Example usage ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./model/triplet_querydoc_model.pth", help="Triplet model path")
    parser.add_argument("--corpus_file", default="./corpus.jsonl", help="Document corpus file (JSONL format, with id and text fields)")
    parser.add_argument("--query", default="Latest AI in Alzheimer's disease", help="Query text")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Load corpus (example: load from JSONL)
    import json
    corpus = []
    with open(args.corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
    doc_texts = [clean_text(doc['text']) for doc in corpus]
    doc_ids = [doc['id'] for doc in corpus]
    
    retriever = HybridRetriever(args.model_path, device=args.device)
    retriever.build_bm25_index(doc_texts, doc_ids)
    
    result = retriever.retrieve(args.query, top_k=args.top_k)
    print("Top results:")
    for idx, (doc_id, score) in enumerate(result['results']):
        print(f"{idx+1}. {doc_id} (score: {score:.4f})")
    print("\nTop contributing MeSH terms for the best document:")
    for mesh, score in result['top_contributing_mesh']:
        print(f"  {mesh}: {score:.4f}")
