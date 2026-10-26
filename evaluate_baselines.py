"""
evaluate_baselines.py
Dedicated to evaluating keyword-document retrieval performance.
Constructs queries from RELISH data (using title or abstract),
and evaluates the ranking ability of each model.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import random
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer, CrossEncoder
import unicodedata
import re
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from relish_dataset import RELISHDataset
from hypergraph_builder import HypergraphBuilder
from gn_scorer_paralle import HypergraphGNNScorer

nltk.download('punkt', quiet=True)
random.seed(42)

# ==================== Text cleaning ====================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==================== Evaluation functions ====================
def evaluate_ranking(true_relevance, pred_scores, k=10, query_idx=0):
    if query_idx < 5:  # only print the first 5 queries
        print(f"Query {query_idx}: total candidates={len(true_relevance)}, "
              f"positive count={np.sum(np.array(true_relevance)>=1)}, "
              f"top 10 labels={np.array(true_relevance)[np.argsort(pred_scores)[-10:][::-1]]}")

    """Compute MRR, nDCG@k, Precision@k, Recall@k"""
    true = np.array(true_relevance)
    pred = np.array(pred_scores)

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

    return {
        'MRR': mrr,
        f'nDCG@{k}': ndcg,
        f'Precision@{k}': precision,
        f'Recall@{k}': recall
    }

# ==================== Keyword-Document Evaluator ====================
class KeywordRetrievalEvaluator:
    def __init__(
        self,
        dataset: RELISHDataset,
        device='cuda',
        num_queries=20,
        num_negatives_per_query=100,
        query_source='title',  # 'title' or 'abstract'
        builder = None
    ):
        """
        Args:
            dataset: RELISHDataset instance
            device: device
            num_queries: number of constructed queries
            num_negatives_per_query: number of negative samples per query
            query_source: source of query text, 'title' or 'abstract'
        """
        self.dataset = dataset
        self.device = device
        self.num_queries = num_queries
        self.num_negatives = num_negatives_per_query
        self.query_source = query_source
        self.builder = builder if builder is not None else HypergraphBuilder()
        self.pmids = list(dataset.pmid_to_doc.keys())
        self.pmid_to_idx = {pid: i for i, pid in enumerate(self.pmids)}
        self.total_docs = len(self.pmids)

        # Build query set: select PMIDs with enough relevant documents
        self._build_queries()

    def _build_queries(self):
        """Build query-document pair dataset"""
        print("Building query-document pairs...")
        rel_df = self.dataset.pairs

        # Count the number of relevant documents for each PMID
        pmid_to_relevant = {}
        for _, row in rel_df.iterrows():
            p1, p2, rel = row['PMID1'], row['PMID2'], row['Relevance']
            if rel >= 1:  # only consider relevant (1 or 2)
                pmid_to_relevant.setdefault(p1, set()).add(p2)
                pmid_to_relevant.setdefault(p2, set()).add(p1)

        # Filter PMIDs with enough relevant documents
        candidate_pmids = [p for p, rels in pmid_to_relevant.items() if len(rels) >= 2]
        if len(candidate_pmids) < self.num_queries:
            print(f"Warning: only {len(candidate_pmids)} PMIDs have enough relevant documents; using all of them")
            selected_pmids = candidate_pmids
        else:
            selected_pmids = random.sample(candidate_pmids, self.num_queries)

        self.query_pmids = selected_pmids
        print(f"Selected {len(self.query_pmids)} query PMIDs")

        # Build positive and negative samples for each query
        self.query_data = []
        for q_pmid in self.query_pmids:
            # Get relevant documents
            relevant_pmids = list(pmid_to_relevant.get(q_pmid, set()))
            if len(relevant_pmids) < 1:
                continue

            # Sample negative documents
            all_other_pmids = [p for p in self.pmids if p != q_pmid and p not in relevant_pmids]
            if len(all_other_pmids) > self.num_negatives:
                negative_pmids = random.sample(all_other_pmids, self.num_negatives)
            else:
                negative_pmids = all_other_pmids

            # Construct candidate set (positive + negative)
            candidate_pmids = relevant_pmids + negative_pmids
            candidate_indices = [self.pmid_to_idx[p] for p in candidate_pmids if p in self.pmid_to_idx]

            # Build true relevance vector
            truth = np.zeros(len(candidate_indices))
            for i, pid in enumerate(candidate_pmids):
                if pid in relevant_pmids:
                    # Check relevance score (0/1/2)
                    score = 1  # default to 1
                    for _, row in rel_df.iterrows():
                        if (row['PMID1'] == q_pmid and row['PMID2'] == pid) or \
                           (row['PMID1'] == pid and row['PMID2'] == q_pmid):
                            score = row['Relevance']
                            break
                    truth[i] = score

            # Get query text
            if self.query_source == 'title':
                raw = self.dataset.pmid_to_doc[q_pmid]['title_readable']
                # Add MeSH IDs (consistent with training)
                mesh_ids = self.builder.extract_mesh_ids(raw)  # requires builder instance
                if mesh_ids:
                    mesh_str = ' '.join(mesh_ids[:5])
                    query_text = f"{raw} {mesh_str}"
                else:
                    query_text = raw
            else:
                query_text = self.dataset.pmid_to_doc[q_pmid]['abstract_readable']

            self.query_data.append({
                'query_pmid': q_pmid,
                'query_text': clean_text(query_text),
                'candidate_pmids': candidate_pmids,
                'candidate_indices': candidate_indices,
                'truth': truth
            })
        print(f"Query {q_pmid}: relevant={len(relevant_pmids)}, negatives={len(negative_pmids)}, total={len(candidate_pmids)}")
        print(f"Successfully built {len(self.query_data)} query-document pairs")
        # Print statistics
        for qd in self.query_data:
            pos_count = np.sum(qd['truth'] >= 1)
            print(f"Query {qd['query_pmid']}: positive samples {pos_count}, negative samples {len(qd['truth']) - pos_count}")

    def evaluate_bm25(self, k=10):
        print("Running BM25 keyword retrieval ...")
        # Build BM25 index for all documents
        doc_texts = [clean_text(self.dataset.pmid_to_doc[pid]['full_readable']) for pid in self.pmids]
        tokenized_docs = [word_tokenize(doc.lower()) for doc in doc_texts]
        bm25 = BM25Okapi(tokenized_docs)

        all_metrics = []
        for q_data in tqdm(self.query_data, desc="BM25 queries"):
            query_text = q_data['query_text']
            query_tokens = word_tokenize(query_text.lower())
            # Score all documents
            all_scores = bm25.get_scores(query_tokens)
            # Extract scores only for candidate documents
            scores = [all_scores[idx] for idx in q_data['candidate_indices']]
            truth = q_data['truth']
            metrics = evaluate_ranking(truth, scores, k=k)
            all_metrics.append(metrics)

        avg_metrics = {m: np.mean([d[m] for d in all_metrics]) for m in all_metrics[0].keys()}
        return avg_metrics

    def evaluate_dense(self, model_name='all-MiniLM-L6-v2', k=10):
        print(f"Running dense retrieval ({model_name}) ...")
        model = SentenceTransformer(model_name, device=self.device)

        # Encode all documents
        print("Encoding documents...")
        doc_texts = [clean_text(self.dataset.pmid_to_doc[pid]['full_readable']) for pid in self.pmids]
        doc_embeddings = model.encode(doc_texts, show_progress_bar=True, convert_to_tensor=True)

        all_metrics = []
        for q_data in tqdm(self.query_data, desc="Dense queries"):
            query_text = q_data['query_text']
            query_emb = model.encode(query_text, convert_to_tensor=True)
            all_scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), doc_embeddings, dim=1).cpu().numpy()
            scores = [all_scores[idx] for idx in q_data['candidate_indices']]
            truth = q_data['truth']
            metrics = evaluate_ranking(truth, scores, k=k)
            all_metrics.append(metrics)

        avg_metrics = {m: np.mean([d[m] for d in all_metrics]) for m in all_metrics[0].keys()}
        return avg_metrics

    def evaluate_cross_encoder(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', k=10):
        print(f"Running cross-encoder ({model_name}) ...")
        cross_encoder = CrossEncoder(model_name, device=self.device)

        all_metrics = []
        for q_data in tqdm(self.query_data, desc="Cross-encoder queries"):
            query_text = q_data['query_text']
            # Rerank only candidate documents
            candidate_texts = [
                clean_text(self.dataset.pmid_to_doc[pid]['full_readable'])
                for pid in q_data['candidate_pmids']
            ]
            pairs = [(query_text, doc) for doc in candidate_texts]
            scores = cross_encoder.predict(pairs)
            truth = q_data['truth']
            metrics = evaluate_ranking(truth, scores, k=k)
            all_metrics.append(metrics)

        avg_metrics = {m: np.mean([d[m] for d in all_metrics]) for m in all_metrics[0].keys()}
        return avg_metrics

    def evaluate_your_model(self, model_path, builder, k=10):
        print("Running hypergraph GNN model keyword retrieval...")
        model = HypergraphGNNScorer(node_feat_dim=200, hidden_dim=128)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        nlp = builder.nlp

        def get_query_embedding(query_text):
            doc = nlp(query_text)
            vecs = [t.vector for t in doc if t.has_vector]
            if vecs:
                return torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32)
            else:
                return torch.zeros(200, dtype=torch.float32)

        all_metrics = []
        for q_data in tqdm(self.query_data, desc="Hypergraph GNN queries"):
            query_text = q_data['query_text']
            query_emb = get_query_embedding(query_text).unsqueeze(0).to(self.device)
            scores = []

            for pid in q_data['candidate_pmids']:
                doc_text = clean_text(self.dataset.pmid_to_doc[pid]['full_readable'])
                hypergraph = builder.build_hypergraph(doc_text, query_text)
                node_feat = hypergraph['node_features'].unsqueeze(0).to(self.device)
                edge_idx = hypergraph['edge_index'].to(self.device)
                with torch.no_grad():
                    logits = model(node_feat, edge_idx, query_emb, batch=None)
                    # Use logits for ranking (without sigmoid)
                    score = logits.item()
                scores.append(score)

            truth = q_data['truth']
            metrics = evaluate_ranking(truth, scores, k=k)
            all_metrics.append(metrics)

        avg_metrics = {m: np.mean([d[m] for d in all_metrics]) for m in all_metrics[0].keys()}
        return avg_metrics

# ==================== Main program ====================
def main():
    parser = argparse.ArgumentParser(description="Keyword-document retrieval evaluation")
    parser.add_argument("--npy_path", default="./train_annotated.npy")
    parser.add_argument("--relevance_path", default="./relevance_matrix.tsv")
    parser.add_argument("--mesh_xml", default="./desc2026.xml")
    parser.add_argument("--model_path", default="./model/best_mini_model.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_queries", type=int, default=20, help="number of queries")
    parser.add_argument("--num_negatives", type=int, default=100, help="number of negative samples per query")
    parser.add_argument("--query_source", type=str, default="title", choices=["title", "abstract"],
                        help="source of query text")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
    help="which split of the dataset to use")
    parser.add_argument("--skip_bm25", action="store_true")
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_cross", action="store_true")
    parser.add_argument("--skip_your", action="store_true")
    args = parser.parse_args()

    print("Loading RELISH dataset...")
    dataset = RELISHDataset(
        npy_path=args.npy_path,
        relevance_path=args.relevance_path,
        mesh_xml_path=Path(args.mesh_xml),
        split=args.split,
        normalize_labels=False
    )
    print(f"Total documents: {len(dataset.pmid_to_doc)}")
    print(f"Relevance pairs: {len(dataset.pairs)}")
    builder = HypergraphBuilder()
    evaluator = KeywordRetrievalEvaluator(
        dataset,
        device=args.device,
        num_queries=args.num_queries,
        num_negatives_per_query=args.num_negatives,
        query_source=args.query_source,
        builder=builder
    )

    results = {}
    if not args.skip_bm25:
        results['BM25'] = evaluator.evaluate_bm25(k=args.k)
    if not args.skip_dense:
        results['Dense (all-MiniLM)'] = evaluator.evaluate_dense(k=args.k)
    if not args.skip_cross:
        results['Cross-Encoder (MiniLM)'] = evaluator.evaluate_cross_encoder(k=args.k)
    if not args.skip_your:
        builder = HypergraphBuilder()
        results['Your HypergraphGNN'] = evaluator.evaluate_your_model(args.model_path, builder, k=args.k)

    if results:
        print("\n" + "="*80)
        print(f"Keyword-document retrieval evaluation results (k={args.k}, {args.num_queries} queries):")
        print("="*80)
        header = ["Model"] + list(next(iter(results.values())).keys())
        rows = [[name] + [f"{metrics[m]:.4f}" for m in metrics.keys()] for name, metrics in results.items()]
        col_widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
        fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*header))
        print("-"*sum(col_widths)+"-"*(len(col_widths)-1)*3)
        for row in rows:
            print(fmt.format(*row))
        print("="*80)
        pd.DataFrame([{'Model': name, **metrics} for name, metrics in results.items()]).to_csv('keyword_retrieval_results.csv', index=False)
        print("Results saved to keyword_retrieval_results.csv")
    else:
        print("No evaluation was completed successfully.")

if __name__ == "__main__":
    main()
