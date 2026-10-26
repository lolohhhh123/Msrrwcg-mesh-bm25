"""
evaluate_keyword_retrieval.py
专门用于评估关键词-文档检索性能。
从RELISH数据中构造查询（使用标题或摘要），并评估各模型的排序能力。
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

# ==================== 文本清洗 ====================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==================== 评估函数 ====================
def evaluate_ranking(true_relevance, pred_scores, k=10, query_idx=0):
    if query_idx < 5:  # 只打印前5个查询
        print(f"Query {query_idx}: total candidates={len(true_relevance)}, "
              f"positive count={np.sum(np.array(true_relevance)>=1)}, "
              f"top 10 labels={np.array(true_relevance)[np.argsort(pred_scores)[-10:][::-1]]}")

    """计算 MRR, nDCG@k, Precision@k, Recall@k"""
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

# ==================== 关键词-文档评估器 ====================
class KeywordRetrievalEvaluator:
    def __init__(
        self,
        dataset: RELISHDataset,
        device='cuda',
        num_queries=20,
        num_negatives_per_query=100,
        query_source='title',  # 'title' 或 'abstract'
        builder = None
    ):
        """
        Args:
            dataset: RELISHDataset 实例
            device: 设备
            num_queries: 构造的查询数量
            num_negatives_per_query: 每个查询的负样本数量
            query_source: 查询文本来源 'title' 或 'abstract'
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

        # 构建查询集：选择有足够相关文档的PMID
        self._build_queries()

    def _build_queries(self):
        """构建查询-文档对数据集"""
        print("构建查询-文档对...")
        rel_df = self.dataset.pairs

        # 统计每个PMID的相关文档数量
        pmid_to_relevant = {}
        for _, row in rel_df.iterrows():
            p1, p2, rel = row['PMID1'], row['PMID2'], row['Relevance']
            if rel >= 1:  # 只考虑相关（1或2）
                pmid_to_relevant.setdefault(p1, set()).add(p2)
                pmid_to_relevant.setdefault(p2, set()).add(p1)

        # 筛选有足够相关文档的PMID
        candidate_pmids = [p for p, rels in pmid_to_relevant.items() if len(rels) >= 2]
        if len(candidate_pmids) < self.num_queries:
            print(f"警告: 只有 {len(candidate_pmids)} 个PMID有足够相关文档，将使用全部")
            selected_pmids = candidate_pmids
        else:
            selected_pmids = random.sample(candidate_pmids, self.num_queries)

        self.query_pmids = selected_pmids
        print(f"选择了 {len(self.query_pmids)} 个查询PMID")

        # 构建每个查询的正负样本
        self.query_data = []
        for q_pmid in self.query_pmids:
            # 获取相关文档
            relevant_pmids = list(pmid_to_relevant.get(q_pmid, set()))
            if len(relevant_pmids) < 1:
                continue

            # 采样负样本
            all_other_pmids = [p for p in self.pmids if p != q_pmid and p not in relevant_pmids]
            if len(all_other_pmids) > self.num_negatives:
                negative_pmids = random.sample(all_other_pmids, self.num_negatives)
            else:
                negative_pmids = all_other_pmids

            # 构造候选集 (正样本 + 负样本)
            candidate_pmids = relevant_pmids + negative_pmids
            candidate_indices = [self.pmid_to_idx[p] for p in candidate_pmids if p in self.pmid_to_idx]

            # 构建真实相关性向量
            truth = np.zeros(len(candidate_indices))
            for i, pid in enumerate(candidate_pmids):
                if pid in relevant_pmids:
                    # 检查相关性分数（0/1/2）
                    score = 1  # 默认为1
                    for _, row in rel_df.iterrows():
                        if (row['PMID1'] == q_pmid and row['PMID2'] == pid) or \
                           (row['PMID1'] == pid and row['PMID2'] == q_pmid):
                            score = row['Relevance']
                            break
                    truth[i] = score

            # 获取查询文本
            if self.query_source == 'title':
                raw = self.dataset.pmid_to_doc[q_pmid]['title_readable']
                # 添加 MeSH ID（与训练一致）
                mesh_ids = self.builder.extract_mesh_ids(raw)  # 需要 builder 实例
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
        print(f"成功构建 {len(self.query_data)} 个查询-文档对")
        # 打印统计信息
        for qd in self.query_data:
            pos_count = np.sum(qd['truth'] >= 1)
            print(f"查询 {qd['query_pmid']}: 正样本 {pos_count}, 负样本 {len(qd['truth']) - pos_count}")

    def evaluate_bm25(self, k=10):
        print("运行 BM25 关键词检索 ...")
        # 构建全量文档的BM25索引
        doc_texts = [clean_text(self.dataset.pmid_to_doc[pid]['full_readable']) for pid in self.pmids]
        tokenized_docs = [word_tokenize(doc.lower()) for doc in doc_texts]
        bm25 = BM25Okapi(tokenized_docs)

        all_metrics = []
        for q_data in tqdm(self.query_data, desc="BM25查询"):
            query_text = q_data['query_text']
            query_tokens = word_tokenize(query_text.lower())
            # 对所有文档打分
            all_scores = bm25.get_scores(query_tokens)
            # 只提取候选文档的分数
            scores = [all_scores[idx] for idx in q_data['candidate_indices']]
            truth = q_data['truth']
            metrics = evaluate_ranking(truth, scores, k=k)
            all_metrics.append(metrics)

        avg_metrics = {m: np.mean([d[m] for d in all_metrics]) for m in all_metrics[0].keys()}
        return avg_metrics

    def evaluate_dense(self, model_name='all-MiniLM-L6-v2', k=10):
        print(f"运行稠密检索 ({model_name}) ...")
        model = SentenceTransformer(model_name, device=self.device)

        # 编码所有文档
        print("编码文档...")
        doc_texts = [clean_text(self.dataset.pmid_to_doc[pid]['full_readable']) for pid in self.pmids]
        doc_embeddings = model.encode(doc_texts, show_progress_bar=True, convert_to_tensor=True)

        all_metrics = []
        for q_data in tqdm(self.query_data, desc="稠密查询"):
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
        print(f"运行交叉编码器 ({model_name}) ...")
        cross_encoder = CrossEncoder(model_name, device=self.device)

        all_metrics = []
        for q_data in tqdm(self.query_data, desc="交叉编码器查询"):
            query_text = q_data['query_text']
            # 只对候选文档重排
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
        print("运行超图GNN模型关键词检索...")
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
        for q_data in tqdm(self.query_data, desc="超图GNN查询"):
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
                    # 使用 logits 排序（不经过sigmoid）
                    score = logits.item()
                scores.append(score)

            truth = q_data['truth']
            metrics = evaluate_ranking(truth, scores, k=k)
            all_metrics.append(metrics)

        avg_metrics = {m: np.mean([d[m] for d in all_metrics]) for m in all_metrics[0].keys()}
        return avg_metrics

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="关键词-文档检索评估")
    parser.add_argument("--npy_path", default="./train_annotated.npy")
    parser.add_argument("--relevance_path", default="./relevance_matrix.tsv")
    parser.add_argument("--mesh_xml", default="./desc2026.xml")
    parser.add_argument("--model_path", default="./model/best_mini_model.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_queries", type=int, default=20, help="查询数量")
    parser.add_argument("--num_negatives", type=int, default=100, help="每个查询的负样本数")
    parser.add_argument("--query_source", type=str, default="title", choices=["title", "abstract"],
                        help="查询文本来源")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
    help="使用数据集的哪个划分")
    parser.add_argument("--skip_bm25", action="store_true")
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_cross", action="store_true")
    parser.add_argument("--skip_your", action="store_true")
    args = parser.parse_args()

    print("加载RELISH数据集...")
    dataset = RELISHDataset(
        npy_path=args.npy_path,
        relevance_path=args.relevance_path,
        mesh_xml_path=Path(args.mesh_xml),
        split=args.split,
        normalize_labels=False
    )
    print(f"文档总数: {len(dataset.pmid_to_doc)}")
    print(f"相关性对: {len(dataset.pairs)}")
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
        print(f"关键词-文档检索评估结果 (k={args.k}, {args.num_queries}个查询):")
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
        print("结果已保存到 keyword_retrieval_results.csv")
    else:
        print("没有成功完成任何评估。")

if __name__ == "__main__":
    main()