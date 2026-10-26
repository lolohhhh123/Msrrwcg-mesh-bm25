"""
train_mesh_cooccurrence.py
使用全量训练集构建MeSH共现矩阵，并评估其在测试集上的检索性能。
"""

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
from pathlib import Path

from relish_dataset import RELISHDataset
from hypergraph_builder import HypergraphBuilder

def build_cooccurrence_matrix(dataset, builder):
    """构建MeSH共现矩阵"""
    print("提取训练集MeSH集合...")
    pmid_to_mesh = {}
    all_mesh_ids = set()
    for pid, doc_info in tqdm(dataset.pmid_to_doc.items(), desc="Extracting MeSH"):
        full_text = doc_info['full_readable']
        mesh_ids = set(builder.extract_mesh_ids(full_text))
        pmid_to_mesh[pid] = mesh_ids
        all_mesh_ids.update(mesh_ids)
    
    mesh_vocab = sorted(list(all_mesh_ids))
    mesh_to_idx = {m: i for i, m in enumerate(mesh_vocab)}
    vocab_size = len(mesh_vocab)
    print(f"MeSH词汇表大小: {vocab_size}")

    # 统计共现
    print("统计MeSH共现...")
    cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    rel_pairs = dataset.pairs
    for _, row in tqdm(rel_pairs.iterrows(), total=len(rel_pairs), desc="Processing pairs"):
        if row['Relevance'] < 1:
            continue
        p1, p2 = row['PMID1'], row['PMID2']
        mesh1 = pmid_to_mesh.get(p1, set())
        mesh2 = pmid_to_mesh.get(p2, set())
        # 只统计两个文档共同的MeSH（交集）
        common = mesh1.intersection(mesh2)
        for m in common:
            i = mesh_to_idx[m]
            for n in common:
                j = mesh_to_idx[n]
                cooccurrence[i, j] += 1
    
    # 对称归一化（可选）
    # 这里不归一化，保留原始频次
    print(f"共现矩阵构建完成，非零元素数: {np.count_nonzero(cooccurrence)}")
    return mesh_vocab, mesh_to_idx, cooccurrence

def score_query_document(q_mesh_set, doc_mesh_set, cooccurrence, mesh_to_idx):
    """计算查询与文档的相关性分数（基于共现权重）"""
    common = q_mesh_set.intersection(doc_mesh_set)
    if not common:
        return 0.0
    total_score = 0.0
    for m in common:
        idx = mesh_to_idx[m]
        # 使用查询中所有MeSH与文档中所有MeSH的共现总和
        # 这里简化为：查询MeSH与文档MeSH的共现平均值
        for n in doc_mesh_set:
            if n in mesh_to_idx:
                total_score += cooccurrence[idx, mesh_to_idx[n]]
    return total_score / (len(q_mesh_set) * len(doc_mesh_set))  # 归一化

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", default="./train_annotated.npy")
    parser.add_argument("--relevance_path", default="./relevance_matrix.tsv")
    parser.add_argument("--mesh_xml", default="./desc2026.xml")
    args = parser.parse_args()

    # 加载训练集
    print("加载训练集...")
    train_dataset = RELISHDataset(
        npy_path=args.npy_path,
        relevance_path=args.relevance_path,
        mesh_xml_path=Path(args.mesh_xml),
        split='train',
        normalize_labels=False
    )
    builder = HypergraphBuilder()
    mesh_vocab, mesh_to_idx, cooccurrence = build_cooccurrence_matrix(train_dataset, builder)

    # 保存共现矩阵供评估使用
    with open("mesh_cooccurrence.pkl", "wb") as f:
        pickle.dump({
            'mesh_vocab': mesh_vocab,
            'mesh_to_idx': mesh_to_idx,
            'cooccurrence': cooccurrence
        }, f)
    print("共现矩阵已保存到 mesh_cooccurrence.pkl")

if __name__ == "__main__":
    main()