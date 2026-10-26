"""
train_triplet_query_doc_precompute.py
预计算所有三元组的特征，训练极速版。
使用新版 hypergraph_builder 的 doc_cache 加速，并支持多进程并行和断点续传。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
import multiprocessing as mp
from functools import partial
import os
warnings.filterwarnings('ignore')

from relish_dataset import RELISHDataset
from hypergraph_builder import HypergraphBuilder
from gn_scorer_paralle import HypergraphGNNScorer
from evaluate_baselines import clean_text

def get_query_embedding(builder, query_text, query_mesh_ids=None, dim=200):
    if query_mesh_ids and builder.mesh_embeddings is not None:
        vecs = []
        for m in query_mesh_ids:
            if m in builder.mesh_to_idx:
                vecs.append(builder.mesh_embeddings[builder.mesh_to_idx[m]])
        if vecs:
            return np.mean(vecs, axis=0).astype(np.float32)
    return builder._hash_text_to_vector(query_text.lower(), dim)

# ========== 验证函数 ==========
def evaluate_on_val(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            q_embs = batch['q_embs'].to(device)
            pos_node_feats = batch['pos_node_feats']
            neg_node_feats = batch['neg_node_feats']
            pos_edge_indices = batch['pos_edge_indices']
            neg_edge_indices = batch['neg_edge_indices']

            for i in range(q_embs.size(0)):
                pos_feat = pos_node_feats[i].unsqueeze(0).to(device)
                neg_feat = neg_node_feats[i].unsqueeze(0).to(device)
                pos_edge = pos_edge_indices[i].to(device)
                neg_edge = neg_edge_indices[i].to(device)
                q_emb = q_embs[i].unsqueeze(0).to(device)

                pos_emb = model.get_document_embedding(pos_feat, pos_edge)
                neg_emb = model.get_document_embedding(neg_feat, neg_edge)
                q_emb_proj = model.query_proj(q_emb)

                pos_score = torch.sum(q_emb_proj * pos_emb, dim=-1)
                neg_score = torch.sum(q_emb_proj * neg_emb, dim=-1)

                if pos_score.item() > neg_score.item():
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0

# ========== 数据集 ==========
class PrecomputedTripletDataset(Dataset):
    def __init__(self, base_dataset, builder, num_negatives=5, query_source='title',
                 precompute_dir='./precomputed_triplets', num_workers=8, batch_size_precompute=10000):
        self.base = base_dataset
        self.builder = builder
        self.num_negatives = num_negatives
        self.query_source = query_source
        self.precompute_dir = Path(precompute_dir)
        self.precompute_dir.mkdir(exist_ok=True, parents=True)
        self.num_workers = num_workers
        self.batch_size_precompute = batch_size_precompute

        # 构建三元组元数据
        self._build_triplets()

        # 构建文档缓存（使用新版 builder 的 precompute_doc_cache）
        self.doc_cache = self._build_doc_cache()

        # 预计算三元组特征（并行 + 断点续传）
        self.data = self._precompute_features_parallel()

    def _build_triplets(self):
        rel_df = self.base.pairs
        self.rel_pairs = [(row['PMID1'], row['PMID2'], row['Relevance'])
                         for _, row in rel_df.iterrows() if row['Relevance'] >= 1]
        pmid_to_pos = {}
        for p1, p2, rel in self.rel_pairs:
            pmid_to_pos.setdefault(p1, []).append((p2, rel))
            pmid_to_pos.setdefault(p2, []).append((p1, rel))

        self.all_pmids = list(self.base.pmid_to_doc.keys())
        self.pmid_to_doc = self.base.pmid_to_doc

        self.triplet_meta = []
        for q_pmid, pos_list in pmid_to_pos.items():
            #title = self.pmid_to_doc[q_pmid]['title_readable']
            #abstract = self.pmid_to_doc[q_pmid]['abstract_readable']
            #abs_words = abstract.split()[:300]
            #abs_short = ' '.join(abs_words)
            #query_text = title + " " + abs_short
            title = self.pmid_to_doc[q_pmid]['title_readable']
            query_text = title   # title only

            for pos_pmid, rel_score in pos_list[:3]:
                neg_candidates = [p for p in self.all_pmids if p != q_pmid and p not in pmid_to_pos]
                if len(neg_candidates) >= self.num_negatives:
                    neg_pmids = random.sample(neg_candidates, self.num_negatives)
                else:
                    neg_pmids = neg_candidates
                for neg_pmid in neg_pmids:
                    self.triplet_meta.append({
                        'query_text': query_text,
                        'pos_pmid': pos_pmid,
                        'neg_pmid': neg_pmid
                    })
        print(f"构建了 {len(self.triplet_meta)} 个三元组")

    def _build_doc_cache(self):
        """使用新版 builder.precompute_doc_cache 预计算所有文档的缓存"""
        cache_file = self.precompute_dir / 'doc_cache_full.pkl'
        if cache_file.exists():
            print("加载文档缓存...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print("构建文档缓存（首次运行，约需1-2小时，但仅一次）...")
        doc_cache = {}
        max_chars = 1500  # 截断到安全长度
        skipped = 0
        for pid, doc_info in tqdm(self.pmid_to_doc.items(), desc="Caching docs"):
            try:
                text = doc_info['full_readable']
                if len(text) > max_chars:
                    text = text[:max_chars]  # 截断
                doc_cache[pid] = self.builder.precompute_doc_cache(text)
            except Exception as e:
                print(f"警告: 跳过文档 {pid}，错误: {e}")
                skipped += 1
                continue
        print(f"缓存完成，跳过了 {skipped} 个文档")
        with open(cache_file, 'wb') as f:
            pickle.dump(doc_cache, f)
        print("文档缓存完成")
        return doc_cache

    # ---------- 并行处理 ----------
    @staticmethod
    def _process_batch(batch_data, precompute_dir, cooccurrence_path):
        import time
        pid = os.getpid()
        print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 子进程开始处理批次", flush=True)

        # 加载 builder
        if not hasattr(PrecomputedTripletDataset._process_batch, 'builder'):
            print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 正在创建 HypergraphBuilder...", flush=True)
            PrecomputedTripletDataset._process_batch.builder = HypergraphBuilder(
                cooccurrence_path=cooccurrence_path
            )
            print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] HypergraphBuilder 创建完成", flush=True)

        # 加载 doc_cache（独立加载）
        if not hasattr(PrecomputedTripletDataset._process_batch, 'doc_cache'):
            print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 正在加载文档缓存...", flush=True)
            cache_path = Path(precompute_dir) / 'doc_cache_full.pkl'
            with open(cache_path, 'rb') as f:
                PrecomputedTripletDataset._process_batch.doc_cache = pickle.load(f)
            print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 文档缓存加载完成", flush=True)

        builder = PrecomputedTripletDataset._process_batch.builder
        doc_cache = PrecomputedTripletDataset._process_batch.doc_cache
        nlp = builder.nlp

        print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 开始处理 {len(batch_data)} 个三元组", flush=True)
        results = []
        for idx, meta in enumerate(batch_data):
            if idx == 1:
                print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 已处理 {idx}/{len(batch_data)} 个三元组", flush=True)                
            if idx % 1000 == 0:
                print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 已处理 {idx}/{len(batch_data)} 个三元组", flush=True)

            q_text = meta['query_text']
            pos_pmid = meta['pos_pmid']
            neg_pmid = meta['neg_pmid']

            # 查询嵌入
            try:
                print(f"...查询嵌入开始", flush=True)
                # 提取 query 的 MeSH
                query_mesh_ids = builder.extract_mesh_ids(q_text) if q_text else []
                q_emb_np = get_query_embedding(builder, q_text, query_mesh_ids)
                q_emb = torch.tensor(q_emb_np, dtype=torch.float32)
                print(f"...查询嵌入完成 (mesh_count={len(query_mesh_ids)})", flush=True)

                # 构建正文档超图
                print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 构建正文档超图 (pos_pmid={pos_pmid}) 开始", flush=True)
                pos_hg = builder.build_hypergraph(
                    document_text=None,
                    query_text=q_text,
                    include_mesh=True,
                    doc_cache=doc_cache[pos_pmid]
                )
                print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 正文档超图构建完成", flush=True)

                # 构建负文档超图
                print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 构建负文档超图 (neg_pmid={neg_pmid}) 开始", flush=True)
                neg_hg = builder.build_hypergraph(
                    document_text=None,
                    query_text=q_text,
                    include_mesh=True,
                    doc_cache=doc_cache[neg_pmid]
                )
                print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 负文档超图构建完成", flush=True)

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 构建超图失败: {e}", flush=True)
                continue

            results.append({
                'q_emb': q_emb,
                'pos_node_feat': pos_hg['node_features'],
                'pos_edge_idx': pos_hg['edge_index'],
                'neg_node_feat': neg_hg['node_features'],
                'neg_edge_idx': neg_hg['edge_index']
            })

        print(f"[{time.strftime('%H:%M:%S')}][PID {pid}] 批次处理完成，共 {len(results)} 个结果", flush=True)
        return results

    def _precompute_features_parallel(self):
        """并行预计算，支持断点续传"""
        final_file = self.precompute_dir / f'triplet_features_{len(self.triplet_meta)}.pth'
        if final_file.exists():
            print(f"从最终缓存加载: {final_file}")
            return torch.load(final_file, map_location='cpu')

        # 检查已完成的批次
        completed_batches = []
        batch_pattern = f'triplet_batch_*_of_{len(self.triplet_meta)}.pkl'
        batch_files = sorted(self.precompute_dir.glob(batch_pattern))
        if batch_files:
            print(f"发现 {len(batch_files)} 个已完成批次，将从中断处继续...")
            for f in batch_files:
                parts = f.stem.split('_')
                if len(parts) >= 4:
                    start = int(parts[2])
                    end = int(parts[3])
                    completed_batches.append((start, end))
            completed_batches.sort()
            last_end = 0
            for start, end in completed_batches:
                if start > last_end:
                    break
                if end > last_end:
                    last_end = end
            start_idx = last_end
        else:
            start_idx = 0

        total = len(self.triplet_meta)
        if start_idx >= total:
            print("所有批次已完成，正在合并...")
            return self._merge_batches()

        print(f"从索引 {start_idx} 继续（共 {total} 个三元组）...")

        # 准备批次
        batch_size = self.batch_size_precompute
        batches = []
        for i in range(start_idx, total, batch_size):
            end = min(i + batch_size, total)
            batches.append(self.triplet_meta[i:end])

        # 并行处理
        print(f"启动 {self.num_workers} 个进程，共 {len(batches)} 个批次...")
        cooccurrence_path = self.builder.cooccurrence_path  # 传递给子进程

        # 使用 partial 固定参数
        func = partial(
            self._process_batch,
            precompute_dir=self.precompute_dir,
            cooccurrence_path=self.builder.cooccurrence_path
        )

        pool = mp.Pool(processes=self.num_workers)
        # 使用 imap 逐步获取结果
        for i, result in enumerate(tqdm(
            pool.imap(func, batches),
            total=len(batches),
            desc="Processing batches"
        )):
            if result is None:
                continue
            # 保存批次
            start_idx_batch = start_idx + i * batch_size
            end_idx_batch = min(start_idx_batch + batch_size, total)
            batch_file = self.precompute_dir / f'triplet_batch_{start_idx_batch}_{end_idx_batch}.pkl'
            with open(batch_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"批次 {start_idx_batch}-{end_idx_batch} 已保存")
        pool.close()
        pool.join()

        # 合并所有批次
        return self._merge_batches()

    def _merge_batches(self):
        """合并所有批次文件"""
        print("合并所有批次...")
        all_data = []
        batch_pattern = f'triplet_batch_*_of_{len(self.triplet_meta)}.pkl'
        batch_files = sorted(self.precompute_dir.glob(batch_pattern))
        for f in batch_files:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                all_data.extend(data)
            # 可选：删除批次文件以节省空间
            # os.remove(f)
        final_file = self.precompute_dir / f'triplet_features_{len(self.triplet_meta)}.pth'
        torch.save(all_data, final_file)
        print(f"合并完成，保存至 {final_file}")
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ========== Collate 函数 ==========
def collate_triplet_precomputed(batch):
    q_embs = torch.stack([item['q_emb'] for item in batch])
    pos_node_feats = [item['pos_node_feat'] for item in batch]
    pos_edge_indices = [item['pos_edge_idx'] for item in batch]
    neg_node_feats = [item['neg_node_feat'] for item in batch]
    neg_edge_indices = [item['neg_edge_idx'] for item in batch]
    return {
        'q_embs': q_embs,
        'pos_node_feats': pos_node_feats,
        'pos_edge_indices': pos_edge_indices,
        'neg_node_feats': neg_node_feats,
        'neg_edge_indices': neg_edge_indices
    }

# ========== 训练函数 ==========
def train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, device, epochs, patience=5):
    best_val_mrr = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Triplet)")
        for batch in pbar:
            optimizer.zero_grad()

            q_embs = batch['q_embs'].to(device)
            pos_node_feats = batch['pos_node_feats']
            neg_node_feats = batch['neg_node_feats']
            pos_edge_indices = batch['pos_edge_indices']
            neg_edge_indices = batch['neg_edge_indices']

            batch_loss = 0.0
            valid_triplets = 0
            for i in range(q_embs.size(0)):
                pos_feat = pos_node_feats[i].unsqueeze(0).to(device)
                neg_feat = neg_node_feats[i].unsqueeze(0).to(device)
                pos_edge = pos_edge_indices[i].to(device)
                neg_edge = neg_edge_indices[i].to(device)
                q_emb = q_embs[i].unsqueeze(0).to(device)

                pos_emb = model.get_document_embedding(pos_feat, pos_edge)
                neg_emb = model.get_document_embedding(neg_feat, neg_edge)
                q_emb_proj = model.query_proj(q_emb)

                loss = criterion(q_emb_proj, pos_emb, neg_emb)
                batch_loss += loss
                valid_triplets += 1

            if valid_triplets == 0:
                continue
            batch_loss = batch_loss / valid_triplets
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()
            pbar.set_postfix({'loss': batch_loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Triplet Epoch {epoch+1} training loss: {avg_loss:.4f}")

        if scheduler:
            scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0:
            val_mrr = evaluate_on_val(model, val_loader, device)
            print(f"Val MRR: {val_mrr:.4f}")
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"  -> New best model saved (MRR: {best_val_mrr:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# ========== 主程序 ==========
def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", default="./train_annotated.npy")
    parser.add_argument("--relevance_path", default="./relevance_matrix.tsv")
    parser.add_argument("--mesh_xml", default="./desc2026.xml")
    parser.add_argument("--num_negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_prefix", type=str, default="triplet_querydoc")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_precompute", type=int, default=10000)
    args = parser.parse_args()

    print("加载数据集...")
    dataset = RELISHDataset(
        npy_path=args.npy_path,
        relevance_path=args.relevance_path,
        mesh_xml_path=Path(args.mesh_xml),
        split='train',
        normalize_labels=False
    )
    builder = HypergraphBuilder(cooccurrence_path="./mesh_cooccurrence.pkl")

    # 预计算数据集（并行+断点续传）
    train_dataset = PrecomputedTripletDataset(
        dataset, builder,
        num_negatives=args.num_negatives,
        query_source='title',
        precompute_dir='./precomputed_triplets',
        num_workers=args.num_workers,
        batch_size_precompute=args.batch_precompute
    )
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    print(f"训练集: {train_size}, 验证集: {val_size}")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_triplet_precomputed, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_triplet_precomputed, num_workers=0, pin_memory=True)

    model = HypergraphGNNScorer(node_feat_dim=200, hidden_dim=256, num_classes=1)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    trained_model = train_model(
        train_loader, val_loader, model, optimizer, scheduler, criterion, device, args.epochs, patience=5
    )

    save_path = f"./model/{args.model_prefix}_model.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    main()