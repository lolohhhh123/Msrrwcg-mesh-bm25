# hypergraph_builder.py
import torch
import re
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from mesh_translator import MeshTranslator
from pathlib import Path
import pickle
import hashlib
from scispacy.linking import EntityLinker
import time
import os, sys
class HypergraphBuilder:
    def __init__(self, cooccurrence_path="/home/hong/GNN/mesh_cooccurrence.pkl"):
        print(f"[{time.strftime('%H:%M:%S')}] 开始初始化 HypergraphBuilder (PID={os.getpid()})", flush=True)
    
        # 加载 scispacy 模型
        print(f"[{time.strftime('%H:%M:%S')}] 正在加载 scispacy 模型...", flush=True)
        self.nlp = spacy.load("en_core_sci_scibert")
        print(f"[{time.strftime('%H:%M:%S')}] scispacy 模型加载完成", flush=True)
    
        print(f"[{time.strftime('%H:%M:%S')}] 正在添加 scispacy_linker...", flush=True)
        if not self.nlp.has_pipe("scispacy_linker"):
            self.nlp.add_pipe("scispacy_linker", config={"linker_name": "mesh"})
        self.linker = self.nlp.get_pipe("scispacy_linker")
        print(f"[{time.strftime('%H:%M:%S')}] scispacy_linker 添加完成", flush=True)
    
        # 其他初始化...
        print(f"[{time.strftime('%H:%M:%S')}] 正在加载共现矩阵...", flush=True)
        self.cooccurrence_path = cooccurrence_path
        self.cooccurrence = None
        self.mesh_to_idx = {}

        if cooccurrence_path:
            try:
                with open(cooccurrence_path, 'rb') as f:
                    data = pickle.load(f)
                    self.cooccurrence = data['cooccurrence']
                    self.mesh_to_idx = data['mesh_to_idx']
                print(f"[{time.strftime('%H:%M:%S')}] 加载共现矩阵，词汇表大小: {self.cooccurrence.shape[0]}", flush=True)
            except FileNotFoundError:
                print(f"警告: 未找到共现矩阵文件 {cooccurrence_path}，将使用默认零向量", flush=True)
        else:
            print("未提供共现矩阵路径，MeSH节点将使用零向量", flush=True)

        if self.cooccurrence is not None:
            svd_cache_path = "/home/hong/GNN/mesh_svd_cache.pkl"
            if os.path.exists(svd_cache_path):
                print(f"Loading cached SVD embeddings from {svd_cache_path}")
                with open(svd_cache_path, "rb") as f:
                    self.mesh_embeddings = pickle.load(f)
                print(f"  mesh_embeddings shape: {self.mesh_embeddings.shape}")
            else:
                from sklearn.decomposition import TruncatedSVD
                print("Precomputing SVD mesh embeddings...")
                svd = TruncatedSVD(n_components=200, random_state=42)
                self.mesh_embeddings = svd.fit_transform(self.cooccurrence)
                print(f"  mesh_embeddings shape: {self.mesh_embeddings.shape}")
                print(f"  explained variance: {svd.explained_variance_ratio_.sum():.3f}")
        else:
            self.mesh_embeddings = None
        self.tfidf = TfidfVectorizer(max_features=50, stop_words=None)  # 停用词已在 noun_chunks 中过滤
        self.mesh_translator = MeshTranslator(Path("./desc2026.xml"))
        self.mesh_pattern = re.compile(r'(meshd\d+)', re.IGNORECASE)
        
        # 缓存：键为文本的 SHA1 哈希值，值为提取到的 MeSH ID 列表
        self.mesh_cache = {}
        self.cache_threshold = 0.5

    def extract_keywords(self, text, n=10):
        """从自由文本中提取关键词（仅用于非缓存模式）"""
        doc = self.nlp(text)
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
        if not noun_chunks:
            return []
        # 过滤纯停用词（可选）
        # from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        # filtered = [chunk for chunk in noun_chunks if chunk not in ENGLISH_STOP_WORDS]
        # if not filtered: return []
        tfidf_matrix = self.tfidf.fit_transform([" ".join(noun_chunks)])
        feature_names = self.tfidf.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = np.argsort(scores)[-n:][::-1]
        return [feature_names[i] for i in top_indices if scores[i] > 0]

    def extract_mesh_ids(self, text):
        """从自由文本中提取 MeSH ID（优先 scispacy，并缓存结果）"""
        if not text:
            return []
        existing = re.findall(r'(meshd\d+)', text, re.IGNORECASE)
        if existing:
            return list(set([mid.lower() for mid in existing]))

        cache_key = hashlib.sha1(text.encode('utf-8')).hexdigest()
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key]

        doc = self.nlp(text)
        mesh_ids = set()
        for ent in doc.ents:
            for cui, score in ent._.kb_ents:
                if score < self.cache_threshold:
                    continue
                entity = self.linker.kb.cui_to_entity[cui]
                concept_id = entity.concept_id
                if concept_id and (concept_id.startswith('D') or concept_id.startswith('C')):
                    mesh_ids.add('meshd' + concept_id.lower())
        result = list(mesh_ids)
        self.mesh_cache[cache_key] = result
        return result

    @staticmethod
    def extract_mesh_ids_regex(text):
        if not hasattr(HypergraphBuilder, '_mesh_pattern'):
            HypergraphBuilder._mesh_pattern = re.compile(r'(meshd\d+)', re.IGNORECASE)
        return list(set(re.findall(HypergraphBuilder._mesh_pattern, text.lower())))

    def build_hypergraph(self, document_text, query_text, include_mesh=True, doc_cache=None):
        """
        构建超图，支持传入预计算的文档缓存（doc_cache）。
        如果 doc_cache 为 None，则实时解析 document_text。
        doc_cache 应包含：
            'mesh_ids': list of str (如 ['meshd000544', ...])
            'keywords': list of str (普通关键词)
            'sent_texts': list of str (每个句子的文本)
            'sent_vectors': list of numpy arrays (每个句子的向量)
        """
        # ---------- 解析查询（必须实时，因为查询很短） ----------
        # 查询的关键词和 MeSH 仍然需要提取，因为每个查询不同
        query_keywords = self.extract_keywords(query_text, n=10) if query_text else []
        query_mesh_ids = self.extract_mesh_ids(query_text) if include_mesh else []

        # ---------- 文档部分 ----------
        if doc_cache is not None:
            # 使用缓存数据
            doc_keywords = doc_cache.get('keywords', [])
            doc_mesh_ids = doc_cache.get('mesh_ids', [])
            sentences = doc_cache.get('sent_texts', [])
            sent_vectors = doc_cache.get('sent_vectors', [])
        else:
            # 实时解析文档
            doc = self.nlp(document_text)
            sentences = [sent.text for sent in doc.sents]
            sent_vectors = [sent.vector for sent in doc.sents]
            doc_keywords = self.extract_keywords(document_text, n=20)
            doc_mesh_ids = self.extract_mesh_ids(document_text) if include_mesh else []

        # ---------- 构建超图 ----------
        all_nodes = list(set(doc_keywords + query_mesh_ids + doc_mesh_ids))
        if not all_nodes:
            all_nodes = ["[EMPTY]"]

        node_to_idx = {node: i for i, node in enumerate(all_nodes)}

        # 构建超边：每个句子连接包含的节点
        hyperedges = []
        hyperedge_features = []
        for sent_idx, sent_text in enumerate(sentences):
            sent_lower = sent_text.lower()
            connected_nodes = []
            for node in all_nodes:
                if node in sent_lower:
                    connected_nodes.append(node_to_idx[node])
                elif node.startswith('meshd') and node.lower() in sent_lower:
                    connected_nodes.append(node_to_idx[node])
            if connected_nodes:
                hyperedges.append(connected_nodes)
                # 使用缓存的向量（如果存在），否则用零向量（但缓存模式下应有）
                if doc_cache is not None and sent_idx < len(sent_vectors):
                    hyperedge_features.append(sent_vectors[sent_idx])
                else:
                    # 实时模式下，我们可能没有预存向量，但我们可以通过 nlp 获取，但为了速度，我们保留原有逻辑
                    # 但在实时模式下，doc 对象还存在，可以获取，但这里为了统一，我们使用零向量
                    # 但为了不损失性能，我们仍然尝试获取
                    hyperedge_features.append(np.zeros(200))  # 占位，后续可改进

        if not hyperedges:
            hyperedges.append(list(range(len(all_nodes))))
            hyperedge_features.append(np.zeros(200))

        # 关联矩阵 H
        H = torch.zeros((len(all_nodes), len(hyperedges)))
        for j, edge in enumerate(hyperedges):
            for node_idx in edge:
                H[node_idx, j] = 1

        # 节点特征
        node_features = self._get_node_features(all_nodes, include_mesh)

        # 边索引
        edge_index = []
        for i, edge in enumerate(hyperedges):
            for node in edge:
                edge_index.append([node, i])
        edge_index = torch.tensor(edge_index).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)

        return {
            'H': H,
            'edge_index': edge_index,
            'node_features': node_features,
            'hyperedge_features': torch.tensor(np.array(hyperedge_features), dtype=torch.float),
            'node_names': all_nodes,
            'num_nodes': len(all_nodes),
            'num_hyperedges': len(hyperedges)
        }

    def _get_node_features(self, nodes, include_mesh):
        """用 SVD embedding（MeSH）或 hash 向量（关键词）生成节点特征"""
        features = []
        for node in nodes:
            if node.startswith('meshd') and include_mesh:
                if self.mesh_embeddings is not None and node in self.mesh_to_idx:
                    idx = self.mesh_to_idx[node]
                    feat = self.mesh_embeddings[idx]  # [200]
                else:
                    feat = self._hash_text_to_vector(node, 200)
            else:
                feat = self._hash_text_to_vector(node, 200)
            features.append(feat)
        return torch.tensor(np.array(features), dtype=torch.float)

    def _hash_text_to_vector(self, text, dim=200):
        """字符 n-gram 哈希，确定性生成向量"""
        import hashlib
        vec = np.zeros(dim, dtype=np.float32)
        text = text.lower()
        for n in (2, 3, 4):
            for i in range(max(1, len(text) - n + 1)):
                gram = text[i:i+n] if len(text) >= n else text
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
                vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def precompute_doc_cache(self, document_text):
        """
        为给定的文档文本生成缓存字典，供 build_hypergraph 使用。
        """
        max_tokens = 450
        tokens = list(self.nlp.tokenizer(document_text))
        if len(tokens) > max_tokens:
            truncated = ''.join([t.text_with_ws for t in tokens[:max_tokens]])
            document_text = truncated
        doc = self.nlp(document_text)
        sentences = list(doc.sents)
        sent_texts = [sent.text for sent in sentences]
        sent_vectors = [sent.vector for sent in sentences]
        doc_keywords = self.extract_keywords(document_text, n=20)
        doc_mesh_ids = self.extract_mesh_ids(document_text)
        return {
            'mesh_ids': doc_mesh_ids,
            'keywords': doc_keywords,
            'sent_texts': sent_texts,
            'sent_vectors': sent_vectors
        }