# relish_dataset.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from mesh_translator import MeshTranslator
from tqdm import tqdm   # 用于显示进度条

class RELISHDataset(Dataset):
    def __init__(self, 
                 npy_path: str, 
                 relevance_path: str,
                 mesh_xml_path: Path,
                 split: str = 'train',
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 normalize_labels: bool = False):
        """
        Args:
            npy_path: .npy文件路径（例如 train_annotated.npy）
            relevance_path: relevance_matrix.tsv 路径
            mesh_xml_path: MeSH XML 文件路径（用于构建翻译器）
            split: 'train', 'val', 'test'
            test_size, val_size, random_state: 用于划分数据集
            normalize_labels: 是否将标签归一化到 [0,1]
        """
        # 初始化 MeSH 翻译器
        self.translator = MeshTranslator(mesh_xml_path)

        # 1. 加载文档token数据
        data = np.load(npy_path, allow_pickle=True)
        self.pmid_to_doc = {}
        print("正在加载并翻译文档...")
        for row in tqdm(data, desc="处理文档"):
            pmid = int(row[0])
            title_tokens = row[1]
            abstract_tokens = row[2]
            # 原始文本（含 MeSH ID）
            title_raw = ' '.join(title_tokens)
            abstract_raw = ' '.join(abstract_tokens)
            full_raw = f"{title_raw}. {abstract_raw}"

            # 一次性翻译成可读文本（仅一次，避免后续重复）
            title_readable = self.translator.replace_mesh_ids_in_text(title_raw)
            abstract_readable = self.translator.replace_mesh_ids_in_text(abstract_raw)
            full_readable = f"{title_readable}. {abstract_readable}"

            self.pmid_to_doc[pmid] = {
                'title_raw': title_raw,
                'abstract_raw': abstract_raw,
                'full_raw': full_raw,
                'title_readable': title_readable,
                'abstract_readable': abstract_readable,
                'full_readable': full_readable
            }

        # 2. 加载relevance矩阵
        rel_df = pd.read_csv(relevance_path, sep='\t')
        valid_mask = rel_df['PMID1'].isin(self.pmid_to_doc.keys()) & \
                     rel_df['PMID2'].isin(self.pmid_to_doc.keys())
        rel_df = rel_df[valid_mask].reset_index(drop=True)

        # 3. 划分数据集（按PMID划分，确保文档对不跨集）
        pmids_unique = list(set(rel_df['PMID1']).union(set(rel_df['PMID2'])))
        train_pmids, temp_pmids = train_test_split(pmids_unique, test_size=test_size+val_size, random_state=random_state)
        val_pmids, test_pmids = train_test_split(temp_pmids, test_size=test_size/(test_size+val_size), random_state=random_state)

        if split == 'train':
            keep_pmids = set(train_pmids)
        elif split == 'val':
            keep_pmids = set(val_pmids)
        elif split == 'test':
            keep_pmids = set(test_pmids)
        else:
            raise ValueError("split must be train/val/test")

        mask = rel_df['PMID1'].isin(keep_pmids) & rel_df['PMID2'].isin(keep_pmids)
        self.pairs = rel_df[mask].reset_index(drop=True)

        # 4. 标签处理
        raw_labels = self.pairs['Relevance'].values.astype(float)
        if normalize_labels:
            self.labels = raw_labels / 2.0
        else:
            self.labels = raw_labels

        print(f"{split} set: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        pmid1 = row['PMID1']
        pmid2 = row['PMID2']
        label = self.labels[idx]

        # 直接使用预处理好的可读文本
        doc1 = self.pmid_to_doc[pmid1]
        doc2 = self.pmid_to_doc[pmid2]

        query_text = doc1['full_readable']
        doc_text = doc2['full_readable']

        return {
            'query': query_text,
            'document': doc_text,
            'label': torch.tensor(label, dtype=torch.float32)
        }