# gn_scorer_paralle.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class HypergraphGNNScorer(nn.Module):
    def __init__(self, node_feat_dim=200, hidden_dim=256, num_classes=1):
        super(HypergraphGNNScorer, self).__init__()
        
        # 2 layer MLP
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 2 layer MLP
        self.query_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 4 layer MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.set_deterministic()

    def set_deterministic(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, node_features, hyperedge_index, query_embedding, batch=None):
        if node_features.dim() == 3:
            node_features = node_features.squeeze(0)
        
        x = self.node_proj(node_features)  # [num_nodes, hidden_dim]
        
        if batch is not None:
            doc_embedding = global_mean_pool(x, batch)
        else:
            doc_embedding = torch.mean(x, dim=0, keepdim=True)
        
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        query_embedding = self.query_proj(query_embedding)
        
        if doc_embedding.size(0) == 1 and query_embedding.size(0) > 1:
            doc_embedding = doc_embedding.repeat(query_embedding.size(0), 1)
        elif doc_embedding.size(0) != query_embedding.size(0):
            raise ValueError(f"Batch size mismatch: doc {doc_embedding.size(0)} vs query {query_embedding.size(0)}")
        
        combined_feat = torch.cat([query_embedding, doc_embedding], dim=-1)
        scores = self.score_mlp(combined_feat)
        return scores
    
    def get_document_embedding(self, node_features, hyperedge_index, batch=None):
        if node_features.dim() == 3:
            if batch is None:
                batch = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
            node_features = node_features.view(-1, node_features.size(-1))
        x = self.node_proj(node_features)
        if batch is not None:
            doc_embedding = global_mean_pool(x, batch)
        else:
            doc_embedding = torch.mean(x, dim=0, keepdim=True)
        return doc_embedding
