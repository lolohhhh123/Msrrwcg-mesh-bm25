"""
precompute_ai_ad_features.py
Precomputes hypergraph features for the AI+AD corpus.
MeSH node features are initialized using the full-PubMed co-occurrence matrix,
and each document's own title is used as the query.
"""

import json
import pickle
from pathlib import Path
from tqdm import tqdm
from hypergraph_builder import HypergraphBuilder

def main():
    corpus_path = "./ai_ad/ai_ad_corpus_fixed.jsonl"  # change to your actual path
    out_path = "./ai_ad/ai_ad_doc_features.pkl"       # change to your actual path
    cooc_path = "./mesh_cooccurrence.pkl"             # change to your actual path

    builder = HypergraphBuilder(cooccurrence_path=cooc_path)

    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"Loaded {len(docs)} documents.")

    doc_features = {}
    skipped = 0

    for d in tqdm(docs, desc="Building hypergraphs"):
        pmid = d["pmid"]
        text = (d.get("title") or "") + " " + (d.get("abstract") or "")
        query_text = d.get("title") or ""

        if not text.strip():
            # Store an empty hypergraph so that the pmid is never missing
            import torch
            doc_features[pmid] = {
            "node_features": torch.zeros((1, 200)),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            }
            continue

        try:
            hg = builder.build_hypergraph(
                document_text=text,
                query_text=query_text,
                include_mesh=True
            )
            doc_features[pmid] = {
                "node_features": hg["node_features"],
                "edge_index": hg["edge_index"],
            }
        except Exception as e:
            print(f"Skip {pmid}: {e}")
            import torch
            doc_features[pmid] = {
                "node_features": torch.zeros((1, 200)),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            }

    with open(out_path, "wb") as f:
        pickle.dump(doc_features, f)

    print(f"\nSaved {len(doc_features)} doc features to {out_path}")
    print(f"Skipped: {skipped}")

if __name__ == "__main__":
    main()
