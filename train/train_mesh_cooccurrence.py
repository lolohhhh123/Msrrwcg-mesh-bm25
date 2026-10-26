"""
train_mesh_cooccurrence.py
Builds the MeSH co-occurrence matrix using the full training set,
and evaluates its retrieval performance on the test set.
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
    """Build the MeSH co-occurrence matrix"""
    print("Extracting MeSH sets from the training set...")
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
    print(f"MeSH vocabulary size: {vocab_size}")

    # Count co-occurrences
    print("Computing MeSH co-occurrences...")
    cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    rel_pairs = dataset.pairs
    for _, row in tqdm(rel_pairs.iterrows(), total=len(rel_pairs), desc="Processing pairs"):
        if row['Relevance'] < 1:
            continue
        p1, p2 = row['PMID1'], row['PMID2']
        mesh1 = pmid_to_mesh.get(p1, set())
        mesh2 = pmid_to_mesh.get(p2, set())
        # Only count MeSH terms shared by both documents (intersection)
        common = mesh1.intersection(mesh2)
        for m in common:
            i = mesh_to_idx[m]
            for n in common:
                j = mesh_to_idx[n]
                cooccurrence[i, j] += 1
    
    # Symmetric normalization (optional)
    # No normalization here; keep the original frequencies
    print(f"Co-occurrence matrix construction complete, number of non-zero elements: {np.count_nonzero(cooccurrence)}")
    return mesh_vocab, mesh_to_idx, cooccurrence

def score_query_document(q_mesh_set, doc_mesh_set, cooccurrence, mesh_to_idx):
    """Compute the relevance score between a query and a document (based on co-occurrence weights)"""
    common = q_mesh_set.intersection(doc_mesh_set)
    if not common:
        return 0.0
    total_score = 0.0
    for m in common:
        idx = mesh_to_idx[m]
        # Use the sum of co-occurrences between all MeSH terms in the query
        # and all MeSH terms in the document.
        # Simplified here as: the average co-occurrence between query MeSH and document MeSH.
        for n in doc_mesh_set:
            if n in mesh_to_idx:
                total_score += cooccurrence[idx, mesh_to_idx[n]]
    return total_score / (len(q_mesh_set) * len(doc_mesh_set))  # normalization

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", default="./train_annotated.npy")
    parser.add_argument("--relevance_path", default="./relevance_matrix.tsv")
    parser.add_argument("--mesh_xml", default="./desc2026.xml")
    args = parser.parse_args()

    # Load training set
    print("Loading training set...")
    train_dataset = RELISHDataset(
        npy_path=args.npy_path,
        relevance_path=args.relevance_path,
        mesh_xml_path=Path(args.mesh_xml),
        split='train',
        normalize_labels=False
    )
    builder = HypergraphBuilder()
    mesh_vocab, mesh_to_idx, cooccurrence = build_cooccurrence_matrix(train_dataset, builder)

    # Save the co-occurrence matrix for evaluation use
    with open("mesh_cooccurrence.pkl", "wb") as f:
        pickle.dump({
            'mesh_vocab': mesh_vocab,
            'mesh_to_idx': mesh_to_idx,
            'cooccurrence': cooccurrence
        }, f)
    print("Co-occurrence matrix saved to mesh_cooccurrence.pkl")

if __name__ == "__main__":
    main()
