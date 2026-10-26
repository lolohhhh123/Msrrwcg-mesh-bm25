# fetch_pubmed_demo.py
import argparse
import logging
import torch
import pandas as pd
import numpy as np
import ftfy
from tqdm import tqdm
from pathlib import Path

from enhanced_data_fetcher import EnhancedPubMedFetcher
from hypergraph_builder import HypergraphBuilder
from gn_scorer_paralle import HypergraphGNNScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean garbled characters in the text"""
    if not isinstance(text, str):
        return text
    # Use ftfy to repair
    try:
        text = ftfy.fix_text(text)
    except Exception:
        pass
    # Optional: custom replacements (e.g. A帣 -> Aβ)
    # More mappings can be added here, adjusted to the actual situation
    replacements = {
        'A帣': 'Aβ',
        # Add other common garbled-character mappings
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
def get_query_embedding(query, nlp):
    doc = nlp(query)
    vecs = [t.vector for t in doc if t.has_vector]
    if vecs:
        return torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32)
    else:
        return torch.zeros(200, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Retrieve articles from PubMed and score their relevance with a trained model")
    parser.add_argument("--query", type=str, required=True, help="Search keywords, e.g. 'Alzheimer's disease'")
    parser.add_argument("--email", type=str, required=True,default="your@example.com", help="PubMed API email")
    parser.add_argument("--api_key", type=str, default="your_api_key", help="PubMed API Key")
    parser.add_argument("--max_records", type=int, default=200, help="Maximum number of articles to retrieve from PubMed (default 200)")
    parser.add_argument("--top_k", type=int, default=20, help="Show the top-K articles (default 20)")
    parser.add_argument("--sort_by", type=str, choices=['relevance', 'date'], default='relevance',
                        help="Sort criterion: 'relevance' (relevance) or 'date' (publication date - not yet wired up)")
    parser.add_argument("--show_abstract", action='store_true', help="Show abstracts (only for the first 5 articles)")
    parser.add_argument("--model_path", type=str, default="./model/best_mini_model.pth", help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size when scoring")
    args = parser.parse_args()

    # 1. Fetch articles from PubMed
    fetcher = EnhancedPubMedFetcher(email=args.email, api_key=args.api_key)
    logger.info(f"Retrieving from PubMed with keyword: {args.query}")
    df = fetcher.fetch_articles_by_keywords_paginated(
        keywords_list=[args.query],
        max_total_records=args.max_records,
        batch_size=100
    )
    if df.empty:
        logger.error("No articles retrieved")
        return
    logger.info(f"Retrieved {len(df)} articles")

    # 2. Load model and hypergraph builder
    logger.info("Loading model...")
    model = HypergraphGNNScorer(node_feat_dim=200, hidden_dim=128)
    device = torch.device(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    builder = HypergraphBuilder()
    nlp = builder.nlp

    # 3. Score each article (using the trained model)
    logger.info("Starting scoring...")
    scores = []
    # If sorting by date, parse the date field in advance (assuming df has publication_date)
    # But our fetcher may not parse the date; parsing logic can be added
    # For simplicity, we only sort by relevance; date sorting can be extended later
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        doc_text = row['full_text']
        # Build hypergraph
        hypergraph = builder.build_hypergraph(doc_text, args.query)
        node_features = hypergraph['node_features'].to(device)
        edge_index = hypergraph['edge_index'].to(device)
        query_emb = get_query_embedding(args.query, nlp).unsqueeze(0).to(device)
        with torch.no_grad():
            # The model returns logits; apply sigmoid to get the probability
            logits = model(node_features.unsqueeze(0), edge_index, query_emb, batch=None)
            prob = torch.sigmoid(logits).item()
        scores.append(prob)
    
    df['relevance_score'] = scores

    # 4. Sort
    if args.sort_by == 'relevance':
        df_sorted = df.sort_values('relevance_score', ascending=False)
    elif args.sort_by == 'date':
        # Need to parse the date; assume the date is in the 'publication_date' column with possibly varied formats
        # For simplicity, assume 'publication_date' is a string and try to parse it
        if 'publication_date' in df.columns:
            df['pub_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            df_sorted = df.sort_values('pub_date', ascending=False)
        else:
            logger.warning("Publication date field not found; falling back to relevance sorting")
            df_sorted = df.sort_values('relevance_score', ascending=False)
    else:
        df_sorted = df.sort_values('relevance_score', ascending=False)

    # 5. Display results
    top_k = min(args.top_k, len(df_sorted))
    print(f"\nTop {top_k} articles (sorted by {args.sort_by}):")
    for i in range(top_k):
        row = df_sorted.iloc[i]
        title = clean_text(row['title'])
        print(f"\n{i+1}. PMID: {row['pmid']}")
        print(f"   Title: {row['title']}")
        if i < 5 and args.show_abstract:
            abstract = clean_text(row.get('abstract', ''))
            if abstract:
                print(f"   Abstract: {abstract}")
        print(f"   Relevance score: {row['relevance_score']:.4f}")
        # If sorting by date, show the date
        if args.sort_by == 'date' and 'pub_date' in df_sorted.columns:
            print(f"   Publication date: {row['pub_date'].strftime('%Y-%m-%d') if pd.notna(row['pub_date']) else 'Unknown'}")

    # 6. Save results
    df_sorted['title_clean'] = df_sorted['title'].apply(clean_text)
    df_sorted['abstract_clean'] = df_sorted['abstract'].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    output_csv = f"pubmed_results_{args.query.replace(' ', '_')}.csv"
    df_sorted.to_csv(output_csv, index=False)
    logger.info(f"All results saved to {output_csv}")

if __name__ == "__main__":
    main()
