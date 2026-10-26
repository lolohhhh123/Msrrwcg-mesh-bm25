"""
sensitivity_analysis.py
对 2.5.4 的 3 个阈值做网格敏感性分析，验证 AI × GRN 缺口的稳健性。
复用 extract_concept_level_gaps.py 的 Step 1–7，只替换 Step 8–10。
"""
import json
import pickle
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from hypergraph_builder import HypergraphBuilder
from evaluate_baselines import clean_text


# ============================================================
# 配置
# ============================================================
CORPUS_PATH = "/home/hong/GNN/ai_ad/ai_ad_corpus_fixed.jsonl"
MINE_THE_GAP_PATH = "mine_the_gap_pairs.tsv"
COOC_PATH = "/home/hong/GNN/mesh_cooccurrence.pkl"
CACHE_PATH = "sensitivity_cache.pkl"          # 中间结果缓存，避免重复跑 Step 1-7
OUTPUT_CSV = "sensitivity_analysis.csv"
OUTPUT_JSON = "sensitivity_analysis.json"

# 目标：AI × GRN 的 MeSH IDs（从之前的对话中确认）
AI_MESH = "meshd001185"   # Artificial Intelligence
GRN_MESH = "meshd053263"  # Gene Regulatory Networks

# 阈值网格（4 × 4 × 4 = 64 种组合）
E_MIN_GRID = [1, 3, 5, 10]       # 最低 expected co-occurrence
O_MAX_GRID = [0, 1, 2, 3]        # 最高 observed co-occurrence
DOC_MIN_GRID = [1, 2, 3, 5]      # 最少 supporting documents

# 基线设置（论文里当前的阈值）
BASELINE = (3, 2, 3)


# ============================================================
# 辅助函数
# ============================================================
def normalize_name(s):
    return s.lower().replace(" ", "").replace(",", "").replace("-", "").strip()


def load_mine_the_gap(tsv_path, name_to_meshid):
    pairs = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            id1 = name_to_meshid.get(normalize_name(parts[0].strip()))
            id2 = name_to_meshid.get(normalize_name(parts[1].strip()))
            if id1 and id2:
                pairs.add(frozenset([id1, id2]))
    return pairs


# ============================================================
# Step 1–7：复用 extract_concept_level_gaps.py（带缓存）
# ============================================================
def build_candidate_pool():
    """
    复用 extract_concept_level_gaps.py 的 Step 1–7，
    返回未过滤的全量候选池 gap_doc_support / gap_query_support，
    以及每个 pair 的 (expected, observed) 统计。
    """
    if Path(CACHE_PATH).exists():
        print(f"Loading cached pool from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    # ---- Step 1: 加载 corpus ----
    print("Step 1: Loading corpus...")
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    pmids = [d["pmid"] for d in docs]
    doc_mesh_list = [d["mesh"] for d in docs]
    N = len(docs)

    # ---- Step 2: 构建共现矩阵 ----
    print("Step 2: Building co-occurrence matrix...")
    mesh_sets = [set(m) for m in doc_mesh_list]
    vocab = sorted({m for s in mesh_sets for m in s})
    mesh_to_idx = {m: i for i, m in enumerate(vocab)}
    V = len(vocab)
    cooc = np.zeros((V, V), dtype=np.int32)
    freq = np.zeros(V, dtype=np.int32)
    for s in mesh_sets:
        idxs = [mesh_to_idx[m] for m in s]
        for i in idxs:
            freq[i] += 1
            for j in idxs:
                cooc[i, j] += 1

    # ---- Step 3: name→ID 映射 ----
    print("Step 3: Building name -> ID mapping...")
    builder = HypergraphBuilder(cooccurrence_path=COOC_PATH)
    name_to_meshid = {}
    for d_id, name in builder.mesh_translator.mapping.items():
        if not isinstance(d_id, str) or not d_id.startswith("D"):
            continue
        meshd_id = "meshd" + d_id[1:].lower()   # D000544 -> meshd000544
        name_to_meshid[normalize_name(name)] = meshd_id
    print(f"  Built {len(name_to_meshid)} name -> ID entries")

    # ---- Step 4: 加载 Mine the Gap! 25 pairs ----
    print("Step 4: Loading Mine the Gap! pairs...")
    mine_the_gap_set = load_mine_the_gap(MINE_THE_GAP_PATH, name_to_meshid)

    # ---- Step 5: 构建 BM25 ----
    print("Step 5: Building BM25...")
    tokenized = [clean_text(d["title"] + " " + d["abstract"]).lower().split()
                 for d in docs]
    bm25 = BM25Okapi(tokenized)

    # ---- Step 6: cooc_scores 函数 ----
    def cooc_scores(q_mesh):
        q_idx = [mesh_to_idx[m] for m in q_mesh if m in mesh_to_idx]
        if not q_idx:
            return np.zeros(len(docs))
        scores = np.zeros(len(docs))
        for k, d_mesh in enumerate(doc_mesh_list):
            d_idx = [mesh_to_idx[m] for m in d_mesh if m in mesh_to_idx]
            if not d_idx:
                continue
            overlap = len(set(q_idx) & set(d_idx))
            scores[k] = overlap / (len(q_idx) + 1e-8)
        return scores

    # ---- Step 7: 收集所有 BM25-only 文档里的 MeSH pair（不过滤！）----
    print("Step 7: Extracting candidate pairs from BM25-only documents...")
    top_k = 20
    queries = [d for d in docs if len(d["mesh"]) >= 3][:50]
    gap_doc_support = defaultdict(set)
    gap_query_support = defaultdict(set)

    for qi, q in enumerate(tqdm(queries, desc="Extracting")):
        q_mesh = q["mesh"]
        bm25_scores = bm25.get_scores(clean_text(q["title"]).lower().split())
        bm25_top = set(np.argsort(bm25_scores)[-top_k:])
        cooc_s = cooc_scores(q_mesh)
        cooc_top = set(np.argsort(cooc_s)[-top_k:])
        bm25_only = bm25_top - cooc_top

        for doc_idx in bm25_only:
            doc_mesh = doc_mesh_list[doc_idx]
            doc_pmid = pmids[doc_idx]
            for qm in q_mesh:
                for dm in doc_mesh:
                    if qm == dm:
                        continue
                    pair = frozenset([qm, dm])
                    gap_doc_support[pair].add(doc_pmid)
                    gap_query_support[pair].add(qi)

    print(f"Collected {len(gap_doc_support)} candidate pairs (before threshold filtering)")

    # ---- 预先计算每个 pair 的 (expected, observed) ----
    print("Precomputing (expected, observed) for all pairs...")
    pair_stats = {}
    for pair in tqdm(gap_doc_support, desc="Precomputing"):
        pair_t = tuple(sorted(pair))
        if len(pair_t) != 2:
            continue
        i, j = mesh_to_idx[pair_t[0]], mesh_to_idx[pair_t[1]]
        expected = float(freq[i] * freq[j] / N)
        observed = int(cooc[i, j])
        pair_stats[pair] = {"expected": expected, "observed": observed}

    cache = {
        "gap_doc_support": gap_doc_support,
        "gap_query_support": gap_query_support,
        "pair_stats": pair_stats,
        "mesh_to_idx": mesh_to_idx,
        "mine_the_gap_set": mine_the_gap_set,
        "builder_mapping": builder.mesh_translator.mapping,
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"Cached to {CACHE_PATH}")
    return cache


# ============================================================
# Step 8：敏感性分析（64 组阈值）
# ============================================================
def run_sensitivity_analysis(cache):
    gap_doc_support = cache["gap_doc_support"]
    gap_query_support = cache["gap_query_support"]
    pair_stats = cache["pair_stats"]
    mine_the_gap_set = cache["mine_the_gap_set"]
    builder_mapping = cache["builder_mapping"]

    ai_grn_key = frozenset([AI_MESH, GRN_MESH])

    results = []
    for E_min, O_max, doc_min in itertools.product(E_MIN_GRID, O_MAX_GRID, DOC_MIN_GRID):
        # 过滤
        filtered = []
        for pair, docs in gap_doc_support.items():
            if len(pair) != 2:
                continue
            stats = pair_stats.get(pair)
            if stats is None:
                continue
            if stats["expected"] < E_min:
                continue
            if stats["observed"] > O_max:
                continue
            if len(docs) < doc_min:
                continue
            filtered.append({
                "pair": pair,
                "expected": stats["expected"],
                "observed": stats["observed"],
                "n_docs": len(docs),
                "n_queries": len(gap_query_support[pair]),
                "in_mtg": pair in mine_the_gap_set,
            })

        # 排序（按 expected 降序）
        filtered.sort(key=lambda x: -x["expected"])

        # AI × GRN 的位置
        ai_grn_rank = None
        for rank, item in enumerate(filtered, start=1):
            if item["pair"] == ai_grn_key:
                ai_grn_rank = rank
                break

        n_total = len(filtered)
        n_confirmed = sum(1 for x in filtered if x["in_mtg"])
        n_new = n_total - n_confirmed

        results.append({
            "E_min": E_min,
            "O_max": O_max,
            "doc_min": doc_min,
            "n_total": n_total,
            "n_confirmed": n_confirmed,
            "n_new": n_new,
            "ai_grn_present": ai_grn_rank is not None,
            "ai_grn_rank": ai_grn_rank,
            "is_baseline": (E_min, O_max, doc_min) == BASELINE,
        })

    return results


# ============================================================
# Step 9–10：输出与摘要
# ============================================================
def summarize(results):
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull results saved to {OUTPUT_CSV}")

    # 保存 JSON（只保留 AI × GRN 相关信息 + 基线）
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "n_combinations": len(results),
            "baseline": BASELINE,
            "ai_grn_appears_in": int(df["ai_grn_present"].sum()),
            "ai_grn_rank_distribution": df["ai_grn_rank"].dropna().astype(int).tolist(),
            "n_new_gaps_range": [int(df["n_new"].min()), int(df["n_new"].max())],
            "n_total_gaps_range": [int(df["n_total"].min()), int(df["n_total"].max())],
        }, f, indent=2)
    print(f"Summary saved to {OUTPUT_JSON}")

    # 终端打印
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total threshold combinations: {len(results)}")
    print(f"AI × GRN present in: {df['ai_grn_present'].sum()} / {len(results)} combinations")
    print(f"AI × GRN rank range: {df['ai_grn_rank'].min():.0f} – {df['ai_grn_rank'].max():.0f}")
    print(f"Total gaps range: {df['n_total'].min()} – {df['n_total'].max()}")
    print(f"New gaps range: {df['n_new'].min()} – {df['n_new'].max()}")

    print(f"\n=== Baseline ({BASELINE[0]}, {BASELINE[1]}, {BASELINE[2]}) ===")
    baseline_row = df[df["is_baseline"]].iloc[0]
    print(f"  Total gaps: {baseline_row['n_total']}")
    print(f"  Confirmed:  {baseline_row['n_confirmed']}")
    print(f"  New:        {baseline_row['n_new']}")
    print(f"  AI × GRN:   present={baseline_row['ai_grn_present']}, rank={baseline_row['ai_grn_rank']}")

    print(f"\n=== All combinations where AI × GRN is present ===")
    present = df[df["ai_grn_present"]].sort_values("ai_grn_rank")
    for _, row in present.iterrows():
        print(f"  E≥{row['E_min']:2d}, O≤{row['O_max']}, docs≥{row['doc_min']}: "
              f"rank={row['ai_grn_rank']:3.0f}, total={row['n_total']:3d}, new={row['n_new']:3d}")


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("Building candidate pool (Step 1–7, will be cached)...")
    print("=" * 70)
    cache = build_candidate_pool()

    print("\n" + "=" * 70)
    print("Running sensitivity analysis (64 threshold combinations)...")
    print("=" * 70)
    results = run_sensitivity_analysis(cache)

    print("\n" + "=" * 70)
    print("Summarizing...")
    print("=" * 70)
    summarize(results)


if __name__ == "__main__":
    main()