"""
extract_concept_level_gaps.py
从三路 ranking 的 divergence 提取 concept-level MeSH pair gaps，
并与 Mine the Gap! 的 25 个缺口对比。
"""
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from hypergraph_builder import HypergraphBuilder
from evaluate_baselines import clean_text


def normalize_name(s):
    """MeSH 名称标准化，用于字符串对比"""
    return s.lower().replace(" ", "").replace(",", "").replace("-", "").strip()


def load_mine_the_gap(tsv_path, name_to_meshid):
    """
    读取 Mine the Gap! 的 25 个 pair，
    用 name_to_meshid 映射把名称转成 meshd ID。
    返回 set of frozenset({meshd_i, meshd_j})。
    """
    pairs = set()
    unmapped = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            name1, name2 = parts[0].strip(), parts[1].strip()

            id1 = name_to_meshid.get(normalize_name(name1))
            id2 = name_to_meshid.get(normalize_name(name2))

            if id1 and id2:
                pairs.add(frozenset([id1, id2]))
            else:
                unmapped.append((name1, name2, id1, id2))
    return pairs, unmapped


def main():
    corpus_path = "/home/hong/GNN/ai_ad/ai_ad_corpus_fixed.jsonl"
    mine_the_gap_path = "mine_the_gap_pairs.tsv"
    output_path = "concept_level_gaps.json"
    cooc_path = "/home/hong/GNN/mesh_cooccurrence.pkl"

    top_k = 20
    min_expected = 3.0        # 放宽到 3.0，抓更多候选
    max_observed = 2
    min_doc_freq = 3

    # 1. 加载语料
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    pmids = [d["pmid"] for d in docs]
    doc_mesh_list = [d["mesh"] for d in docs]
    N = len(docs)

    # 2. 构建 MeSH 词表和共现矩阵
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

    # 3. 建 name_to_meshid 映射（用于 Mine the Gap! 对比）
    print("Building MeSH name -> ID mapping from desc2026.xml...")
    builder = HypergraphBuilder(cooccurrence_path=cooc_path)
    name_to_meshid = {}
    for meshd_id, name in builder.mesh_translator.mapping.items():
        # meshd_id 形如 "MESHD000544" 或 "meshd000544"
        if not isinstance(meshd_id, str):
            continue
        mid = meshd_id.lower()
        if mid.startswith("meshd"):
            name_to_meshid[normalize_name(name)] = mid
    print(f"  Built {len(name_to_meshid)} name -> ID entries")

    # 4. 加载 Mine the Gap! 的 25 个 pair
    mine_the_gap_set, unmapped = load_mine_the_gap(mine_the_gap_path, name_to_meshid)
    print(f"Mine the Gap! pairs loaded: {len(mine_the_gap_set)}")
    if unmapped:
        print(f"  WARNING: {len(unmapped)} pairs could not be mapped:")
        for n1, n2, i1, i2 in unmapped[:5]:
            print(f"    '{n1}' × '{n2}' -> {i1}, {i2}")

    # 5. BM25
    tokenized = [clean_text(d["title"] + " " + d["abstract"]).lower().split()
                 for d in docs]
    bm25 = BM25Okapi(tokenized)

    # 6. Cooc 打分
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

    # 7. 对每个 query，找 BM25-only 文档中的 MeSH pair
    queries = [d for d in docs if len(d["mesh"]) >= 3][:50]
    gap_doc_support = defaultdict(set)
    gap_query_support = defaultdict(set)

    for qi, q in enumerate(tqdm(queries, desc="Extracting gaps")):
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

    # 8. 过滤
    candidate_gaps = []
    for pair, doc_set in gap_doc_support.items():
        pair = tuple(sorted(pair))
        if len(pair) != 2:
            continue
        i, j = mesh_to_idx[pair[0]], mesh_to_idx[pair[1]]
        f_i, f_j = freq[i], freq[j]
        expected = f_i * f_j / N
        observed = cooc[i, j]

        if expected >= min_expected and observed <= max_observed \
           and len(doc_set) >= min_doc_freq:
            # 查是否在 Mine the Gap! 结果中
            pair_key = frozenset([pair[0], pair[1]])
            in_mtg = pair_key in mine_the_gap_set

            # 翻译 MeSH 名称（用 builder 的 mapping）
            name1 = builder.mesh_translator.mapping.get(pair[0].upper(), pair[0])
            name2 = builder.mesh_translator.mapping.get(pair[1].upper(), pair[1])

            candidate_gaps.append({
                "mesh_id_1": pair[0],
                "mesh_id_2": pair[1],
                "mesh_name_1": name1,
                "mesh_name_2": name2,
                "expected_cooc": float(expected),
                "observed_cooc": int(observed),
                "n_supporting_docs": len(doc_set),
                "n_supporting_queries": len(gap_query_support[pair_key]),
                "in_mine_the_gap": in_mtg,
                "supporting_pmids": sorted(doc_set)[:5],
            })

    candidate_gaps.sort(key=lambda x: -x["expected_cooc"])

    # 9. 输出
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidate_gaps, f, indent=2, ensure_ascii=False)

    # 10. 统计
    total = len(candidate_gaps)
    in_mtg = sum(1 for g in candidate_gaps if g["in_mine_the_gap"])
    new_gaps = total - in_mtg

    print(f"\n=== Summary ===")
    print(f"Total concept-level gaps identified: {total}")
    print(f"Already in Mine the Gap! (25 pairs): {in_mtg}")
    print(f"NEW gaps (Mine the Gap! missed)    : {new_gaps}")

    print(f"\n=== Top 15 NEW concept-level gaps ===")
    for g in candidate_gaps:
        if not g["in_mine_the_gap"]:
            print(f"  {g['mesh_name_1']} × {g['mesh_name_2']}")
            print(f"    E={g['expected_cooc']:.1f}, O={g['observed_cooc']}, "
                  f"docs={g['n_supporting_docs']}, queries={g['n_supporting_queries']}")

    print(f"\n=== Top 10 confirmed by Mine the Gap! ===")
    confirmed = [g for g in candidate_gaps if g["in_mine_the_gap"]]
    for g in confirmed[:10]:
        print(f"  {g['mesh_name_1']} × {g['mesh_name_2']} "
              f"(E={g['expected_cooc']:.1f}, docs={g['n_supporting_docs']})")


if __name__ == "__main__":
    main()