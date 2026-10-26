"""
translate_and_compare.py
Translates the meshd IDs in concept_level_gaps.json into MeSH names,
compares them against the 25 pairs from Mine the Gap!, and outputs the full table.
"""
import json
from pathlib import Path
from mesh_translator import MeshTranslator


def normalize(s):
    """Normalize a MeSH name for string comparison"""
    return s.lower().replace(" ", "").replace(",", "").replace("-", "").strip()


def main():
    gaps_path = "concept_level_gaps.json"
    mtg_tsv_path = "mine_the_gap_pairs.tsv"
    output_path = "concept_level_gaps_final.json"

    # ---------- 1. Load the mapping from desc2026.xml ----------
    print("Loading MeshTranslator...")
    mt = MeshTranslator(Path("./desc2026.xml"))
    mapping = mt.mapping
    print(f"  Mapping size: {len(mapping)}")
    print(f"  Sample: {list(mapping.items())[:2]}")

    # ---------- 2. Build bidirectional mappings ----------
    d_to_meshd = {}
    meshd_to_name = {}
    name_to_meshd = {}

    for d_id, name in mapping.items():
        if not isinstance(d_id, str) or not isinstance(name, str):
            continue
        d_id = d_id.strip()
        if len(d_id) < 2 or d_id[0] not in ("D", "C"):
            continue

        # Key fix: strip the leading D/C, then prepend "meshd"
        # D000368 -> meshd000368
        # C0011914 -> meshc0011914
        prefix = "meshd" if d_id[0] == "D" else "meshc"
        meshd_id = prefix + d_id[1:].lower()

        d_to_meshd[d_id] = meshd_id
        meshd_to_name[meshd_id] = name
        name_to_meshd[normalize(name)] = meshd_id

    print(f"  meshd_to_name entries: {len(meshd_to_name)}")
    print(f"  name_to_meshd entries: {len(name_to_meshd)}")
    # Verify it
    print(f"  Sanity check: D000368 -> {d_to_meshd.get('D000368')} -> "
          f"{meshd_to_name.get('meshd000368')}")
    print(f"  Sanity check: D000544 -> {d_to_meshd.get('D000544')} -> "
          f"{meshd_to_name.get('meshd000544')}")
    # ---------- 3. Load the 25 pairs from Mine the Gap! ----------
    mtg_pairs = set()       # set of frozenset({meshd_id1, meshd_id2})
    mtg_unmapped = []

    with open(mtg_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            n1, n2 = parts[0].strip(), parts[1].strip()

            id1 = name_to_meshd.get(normalize(n1))
            id2 = name_to_meshd.get(normalize(n2))

            if id1 and id2:
                mtg_pairs.add(frozenset([id1, id2]))
            else:
                mtg_unmapped.append((n1, n2, id1, id2))

    print(f"\nMine the Gap! pairs successfully mapped: {len(mtg_pairs)} / 25")
    if mtg_unmapped:
        print(f"  Unmapped ({len(mtg_unmapped)}):")
        for n1, n2, i1, i2 in mtg_unmapped:
            print(f"    '{n1}' × '{n2}' -> {i1}, {i2}")

    # ---------- 4. Load the 76 gaps and translate ----------
    with open(gaps_path, "r", encoding="utf-8") as f:
        gaps = json.load(f)
    print(f"\nLoaded {len(gaps)} candidate gaps")

    for g in gaps:
        id1, id2 = g["mesh_id_1"], g["mesh_id_2"]
        g["mesh_name_1"] = meshd_to_name.get(id1, f"[UNMAPPED:{id1}]")
        g["mesh_name_2"] = meshd_to_name.get(id2, f"[UNMAPPED:{id2}]")
        pair_key = frozenset([id1, id2])
        g["in_mine_the_gap"] = pair_key in mtg_pairs

    # ---------- 5. Output ----------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2, ensure_ascii=False)

    # ---------- 6. Print statistics ----------
    total = len(gaps)
    confirmed = [g for g in gaps if g["in_mine_the_gap"]]
    new_gaps = [g for g in gaps if not g["in_mine_the_gap"]]

    print(f"\n{'='*70}")
    print(f"Total candidate gaps          : {total}")
    print(f"Confirmed by Mine the Gap!    : {len(confirmed)}")
    print(f"NEW gaps (Mine the Gap missed): {len(new_gaps)}")
    print(f"{'='*70}")

    print(f"\n=== Top 15 NEW concept-level gaps ===")
    for g in new_gaps[:15]:
        print(f"  {g['mesh_name_1']} × {g['mesh_name_2']}")
        print(f"    E={g['expected_cooc']:.1f}, O={g['observed_cooc']}, "
              f"docs={g['n_supporting_docs']}, queries={g['n_supporting_queries']}")

    print(f"\n=== Confirmed by Mine the Gap! ===")
    for g in confirmed:
        print(f"  {g['mesh_name_1']} × {g['mesh_name_2']} "
              f"(E={g['expected_cooc']:.1f}, O={g['observed_cooc']}, "
              f"docs={g['n_supporting_docs']})")

    # ---------- 7. Output a summary CSV table ----------
    import csv
    csv_path = "concept_level_gaps_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["MeSH 1", "MeSH 2", "Expected", "Observed",
                         "Support Docs", "Support Queries",
                         "In Mine the Gap!"])
        for g in gaps:
            writer.writerow([
                g["mesh_name_1"], g["mesh_name_2"],
                f"{g['expected_cooc']:.1f}", g["observed_cooc"],
                g["n_supporting_docs"], g["n_supporting_queries"],
                "YES" if g["in_mine_the_gap"] else "NO",
            ])
    print(f"\nCSV table saved to: {csv_path}")
    print(f"Full JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
