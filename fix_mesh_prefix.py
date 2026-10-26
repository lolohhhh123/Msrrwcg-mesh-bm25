"""
fix_mesh_prefix.py
把 meshddXXXXXX 改成 meshdXXXXXX
"""
import json

input_path = "/home/hong/GNN/ai_ad/ai_ad_corpus_dedup.jsonl"
output_path = "/home/hong/GNN/ai_ad/ai_ad_corpus_fixed.jsonl"

n_fixed = 0
with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        d = json.loads(line)
        new_mesh = []
        for m in d["mesh"]:
            if m.startswith("meshdd"):
                m = "meshd" + m[6:]   # 去掉多余的 d
                n_fixed += 1
            new_mesh.append(m)
        d["mesh"] = new_mesh
        fout.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"Fixed {n_fixed} MeSH entries")
print(f"Output: {output_path}")