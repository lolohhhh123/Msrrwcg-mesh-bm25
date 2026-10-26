"""
bootstrap_ci.py
对 analyze_multi_signal_ranking.py 输出的 per-query 统计做 bootstrap 置信区间。
支持：
  (1) 单 metric 的 percentile bootstrap CI
  (2) 两个 metric 的 paired bootstrap（差值 CI）
输出：
  - bootstrap_ci.json    (完整数据)
  - bootstrap_ci.txt     (论文可直接引用的文本)
  - bootstrap_ci.tex     (LaTeX 表格)
"""
import json
import argparse
import numpy as np
from pathlib import Path


# ============================================================
# 核心：bootstrap 函数
# ============================================================
def bootstrap_ci(values, n_bootstrap=10000, ci=95, seed=42):
    """
    Percentile bootstrap 置信区间。
    values: 每个 query 的 metric 值（list of float）
    返回: dict(mean, ci_lower, ci_upper, n_queries, n_bootstrap, ci_level)
    """
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[b] = np.mean(sample)
    alpha = (100 - ci) / 2
    lower = float(np.percentile(means, alpha))
    upper = float(np.percentile(means, 100 - alpha))
    return {
        "mean": float(np.mean(values)),
        "ci_lower": lower,
        "ci_upper": upper,
        "n_queries": int(n),
        "n_bootstrap": int(n_bootstrap),
        "ci_level": int(ci),
        "excludes_zero": bool(lower > 0 or upper < 0),
    }


def paired_bootstrap_diff(values_a, values_b, n_bootstrap=10000, ci=95, seed=42):
    """
    Paired bootstrap：两个 per-query metric 的差值（mean_a − mean_b）及其 CI。
    用于回答"两个 metric 是否显著不同"。
    """
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    assert len(a) == len(b), "paired bootstrap 要求两个 metric 来自同一批 query"
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(a[idx]) - np.mean(b[idx])
    alpha = (100 - ci) / 2
    lower = float(np.percentile(diffs, alpha))
    upper = float(np.percentile(diffs, 100 - alpha))
    # 双尾 p-value：差值中 <=0 或 >=0 的比例 × 2
    p_two_sided = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "ci_lower": lower,
        "ci_upper": upper,
        "p_two_sided": p_two_sided,
        "n_queries": int(n),
        "excludes_zero": bool(lower > 0 or upper < 0),
    }


# ============================================================
# 格式化输出
# ============================================================
def format_paper_string(metric, stats):
    """输出 'Spearman_BM25_Cooc: 0.0700 (95% CI: [0.0312, 0.1105], n = 50)'。"""
    return (f"{metric}: {stats['mean']:.4f} "
            f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}], "
            f"n = {stats['n_queries']} queries)")


def format_latex_table(single_results, paired_results, metrics_to_report):
    """生成可直接粘进论文的 LaTeX 表格。"""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Bootstrap 95\% confidence intervals for inter-signal rank correlations and top-$k$ overlap.}",
        r"\label{tab:bootstrap}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Metric & Mean & 95\% CI & $n$ \\",
        r"\hline",
    ]
    for m in metrics_to_report:
        if m in single_results:
            s = single_results[m]
            pretty = m.replace("_", "–")
            lines.append(
                f"{pretty} & {s['mean']:.4f} & [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}] & {s['n_queries']} \\\\"
            )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_query_stats", default="./multi_signal_analysis/per_query_stats.json")
    parser.add_argument("--output_json", default="./multi_signal_analysis/bootstrap_ci.json")
    parser.add_argument("--output_txt", default="./multi_signal_analysis/bootstrap_ci.txt")
    parser.add_argument("--output_latex", default="./multi_signal_analysis/bootstrap_ci.tex")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. 读取 per-query 数据
    with open(args.per_query_stats, "r") as f:
        per_query = json.load(f)
    print(f"Loaded {len(per_query)} metrics from {args.per_query_stats}")
    print(f"Each metric has {len(next(iter(per_query.values())))} per-query values\n")

    # 2. 单 metric bootstrap
    single_results = {}
    for metric, values in per_query.items():
        stats = bootstrap_ci(values, n_bootstrap=args.n_bootstrap, seed=args.seed)
        if stats is not None:
            single_results[metric] = stats

    # 3. Paired bootstrap（同一批 query 上两个 metric 的差值）
    paired_results = {}
    pairs_to_test = [
        ("Spearman_BM25_Cooc", "Spearman_BM25_GNN"),
        ("Spearman_BM25_Cooc", "Spearman_Cooc_GNN"),
        ("Jaccard@10_BM25_Cooc", "Jaccard@10_BM25_GNN"),
        ("Jaccard@10_BM25_Cooc", "Jaccard@10_Cooc_GNN"),
    ]
    for a, b in pairs_to_test:
        if a in per_query and b in per_query:
            diff = paired_bootstrap_diff(per_query[a], per_query[b],
                                         n_bootstrap=args.n_bootstrap, seed=args.seed)
            if diff is not None:
                paired_results[f"{a}_vs_{b}"] = diff

    # 4. 保存 JSON
    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "single_metrics": single_results,
            "paired_differences": paired_results,
        }, f, indent=2)

    # 5. 人读文本
    metrics_to_report = [
        "Spearman_BM25_Cooc",
        "Spearman_BM25_GNN",
        "Spearman_Cooc_GNN",
        "Kendall_BM25_Cooc",
        "Jaccard@5_BM25_Cooc",
        "Jaccard@10_BM25_Cooc",
        "Jaccard@20_BM25_Cooc",
        "Jaccard@50_BM25_Cooc",
        "Jaccard@10_BM25_GNN",
        "Jaccard@10_Cooc_GNN",
    ]
    txt_lines = []
    for m in metrics_to_report:
        if m in single_results:
            line = format_paper_string(m, single_results[m])
            print(line)
            txt_lines.append(line)

    # 6. Paired 差值也打印
    print("\n=== Paired bootstrap differences ===")
    for k, v in paired_results.items():
        line = (f"{k}: Δ = {v['mean_diff']:.4f} "
                f"(95% CI: [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}], "
                f"p = {v['p_two_sided']:.4f})")
        print(line)
        txt_lines.append(line)

    with open(args.output_txt, "w") as f:
        f.write("\n".join(txt_lines))

    # 7. LaTeX
    latex = format_latex_table(single_results, paired_results, metrics_to_report)
    with open(args.output_latex, "w") as f:
        f.write(latex)

    # 8. 关键结论
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    if "Spearman_BM25_Cooc" in single_results:
        s = single_results["Spearman_BM25_Cooc"]
        print(f"BM25 vs Cooc Spearman: ρ = {s['mean']:.3f} "
              f"(95% CI: [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}])")
        if s["excludes_zero"]:
            print("  → CI excludes 0: the correlation is statistically distinguishable from zero.")
        else:
            print("  → CI includes 0: the correlation is NOT statistically distinguishable from zero.")
    if "Jaccard@10_BM25_Cooc" in single_results:
        s = single_results["Jaccard@10_BM25_Cooc"]
        print(f"BM25 vs Cooc Jaccard@10: {s['mean']:.3f} "
              f"(95% CI: [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}])")
    if "Spearman_BM25_Cooc_vs_Spearman_BM25_GNN" in paired_results:
        d = paired_results["Spearman_BM25_Cooc_vs_Spearman_BM25_GNN"]
        print(f"ρ(BM25,Cooc) − ρ(BM25,GNN): Δ = {d['mean_diff']:.3f} "
              f"(95% CI: [{d['ci_lower']:.3f}, {d['ci_upper']:.3f}], p = {d['p_two_sided']:.3f})")
        if d["excludes_zero"]:
            print("  → BM25–Cooc correlation is significantly different from BM25–GNN correlation.")

    print(f"\nOutputs:")
    print(f"  {args.output_json}")
    print(f"  {args.output_txt}")
    print(f"  {args.output_latex}")


if __name__ == "__main__":
    main()