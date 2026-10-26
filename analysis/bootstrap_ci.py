"""
bootstrap_ci.py
Computes bootstrap confidence intervals for the per-query statistics
output by analyze_multi_signal_ranking.py.
Supports:
  (1) Percentile bootstrap CI for a single metric
  (2) Paired bootstrap for two metrics (difference CI)
Outputs:
  - bootstrap_ci.json    (full data)
  - bootstrap_ci.txt     (paper-ready text)
  - bootstrap_ci.tex     (LaTeX table)
"""
import json
import argparse
import numpy as np
from pathlib import Path


# ============================================================
# Core: bootstrap functions
# ============================================================
def bootstrap_ci(values, n_bootstrap=10000, ci=95, seed=42):
    """
    Percentile bootstrap confidence interval.
    values: per-query metric values (list of float)
    Returns: dict(mean, ci_lower, ci_upper, n_queries, n_bootstrap, ci_level)
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
    Paired bootstrap: the difference between two per-query metrics
    (mean_a − mean_b) and its CI.
    Used to answer "are the two metrics significantly different?".
    """
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    assert len(a) == len(b), "paired bootstrap requires both metrics to come from the same set of queries"
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
    # Two-sided p-value: proportion of diffs <=0 or >=0, times 2
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
# Formatting output
# ============================================================
def format_paper_string(metric, stats):
    """Produces 'Spearman_BM25_Cooc: 0.0700 (95% CI: [0.0312, 0.1105], n = 50)'."""
    return (f"{metric}: {stats['mean']:.4f} "
            f"(95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}], "
            f"n = {stats['n_queries']} queries)")


def format_latex_table(single_results, paired_results, metrics_to_report):
    """Generates a LaTeX table that can be pasted directly into a paper."""
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
# Main flow
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

    # 1. Load per-query data
    with open(args.per_query_stats, "r") as f:
        per_query = json.load(f)
    print(f"Loaded {len(per_query)} metrics from {args.per_query_stats}")
    print(f"Each metric has {len(next(iter(per_query.values())))} per-query values\n")

    # 2. Single-metric bootstrap
    single_results = {}
    for metric, values in per_query.items():
        stats = bootstrap_ci(values, n_bootstrap=args.n_bootstrap, seed=args.seed)
        if stats is not None:
            single_results[metric] = stats

    # 3. Paired bootstrap (difference between two metrics on the same set of queries)
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

    # 4. Save JSON
    out_dir = Path(args.output_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "single_metrics": single_results,
            "paired_differences": paired_results,
        }, f, indent=2)

    # 5. Human-readable text
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

    # 6. Also print paired differences
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

    # 8. Key findings
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
