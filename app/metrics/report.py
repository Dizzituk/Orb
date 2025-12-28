# FILE: app/metrics/report.py
"""
Metrics Report Generator

Aggregates job metrics and generates cost/accuracy reports.

Usage:
    python -m app.metrics.report --days 7
    python -m app.metrics.report --date 2024-12-28
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from collections import defaultdict


def load_metrics(metrics_dir: Path, days: int = 7) -> List[Dict[str, Any]]:
    """Load metrics from JSONL files."""
    metrics = []
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    for filepath in sorted(metrics_dir.glob("metrics_*.jsonl")):
        # Extract date from filename
        date_str = filepath.stem.replace("metrics_", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                continue
        except ValueError:
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return metrics


def generate_report(metrics: List[Dict[str, Any]]) -> str:
    """Generate a summary report."""
    if not metrics:
        return "No metrics data found."
    
    lines = []
    lines.append("=" * 70)
    lines.append("ORB PIPELINE METRICS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall stats
    total_jobs = len(metrics)
    successful_jobs = sum(1 for m in metrics if m.get("success"))
    total_cost = sum(m.get("total_cost_usd", 0) for m in metrics)
    
    lines.append(f"Period: {metrics[0].get('started_at', 'N/A')[:10]} to {metrics[-1].get('started_at', 'N/A')[:10]}")
    lines.append(f"Total Jobs: {total_jobs}")
    lines.append(f"Successful: {successful_jobs} ({100*successful_jobs/total_jobs:.1f}%)")
    lines.append(f"Total Cost: ${total_cost:.4f}")
    lines.append(f"Avg Cost/Job: ${total_cost/total_jobs:.4f}")
    lines.append("")
    
    # Critique loop analysis
    lines.append("-" * 70)
    lines.append("CRITIQUE LOOP ANALYSIS")
    lines.append("-" * 70)
    
    iterations = [m.get("total_critique_iterations", 0) for m in metrics]
    critique_passed = sum(1 for m in metrics if m.get("critique_passed"))
    
    lines.append(f"Jobs with critique pass: {critique_passed}/{total_jobs}")
    lines.append(f"Avg iterations to pass: {sum(iterations)/len(iterations):.2f}")
    lines.append(f"Max iterations: {max(iterations)}")
    
    # Distribution
    iter_dist = defaultdict(int)
    for i in iterations:
        iter_dist[i] += 1
    lines.append("Iteration distribution:")
    for i in sorted(iter_dist.keys()):
        pct = 100 * iter_dist[i] / total_jobs
        bar = "█" * int(pct / 2)
        lines.append(f"  {i} iterations: {iter_dist[i]:3d} ({pct:5.1f}%) {bar}")
    lines.append("")
    
    # Stage 3 verification
    lines.append("-" * 70)
    lines.append("STAGE 3 VERIFICATION")
    lines.append("-" * 70)
    
    stage3_passed = sum(1 for m in metrics if m.get("stage3_verified"))
    lines.append(f"Passed: {stage3_passed}/{total_jobs} ({100*stage3_passed/total_jobs:.1f}%)")
    lines.append("")
    
    # Cost breakdown by stage
    lines.append("-" * 70)
    lines.append("COST BREAKDOWN BY STAGE")
    lines.append("-" * 70)
    
    stage_costs = defaultdict(float)
    stage_counts = defaultdict(int)
    
    for m in metrics:
        for stage in m.get("stages", []):
            stage_name = stage.get("stage", "unknown")
            stage_costs[stage_name] += stage.get("cost_usd", 0)
            stage_counts[stage_name] += 1
    
    for stage in sorted(stage_costs.keys()):
        cost = stage_costs[stage]
        count = stage_counts[stage]
        avg = cost / count if count > 0 else 0
        pct = 100 * cost / total_cost if total_cost > 0 else 0
        lines.append(f"  {stage:20s}: ${cost:8.4f} ({pct:5.1f}%) | avg ${avg:.4f}/call")
    lines.append("")
    
    # Model usage
    lines.append("-" * 70)
    lines.append("MODEL USAGE")
    lines.append("-" * 70)
    
    model_costs = defaultdict(float)
    model_tokens = defaultdict(lambda: {"input": 0, "output": 0})
    
    for m in metrics:
        for stage in m.get("stages", []):
            model = stage.get("model", "unknown")
            model_costs[model] += stage.get("cost_usd", 0)
            model_tokens[model]["input"] += stage.get("input_tokens", 0)
            model_tokens[model]["output"] += stage.get("output_tokens", 0)
    
    for model in sorted(model_costs.keys(), key=lambda x: model_costs[x], reverse=True):
        cost = model_costs[model]
        tokens = model_tokens[model]
        pct = 100 * cost / total_cost if total_cost > 0 else 0
        lines.append(f"  {model:35s}: ${cost:8.4f} ({pct:5.1f}%)")
        lines.append(f"    tokens: {tokens['input']:,} in / {tokens['output']:,} out")
    lines.append("")
    
    # Recommendations
    lines.append("-" * 70)
    lines.append("OPTIMIZATION OPPORTUNITIES")
    lines.append("-" * 70)
    
    avg_iterations = sum(iterations) / len(iterations) if iterations else 0
    if avg_iterations > 2:
        lines.append(f"⚠ High avg iterations ({avg_iterations:.1f}): Consider improving draft quality")
    
    stage3_rate = stage3_passed / total_jobs if total_jobs > 0 else 0
    if stage3_rate < 0.95:
        lines.append(f"⚠ Stage 3 pass rate low ({100*stage3_rate:.1f}%): Check spec echo injection")
    
    # Find most expensive stage
    if stage_costs:
        most_expensive = max(stage_costs.keys(), key=lambda x: stage_costs[x])
        lines.append(f"💰 Most expensive stage: {most_expensive} (${stage_costs[most_expensive]:.4f})")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate metrics report")
    parser.add_argument("--days", type=int, default=7, help="Number of days to include")
    parser.add_argument("--dir", type=str, default="metrics", help="Metrics directory")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()
    
    metrics_dir = Path(args.dir)
    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        return
    
    metrics = load_metrics(metrics_dir, days=args.days)
    report = generate_report(metrics)
    
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()