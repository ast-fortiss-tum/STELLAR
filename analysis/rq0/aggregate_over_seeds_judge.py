import os
import json
import pandas as pd
import numpy as np
import argparse

# ---------------- Helper functions ----------------
def collect_json_files(base_dir: str):
    """Recursively collect all evaluation_results.json files."""
    json_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f == "evaluation_results.json":
                json_files.append(os.path.join(root, f))
    return json_files


def aggregate_metrics(json_files, n_runs=None):
    """
    Aggregates metrics across all seeds/runs.
    Only uses the first `n_runs` per model if specified.
    Returns a dictionary:
        { model_name: {metric_name: (mean, std), ...}, ... }
    """
    model_data = {}

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for model_name, metrics in data.items():
            model_data.setdefault(model_name, []).append(metrics)

    aggregated_results = {}

    for model_name, runs in model_data.items():
        # Limit runs if n_runs is specified
        runs_to_use = runs[:n_runs] if n_runs else runs

        # ---- Aggregate overall metrics ----
        overall_metrics = {}
        overall_keys = runs_to_use[0]['overall'].keys()
        for key in overall_keys:
            if key == "variance_per_dimension":
                per_dim_values = np.array([r['overall'][key] for r in runs_to_use])
                mean_val = np.nanmean(per_dim_values, axis=0).tolist()
                std_val = np.nanstd(per_dim_values, axis=0).tolist()
            elif isinstance(runs_to_use[0]['overall'][key], (int, float)):
                values = [r['overall'][key] for r in runs_to_use]
                mean_val = float(np.nanmean(values))
                std_val = float(np.nanstd(values))
            else:
                continue
            overall_metrics[key + "_mean"] = mean_val
            overall_metrics[key + "_std"] = std_val

        aggregated_results[model_name] = overall_metrics

    return aggregated_results


def save_to_csv(aggregated_results, output_file):
    df = pd.DataFrame.from_dict(aggregated_results, orient="index")
    df.index.name = "Model"
    df.to_csv(output_file, float_format="%.4f")
    print(f"Aggregated metrics saved to {output_file}")


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across multiple seeds/runs per model."
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Base directory containing evaluation_results.json files (recursively searched)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./model_metrics_summary.csv",
        help="Output CSV file for aggregated metrics."
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=None,
        help="Number of runs per model to use for aggregation. Default: use all."
    )

    args = parser.parse_args()

    json_files = collect_json_files(args.eval_dir)
    if not json_files:
        print(f"No evaluation_results.json files found in {args.eval_dir}")
        exit(1)

    aggregated_results = aggregate_metrics(json_files, n_runs=args.n_runs)
    save_to_csv(aggregated_results, args.output_csv)
