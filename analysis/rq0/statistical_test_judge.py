import os
import json
import pandas as pd
import numpy as np
import argparse
import itertools
from scipy.stats import wilcoxon


# ---------------- Helper functions ----------------
def collect_json_files(base_dir: str):
    """Recursively collect all evaluation_results.json files."""
    json_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f == "evaluation_results.json":
                json_files.append(os.path.join(root, f))
    return json_files


# ---------------- Statistical testing ----------------
def vargha_delaney_A12(a, b):
    """
    Compute Vargha–Delaney Â12 effect size for samples a and b.
    Returns (A12_value, interpretation_str).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n1 = a.size
    n2 = b.size
    if n1 == 0 or n2 == 0:
        return np.nan, "undefined"

    comp = np.subtract.outer(a, b)
    wins = np.sum(comp > 0)
    ties = np.sum(comp == 0)
    A12 = (wins + 0.5 * ties) / (n1 * n2)

    d = abs(A12 - 0.5)
    if d < 0.06:
        mag = "negligible"
    elif d < 0.14:
        mag = "small"
    elif d < 0.21:
        mag = "medium"
    else:
        mag = "large"

    if A12 > 0.5:
        direction = "Model1 > Model2"
    elif A12 < 0.5:
        direction = "Model1 < Model2"
    else:
        direction = "no difference"

    return float(A12), f"{mag}; {direction}"


def perform_statistical_tests(json_files, n_runs=None):
    """
    Perform pairwise Wilcoxon signed-rank tests and Vargha–Delaney Â12
    only for 'f1' and 'time' metrics across models.
    Optionally only use the first `n_runs` per model.
    Returns a pandas DataFrame with results.
    """
    detailed_data = {}
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for model_name, metrics in data.items():
            detailed_data.setdefault(model_name, []).append(metrics)

    model_names = sorted(detailed_data.keys())
    if not model_names:
        return pd.DataFrame()

    metrics_of_interest = ["f1", "time"]
    rows = []

    for m1, m2 in itertools.combinations(model_names, 2):
        runs1 = detailed_data[m1][:n_runs] if n_runs else detailed_data[m1]
        runs2 = detailed_data[m2][:n_runs] if n_runs else detailed_data[m2]

        for metric in metrics_of_interest:
            vals1 = [r['overall'][metric] for r in runs1
                     if 'overall' in r and metric in r['overall'] and isinstance(r['overall'][metric], (int, float))]
            vals2 = [r['overall'][metric] for r in runs2
                     if 'overall' in r and metric in r['overall'] and isinstance(r['overall'][metric], (int, float))]

            if len(vals1) == 0 or len(vals2) == 0:
                continue

            n = min(len(vals1), len(vals2))
            wilcoxon_p = np.nan
            try:
                if n >= 2:
                    stat, wilcoxon_p = wilcoxon(vals1[:n], vals2[:n])
            except ValueError:
                wilcoxon_p = 1.0
            except Exception:
                wilcoxon_p = np.nan

            A12, A12_mag = vargha_delaney_A12(vals1, vals2)

            rows.append({
                "Model_1": m1,
                "Model_2": m2,
                "Metric": metric,
                "n_model1": len(vals1),
                "n_model2": len(vals2),
                "p": wilcoxon_p,
                "A12": A12,
                "A12_interpretation": A12_mag
            })

    return pd.DataFrame(rows)


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform Wilcoxon and Vargha–Delaney A12 tests for F1 and Time with optional run limit."
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
        default="./statistical_tests_results.csv",
        help="Output CSV file for statistical test results."
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=None,
        help="Number of runs per model to use in the statistical test. Default: use all."
    )

    args = parser.parse_args()

    json_files = collect_json_files(args.eval_dir)
    if not json_files:
        print(f"No evaluation_results.json files found in {args.eval_dir}")
        exit(1)

    stats_df = perform_statistical_tests(json_files, n_runs=args.n_runs)
    if not stats_df.empty:
        stats_df.to_csv(args.output_csv, index=False, float_format="%.4f")
        print(f"Statistical test results saved to {args.output_csv}")
    else:
        print("No statistical test results (no numeric metrics or insufficient data).")
