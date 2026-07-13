import os
import pandas as pd
import numpy as np
import ast
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score

def safe_parse_scores(x, key=None):
    try:
        parsed = ast.literal_eval(x)
        if key:
            scores = list(parsed[key].values())
        else:
            scores = list(parsed)
        cleaned = [np.nan if str(v).upper() == "N/A" else v for v in scores]
        if any(pd.isna(cleaned)):
            return np.nan
        return cleaned
    except Exception:
        return np.nan

def safe_parse_score_samples(x):
    try:
        parsed = ast.literal_eval(x)
        if not parsed or not isinstance(parsed, list):
            return np.nan
        cleaned = []
        for sample in parsed:
            if not isinstance(sample, list):
                sample = [sample]
            cleaned_sample = [float(v) if str(v).upper() != "N/A" else np.nan for v in sample]
            cleaned.append(cleaned_sample)
        return cleaned
    except Exception:
        return np.nan

def load_ground_truth(gt_csv_path: str) -> pd.DataFrame:
    gt_df = pd.read_csv(gt_csv_path, encoding="utf-8")
    required_cols = {"Question", "Response", "R", "D", "P"}
    if not required_cols.issubset(gt_df.columns):
        raise ValueError(f"Ground truth CSV must contain columns {required_cols}")
    gt_df = gt_df[["Question", "Response", "R", "D", "P"]].copy()
    gt_df["Question"] = gt_df["Question"].astype(str).str.strip()
    gt_df["Response"] = gt_df["Response"].astype(str).str.strip()
    for col in ["R", "D", "P"]:
        gt_df[col] = pd.to_numeric(gt_df[col], errors="coerce")
    return gt_df

def evaluate_metrics_results(csv_path: str,
                             gt_csv_path: str,
                             json_output_path: str,
                             plot_errors_file: str,
                             plot_efficiency_file: str,
                             plot_f1_file: str) -> None:

    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.dropna(subset=['Scores', 'Score_samples', 'Model', 'Time'])

    if "Question" not in df.columns or "Answer" not in df.columns:
        raise ValueError("Judge file must contain 'Question' and 'Answer' columns")

    df["Question"] = df["Question"].astype(str).str.strip()
    df["Answer"] = df["Answer"].astype(str).str.strip()

    # Output folder based on F1 file
    output_folder = os.path.dirname(plot_f1_file)
    os.makedirs(output_folder, exist_ok=True)

    # Load ground truth and merge
    gt_df = load_ground_truth(gt_csv_path)

    print("Number of rows in df (predictions):", len(df))
    print("Number of rows in gt_df (ground truth):", len(gt_df))

    df = df.merge(gt_df, left_on=["Question", "Answer"], right_on=["Question", "Response"], how="inner")

    df['Answer_scores'] = df.apply(lambda row: [row['R'], row['D'], row['P']], axis=1)
    df['Pred_scores'] = df['Scores'].apply(lambda x: safe_parse_scores(x))
    df['Score_samples_parsed'] = df['Score_samples'].apply(safe_parse_score_samples)
    df = df.dropna(subset=['Answer_scores', 'Pred_scores', 'Score_samples_parsed']).reset_index(drop=True)

    category_labels = ["Request-oriented", "Directness", "Proactivity"]

    results = {}
    models = df['Model'].unique()
    avg_times, overall_f1_scores, overall_num_tests = [], [], []

    # ---------- Prediction error plot ----------
    fig_err, axes_err = plt.subplots(1, len(models), figsize=(6 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes_err = [axes_err]

    for ax, model in zip(axes_err, models):
        model_df = df[df['Model'] == model]
        model_result, per_category = {}, {}
        mean_distances, std_distances, num_tests_per_category = [], [], []
        variance_per_dimension = np.full(len(category_labels), np.nan)

        for i, cat_label in enumerate(category_labels):
            y_true = np.array([scores[i] for scores in model_df['Answer_scores']])
            y_pred = np.array([scores[i] for scores in model_df['Pred_scores']])
            diffs = y_pred - y_true

            # Variance across repeated samples
            var_per_question = []
            for samples in model_df['Score_samples_parsed']:
                samples_i = [round(s[i]) for s in samples if s]
                if len(samples_i) > 1:
                    var_per_question.append(float(np.var(samples_i)))
            if var_per_question:
                variance_per_dimension[i] = float(np.mean(var_per_question))

            # Metrics
            f1 = f1_score(y_true, y_pred, average='macro')
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro')
            avg_distance = float(np.mean(np.abs(diffs)))
            std_distance = float(np.std(np.abs(diffs)))
            n_tests = len(y_true)

            per_category[cat_label] = {
                'num_tests': n_tests,
                'f1': f1,
                'accuracy': acc,
                'precision': prec,
                'avg_distance': avg_distance,
                'std_distance': std_distance
            }

            mean_distances.append(avg_distance)
            std_distances.append(std_distance)
            num_tests_per_category.append(n_tests)

        model_result['per_category'] = per_category

        # Overall metrics
        y_true_all = [score for sublist in model_df['Answer_scores'] for score in sublist]
        y_pred_all = [score for sublist in model_df['Pred_scores'] for score in sublist]
        diffs_all = np.array(y_pred_all) - np.array(y_true_all)
        avg_time = float(model_df['Time'].mean())
        avg_times.append(avg_time)
        overall_f1 = f1_score(y_true_all, y_pred_all, average='macro')
        overall_f1_scores.append(overall_f1)
        overall_num_tests.append(len(y_true_all))

        model_result['overall'] = {
            'num_tests': len(y_true_all),
            'f1': overall_f1,
            'accuracy': accuracy_score(y_true_all, y_pred_all),
            'precision': precision_score(y_true_all, y_pred_all, average='macro'),
            'avg_distance': float(np.mean(np.abs(diffs_all))),
            'std_distance': float(np.std(np.abs(diffs_all))),
            'avg_time': avg_time,
            'variance_per_dimension': variance_per_dimension.tolist()
        }

        results[model] = model_result

        # Plot prediction errors
        x = np.arange(len(category_labels))
        ax.bar(x, mean_distances, 0.6, yerr=std_distances, capsize=5, color='skyblue', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(category_labels, rotation=15)
        ax.set_title(f"Model: {model} (n={sum(num_tests_per_category)} tests)")
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_ylabel("Avg Prediction Error")

    plt.tight_layout()
    plt.savefig(plot_errors_file, format="png", dpi=300)
    plt.close()

    # ---------- Overall F1 plot ----------
    fig_f1, ax_f1 = plt.subplots(figsize=(8, 5))
    bars_f1 = ax_f1.bar(models, overall_f1_scores, color='green', edgecolor='black')
    for bar, f1_val, n_tests in zip(bars_f1, overall_f1_scores, overall_num_tests):
        ax_f1.text(bar.get_x() + bar.get_width() / 2, f1_val + 0.01, f"{f1_val:.2f}",
                   ha='center', va='bottom', fontsize=10, fontweight="bold")
        ax_f1.text(bar.get_x() + bar.get_width() / 2, f1_val / 2, f"n={n_tests}",
                   ha='center', va='center', fontsize=9, color="white")
    ax_f1.set_ylabel("Overall F1 Score")
    ax_f1.set_ylim(0, 1.05)
    ax_f1.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_f1_file, format="png", dpi=300)
    plt.close()

    # ---------- Efficiency vs Accuracy ----------
    fig_eff, ax_eff = plt.subplots(figsize=(8, 5))
    for model in models:
        avg_time = results[model]['overall']['avg_time']
        overall_acc = results[model]['overall']['accuracy']
        ax_eff.scatter(avg_time, overall_acc, label=model, s=100)
        ax_eff.text(avg_time, overall_acc, model, fontsize=9, ha='right', va='bottom')
    ax_eff.set_xlabel("Average Inference Time (s)")
    ax_eff.set_ylabel("Overall Accuracy")
    ax_eff.set_title("Model Efficiency vs Accuracy")
    ax_eff.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_efficiency_file, dpi=300)
    plt.close()

    # ---------- Per-dimension F1 plot ----------
    bar_width = 0.6
    fig_width_per_dim = 5
    fig_height = 5
    label_rotation = 30
    font_size_labels = 10
    font_size_values = 9

    num_models = len(models)
    num_dims = len(category_labels)
    fig_f1_dim, axes_f1_dim = plt.subplots(1, num_dims, figsize=(fig_width_per_dim * num_dims, fig_height), sharey=True)
    if num_dims == 1:
        axes_f1_dim = [axes_f1_dim]

    for ax, cat_label in zip(axes_f1_dim, category_labels):
        x = np.arange(num_models)
        for i, model in enumerate(models):
            f1_val = results[model]['per_category'][cat_label]['f1']
            ax.bar(x[i], f1_val, width=bar_width, color='orange', edgecolor='black')
            ax.text(x[i], f1_val + 0.01, f"{f1_val:.2f}", ha='center', va='bottom', fontsize=font_size_values)
        ax.set_title(f"F1 Score: {cat_label}")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=label_rotation, ha='right', fontsize=font_size_labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 Score")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    dim_f1_file = os.path.join(output_folder, "f1_per_dimension_separate.png")
    plt.savefig(dim_f1_file, dpi=300)
    plt.close()

    # ---------- Save results ----------
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation complete. Results saved to {json_output_path}")
    print(f"Per-dimension F1 plot saved to {dim_f1_file}")

# ---------- CLI entry point ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model metrics against ground truth scores.")
    parser.add_argument("--csv_path", required=True, help="Path to judge evaluation CSV file.")
    parser.add_argument("--gt_csv_path", required=True, help="Path to ground truth CSV file.")
    parser.add_argument("--json_output_path", default="./judge_eval/out/temp/evaluation_results.json", help="Output JSON file for metrics.")
    parser.add_argument("--plot_errors_file", default="./judge_eval/out/temp/prediction_errors.png", help="Output PNG for prediction errors.")
    parser.add_argument("--plot_efficiency_file", default="./judge_eval/out/temp/model_efficiency.png", help="Output PNG for model efficiency.")
    parser.add_argument("--plot_f1_file", default="./judge_eval/out/temp/model_f1_score.png", help="Output PNG for F1 score.")
    args = parser.parse_args()

    evaluate_metrics_results(
        csv_path=args.csv_path,
        gt_csv_path=args.gt_csv_path,
        json_output_path=args.json_output_path,
        plot_errors_file=args.plot_errors_file,
        plot_efficiency_file=args.plot_efficiency_file,
        plot_f1_file=args.plot_f1_file
    )
