import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

# Find all metrics.json files in subfolders
folder_name = "../opensbt-results/judge-eval/request_response/"
json_files = glob.glob(f"{folder_name}/judge-eval*/metrics.json", recursive=True)
save_folder = f"{folder_name}/combined_eval/"

Path(save_folder).mkdir(exist_ok= True, parents=True)

# Flatten into list of (config, model, metrics)
entries = []
configs = []
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        folder_name = os.path.basename(os.path.dirname(file))
        # Remove trailing timestamp
        config_name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", "", folder_name)
        configs.append(config_name)
        for model, metrics in data.items():
            overall = metrics["overall"]
            model_clean = re.sub(r"\s*\(.*?\)", "", model)
            # Replace 'judge-eval-paper-*' with 'judges' in x-axis labels
            label = f"{config_name.replace('judge-eval-paper-','judges')}\n{model_clean}"
            entries.append((label, config_name, overall))

# Unique configs for coloring
unique_configs = sorted(set(c for _, c, _ in entries))
colors = {cfg: plt.cm.tab10(i % 10) for i, cfg in enumerate(unique_configs)}

# Metrics to plot
metrics_to_plot = ["f1", "accuracy", "avg_time"]
titles = {
    "f1": "F1 Score across Judge Configurations and Models",
    "accuracy": "Accuracy across Judge Configurations and Models",
    "avg_time": "Average Time across Judge Configurations and Models"
}
ylabels = {
    "f1": "F1 Score",
    "accuracy": "Accuracy",
    "avg_time": "Time (s)"
}

labels = [e[0] for e in entries]

for metric in metrics_to_plot:
    values = [e[2][metric] for e in entries]
    cfgs = [e[1] for e in entries]

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(labels, values, color=[colors[c] for c in cfgs])

    # Annotate bars with their values
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x position: center of the bar
            height,                             # y position: top of the bar
            f"{height:.2f}",                     # format value
            ha='center', va='bottom', fontsize=8
        )

    ax.set_title(titles[metric])
    ax.set_ylabel(ylabels[metric])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")

    # Legend outside to the right
    legend_handles = [plt.Rectangle((0,0),1,1,color=colors[cfg]) for cfg in unique_configs]
    ax.legend(legend_handles, unique_configs, title="Judge Config",
              bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_folder + os.sep + metric + ".png", format="png")
