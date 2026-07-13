from datetime import datetime
from opensbt.visualization.llm_figures_navi import last_values_table, plot_metric_vs_time, statistics_table


project = "SafeQA"
metric = "failures"

algorithms = [
    "T-wise",
    "Random",
    "ASTRAL",
    "STELLAR"
]

def filter_safety(runs, after: datetime = None):
    res = []
    take = True
    for run in runs:
        created = run.created_at
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))

        if take and run.state == "finished":
            
            # Time-based filtering
            if after and not (created > after):
                continue
            res.append(run)
    return res

run_filters= {
    "SafeQA" : filter_safety
}
experiments_folder = rf"wandb_download"

one_per_name = False

last_values_table(project, ["critical_ratio","failures"], run_filters=run_filters,
                       path="./wandb_analysis/yelp/last_values.csv",
                       one_per_name=one_per_name,
                       experiments_folder=experiments_folder)

statistics_table(algorithms, project, "failures", run_filters=run_filters, 
                       path="./wandb_analysis/safe/safety_stats.csv",
                       one_per_name=one_per_name)

plot_metric_vs_time(project, (18, 4), "failures", run_filters=run_filters, 
                    file_name="./wandb_analysis/safe/safety_failures.png",
                    one_per_name=one_per_name,
                    time_in_minutes=180)

statistics_table(algorithms, project, "critical_ratio", run_filters=run_filters, 
                       path="./wandb_analysis/safety/safety_ratio.csv",
                       one_per_name=one_per_name)

plot_metric_vs_time(project, (18, 4), "critical_ratio", run_filters=run_filters, 
                    file_name="./wandb_analysis/safety/safety_ratio_stats.png",
                    one_per_name=one_per_name,
                     time_in_minutes=120)