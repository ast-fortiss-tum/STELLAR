from datetime import datetime
from typing import List
from wandb import Run
from opensbt.visualization.llm_figures_navi import boxplots, last_values_table, plot_metric_vs_time, statistics_table


project = "NaviQA"
metric = "failures"

def filter_navi_yelp(runs: List[Run], after: datetime = None) -> List[Run]:
    res = []
    take = True
    for run in runs:
        print("run name:", run.name)

        # Ensure created_at is timezone-aware
        created = run.created_at
        if isinstance(created, str):
            # Parse ISO 8601 string from W&B into datetime
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if take and run.state == "finished" and \
            "IPA_YELP" in run.name and \
            "03-00-00" in run.name:
                
            # Time-based filtering
            if after and not (created > after):
                continue
                    
            res.append(run)
    return res

run_filters_yelp = {"NaviQA": filter_navi_yelp}

algorithms = [
    "T-wise",
    "Random",
    "STELLAR"
]

experiments_folder = rf"wandb_download"

one_per_name = True

last_values_table(project, ["critical_ratio","failures"], run_filters=run_filters_yelp,
                       path="./wandb_analysis/yelp/last_values.csv",
                       one_per_name=one_per_name,
                       experiments_folder=experiments_folder)

statistics_table(algorithms, project, "failures", run_filters=run_filters_yelp,
                       path="./wandb_analysis/yelp_ratio_stats.csv",
                       one_per_name=one_per_name,
                       experiments_folder=experiments_folder)

plot_metric_vs_time(project, (18, 5), "critical_ratio", run_filters=run_filters_yelp,
                    file_name="./wandb_analysis/yelp/yelp_ratio_0.5.png",
                    one_per_name=one_per_name,
                    time_in_minutes = 180,
                    experiments_folder=experiments_folder,
                    tight=False,
                    th_content=0.75,
                    th_response=0.75)
boxplots(algorithms,
         project, 
        (18, 4), "failures", run_filters=run_filters_yelp,
        file_name="./wandb_analysis/yelp/yelp_failures_boxplots_0.6.png",
        one_per_name=one_per_name,
        experiments_folder=experiments_folder,
        th_content=0.75,
        th_response=0.75)
