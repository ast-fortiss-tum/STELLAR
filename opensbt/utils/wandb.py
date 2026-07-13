import io
import time
import os
from pathlib import Path

import pandas as pd
from pymoo.core.algorithm import Algorithm

import wandb


def wandb_log_csv(filename):
    # log only if file is not empty
    if (
        Path(filename).exists()
        and Path(filename).is_file()
        and os.stat(filename).st_size > 0
    ):
        try:
            df = pd.read_csv(filename)
            metric_table = wandb.Table(dataframe=df)
            metric_table_artifact = wandb.Artifact("metric_history", type="dataset")
            metric_table_artifact.add(metric_table, "metric_table")
            metric_table_artifact.add_file(filename)
            wandb.log({"log": metric_table})
            wandb.log_artifact(metric_table_artifact, name=filename)
        except io.UnsupportedOperation:
            print(f"Cannot log {filename}. Check if it is in .csv format.")
    else:
        print(f"{filename} does not exist or it is empty.")



def logging_callback(algorithm: Algorithm):
    all_population = algorithm.pop
    critical_all, _ = all_population.divide_critical_non_critical()
    wandb.log(
        {
            "population_size": len(all_population),
            "failures": len(critical_all),
            "critical_ratio": len(critical_all) / len(all_population),
            "timestamp": time.time()
        }
    )

def logging_callback_archive(algorithm: Algorithm):
    if hasattr(algorithm, "archive") and algorithm.archive is not None:
        all_population = algorithm.archive
        critical_all, _ = all_population.divide_critical_non_critical()
        wandb.log(
            {
                "test_size": len(all_population),
                "failures": len(critical_all),
                "critical_ratio": len(critical_all) / len(all_population) if len(all_population) > 0 else 0.0,
                "timestamp": time.time()
            }
        )

class TableCallback:
    def __init__(self):
        self.table = wandb.Table(columns=["test_size", "failures", "critical_ratio", "timestamp"],
                                 log_mode="MUTABLE")

    def log(self, algorithm: Algorithm):
        all_population = algorithm.archive
        critical_all, _ = all_population.divide_critical_non_critical()
        self.table.add_data(
            len(all_population),
            len(critical_all),
            len(critical_all) / len(all_population) if len(all_population) > 0 else 0.0,
            time.time()
        )
        wandb.log({"Summary Table": self.table})

    def __getstate__(self):
        """Return state for pickling (skip unpicklable parts)."""
        state = self.__dict__.copy()
        state["table"] = None
        return state

    def __setstate__(self, state):
        """Recreate skipped attributes after unpickling."""
        self.__dict__.update(state)
        if self.table is None:
            self.table = wandb.Table(columns=["test_size", "failures", "critical_ratio", "timestamp"],
                                 log_mode="MUTABLE")
