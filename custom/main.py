"""Custom LUNAR use case — entry point.

Run the search with a preset (fully offline by default):

    # from the LUNAR repository root
    python -m custom.main --preset test        # fast smoke test
    python -m custom.main --preset default      # default settings

Override individual hyperparameters on top of a preset, e.g.:

    python -m custom.main --preset default --algorithm random --seed 7

List the presets / hyperparameters:

    python -m custom.main --list

Results (fail rate, failing inputs, metrics) are written under ``./results``.
"""
import argparse
import csv
import json
from pathlib import Path

from custom.presets import PRESETS, describe_hyperparameters, get_preset
from custom.run_setup import (
    build_fitness_and_critical,
    build_optimizer,
    build_problem,
    build_search_config,
    configure_runtime,
    log,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a custom LUNAR search.")
    parser.add_argument("--preset", choices=list(PRESETS), default="test",
                        help="Hyperparameter preset to start from.")
    parser.add_argument("--list", action="store_true",
                        help="Print the presets and hyperparameters, then exit.")
    # Optional overrides (None -> keep the preset value).
    parser.add_argument("--algorithm", choices=["nsga2", "random"], default=None)
    parser.add_argument("--population_size", type=int, default=None)
    parser.add_argument("--n_generations", type=int, default=None)
    parser.add_argument("--max_time", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generator_llm", type=str, default=None,
                        help="Generator model name ('mock' keeps it offline).")
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Optional W&B organization or user name that owns the project.")
    parser.add_argument("--results_folder", type=str, default=None)
    return parser.parse_args()


def report_execution_metrics(results_folder: str) -> dict:
    """Summarise fail rate and failing inputs from the results folder."""
    results_path = Path(results_folder)
    metrics: dict = {}

    summary_path = results_path / "summary_results.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Attribute") == "Ratio Critical/All scenarios (duplicate free)":
                    metrics["fail_rate"] = float(row["Value"])
                    break

    critical_path = results_path / "all_critical_utterances.json"
    if critical_path.exists():
        with critical_path.open("r", encoding="utf-8") as handle:
            critical = json.load(handle)
        metrics["num_failures"] = len(critical)
        labels = set()
        for entry in critical:
            conversation = entry.get("utterance") or entry.get("conversation") or {}
            content = conversation.get("content_input", {}) or {}
            debug = (conversation.get("raw_output", {}) or {}).get("_fitness_debug", {})
            intent = content.get("intent") or debug.get("expected_intent")
            if intent:
                labels.add(intent)
        metrics["num_distinct_failing_intents"] = len(labels)

    (results_path / "overall_metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=4))
    return metrics


def main() -> None:
    args = parse_args()

    if args.list:
        print(describe_hyperparameters())
        return

    hp = get_preset(
        args.preset,
        algorithm=args.algorithm,
        population_size=args.population_size,
        n_generations=args.n_generations,
        max_time=args.max_time,
        seed=args.seed,
        generator_llm=args.generator_llm,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        results_folder=args.results_folder,
    )

    configure_runtime(hp)
    log.info(f"Starting custom search with preset '{args.preset}': {hp}")

    config = build_search_config(hp)
    fitness, critical = build_fitness_and_critical()
    problem = build_problem(hp, config, fitness, critical)
    optimizer = build_optimizer(hp, problem, config)

    result = optimizer.run()
    result.write_results(
        results_folder=optimizer.save_folder,
        params=optimizer.parameters,
        search_config=config,
    )

    report_execution_metrics(optimizer.save_folder)
    log.info("====== Search time: " + str("%.2f" % result.exec_time) + " sec")


if __name__ == "__main__":
    main()
