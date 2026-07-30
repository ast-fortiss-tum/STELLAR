import csv
import json
from pathlib import Path

from tarot.run_test_setup import (
    build_fitness_and_critical,
    build_optimizer,
    build_problem,
    build_search_config,
    configure_runtime,
    log,
    parse_args,
)


def report_execution_metrics(results_folder: str) -> dict:
    results_path = Path(results_folder)

    summary_results_path = results_path / "summary_results.csv"
    all_critical_utterances_path = results_path / "all_critical_utterances.json"
    llm_usage_summary_path = results_path / "llm_usage_summary.json"

    fail_rate = None
    with summary_results_path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("Attribute") == "Ratio Critical/All scenarios (duplicate free)":
                fail_rate = float(row["Value"])
                break

    if fail_rate is None:
        raise ValueError(
            "Could not find 'Ratio Critical/All scenarios (duplicate free)' in summary_results.csv"
        )

    with all_critical_utterances_path.open("r", encoding="utf-8") as json_file:
        all_critical_utterances = json.load(json_file)

    diverse_intents = set()
    for entry in all_critical_utterances:
        utterance = entry.get("utterance", {})
        content_input = utterance.get("content_input", {})
        intent = content_input.get("intent")
        if intent:
            diverse_intents.add(intent)

    with llm_usage_summary_path.open("r", encoding="utf-8") as json_file:
        llm_usage_summary = json.load(json_file)

    tokens_spent = llm_usage_summary["total"]["tokens"]

    metrics = {
        "fail_rate": fail_rate,
        "num_diverse_fails": len(diverse_intents),
        "tokens_spent": tokens_spent,
    }

    metrics_path = results_path / "overall_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=4)

    print(json.dumps(metrics, indent=4))

    return metrics


def main() -> None:
    args = parse_args()

    configure_runtime()
    log.info("Starting NLU search experiment")

    config = build_search_config(args)
    fitness, critical = build_fitness_and_critical()
    problem = build_problem(args, config, fitness, critical)

    optimizer = build_optimizer(args, problem, config)

    result = optimizer.run()
    result.write_results(
        results_folder=optimizer.save_folder,
        params=optimizer.parameters,
        search_config=config,
    )

    report_execution_metrics(optimizer.save_folder)

    log.info("====== Algorithm search time: " + str("%.2f" % result.exec_time) + " sec")


if __name__ == "__main__":
    main()
