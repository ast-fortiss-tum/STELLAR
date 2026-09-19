#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python}"
entity="${WANDB_ENTITY:-opentest}"
project="stellar-safety-algorithm-smoke"
status=0
for algorithm in rs gs nsga2 nsga2d nsga2ds; do
    if ! "$python_bin" run_tests_safety.py \
        --algorithm "$algorithm" \
        --population_size 2 \
        --n_generations 1 \
        --max_time 00:00:30 \
        --no_rag \
        --wandb_entity "$entity" \
        --wandb_project "$project" \
        "$@"; then
        printf 'Algorithm failed: %s\n' "$algorithm" >&2
        status=1
    fi
done

exit "$status"