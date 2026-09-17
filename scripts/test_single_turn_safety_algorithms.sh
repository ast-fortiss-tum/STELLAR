#!/bin/bash
set -euo pipefail

project="stellar-safety-algorithm-smoke"
entity="${WANDB_ENTITY:-opentest}"
algorithms=(rs gs nsga2 nsga2d nsga2ds)

for algorithm in "${algorithms[@]}"; do
    python main.py \
        --case_study safety \
        --mode single-turn \
        --algorithm "$algorithm" \
        --n 2 \
        --i 1 \
        --max_time 00:00:30 \
        --llm_ipa gpt-5-chat \
        --llm_judge gpt-4o-mini \
        --llm_generator gpt-4o-mini \
        --use_wandb \
        --wandb_project "$project" \
        --wandb_entity "$entity" \
        "$@"
done