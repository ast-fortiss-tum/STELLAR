python -m scripts.aggregate_over_seeds_judge \
        --eval_dir "judge-eval\tmp\sample-1-majority" \
        --output_csv "judge-eval\tmp\sample-1-majority\combined.csv" \
        --n_runs 6

python -m scripts.aggregate_over_seeds_judge \
        --eval_dir "judge-eval\tmp\sample-3-majority" \
        --output_csv "judge-eval\tmp\sample-3-majority\combined.csv" \
        --n_runs 6