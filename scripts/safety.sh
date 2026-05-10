python run_tests_safety.py \
        --sut "gpt-4o-mini" \
        --population_size 20 \
        --n_generations 100 \
        --algorithm "nsga2" \
        --max_time "02:00:00" \
        --features_config "configs/safety_features.json" \
        --no_wandb \
        --seed 1