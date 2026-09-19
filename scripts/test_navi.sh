DEPLOYMENT_NAME="llama3.2" python run_tests_navi.py \
        --sut "IPA_YELP" \
        --population_size 20 \
        --n_generations 5 \
        --algorithm "nsga2" \
        --max_time "03:00:00" \
        --features_config "configs/navi_features.json"\
        --no_wandb \
        --use_rag \
        --seed 1
