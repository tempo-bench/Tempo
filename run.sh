#!/bin/bash

# Define the domains and models to evaluate
DOMAINS=(
    'bitcoin' 'cardano' 'economics' 'genealogy' 'history' 'hsm' 
    'iota' 'law' 'monero' 'politics' 'quant' 'travel' 'workplace'
)
#'sf' 'qwen' 'qwen2' 'e5' 'bm25' 'sbert' 'bge' 'inst-l' 'inst-xl' 
    # 'grit' 'cohere' 'voyage' 'openai' 'google' 'diver-retriever' 
    # 'contriever' 'reasonir' 'm2' 'rader' 'nomic'
MODELS=(
    'bm25' 'contriever' 'bge'
)

# Directory for outputs and cache
OUTPUT_DIR="outputs"
CACHE_DIR="cache"
CONFIG_DIR="configs"

# Loop over each domain and model to run the evaluation
for domain in "${DOMAINS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Running evaluation for domain: $domain with model: $model"

        # Base command
        CMD="python run.py \
            --task $domain \
            --model $model \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --config_dir $CONFIG_DIR"

        # Add model-specific arguments if needed
        if [ "$model" == "reasonir" ]; then
            CMD="$CMD --encode_batch_size 1"
        else
            CMD="$CMD --encode_batch_size 1024"
        fi

        # To avoid re-running and overwriting results, check if the output exists
        SCORE_FILE="outputs/$model/$domain/scores.json"
        if [ -f "$SCORE_FILE" ]; then
            echo "Scores for $model on $domain already exist. Skipping."
            continue
        fi

        # Run the command
        eval $CMD

        # Optional: Add a delay to avoid rate-limiting if using API-based models
        # sleep 1
    done
done

echo "All evaluations complete."