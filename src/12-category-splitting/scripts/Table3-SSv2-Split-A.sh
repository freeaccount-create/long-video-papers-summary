#!/bin/bash

# ===== args =====
MODEL="./configs/models/SSv2-Split-A.json"
CHECKPOINT="./checkpoints/SSv2-Split-A.pt"
DATASET="ssv2"
LABEL_DIR="./benchmark/SSv2-Split/A"
VIDEO_DIR="./video/ssv2"
WEIGHT_INIT="coarse_grained_class_weight"
MODIFIERS_IN_BASE_MODEL="./modifiers/modifiers_in_base_model/SSv2-Split-A.json"
MODIFIERS_FOR_NEW_CLASSES="./modifiers/modifiers_for_new_classes/SSv2-Split.json"

OUTPUT_ROOT="./output"

# ===== coarse classes to be split =====
ALGS=("ma" "mr" "vlm")

COARSE_LABELS=(
        "Attaching something to something and Trying but failing to attach something to something because it doesn't stick" 
        "Bending something" 
        "Burying something in something and Digging something out of something" 
        "Closing something and Opening something" 
        "Dropping something with spatial relation" 
        "Letting something roll" 
        "Lifting a surface with something on it" 
        "Lifting something up completely" 
        "Moving something and something" 
        "Picking something up and Pretending to pick something up" 
        "Poking a stack of something" 
        "Twisting something, Twisting (wringing) something wet until water comes out and Pretending or trying and failing to twist something" 
        "Pretending to put something with spatial relation" 
        "Pretending to spread air onto something and Spreading something onto something" 
        "Pretending to squeeze something and Squeezing something" 
        "Pushing something" 
        "Putting multiple things" 
        "Putting something on the table" 
        "Putting something on a surface or onto something" 
        "Showing something with spatial relation" 
        "Something falling" 
        "Spinning something" 
        "Pretending to take something from somewhere and Taking something from somewhere" 
        "Tearing something" 
        "Throwing something in the air" 
        "Tilting something with something on it" 
        "Something colliding with something"
        )

SEEDS=(0 1 3)

# ===== loop =====
for alg in "${ALGS[@]}"; do
    for coarse_label in "${COARSE_LABELS[@]}"; do

        clean_label=$(echo "$coarse_label" | sed 's/[^a-zA-Z0-9._-]/_/g')
        output_dir="${OUTPUT_ROOT}/Table3/SSv2-Split-A/${alg}/${clean_label}"
        mkdir -p "$output_dir"

        if [ "$alg" == "ma" ]; then
            # ma with different seed
            for seed in "${SEEDS[@]}"; do
                sbatch \
                    --output="${output_dir}/%x_%j.out" \
                    --error="${output_dir}/%x_%j.err" \
                    ./scripts/job.sh \
                    "$MODEL" \
                    "$CHECKPOINT" \
                    "$DATASET" \
                    "$LABEL_DIR" \
                    "$VIDEO_DIR" \
                    "$alg" \
                    "$WEIGHT_INIT" \
                    "$coarse_label" \
                    "$MODIFIERS_IN_BASE_MODEL" \
                    "$MODIFIERS_FOR_NEW_CLASSES" \
                    "$seed" \
                    "$output_dir"
            done
        else
            # the result of mr and vlm don't have randomness, so no need to try with different seed
            sbatch \
                --output="${output_dir}/%x_%j.out" \
                --error="${output_dir}/%x_%j.err" \
                ./scripts/job.sh \
                "$MODEL" \
                "$CHECKPOINT" \
                "$DATASET" \
                "$LABEL_DIR" \
                "$VIDEO_DIR" \
                "$alg" \
                "$WEIGHT_INIT" \
                "$coarse_label" \
                "$MODIFIERS_IN_BASE_MODEL" \
                "$MODIFIERS_FOR_NEW_CLASSES" \
                "0" \
                "$output_dir"
        fi

    done
done