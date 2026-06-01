#!/bin/bash

# ===== args =====
MODEL="./configs/models/SSv2-Split-A.json"
CHECKPOINT="./checkpoints/SSv2-Split-A.pt"
DATASET="ssv2"
LABEL_DIR="./benchmark/SSv2-Split/A"
VIDEO_DIR="./video/ssv2"
ALG="ft"

MODIFIERS_IN_BASE_MODEL="./modifiers/modifiers_in_base_model/SSv2-Split-A.json"
MODIFIERS_FOR_NEW_CLASSES="./modifiers/modifiers_for_new_classes/SSv2-Split.json"

OUTPUT_ROOT="./output"

# ===== coarse classes to be split =====
WEIGHT_INITS=("random" "coarse_grained_class_weight" "ma")

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

SEEDS=(0 1 2 3 4 5)

# ===== loop =====
for weight_init in "${WEIGHT_INITS[@]}"; do
  for coarse_label in "${COARSE_LABELS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      clean_label=$(echo "$coarse_label" | sed 's/[^a-zA-Z0-9._-]/_/g')

      output_dir="${OUTPUT_ROOT}/Table4/SSv2-Split-A/${ALG}_${weight_init}/${clean_label}"

      mkdir -p "$output_dir"

      sbatch \
        --output="${output_dir}/%x_%j.out" \
        --error="${output_dir}/%x_%j.err" \
        ./scripts/job.sh \
        "$MODEL" \
        "$CHECKPOINT" \
        "$DATASET" \
        "$LABEL_DIR" \
        "$VIDEO_DIR" \
        "$ALG" \
        "$weight_init" \
        "$coarse_label" \
        "$MODIFIERS_IN_BASE_MODEL" \
        "$MODIFIERS_FOR_NEW_CLASSES" \
        "$seed" \
        "$output_dir"

    done
  done
done