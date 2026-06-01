#!/bin/bash

# ===== args =====
MODEL="./configs/models/SSv2-Split-B.json"
CHECKPOINT="./checkpoints/SSv2-Split-B.pt"
DATASET="ssv2"
LABEL_DIR="./benchmark/SSv2-Split/B"
VIDEO_DIR="./video/ssv2"
ALG="ma"
WEIGHT_INIT="coarse_grained_class_weight"
MODIFIERS_IN_BASE_MODEL="./modifiers/modifiers_in_base_model/SSv2-Split-B.json"
MODIFIERS_FOR_NEW_CLASSES="./modifiers/modifiers_for_new_classes/SSv2-Split.json"

OUTPUT_ROOT="./output"

# ===== coarse classes to be split =====
COARSE_LABELS=(
        "Moving the camera and Turning the camera" 
        "Covering something with something and Uncovering something" 
        "Holding something with spatial relation" 
        "Folding something and Unfolding something" 
        "Lifting up one end of something" 
        "Moving something across a surface" 
        "Moving something" 
        "Piling something up and Stacking number of something" 
        "Plugging something into something and Plugging something into something but pulling it right out as you remove your hand" 
        "Poking a hole" 
        "Pouring something" 
        "Pretending or failing to wipe something off of something and Wiping something off of something" 
        "Pretending to close something without actually closing it and Pretending to open something without actually opening it" 
        "Pretending to scoop something up with something and Scooping something up with something" 
        "Pretending to sprinkle air onto something and Sprinkling something onto something" 
        "Pretending to turn something upside down and Turning something upside down" 
        "Pulling something" 
        "Pulling two ends of something" 
        "Putting something with spatial relation" 
        "Putting something that can't roll onto a slanted surface" 
        "Showing something's properties" 
        "Spilling something with spatial relation" 
        "Pretending to take something out of something and Taking something out of something" 
        "Pretending to throw something and Throwing something" 
        "Throwing something toward something" 
        "Tipping something over and Tipping something with something in it over, so something in it falls out" 
        "Poking something"
        )

SEEDS=(0 1 2)

# ===== loop =====
for coarse_label in "${COARSE_LABELS[@]}"; do
  for seed in "${SEEDS[@]}"; do

    clean_label=$(echo "$coarse_label" | sed 's/[^a-zA-Z0-9._-]/_/g')

    output_dir="${OUTPUT_ROOT}/Table2/SSv2-Split-B/${ALG}/${clean_label}"

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
      "$WEIGHT_INIT" \
      "$coarse_label" \
      "$MODIFIERS_IN_BASE_MODEL" \
      "$MODIFIERS_FOR_NEW_CLASSES" \
      "$seed" \
      "$output_dir"

  done
done