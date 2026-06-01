#!/bin/bash

# ===== args =====
MODEL="./configs/models/FineGym-Split-A.json"
CHECKPOINT="./checkpoints/FineGym-Split-A.pt"
DATASET="finegym"
LABEL_DIR="./benchmark/FineGym-Split/A"
VIDEO_DIR="./video/finegym"
ALG="ma"
WEIGHT_INIT="coarse_grained_class_weight"
MODIFIERS_IN_BASE_MODEL="./modifiers/modifiers_in_base_model/FineGym-Split-A.json"
MODIFIERS_FOR_NEW_CLASSES="./modifiers/modifiers_for_new_classes/FineGym-Split.json"

OUTPUT_ROOT="./output"

# ===== coarse classes to be split =====
COARSE_LABELS=(
        "(VT) round-off, flic-flac with 0.5 turn on, salto forward" 
        "(VT) round-off, flic-flac with 1 turn on, salto backward" 
        "(VT) handspring forward on, salto forward" 
        "(FX) leap" 
        "(FX) jump or hop" 
        "(FX) 2 turn or more" 
        "(FX) 3 turn" 
        "(FX) take-off forward from one or both legs, salto sideward tucked and (FX) aerial cartwheel" 
        "(FX) salto forward" 
        "(FX) arabian double salto" 
        "(BB) ring and arch jump" 
        "(BB) wolf hop or jump and cat leap" 
        "(BB) turns with free leg at horizontal and turns with free leg optional below horizontal" 
        "(BB) salto backward" 
        "(BB) free aerial" 
        "(BB) salto stretched" 
        "(UB) circle backward without turn" 
        "(UB) circle forward with turn" 
        "(UB) hang on high bar" 
        "(UB) transition flight" 
        "(BB) salto forward tucked" 
        "(BB) gainer salto" 
        "(UB) (swing forward) double salto backward"
)

SEEDS=(0 1 2)

# ===== loop =====
for coarse_label in "${COARSE_LABELS[@]}"; do
  for seed in "${SEEDS[@]}"; do

    clean_label=$(echo "$coarse_label" | sed 's/[^a-zA-Z0-9._-]/_/g')

    output_dir="${OUTPUT_ROOT}/Table2/FineGym-Split-A/${ALG}/${clean_label}"

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