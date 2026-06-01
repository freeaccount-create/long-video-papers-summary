#!/bin/bash

# ===== args =====
MODEL="./configs/models/FineGym-Split-B.json"
CHECKPOINT="./checkpoints/FineGym-Split-B.pt"
DATASET="finegym"
LABEL_DIR="./benchmark/FineGym-Split/B"
VIDEO_DIR="./video/finegym"
ALG="ma"
WEIGHT_INIT="coarse_grained_class_weight"
MODIFIERS_IN_BASE_MODEL="./modifiers/modifiers_in_base_model/FineGym-Split-B.json"
MODIFIERS_FOR_NEW_CLASSES="./modifiers/modifiers_for_new_classes/FineGym-Split.json"

OUTPUT_ROOT="./output"

# ===== coarse classes to be split =====
COARSE_LABELS=(
        "(VT) round-off, flic-flac on, salto backward" 
        "(VT) tsukahara" 
        "(VT) handspring forward on" 
        "(FX) leap with turn" 
        "(FX) jump or hop with turn" 
        "(FX) 1 turn" 
        "(FX) salto backward" 
        "(BB) split and straddle jump" 
        "(BB) split and switch leap" 
        "(BB) tuck hop or jump, pick jump and stretched hop or jump" 
        "(BB) turn with legs in 180 split" 
        "(BB) turn in tuck" 
        "(BB) salto sideward tucked" 
        "(BB) flic-flac" 
        "(BB) salto tucked" 
        "(UB) circle backward with turn" 
        "(UB) circle forward without turn" 
        "(UB) over high bar" 
        "(UB) salto forward tucked"
)

SEEDS=(0 1 2)

# ===== loop =====
for coarse_label in "${COARSE_LABELS[@]}"; do
  for seed in "${SEEDS[@]}"; do

    clean_label=$(echo "$coarse_label" | sed 's/[^a-zA-Z0-9._-]/_/g')

    output_dir="${OUTPUT_ROOT}/Table2/FineGym-Split-B/${ALG}/${clean_label}"

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