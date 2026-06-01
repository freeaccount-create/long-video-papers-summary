#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1

# ===== args =====
model=$1
checkpoint=$2
dataset=$3
label_dir=$4
video_dir=$5
alg=$6
weight_init=$7
coarse_grained_text_label=$8
modifiers_in_base_model=$9
modifiers_for_new_classes=${10}
seed=${11}
output_dir=${12}


module purge
module load 2023
module load CUDA/12.1.1

python main.py \
        --model $model \
        --checkpoint $checkpoint \
        --dataset $dataset \
        --label_dir $label_dir \
        --video_dir $video_dir \
        --alg $alg \
        --weight_init $weight_init \
        --coarse_grained_text_label "$coarse_grained_text_label" \
        --modifiers_in_base_model $modifiers_in_base_model \
        --modifiers_for_new_classes $modifiers_for_new_classes \
        --enable_seed \
        --seed $seed \
        --output_dir $output_dir \