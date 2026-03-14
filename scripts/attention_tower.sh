#!/bin/bash
#SBATCH -p GPUExtended              # partition (queue)
#SBATCH -N 1                        # number of nodes
#SBATCH --job-name=att_tower         # job name
#SBATCH -t 1-36:00                  # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out          # STDOUT
#SBATCH -e slurm.%N.%j.err          # STDERR
#SBATCH --gres=gpu:1                # request GPUs

source ~/.bashrc  
conda activate env_t

export CUDA_VISIBLE_DEVICES=1


INPUT_FILE="$1"                  # Path to the input file
TRANSLATION_FILE="$2"            # Path to the translation file
OUTPUT_DIR="$3"                  # Directory for final outputs
MODEL_NAME="$4"                  # Model name
TARGET_LANG="$5"                 # Target language
SUFFIX="$6"                      # Optional suffix: reg, pro, anti...

SECONDS=0

python scripts/src/collect_attention.py \
    --input_file "$INPUT_FILE" \
    --translation_file "$TRANSLATION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --tgt_lang "$TARGET_LANG" \
    --suffix "$SUFFIX" 


# duration
duration=$SECONDS
echo "Job finished in $(($duration / 60)) minutes and $(($duration % 60)) seconds."
