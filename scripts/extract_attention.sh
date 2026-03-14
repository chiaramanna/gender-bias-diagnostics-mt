#!/bin/bash
#SBATCH -p GPUExtended              # partition (queue)
#SBATCH -N 1                        # number of nodes
#SBATCH --job-name=translation_tower  # job name
#SBATCH -t 1-36:00                  # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out          # STDOUT
#SBATCH -e slurm.%N.%j.err          # STDERR
#SBATCH --gres=gpu:1                # request GPUs

source ~/.bashrc  
conda activate env_t

export CUDA_VISIBLE_DEVICES=1

INPUT_FILE="$1"                  # Path to the input file
TRANSLATION_FILE="$2"            # Path to the translation file
RESULTS="$3"                     # Directory for att/vs full results
OUTPUT_FILE_ATT="$4"             # Directory for final outputs
ALIGNMENT_FILE="$5"              # Alignment file
MODEL_NAME="$6"                  # Model name
TARGET_LANG="$7"                 # Target language

SECONDS=0

python scripts/src/extract_attention.py \
    --input_file "$INPUT_FILE" \
    --translation_file "$TRANSLATION_FILE" \
    --results "$RESULTS" \
    --output_file_attention "$OUTPUT_FILE_ATT" \
    --alignment_file "$ALIGNMENT_FILE" \
    --model_id "$MODEL_NAME" \
    --tgt_lang "$TARGET_LANG" 


duration=$SECONDS
echo "Job finished in $(($duration / 60)) minutes and $(($duration % 60)) seconds."
