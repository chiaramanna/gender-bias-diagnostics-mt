#!/bin/bash
#SBATCH -p GPU              # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH --job-name=translation_llm  # job name
#SBATCH -t 0-36:00          # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out  # STDOUT
#SBATCH -e slurm.%N.%j.err  # STDERR
#SBATCH --gres=gpu:1        # request one GPU

#module load python/3.9.7    
export CUDA_VISIBLE_DEVICES=1
source activate env_t

python scripts/src/translate.py \
    --input_file "$1" \
    --output_file "$2" \
    --model_name "$3" \
    --src_lang "$4"\
    --tgt_lang "$5" \
    $( [ "$6" == "True" ] && echo "--quantization" )

duration=$SECONDS
echo "Job finished in $(($duration / 60)) minutes and $(($duration % 60)) seconds."

