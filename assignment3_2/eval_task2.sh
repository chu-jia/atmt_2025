#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=03:00:00        # 3 hours of time
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_eval_task2.out    # New log file for Task 2
#SBATCH --error=err_eval_task2.out     # New error file for Task 2

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Limit CPU threads
export OMP_NUM_THREADS=1

echo "--- Task 2 Evaluation Start (Working Directory: $(pwd)) ---"

# --- STEP 1: Average Checkpoints (using correct filenames) ---
echo "--- Averaging final 3 checkpoints (Epochs 4, 5, 6)... ---"
python average_checkpoints.py \
    --inputs ./task2_checkpoint_avg/checkpoints/checkpoint4_5.856.pt \
             ./task2_checkpoint_avg/checkpoints/checkpoint5_5.594.pt \
             ./task2_checkpoint_avg/checkpoints/checkpoint6_5.462.pt \
    --output ./task2_checkpoint_avg/checkpoints/checkpoint_avg_4-6.pt

# --- STEP 2: Translate (using the averaged model) ---
echo "--- Translating with averaged model... ---"
python ../translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer ../cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer ../cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path ./task2_checkpoint_avg/checkpoints/checkpoint_avg_4-6.pt \
    --output ./task2_checkpoint_avg/output.cz_test.avg.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --batch-size 32

echo "--- Task 2 Evaluation Complete ---"
