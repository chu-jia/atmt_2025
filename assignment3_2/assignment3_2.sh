#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=03:00:00        # 3 hours of time (just in case)
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment3_task1.out  # We will overwrite the old, failed log
#SBATCH --error=err_assignment3_task1.out # A new file for errors

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Limit CPU threads
export OMP_NUM_THREADS=1

echo "--- Task 1: Joint BPE (Evaluation-Only Mode) ---"

# --- STEP 1: PREPARE DATA (SKIPPED) ---
echo "--- Preprocessing is already complete. Skipping. ---"
# python ../preprocess.py \
#     ... (all preprocess commands are commented out) ...


# --- STEP 2: TRAIN (SKIPPED) ---
echo "--- Training is already complete. Skipping. ---"
# python ../train.py \
#     ... (all train commands are commented out) ...


# --- STEP 3: TRANSLATE (CZECH) ---
# This step already worked, but we run it again
echo "--- Evaluating Task 1 (CZ)... ---"
python ../translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer ./tokenizers/cz-en-joint-bpe-16000.model \
    --tgt-tokenizer ./tokenizers/cz-en-joint-bpe-16000.model \
    --checkpoint-path ./checkpoints/checkpoint_best.pt \
    --output ./output.cz_test.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --batch-size 32

# --- STEP 4: TRANSLATE (SLOVAK) [FIXED] ---
# We now use the correct paths we found
echo "--- Evaluating Task 1 (SK)... ---"
python ../translate.py \
    --cuda \
    --input /home/cjia/data/atmt_2025/experiments/sk-en/sk_5k.sk \
    --src-tokenizer ./tokenizers/cz-en-joint-bpe-16000.model \
    --tgt-tokenizer ./tokenizers/cz-en-joint-bpe-16000.model \
    --checkpoint-path ./checkpoints/checkpoint_best.pt \
    --output ./output.sk_test.txt \
    --max-len 300 \
    --bleu \
    --reference /home/cjia/data/atmt_2025/experiments/sk-en/en_5k.en \
    --batch-size 32

echo "--- Task 1 (Joint BPE) Evaluation Complete ---"
