#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=03:00:00        # 3 小时时间
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1                 # 我们请求一个GPU
#SBATCH --output=out_eval.out    # 日志文件

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# 限制CPU线程
export OMP_NUM_THREADS=1

echo "--- 评估开始 (工作目录: $(pwd)) ---"

# --- 任务1 (联合BPE) 评估 ---
echo "--- 正在评估 任务1 (CZ)... ---"
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

echo "--- 正在评估 任务1 (SK)... ---"
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

# --- 任务2 (检查点平均) 评估 ---
echo "--- 正在评估 任务2 (Avg 1-2)... ---"
python ../translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer ../cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer ../cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path ./task2_checkpoint_avg/checkpoints/checkpoint_avg_1-2.pt \
    --output ./task2_checkpoint_avg/output.cz_test.avg.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --batch-size 32

echo "--- 所有评估已完成 ---"
