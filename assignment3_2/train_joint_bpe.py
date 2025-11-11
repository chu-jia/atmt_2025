import sentencepiece as spm
import os
import sys

print("Starting Joint BPE training with Python...")

# task_dir 现在是当前目录 ('.')
task_dir = '.' 

# 路径现在相对于 assignment3_2
input_file = os.path.join(task_dir, 'data/cz-en.train.joint')
model_prefix = os.path.join(task_dir, 'tokenizers/cz-en-joint-bpe-16000')

# 确保文件夹存在
os.makedirs(os.path.join(task_dir, 'tokenizers'), exist_ok=True)

if not os.path.exists(input_file):
    print(f"Error: Input file not found at {input_file}", file=sys.stderr)
    print("Please make sure you have run the 'cat' command to create it.", file=sys.stderr)
else:
    print(f"Found input file: {input_file}")
    print(f"Will save model to: {model_prefix}.model")

    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            pad_id=3,
            vocab_size=16000,
            model_type='bpe',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            pad_piece='<pad>',
            character_coverage=0.9995,
            num_threads=16,

            # --- 解决内存问题的关键 ---
            input_sentence_size=5000000,  # 仅使用 500 万行样本
            shuffle_input_sentence=True   # 随机抽取这 500 万行
        )
        print(f"Success! Model saved to {model_prefix}.model")
    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
