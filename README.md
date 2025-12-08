# ATMT Assignment 5: Improving Decoding in NMT

**Student:** Chu Jia (23-745-912)  
**Course:** Advanced Techniques of Machine Translation (HS25)

---

## 📂 Repository Structure & Submitted Files

This repository contains the code and experimental results for **Assignment 5**.

### 1. Code Changes
* **`seq2seq/decode.py`**: 
    * Optimized `decode` and `beam_search_decode` by moving the Encoder out of the loop (Section 3).
    * Implemented **Length Penalty** logic (Section 4.1).
    * Implemented **Stopping Criteria** (Patience/N-Best) (Section 4.2).
* **`translate.py`**: 
    * Fixed mask generation logic to prevent `CPUBoolType` errors during inference.
    * Updated decoder calls to use keyword arguments for compatibility.

### 2. Experimental Output (Evidence)
All output files generated during the experiments are included:

| File Name | Description | Related Task |
| :--- | :--- | :--- |
| `output_baseline.txt` | Translation of 100 sentences **before optimization** (Slow). | Section 3 |
| `output_optimized.txt` | Translation **after optimization** (Fast). | Section 3 |
| `output_small_optimized.txt` | Speed test on small sample (50 sentences). | Section 3 |
| `output_beam3.txt` | Beam Search (=3$) output showing quality improvement. | Section 3 |
| `output_alpha0.txt` | Translation with Length Penalty $\alpha=0.0$ (Shorter). | Section 4.1 |
| `output_alpha1.txt` | Translation with Length Penalty $\alpha=1.0$ (Longer). | Section 4.1 |
| `output_stop_best.txt` | Result using **Best-One Stopping** criterion. | Section 4.2 |
| `output_stop_patience.txt` | Result using **Patience Stopping** (=3$) criterion. | Section 4.2 |

---

## 🚀 How to Run

To reproduce the translation with optimized beam search:

```bash
python translate.py \
    --checkpoint-path assignment3_2/checkpoints/checkpoint_best.pt \
    --input toy_example/data/raw/test.cz \
    --output output_final.txt \
    --src-tokenizer assignment3_2/tokenizers/cz-en-joint-bpe-16000.model \
    --tgt-tokenizer assignment3_2/tokenizers/cz-en-joint-bpe-16000.model \
    --beam-size 5 \
    --alpha 1.0
```

---

## 📝 Baseline Model
* **Model Used:** Transformer (Joint BPE 16k) from **Assignment 3 Task 1**.
* **Checkpoint:** `assignment3_2/checkpoints/checkpoint_best.pt`

