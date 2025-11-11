# ATMT Assignment 3: Improving a Baseline NMT System

* **Author:** Chu Jia (23-745-912)
* **Tasks:** Task 1 (Joint BPE) & Task 2 (Checkpoint Averaging)

---

## Assignment Overview

This repository contains the code, scripts, and translation outputs for Assignment 3. The goal was to improve upon the baseline Czech-to-English (CZ-EN) NMT model from Assignment 1.

Two improvement methods were implemented:
1.  **Task 1 (Retraining): Joint BPE** 
    * **Goal:** To improve cross-lingual generalization (especially to Slovak) by training a single shared BPE tokenizer for both Czech and English.
2.  **Task 2 (Refining): Checkpoint Averaging** 
    * **Goal:** To produce a more stable and robust model by averaging the parameters of the final 3-5 training epochs.

## Final Results

Both long-running training jobs (`assignment3_2.sh` and `assignment3_task2.sh`) failed prematurely after completing all 7 epochs but before finishing their evaluation steps. The final BLEU scores were successfully recovered by running dedicated evaluation scripts (`eval.sh`, `eval_task2.sh`) on the saved checkpoints.

The final results show a significant relative improvement over the baseline.

| Experiment | Test Set | Baseline BLEU (Ass1) | **Final Test BLEU (Ass3)** |
| :--- | :--- | :--- | :--- |
| **Task 1 (Joint BPE)** | Czech (test.cz) | 0.50 | **1.47** |
| **Task 1 (Joint BPE)** | **Slovak (sk_5k.sk)** | **0.10** | **2.00** |
| **Task 2 (Ckpt Avg.)**| Czech (test.cz) | 0.50 | **2.02** |

### Note on BLEU Scores

The final test set BLEU scores (1.47, 2.02) are numerically low. All evaluation logs produced a `sacrebleu` warning: `It looks like you forgot to detokenize your test data...`. This implies the baseline (0.50) was also calculated on tokenized data.

The key finding is the **relative improvement**:
* **Task 1** improved the Slovak generalization 20-fold (0.10 $\to$ 2.00).
* **Task 2** produced the best model, improving the Czech score 4-fold (0.50 $\to$ 2.02).

---

## File Descriptions

This repository contains all code changes and test set translations as required by the assignment.

### Task 1: Joint BPE

* `train_joint_bpe.py`: A Python script used to train the new 16k-vocabulary joint BPE model. It uses a 5M-sentence random sample to avoid memory errors on the login node.
* `assignment3_2.sh`: The **final evaluation script for Task 1**. This script was modified from the original training script. It skips the (already completed) training and runs the `translate.py` command on the CZ and SK test sets using the correct file paths.
* `output.cz_test.txt`: The final test set translation (5000 sentences) for Task 1 (CZ).
* `output.sk_test.txt`: The final test set translation (5000 sentences) for Task 1 (SK).

### Task 2: Checkpoint Averaging

* `assignment3_task2.sh`: The **original (failed)** training script for Task 2. This log (`out_assignment3_task2.out`) confirmed that all 7 epoch checkpoints were successfully saved before the script failed.
* `average_checkpoints.py`: The utility script used to average checkpoint files.
* `eval_task2.sh`: The **final evaluation script for Task 2**. This script was created to "rescue" the results from the failed job. It correctly averages the final 3 checkpoints (Epochs 4, 5, 6) using their full filenames (e.g., `checkpoint4_5.856.pt`) and then runs `translate.py` on the averaged model.
* `task2_checkpoint_avg/`: This directory holds all outputs for Task 2.
    * `output.cz_test.avg.txt`: The final test set translation (5000 sentences) for the averaged Task 2 model.

