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

