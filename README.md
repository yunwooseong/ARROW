# [Under review] ARROW: Adaptive Reasoning for LLM-based Recommendation with Explainability

**This repository is built on [CoLLM](https://github.com/zyang1580/CoLLM). Please refer to CoLLM's "readme.md" for an overview of the code structure.**

## 1. Prepare the Dataset and requirements

### Datasets

MovieLens-1M :  https://grouplens.org/datasets/movielens/

Amazon-Books : https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews

### Data pre-processing
The data pre-processing is based on CoLLM.

## 2. ARROW example command

### Step 1. Setting Up the Environment and Preparing Vicuna, Pretrained CF Model.
---
### Step 2. Performing ARROW in the 1st Stage.
```
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood_stage_1.yaml > /log_result.log &
```
### Step 3. Performing ARROW in the 2nd Stage.
```
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood_stage_2.yaml > /log_result.log &
```
### Step 4. Performing ARROW in the Inference Stage.
```
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood_stage_eval.yaml > /log_result.log &
```
### Acknowledgements
Our repository is built upon [CoLLM](https://arxiv.org/abs/2310.19488]) and [BinLLM](https://aclanthology.org/2024.acl-long.497/), and we sincerely appreciate the contributions of their authors.
