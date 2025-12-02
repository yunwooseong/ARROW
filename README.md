# [WSDM' 26] ARROW: Adaptive Reasoning for LLM-based Recommendation with Explainability

**This repository is built on [CoLLM](https://github.com/zyang1580/CoLLM). Please refer to CoLLM's "readme.md" for an overview of the code structure.**

## 1. Prepare the requirements and datasets

### Requirements and Environment Setup

Creating a python environment and activate it:
```bash
conda env create -f environment.yml
conda activate minigpt4
```

### Prepare Pretrained Vicuna Weights

This project uses the Vicuna-7B model as its LLM backbone. Please follow the instructions from Mini-GPT4 [PrepareVicuna.md](PrepareVicuna.md) to download and prepare the Vicuna weights.
Once prepared, you must set the correct path to the Vicuna weights in the `"llama_model"` field within the training configuration files (e.g., [train_configs/collm_pretrain_mf_ood_stage_1.yaml](train_configs/collm_pretrain_mf_ood_stage_1.yaml)).


### Prepare and Pre-process the Datasets

Download the datasets from the following links:
- MovieLens-1M : https://grouplens.org/datasets/movielens/
- Amazon-Books : https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews

The data pre-processing is based on CoLLM. You can use the scripts provided in the `./dataset` directory to process the data yourself.

## 2. ARROW example command

### Step 1. Prepare the Pretrained CF Model

Before performing LLMs with collaborative information, you should pre-train a collaborative filtering model
with the following command:
```shell
python baseline_train_mf_ood.py
...
```

### Step 2. Performing ARROW in the 1st Stage.

Set the hyper-parameters in the training config file  [train_configs/collm_pretrain_mf_ood_stage_1.yaml](train_configs/collm_pretrain_mf_ood_stage_1.yaml) as follows:
```
- pretrained_path: pretrained_collab_model_path
- ckpt: None 
```
Then run the following command:
```
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood_stage_1.yaml > /log_result.log &
```

### Step 3. Performing ARROW in the 2nd Stage.

Set the hyper-parameters in the training config file  [train_configs/collm_pretrain_mf_ood_stage_2.yaml](train_configs/collm_pretrain_mf_ood_stage_2.yaml) as follows:
```
- pretrained_path: pretrained_collab_model_path
- ckpt: 1st_stage_checkpoint_best_path
```
Then run the following command:
```
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood_stage_2.yaml > /log_result.log &
```

### Step 4. Performing ARROW in the Inference Stage.

Set the hyper-parameters in the training config file [train_configs/collm_pretrain_mf_ood_stage_eval.yaml](train_configs/collm_pretrain_mf_ood_stage_eval.yaml) as follows:
```
- pretrained_path: pretrained_collab_model_path
- ckpt: 2nd_stage_checkpoint_best_path
```
Then run the following command:
```
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 nohup torchrun --nproc-per-node 2 --master_port=11139 train_collm_mf_din.py  --cfg-path=train_configs/collm_pretrain_mf_ood_stage_eval.yaml > /log_result.log &
```

### Acknowledgements

Our repository is built upon [CoLLM](https://arxiv.org/abs/2310.19488]) and [BinLLM](https://aclanthology.org/2024.acl-long.497/), and we sincerely appreciate the contributions of their authors.

