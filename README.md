<div align="center">

# ARROW: Adaptive Reasoning for LLM-based Recommendation with Explainability

**[Woo-Seong Yun](https://scholar.google.com/citations?user=ZRXyvtMAAAAJ)** &nbsp;·&nbsp; **Min-Seong Kim** &nbsp;·&nbsp; **Yoon-Sik Cho**

<sub>Department of Artificial Intelligence, Chung-Ang University</sub>

*WSDM 2026 (19th ACM International Conference on Web Search and Data Mining), pp. 1283–1287, Boise, ID, USA*

[![Paper](https://img.shields.io/badge/Paper-ACM%20DL-0085CA?logo=acm&logoColor=white)](https://dl.acm.org/doi/10.1145/3773966.3779396)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

</div>

This is the PyTorch implementation for our WSDM 2026 paper:

> ARROW: Adaptive Reasoning for LLM-based Recommendation with Explainability (WSDM, 2026)

<p align="center">
  <img src="assets/overview.png" width="90%" alt="Case study of CoLLM and ARROW on ML-1M">
</p>

## Overview

LLM-based recommenders leverage rich world knowledge, but the semantic gap between linguistic knowledge and collaborative patterns leaves them unable to explain why they recommend an item. To address this limitation, we propose **ARROW** (Adaptive Reasoning for LLM-based RecommendatiOn With explainability), which guides the LLM to first infer the genres a user is likely to enjoy through chain-of-thought prompting and trains this reasoning step jointly with the recommendation objective. The **Adaptive Reasoning Modulator** weights the reasoning loss by the model's own uncertainty, so it reasons harder exactly when it is unsure. On ML-1M and Amazon-Book, ARROW outperforms strong LLM-based baselines on AUC, UAUC and NDCG while producing human-interpretable explanations.

## Requirements

```bash
conda env create -f environment.yml
conda activate minigpt4
```

ARROW uses Vicuna-7B as the LLM backbone. Follow [PrepareVicuna.md](PrepareVicuna.md) to prepare the weights, then set `llama_model` in each file under `train_configs/` to the weight path.

## Datasets

We follow the preprocessing of [CoLLM](https://github.com/zyang1580/CoLLM). Ground-truth genre labels for the reasoning task are generated from item titles with Mistral-Nemo-Instruct.

| Dataset | Train | Valid | Test | Users | Items |
| --- | ---: | ---: | ---: | ---: | ---: |
| ML-1M | 33,891 | 10,401 | 7,331 | 839 | 3,256 |
| Amazon-Book | 727,468 | 25,747 | 25,747 | 22,967 | 34,154 |

Download [MovieLens-1M](https://grouplens.org/datasets/movielens/) and [Amazon-Books](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews), then run the notebooks in `dataset/` to build the splits.

## Training

**Step 1. Pre-train the collaborative filtering model.**

```bash
python baseline_train_mf_ood.py
```

**Step 2. Stage 1 (LoRA tuning).** Set `pretrained_path` to the CF checkpoint and `ckpt: None` in `train_configs/collm_pretrain_mf_ood_stage_1.yaml`, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 torchrun --nproc-per-node 2 --master_port=11139 \
    train_collm_mf_din.py --cfg-path=train_configs/collm_pretrain_mf_ood_stage_1.yaml
```

**Step 3. Stage 2 (collaborative alignment).** Set `ckpt` to the best stage-1 checkpoint in `train_configs/collm_pretrain_mf_ood_stage_2.yaml`, then run the same command with the stage-2 config.

**Step 4. Evaluation.** Set `ckpt` to the best stage-2 checkpoint in `train_configs/collm_pretrain_mf_ood_stage_eval.yaml`, then run the same command with the eval config.

## Results

Bold marks the best result and underline the second best (Table 2 of the paper). All improvements are statistically significant (paired t-test, p ≤ 0.05).

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="3" align="center">ML-1M</th>
      <th colspan="3" align="center">Amazon-Book</th>
    </tr>
    <tr>
      <th align="center">AUC</th>
      <th align="center">UAUC</th>
      <th align="center">NDCG</th>
      <th align="center">AUC</th>
      <th align="center">UAUC</th>
      <th align="center">NDCG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">MF</td>
      <td align="center">0.6482</td>
      <td align="center">0.6361</td>
      <td align="center">0.8447</td>
      <td align="center">0.7119</td>
      <td align="center">0.5554</td>
      <td align="center">0.8194</td>
    </tr>
    <tr>
      <td align="left">SASRec</td>
      <td align="center">0.7055</td>
      <td align="center">0.6885</td>
      <td align="center">0.8612</td>
      <td align="center">0.6829</td>
      <td align="center">0.5800</td>
      <td align="center">0.8244</td>
    </tr>
    <tr>
      <td align="left">ICL</td>
      <td align="center">0.5320</td>
      <td align="center">0.5268</td>
      <td align="center">0.8102</td>
      <td align="center">0.4820</td>
      <td align="center">0.4856</td>
      <td align="center">0.7917</td>
    </tr>
    <tr>
      <td align="left">Prompt4NR</td>
      <td align="center">0.7071</td>
      <td align="center">0.6739</td>
      <td align="center">0.8541</td>
      <td align="center">0.7224</td>
      <td align="center">0.5881</td>
      <td align="center">0.8346</td>
    </tr>
    <tr>
      <td align="left">TALLRec</td>
      <td align="center">0.7072</td>
      <td align="center">0.6743</td>
      <td align="center">0.8578</td>
      <td align="center">0.7209</td>
      <td align="center">0.5814</td>
      <td align="center">0.8361</td>
    </tr>
    <tr>
      <td align="left">CoLLM (MF)</td>
      <td align="center">0.7295</td>
      <td align="center">0.6798</td>
      <td align="center">0.8658</td>
      <td align="center">0.8107</td>
      <td align="center">0.6029</td>
      <td align="center">0.8557</td>
    </tr>
    <tr>
      <td align="left">BinLLM</td>
      <td align="center"><ins>0.7412</ins></td>
      <td align="center">0.6951</td>
      <td align="center"><ins>0.8747</ins></td>
      <td align="center"><ins>0.8186</ins></td>
      <td align="center"><ins>0.6338</ins></td>
      <td align="center"><ins>0.8580</ins></td>
    </tr>
    <tr>
      <td align="left">CoRA</td>
      <td align="center">0.7410</td>
      <td align="center"><ins>0.7061</ins></td>
      <td align="center">0.8728</td>
      <td align="center">0.8109</td>
      <td align="center">0.5975</td>
      <td align="center">0.8413</td>
    </tr>
    <tr>
      <td align="left"><b>ARROW</b></td>
      <td align="center"><b>0.7577</b></td>
      <td align="center"><b>0.7146</b></td>
      <td align="center"><b>0.8792</b></td>
      <td align="center"><b>0.8198</b></td>
      <td align="center"><b>0.6576</b></td>
      <td align="center"><b>0.8653</b></td>
    </tr>
  </tbody>
</table>

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yun2026arrow,
  title     = {ARROW: Adaptive Reasoning for LLM-based Recommendation with Explainability},
  author    = {Woo-Seong Yun and Min-Seong Kim and Yoon-Sik Cho},
  booktitle = {Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  series    = {WSDM '26},
  pages     = {1283--1287},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3773966.3779396}
}
```

## Acknowledgements

This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)] and partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00419201).

Our implementation builds on [CoLLM](https://github.com/zyang1580/CoLLM) and [BinLLM](https://github.com/zyang1580/BinLLM).
