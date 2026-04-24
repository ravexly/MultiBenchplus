<div align="center">

# MultiBench++

### A Unified and Comprehensive Multimodal Fusion Benchmarking Across Specialized Domains

<p>
  <a href="https://arxiv.org/abs/2511.06452">
    <img src="https://img.shields.io/badge/arXiv-2511.06452-b31b1b?style=flat-square" alt="arXiv">
  </a>
  <a href="https://ravexly.github.io/MultiBenchplus/">
    <img src="https://img.shields.io/badge/Project_Page-MultiBench++%20%7C%20Comprehensive%20Multimodal%20Fusion%20Benchmark-2ea44f?style=flat-square" alt="Project Page">
  </a>
</p>

</div>

---

This repository contains the official implementation of **MULTIBENCH++**, a unified and comprehensive benchmark for evaluating **multimodal fusion methods** across specialized domains. It is designed to support fair, reproducible, and systematic comparison of fusion strategies under a consistent experimental pipeline.

By emphasizing controlled encoder settings and broad domain coverage, **MULTIBENCH++** helps researchers better understand how fusion methods behave across different modalities, tasks, and data conditions.

## Requirements

### Dataset

Please follow [DATASET.md](DATASET.md) to prepare the required datasets.

## Getting Started

### 1. Data Preparation

All datasets should be organized under the `data/` directory.

#### Example: Setting up the MAMI dataset

1. Follow [DATASET.md](DATASET.md) to download the MAMI dataset.
2. We also provide a Baidu Netdisk mirror for quicker access:
   - **Link:** [Baidu Netdisk](https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9)
   - **Password:** `8rc9`
3. Place the downloaded data under `data/MAMI/`.

> Note:
> We are actively maintaining and updating the Baidu Netdisk link. Some datasets may not be fully available yet.

```text
ProjectRoot/
|-- data/
|   `-- MAMI/
|-- exper/
|   `-- baseline/
|       |-- mami_baseline.py
|       |-- healthcare_baseline.py
|       `-- ...
`-- ...
```

### 2. Running Experiments

We provide standardized scripts in the `baseline/` directory to facilitate quick reproduction across datasets.

```bash
# Navigate to the baseline directory for a target dataset
cd baseline/MAMI

# Run the experiment
python baseline.py
```

## Key Design Notes

### Encoder Selection Strategy

To ensure a **clean comparison of fusion performance**, our experimental setup follows the principles below:

- **Simplicity First:** We prioritize simple encoders to better isolate the contribution of the fusion mechanism. For pre-extracted features, we use an **Identity Encoder**. In many scenarios, a lightweight **MLP** can further improve performance by projecting features into a more suitable space.
- **Combinatorial Optimization:** The choice of **encoder** and **fusion method** is inherently combinatorial. The effectiveness of a given fusion strategy may vary significantly depending on the underlying encoder architecture.

### Encoder Setup

For BERT-based experiments, please download `bert-base-uncased` from the link below and place it in the project root:

- [bert-base-uncased (Google Drive)](https://drive.google.com/file/d/1ivh-3aHtoqRMwVN4ZOPvPm59pFP93-hD/view)

Expected structure:

```text
ProjectRoot/
|-- bert-base-uncased/
|-- data/
`-- ...
```

## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{liang2021multibench,
  title={MultiBench: Multiscale Benchmarks for Multimodal Representation Learning},
  author={Liang, Paul Pu and Lyu, Yiwei and Fan, Xiang and Wu, Zetian and Cheng, Yun and Wu, Jason and Chen, Leslie Yufan and Wu, Peter and Lee, Michelle A and Zhu, Yuke and Salakhutdinov, Ruslan and Morency, Louis-Philippe},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  year={2021}
}

@inproceedings{xue2026multibench,
  title={MULTIBENCH++: A Unified and Comprehensive Multimodal Fusion Benchmarking Across Specialized Domains},
  author={Leyan Xue and Changqing Zhang and Kecheng Xue and Xiaohong Liu and Guangyu Wang and Zongbo Han},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## Acknowledgements

Our implementation is built upon [MULTIBENCH](https://github.com/pliang279/MultiBench). We sincerely thank the authors for their excellent open-source contribution.
