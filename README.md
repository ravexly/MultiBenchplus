
# MultiBench++

Official code of MULTIBENCH++: A Unified and Comprehensive Multimodal Fusion Benchmarking Across Specialized Domains

## Requirements

### Dataset

Follow [DATASET.md](https://www.google.com/search?q=DATASET.md) to install datasets.

## Get Started

### 1\. Data Preparation

All datasets should be organized under the `data/` directory.

**Example: Setting up the MAMI Dataset**

1.  Follow [DATASET.md](https://www.google.com/search?q=DATASET.md) to download the MAMI dataset. Alternative Download: We also provide a Baidu Netdisk link for quicker access to the datasets:Link: [Baidu Netdisk ](https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9)(Password: 8rc9)
Note: We are actively maintaining and updating this link. Some datasets might not be fully available yet; please stay tuned for upcoming updates.

2.  Place the data into `data/MAMI/`.

<!-- end list -->

```text
ProjectRoot/
├── data/
│   └── MAMI/
├── exper/
│   └── baseline/
│       ├── mami_baseline.py
│       ├── healthcare_baseline.py
│       └── ...
└── ...
```

### 2\. Running Experiments

We provide standardized scripts in the `baseline/` directory to facilitate quick reproduction across all datasets.

```bash
# Navigate to the baseline directory (e.g., MAMI)
cd baseline/MAMI

# Run the specific experiment script 
python baseline.py
```

-----

##  Key Design Notes

### Encoder Selection Strategy

To ensure a **pure comparison of fusion performance**, our experimental setup follows these principles:

  * **Simplicity First:** We prioritize using the simplest possible encoders to isolate the impact of the fusion mechanism. For pre-processed datasets where features are already extracted, we utilize an **Identity Encoder**. In most scenarios, adding a simple MLP (Multi-Layer Perceptron) can directly boost performance by projecting features into a more suitable space.
  * **Combinatorial Optimization:** Please note that the choice of **Encoder** and **Fusion Method** is a combinatorial problem. A specific fusion method's performance may vary significantly depending on the underlying encoder architecture.

### Encoders Setup

For BERT-based experiments, download `bert-base-uncased` ([Google Drive Link](https://drive.google.com/file/d/1ivh-3aHtoqRMwVN4ZOPvPm59pFP93-hD/view)) and place it in the root folder: `bert-base-uncased/`.

-----

### Acknowledgements

Our implementation is built upon [MULTIBENCH](https://github.com/pliang279/MultiBench). We thank the authors for their excellent open-source work.
