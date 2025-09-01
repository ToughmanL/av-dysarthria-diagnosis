# AV-Dysarthria-Diagnosis

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)

**Official implementation of the paper:
*KGMV-Net: Knowledge-guided Multi-View Network for Audiovisual Dysarthria Severity Assessment***

This repository provides an end-to-end **automatic dysarthria severity assessment system** based on deep learning. The system leverages audio-visual multimodal data and incorporates a knowledge-guided mechanism to enhance feature representation and cross-modal integration. It aims to serve as an objective, scalable, and efficient tool to assist clinical diagnosis and rehabilitation.

> **Abstract**: KGMV-Net introduces a knowledge-guided multi-view learning framework that effectively fuses acoustic and visual features, significantly improving the modeling and prediction of dysarthria severity.

---

## 📖 Paper Information

If you find this work useful, please cite our paper:

```bibtex
@article{liu2024kgmv,
  title={KGMV-Net: Knowledge-Guided Multi-View Network for Audiovisual Dysarthria Severity Assessment},
  author={Liu, Xiaokang and Yang, Yudong and Xu, Guorong and Du, Xiaoxia and Su, Rongfeng and Wang, Lan and Yan, Nan},
  journal={SSRN},
  year={2024},
  note={Available at SSRN: https://ssrn.com/abstract=5332691}
}
```

**Paper link**: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5332691)

---

## 🚀 Key Features

* **Multimodal Fusion**: Joint modeling of speech and video signals for improved accuracy and robustness.
* **Knowledge-Guided Learning**: Incorporates prior knowledge to enhance feature representation and cross-modal alignment.
* **End-to-End Pipeline**: Covers the entire workflow from data preparation to training, evaluation, and inference.
* **Modular Design**: Clear code structure for reproducibility, extensibility, and secondary development.

---

## ⚙️ Requirements

* **Operating System**: Linux
* **Programming Language**: Python ≥ 3.8
* **Deep Learning Framework**: PyTorch ≥ 1.12
* **Hardware Requirement**: GPU with ≥ 24 GB memory (e.g., NVIDIA V100 / A100)
* **Dependencies**: listed in `requirements.txt`

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/av-dysarthria-diagnosis.git
   cd av-dysarthria-diagnosis
   ```

2. Create and activate the environment:

   ```bash
   conda create -n dysarthria python=3.9 -y
   conda activate dysarthria
   pip install -r requirements.txt
   ```

---

## 📁 Project Structure

```bash
av-dysarthria-diagnosis/
├── egs/                     # Example experiments
│   ├── msdm/                 # MSDM dataset configuration and scripts
│   ├── conf/                 # Training configuration files
│   ├── local/                # Data preparation scripts
│   ├── networks -> ../../networks/  # Network definitions (symlink)
│   ├── path.sh               # Environment path setup
│   ├── run.sh                # Main entry script (training/evaluation)
│   └── tools -> ../../tools/        # Utility scripts (symlink)
├── networks/                 # Network architectures
│   ├── dataset/              # Data loading and processing modules
│   ├── model/                # KGMV-Net model definition
│   └── utils/                # Utility functions
├── tools/                    # General utility scripts
└── README.md                 # Project documentation
```

---

## 🗃️ Data and Models

* **Training Data**: The model is trained on the **MSDM** dataset. Please obtain the dataset according to its official license agreement.
* **Data Preparation**: Follow the scripts under `egs/msdm/local/` for preprocessing and preparation.
* **Pretrained Models**: If pretrained weights are available, please add the download link and usage instructions here.

---

## 🚀 Usage

The main entry point of the project is the `egs/msdm/run.sh` script.

### Training and Evaluation

1. Ensure the MSDM dataset is correctly prepared.
2. Modify the configuration files in `egs/msdm/conf/` as needed.
3. Run the script:

   ```bash
   cd egs/msdm/
   bash run.sh
   ```

   This will sequentially execute data preparation, model training, and evaluation.

---

## 🤝 Contribution

We welcome contributions from both academia and industry:

* **Bug Reports**: Submit issues via [GitHub Issues](../../issues).
* **Feature Requests**: Propose new ideas through Issues.
* **Code Contributions**: Submit improvements via Pull Requests.

---

## 📄 License

This project is released under the **GNU General Public License v3.0 (GPL-3.0)**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* We thank all co-authors and contributors to this research.
* We acknowledge the providers of the MSDM dataset.

---

## ❓ FAQ

1. **Q: Encounter `CUDA out of memory` error?**
   **A**: Try reducing the `batch_size` in the configuration file or using a GPU with larger memory.
