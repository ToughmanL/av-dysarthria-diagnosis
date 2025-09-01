# AV-Dysarthria-Diagnosis

**语言/Language**: [中文](README.md) | [English](README_EN.md)

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)

**Official implementation of the paper:**  
*KGMV-Net: Knowledge-Guided Multi-View Network for Audiovisual Dysarthria Severity Assessment*

---

## 📌 Overview

This repository provides the official implementation of **KGMV-Net**, a **knowledge-guided multi-view deep learning framework** for **automatic dysarthria severity assessment**.  
The framework systematically integrates **clinical prior knowledge** with **multimodal (acoustic and visual) representations**, enabling robust, interpretable, and scalable evaluation of dysarthria severity.  

> **Abstract**: KGMV-Net introduces a knowledge-guided multi-view feature fusion framework that extracts and integrates dysarthria-relevant cues across acoustic and visual modalities. Extensive experiments on the MSDM dataset demonstrate that KGMV-Net achieves state-of-the-art performance, significantly surpassing existing baselines. These results highlight the potential of KGMV-Net as a reliable and objective tool for clinical applications in dysarthria assessment.

---

## 📖 Paper Information

If you find this repository useful, please cite our work:

```bibtex
@article{liu2024kgmv,
  title={KGMV-Net: Knowledge-Guided Multi-View Network for Audiovisual Dysarthria Severity Assessment},
  author={Liu, Xiaokang and Yang, Yudong and Xu, Guorong and Du, Xiaoxia and Su, Rongfeng and Wang, Lan and Yan, Nan},
  journal={SSRN},
  year={2024},
  note={Available at SSRN: https://ssrn.com/abstract=5332691}
}
```

**Paper link**: [SSRN](https://ssrn.com/abstract=5332691)

---

## 🚀 Key Features

* **Multimodal Fusion**: Joint modeling of speech and facial motion for accurate and robust dysarthria severity prediction.
* **Knowledge-Guided Modules**: Incorporates clinical knowledge to guide feature extraction and cross-modal alignment.
* **End-to-End Pipeline**: From data preparation to training, evaluation, and inference.
* **Research-Oriented Design**: Modular and extensible codebase, facilitating reproducibility and secondary development.

---

## ⚙️ Requirements

* **Operating System**: Linux
* **Programming Language**: Python ≥ 3.8
* **Framework**: PyTorch ≥ 1.12
* **Hardware**: GPU with ≥ 24 GB memory (e.g., NVIDIA V100 / A100)
* **Dependencies**: Specified in `requirements.txt`

---

## 🛠️ Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/your_username/av-dysarthria-diagnosis.git
cd av-dysarthria-diagnosis

conda create -n dysarthria python=3.9 -y
conda activate dysarthria
pip install -r requirements.txt
```


## 📁 Project Structure

```bash
av-dysarthria-diagnosis/
├── egs/                      # Example experiments
│   ├── msdm/                 # MSDM dataset configuration & scripts
│   ├── conf/                 # Training configuration files
│   ├── local/                # Data preparation scripts
│   ├── networks -> ../../networks/   # Network definitions (symlink)
│   ├── path.sh               # Environment path setup
│   ├── run.sh                # Main entry script (training/evaluation)
│   └── tools -> ../../tools/         # Utility scripts (symlink)
├── networks/                 # Model architectures
│   ├── dataset/              # Data loading and processing
│   ├── model/                # KGMV-Net implementation
│   └── utils/                # Utility functions
├── tools/                    # General-purpose utilities
└── README.md                 # Documentation
```

---

## 🗃️ Data and Pretrained Models

* **Dataset**: Experiments are conducted on the **[MSDM](https://huanraozhineng1.github.io/MSDM/)**  dataset. Users must acquire the dataset in accordance with its official license.
* **Data Preparation**: Scripts for preprocessing and preparation are provided under `egs/msdm/local/`.
* **Pretrained Models**: If released, pretrained weights and download instructions will be provided here.

---

## 🚀 Usage

The primary entry point is the script `egs/msdm/run.sh`.

### Training and Evaluation

1. Prepare the MSDM dataset.
2. Adjust the configuration files in `egs/msdm/conf/` as required.
3. Execute the following:

```bash
cd egs/msdm/
bash run.sh
```

This script sequentially performs data preparation, model training, and evaluation.

---

## 🤝 Contributions

We welcome contributions from both academia and industry:

* **Bug Reports**: Submit issues via [GitHub Issues](../../issues).
* **Feature Requests**: Suggest new ideas through Issues.
* **Code Contributions**: Submit Pull Requests for improvements.

---

## 📄 License

This project is distributed under the **GNU General Public License v3.0 (GPL-3.0)**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* We thank all co-authors and collaborators of this research.
* We acknowledge the MSDM dataset providers for enabling this work.

---

## ❓ FAQ

1. **Q: Encounter `CUDA out of memory`?**
   **A**: Reduce `batch_size` in the configuration file or use a GPU with larger memory capacity.
