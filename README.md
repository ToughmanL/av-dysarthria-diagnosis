# AV-Dysarthria-Diagnosis

**语言/Language**: [中文](README.md) | [English](README_EN.md)

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)

**论文官方实现：**  
*KGMV-Net: Knowledge-Guided Multi-View Network for Audiovisual Dysarthria Severity Assessment*  

---

## 📌 项目简介

本仓库提供 **KGMV-Net** 的官方实现。KGMV-Net 是一个 **基于知识引导的多视角深度学习框架**，用于 **自动化构音障碍严重程度评估**。  
该框架系统性地融合了 **临床先验知识** 与 **音频和视觉多模态表征**，实现了稳健、可解释且可扩展的病情评估。  

> **摘要**：KGMV-Net 提出了一种知识引导的多视角特征融合框架，有效提取并整合音频和视觉模态的病症相关特征。在 MSDM 数据集上的大量实验表明，KGMV-Net 在严重程度预测任务中显著优于现有方法，展示了其在临床构音障碍客观评估中的应用潜力。

---

## 📖 论文信息

如果本项目对您的研究有帮助，请引用我们的论文：  

```bibtex
@article{liu2024kgmv,
  title={KGMV-Net: Knowledge-Guided Multi-View Network for Audiovisual Dysarthria Severity Assessment},
  author={Liu, Xiaokang and Yang, Yudong and Xu, Guorong and Du, Xiaoxia and Su, Rongfeng and Wang, Lan and Yan, Nan},
  journal={SSRN},
  year={2024},
  note={Available at SSRN: https://ssrn.com/abstract=5332691}
}
```

**论文链接**: [SSRN](https://ssrn.com/abstract=5332691)

---

## 🚀 核心特性

* **多模态融合**：联合建模语音与面部运动，提高预测的准确性与鲁棒性。
* **知识引导模块**：引入临床知识，提升特征表达能力与跨模态对齐效果。
* **端到端流程**：覆盖数据准备、训练、评估与推理的完整流程。
* **科研导向设计**：模块化与可扩展的代码结构，支持复现与二次开发。

---

## ⚙️ 环境要求

* **操作系统**: Linux
* **编程语言**: Python ≥ 3.8
* **深度学习框架**: PyTorch ≥ 1.12
* **硬件要求**: GPU 显存 ≥ 24 GB（如 NVIDIA V100 / A100）
* **依赖包**: 详见 `requirements.txt`

---

## 🛠️ 安装步骤

克隆代码并配置环境：

```bash
git clone https://github.com/your_username/av-dysarthria-diagnosis.git
cd av-dysarthria-diagnosis

conda create -n dysarthria python=3.9 -y
conda activate dysarthria
pip install -r requirements.txt
```

---

## 📁 项目结构

```bash
av-dysarthria-diagnosis/
├── egs/                      # 实验示例
│   ├── msdm/                 # MSDM 数据集配置与脚本
│   ├── conf/                 # 训练配置文件
│   ├── local/                # 数据预处理脚本
│   ├── networks -> ../../networks/   # 网络定义（符号链接）
│   ├── path.sh               # 环境路径设置
│   ├── run.sh                # 主脚本（训练/评估入口）
│   └── tools -> ../../tools/         # 工具脚本（符号链接）
├── networks/                 # 模型架构
│   ├── dataset/              # 数据加载与处理模块
│   ├── model/                # KGMV-Net 模型实现
│   └── utils/                # 工具函数
├── tools/                    # 通用工具
└── README.md                 # 项目文档
```

---

## 🗃️ 数据与预训练模型

* **训练数据**: 实验基于 **[MSDM](https://huanraozhineng1.github.io/MSDM/)** 数据集。请根据其官方许可获取数据。
* **数据准备**: 使用 `egs/msdm/local/` 下的脚本进行预处理。
* **预训练模型**: 若公开，将在此提供下载链接与使用说明。

---

## 🚀 使用方法

项目的主要入口为 `egs/msdm/run.sh`。

### 训练与评估

1. 确保已准备好 MSDM 数据集；
2. 根据需求修改 `egs/msdm/conf/` 下的配置文件；
3. 运行以下命令：

```bash
cd egs/msdm/
bash run.sh
```

该脚本将依次执行数据准备、模型训练与评估。

---

## 🤝 贡献指南

我们欢迎学术界和工业界的共同参与：

* **Bug 反馈**：请通过 [GitHub Issues](../../issues) 提交；
* **功能建议**：欢迎在 Issues 中提出新想法；
* **代码贡献**：通过 Pull Request 提交改进。

---

## 📄 许可证

本项目遵循 **GNU 通用公共许可证 v3.0 (GPL-3.0)** 开源。详情参见 [LICENSE](LICENSE)。

---

## 🙏 致谢

* 感谢本研究的所有合作者与贡献者；
* 感谢 MSDM 数据集提供方。

---

## ❓ 常见问题

1. **问：出现 `CUDA out of memory` 错误怎么办？**
   **答**：尝试在配置文件中减小 `batch_size`，或使用显存更大的 GPU。
