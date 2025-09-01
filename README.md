# AV-Dysarthria-Diagnosis

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)

**Official implementation of the paper:
*KGMV-Net: Knowledge-guided Multi-View Network for Audiovisual Dysarthria Severity Assessment***

本项目旨在提供一个基于深度学习的端到端 **构音障碍（Dysarthria）严重程度自动评估系统**。系统充分利用音频与视频的多模态信息，并通过知识引导机制提升特征表示与判别能力，为临床诊断和康复治疗提供客观、可扩展的辅助工具。

> **摘要**: KGMV-Net 引入了知识引导的多视角学习框架，有效融合声学与视觉特征，显著提升了对构音障碍严重程度的建模能力和预测性能。

---

## 📖 论文信息

如果您使用本项目，请引用以下论文：

```bibtex
@article{liu2024kgmv,
  title={KGMV-Net: Knowledge-Guided Multi-View Network for Audiovisual Dysarthria Severity Assessment},
  author={Liu, Xiaokang and Yang, Yudong and Xu, Guorong and Du, Xiaoxia and Su, Rongfeng and Wang, Lan and Yan, Nan},
  journal={SSRN},
  year={2024},
  note={Available at SSRN: https://ssrn.com/abstract=5332691}
}
```

**论文链接**: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5332691)

---

## 🚀 主要特性

* **多模态融合**：联合建模语音与视觉信息，提升评估精度与鲁棒性。
* **知识引导机制**：引入先验知识，增强特征表示与跨模态对齐能力。
* **端到端实现**：涵盖数据准备、模型训练、验证与推理的完整流程。
* **模块化设计**：代码结构清晰，便于复现、扩展与二次开发。

---

## ⚙️ 环境依赖

* **操作系统**：Linux
* **编程语言**：Python ≥ 3.8
* **深度学习框架**：PyTorch ≥ 1.12
* **硬件需求**：GPU 显存 ≥ 24 GB (推荐 NVIDIA V100 / A100)
* **依赖库**：详见 `requirements.txt`

---

## 🛠️ 安装步骤

1. 克隆仓库：

   ```bash
   git clone https://github.com/your_username/av-dysarthria-diagnosis.git
   cd av-dysarthria-diagnosis
   ```

2. 创建并激活环境：

   ```bash
   conda create -n dysarthria python=3.9 -y
   conda activate dysarthria
   pip install -r requirements.txt
   ```

---

## 📁 项目结构

```bash
av-dysarthria-diagnosis/
├── egs/                     # 实验示例
│   ├── msdm/                 # MSDM 数据集配置与脚本
│   ├── conf/                 # 模型训练配置文件
│   ├── local/                # 数据准备与处理脚本
│   ├── networks -> ../../networks/  # 网络定义（软链接）
│   ├── path.sh               # 环境路径配置
│   ├── run.sh                # 主入口脚本（训练/评估）
│   └── tools -> ../../tools/        # 工具目录（软链接）
├── networks/                 # 模型与网络结构
│   ├── dataset/              # 数据加载与处理模块
│   ├── model/                # KGMV-Net 模型定义
│   └── utils/                # 辅助工具函数
├── tools/                    # 通用工具脚本
└── README.md                 # 项目说明文档
```

---

## 🗃️ 数据与模型

* **训练数据**：本项目使用 **MSDM** 数据集。请根据其许可协议获取原始数据。
* **数据准备**：参考 `egs/msdm/local/` 目录下的脚本完成数据预处理。
* **预训练模型**：如需使用预训练权重，请在此添加下载链接及使用说明。

---

## 🚀 使用方法

本项目的主要入口为 `egs/msdm/run.sh` 脚本。

### 训练与评估流程

1. 确保已准备好 MSDM 数据集。
2. 修改 `egs/msdm/conf/` 中的配置文件以适应实验需求。
3. 执行脚本：

   ```bash
   cd egs/msdm/
   bash run.sh
   ```

   该脚本将依次完成数据准备、模型训练与评估。

---

## 🤝 贡献指南

我们欢迎来自学术界与工业界的贡献：

* **报告问题**：在 [Issues](../../issues) 中提交 Bug 报告。
* **功能建议**：通过 Issue 讨论新功能与改进点。
* **代码贡献**：提交 Pull Request。

---

## 📄 许可证

本项目基于 **GNU General Public License v3.0 (GPL-3.0)** 开源协议，详见 [LICENSE](LICENSE)。

---

## 🙏 致谢

* 感谢所有合作者对本研究的贡献。
* 感谢 MSDM 数据集提供方的支持。

---

## ❓ 常见问题

1. **Q: 出现 `CUDA out of memory` 错误？**
   **A**: 请尝试减小 `batch_size` 参数，或使用更高显存的 GPU。

