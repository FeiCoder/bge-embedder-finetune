# BGE Embedder Fine-tuning Project

本项目用于基于 `mldr` 数据集对 `BAAI/bge-base-zh-v1.5` 向量模型进行微调（Fine-tuning）和评估。

## 目录结构

```
bge-embedder-finetune/
├── data/                       # 存放训练和测试数据
├── output/                     # 存放微调后的模型权重
├── scripts/
│   ├── prepare_data.py         # 数据预处理脚本：转换 mldr 格式为 FlagEmbedding 训练格式
│   ├── run_finetune.sh         # 模型微调启动脚本 (支持多卡)
│   ├── evaluate.py             # 模型评估脚本 (Recall/NDCG 指标计算)
│   └── run_eval_comparison.sh  # 评估流程控制脚本：对比基座与微调模型
└── README.md
```

## 环境准备

本项目依赖 `FlagEmbedding` 及其相关库。确保已安装 `uv` 包管理器。

```bash
# 安装依赖
uv pip install FlagEmbedding torch transformers datasets accelerate
```

## 使用指南

### 1. 数据准备

将原始数据集转换为微调所需的 JSONL 格式 (`query`, `pos`, `neg`).

```bash
python scripts/prepare_data.py
```
*   输入: `/data/zf/Datasets/mldr-v1.0-zh/`
*   输出: `data/finetune_data.jsonl`

### 2. 模型微调 (Training)

使用 `scripts/run_finetune.sh` 启动训练。
默认脚本已配置为单机多卡 (4x GPU) 模式以加速训练。

```bash
./scripts/run_finetune.sh
```

**主要参数说明:**
*   `--nproc_per_node 4`: 使用的 GPU 数量。
*   `--per_device_train_batch_size`: 单卡 Batch Size。建议根据显存大小调整 (RTX 4090 可设为 32-64)。
*   `--train_group_size`: 每个 Query 对应的正负例总数。
*   `--learning_rate`: 学习率 (默认 1e-5)。

### 3. 模型评估 (Evaluation)

运行评估脚本，该脚本会自动对比 **基座模型** 和 **微调后模型** 的检索性能 (Recall@10, NDCG@10)。

```bash
./scripts/run_eval_comparison.sh
```

*   **加速评估**: 评估脚本会自动检测可用 GPU 并开启 `DataParallel` 进行并行推理。
*   **Batch Size**: 评估脚本默认 Batch Size 较大 (2048) 以利用 RTX 4090 性能。

## 性能记录

| 模型 | Recall@10 | NDCG@10 |
| :--- | :--- | :--- |
| **BGE Base (v1.5)** | 0.1750 | 0.1354 |
| **Finetuned** | *待评估* | *待评估* |

## 常见问题

1.  **CUDA OOM**: 如果在训练时遇到显存不足，请降低 `per_device_train_batch_size` (如 4) 或 `train_group_size`。
2.  **多卡运行错误**: 确保 `CUDA_VISIBLE_DEVICES` 设置包含足够的 GPU ID。
