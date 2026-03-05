# Mirror-MoE-236M

MirrorAI V3 is a 236 million parameter Mixture-of-Experts (MoE) language model trained natively on Apple Silicon using the [MLX framework](https://github.com/ml-explore/mlx). This project demonstrates extreme data efficiency and the power of Sparse MoE for small language models (SLMs).

## 🚀 Live Demo
Experience the model live on HuggingFace: [**Mirror-MoE-236M Space**](https://huggingface.co/spaces/dipeshmajithia/Mirror-MoE-236M)

## 🏗️ Architecture
- **Parameters:** 236M total, ~62M active per token.
- **MoE:** 1 layer of 16 experts with top-2 routing + 1 shared expert.
- **Layers:** 8 transformer blocks.
- **Dim:** 512 (hidden_dim: 1365).
- **Atomic Tool Calling:** Integrated `<call>` and `</call>` tokens for RAG and math tools, bypass BPE fragmenting for higher reliability.

## 📊 Benchmark Results (Real World)
Despite being trained on strictly **~61M tokens** (~164k samples), MirrorAI V3 achieves competitive results compared to models trained on 1000x more data:

| Benchmark | Score |
|-----------|-------|
| **MMLU** | 26.0% |
| **ARC-Easy** | 37.0% |
| **HellaSwag** | 25.5% |
| **TruthfulQA** | 27.0% |

## 🛠️ Installation & Usage

### 1. Requirements
- macOS with Apple Silicon (M1, M2, M3, M4)
- Python 3.11+
- MLX, numpy, tokenizers

```bash
git clone https://github.com/DipeshMajithia/Mirror-MoE-236M
cd Mirror-MoE-236M
pip install -r requirements.txt
```

### 2. Run Chat Interface
Download the model weights from [HuggingFace](https://huggingface.co/dipeshmajithia/Mirror-MoE-236M/blob/main/model.safetensors) and place them in the root directory.

```bash
python3 v3/chat_sota.py model.safetensors
```

## 📚 Training Your Own
The `v3/` directory contains the full pipeline used to recreate this run:
- **`v3/run_v3_train.py`**: The core multi-epoch training script with curriculum loading.
- **`v3/benchmark.py`**: Standardized benchmarking suite for MLX models.
- **`v3/build_sota_dataset.py`**: Pipeline for converting OpenHermes/SlimOrca into MoE-ready formats.

## 👤 Author
**Dipesh Majithia**

Built using the Apple MLX framework.
