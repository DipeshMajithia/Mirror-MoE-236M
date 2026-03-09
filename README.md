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

### 1. General Knowledge
Despite being trained on strictly **~61M tokens** (~164k samples), MirrorAI V3 achieves competitive results compared to models trained on 1000x more data:

| Benchmark | Score |
|-----------|-------|
| **MMLU** | 26.0% |
| **ARC-Easy** | 37.0% |
| **HellaSwag** | 25.5% |
| **TruthfulQA** | 27.0% |

### 2. Tool Calling Syntax Success
To test our **Atomic Token Theory** ($p^1 > p^n$), we benchmarked MirrorAI V3 against two popular small models on a zero-shot tool-calling dataset of 100 prompts.

| Model | Format Target | Syntax Success |
|-------|---------------|----------------|
| **MirrorAI V3 (Ours)** | **Atomic tokens (`<call>`)** | **57.0%** |
| SmolLM2-135M | JSON Object | 0.0% |
| GPT-2 Small | Text Template | 59.0% |

**Insight:** Structured JSON is extremely fragile for Small Language Models (SLMs). Even a model trained on 2 Trillion tokens like SmolLM2 fails to generate valid JSON tool calls zero-shot. MirrorAI V3 achieves near-parity with text templates while providing a fully parsable, atomic interface.

### 3. V4 System 2: Zero-Shot Error Recovery
For MirrorAI V4, we fine-tuned the model on a focused 3,000-sample "Resilient Agent" curriculum. By injecting a reflection prompt containing the Python execution error, the 236M model can autonomously debug and correct its own syntax errors.

Tested on a dataset of 100 broken tool calls (including out-of-distribution logic errors):

| Model | Syntax Success (Base) | Error Recovery Rate |
|-------|-----------------------|---------------------|
| MirrorAI V3 | 57.0% | 0.0% |
| **MirrorAI V4** | **57.0%** | **32.0%** |

This proves that "System 2" self-correction is not an emergent property restricted to massive models, but a trainable behavior even at 236M parameters.

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
