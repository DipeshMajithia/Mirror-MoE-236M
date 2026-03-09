#!/usr/bin/env python3
"""
MirrorAI V3 — Comprehensive Benchmark Suite
Evaluates on standard NLP benchmarks + custom MirrorAI capabilities.
Results formatted for HuggingFace model card.

Benchmarks:
  1. HellaSwag (common-sense sentence completion) — 200 samples
  2. ARC-Easy (science reasoning, multiple choice) — 200 samples
  3. MMLU (knowledge, multiple choice) — 200 samples
  4. TruthfulQA (factual accuracy) — 100 samples
  5. MirrorAI Custom (identity, tool-calling, math, coding, conversation)
"""
import os, sys, json, time, random
import mlx.core as mx
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'v2'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model.transformer import MirrorTransformer, ModelArgs
from tokenizer_wrapper import MirrorTokenizer

V3_ARGS = ModelArgs(
    dim=512, hidden_dim=1365, n_layers=8,
    vocab_size=32002, use_moe=True,
    num_experts=16, num_experts_per_tok=2,
    shared_expert_dim=1365
)

SYSTEM_PROMPT = """You are MirrorAI, a personal AI assistant created by Dipesh Majithia.
You are helpful, friendly, and knowledgeable.
When you need factual information, use: <call>search_knowledge("query")</call>
When you need to perform math calculations, use: <call>calculator("expression")</call>
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly.
"""

# ── Utility Functions ──────────────────────────────────────

def sample_top_p(logits, temperature=0.7, top_p=0.9, repetition_penalty=1.5,
                 generated_ids=None):
    if generated_ids and len(generated_ids) > 0:
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits = logits.at[token_id].multiply(1.0 / repetition_penalty)
            else:
                logits = logits.at[token_id].multiply(repetition_penalty)
    logits = logits / temperature
    logits_np = np.array(logits)
    sorted_indices = np.argsort(-logits_np)
    sorted_logits = logits_np[sorted_indices]
    sorted_logits = sorted_logits - sorted_logits.max()
    probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
    cumsum = np.cumsum(probs)
    cutoff_idx = min(np.searchsorted(cumsum, top_p) + 1, len(probs))
    probs[cutoff_idx:] = 0.0
    chosen_idx = np.random.choice(cutoff_idx, p=probs[:cutoff_idx] / probs[:cutoff_idx].sum())
    return int(sorted_indices[chosen_idx])

def generate(model, tokenizer, prompt, max_tokens=200):
    tokens = tokenizer.encode(prompt)
    generated_ids = []
    for _ in range(max_tokens):
        x = mx.array(tokens)[None, :]
        logits, _, _ = model(x)
        next_logits = logits[0, -1, :]
        next_token = sample_top_p(next_logits, generated_ids=generated_ids)
        if next_token == 3: break
        generated_ids.append(next_token)
        tokens.append(next_token)
        if '</call>' in tokenizer.decode(generated_ids): break
    return tokenizer.decode(generated_ids).strip()

def compute_log_likelihood(model, tokenizer, context, continuation):
    """Compute log-likelihood of continuation given context."""
    ctx_tokens = tokenizer.encode(context)
    cont_tokens = tokenizer.encode(continuation)
    all_tokens = ctx_tokens + cont_tokens
    
    if len(all_tokens) > 512:
        all_tokens = all_tokens[:512]
        cont_len = max(1, len(all_tokens) - len(ctx_tokens))
    else:
        cont_len = len(cont_tokens)
    
    x = mx.array(all_tokens)[None, :]
    logits, _, _ = model(x)
    
    # Compute log-likelihood of the continuation tokens
    log_likelihood = 0.0
    start_pos = len(all_tokens) - cont_len
    for i in range(start_pos, len(all_tokens)):
        token_logits = logits[0, i - 1, :]
        log_probs = mx.log(mx.softmax(token_logits))
        log_likelihood += float(log_probs[all_tokens[i]])
    
    return log_likelihood / cont_len  # Normalize by length

# ── Benchmark 1: HellaSwag ─────────────────────────────────

def download_hellaswag(n=200):
    """Download HellaSwag validation split."""
    print("  📥 Downloading HellaSwag...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
        samples = []
        for i, row in enumerate(ds):
            if len(samples) >= n: break
            samples.append({
                "ctx": row["ctx"],
                "endings": row["endings"],
                "label": int(row["label"]),
            })
        return samples
    except Exception as e:
        print(f"  ⚠️ HellaSwag download failed: {e}")
        return []

def eval_hellaswag(model, tokenizer, samples):
    """Evaluate on HellaSwag — pick most likely continuation."""
    correct = 0
    total = len(samples)
    
    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{total}")
        
        ctx = s["ctx"]
        best_score = float('-inf')
        best_idx = 0
        
        for j, ending in enumerate(s["endings"]):
            score = compute_log_likelihood(model, tokenizer, ctx, " " + ending)
            if score > best_score:
                best_score = score
                best_idx = j
        
        if best_idx == s["label"]:
            correct += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, correct, total

# ── Benchmark 2: ARC-Easy ──────────────────────────────────

def download_arc_easy(n=200):
    """Download ARC-Easy test split."""
    print("  📥 Downloading ARC-Easy...")
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", streaming=True)
        samples = []
        for i, row in enumerate(ds):
            if len(samples) >= n: break
            choices = row["choices"]
            labels = choices["label"]
            texts = choices["text"]
            answer_key = row["answerKey"]
            
            if answer_key not in labels:
                continue
            
            samples.append({
                "question": row["question"],
                "choices": {l: t for l, t in zip(labels, texts)},
                "answer": answer_key,
            })
        return samples
    except Exception as e:
        print(f"  ⚠️ ARC-Easy download failed: {e}")
        return []

def eval_arc_easy(model, tokenizer, samples):
    """Evaluate on ARC-Easy — pick most likely answer."""
    correct = 0
    total = len(samples)
    
    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{total}")
        
        question = s["question"]
        best_score = float('-inf')
        best_label = None
        
        for label, text in s["choices"].items():
            prompt = f"Question: {question}\nAnswer:"
            score = compute_log_likelihood(model, tokenizer, prompt, " " + text)
            if score > best_score:
                best_score = score
                best_label = label
        
        if best_label == s["answer"]:
            correct += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, correct, total

# ── Benchmark 3: MMLU ──────────────────────────────────────

def download_mmlu(n=200):
    """Download MMLU validation samples."""
    print("  📥 Downloading MMLU...")
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="validation", streaming=True)
        samples = []
        for i, row in enumerate(ds):
            if len(samples) >= n: break
            samples.append({
                "question": row["question"],
                "choices": row["choices"],
                "answer": int(row["answer"]),
                "subject": row.get("subject", "unknown"),
            })
        return samples
    except Exception as e:
        print(f"  ⚠️ MMLU download failed: {e}")
        return []

def eval_mmlu(model, tokenizer, samples):
    """Evaluate on MMLU — pick most likely choice."""
    correct = 0
    total = len(samples)
    choice_labels = ["A", "B", "C", "D"]
    
    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{total}")
        
        question = s["question"]
        choices = s["choices"]
        best_score = float('-inf')
        best_idx = 0
        
        for j, choice in enumerate(choices):
            prompt = f"Question: {question}\nAnswer:"
            score = compute_log_likelihood(model, tokenizer, prompt, f" {choice_labels[j]}. {choice}")
            if score > best_score:
                best_score = score
                best_idx = j
        
        if best_idx == s["answer"]:
            correct += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, correct, total

# ── Benchmark 4: TruthfulQA ───────────────────────────────

def download_truthfulqa(n=100):
    """Download TruthfulQA MC1 split."""
    print("  📥 Downloading TruthfulQA...")
    try:
        from datasets import load_dataset
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", streaming=True)
        samples = []
        for i, row in enumerate(ds):
            if len(samples) >= n: break
            mc1_targets = row.get("mc1_targets", {})
            choices = mc1_targets.get("choices", [])
            labels = mc1_targets.get("labels", [])
            if not choices or not labels:
                continue
            answer_idx = labels.index(1) if 1 in labels else 0
            samples.append({
                "question": row["question"],
                "choices": choices,
                "answer": answer_idx,
            })
        return samples
    except Exception as e:
        print(f"  ⚠️ TruthfulQA download failed: {e}")
        return []

def eval_truthfulqa(model, tokenizer, samples):
    """Evaluate on TruthfulQA MC1."""
    correct = 0
    total = len(samples)
    
    for i, s in enumerate(samples):
        if (i + 1) % 25 == 0:
            print(f"    ... {i+1}/{total}")
        
        question = s["question"]
        best_score = float('-inf')
        best_idx = 0
        
        for j, choice in enumerate(s["choices"]):
            prompt = f"Q: {question}\nA:"
            score = compute_log_likelihood(model, tokenizer, prompt, " " + choice)
            if score > best_score:
                best_score = score
                best_idx = j
        
        if best_idx == s["answer"]:
            correct += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, correct, total

# ── Benchmark 5: MirrorAI Custom ──────────────────────────

CUSTOM_TESTS = {
    "identity": [
        ("Who created you?", lambda r: any(w in r.lower() for w in ["dipesh", "majithia"])),
        ("What is your name?", lambda r: any(w in r.lower() for w in ["mirror", "mirrorai"])),
        ("Who are you?", lambda r: any(w in r.lower() for w in ["mirror", "ai", "assistant"])),
        ("Who built you?", lambda r: any(w in r.lower() for w in ["dipesh", "majithia"])),
        ("What are you?", lambda r: any(w in r.lower() for w in ["mirror", "ai", "assistant"])),
    ],
    "tool_calling_rag": [
        ("What is the capital of France?", lambda r: "<call>search_knowledge" in r),
        ("Tell me about DNA.", lambda r: "<call>search_knowledge" in r),
        ("Who was the first president of the USA?", lambda r: "<call>search_knowledge" in r or "washington" in r.lower()),
        ("What is photosynthesis?", lambda r: "<call>search_knowledge" in r),
        ("Tell me about the solar system.", lambda r: "<call>search_knowledge" in r),
    ],
    "tool_calling_math": [
        ("What is 125 + 372?", lambda r: "<call>calculator" in r or "497" in r),
        ("Calculate 15 * 12.", lambda r: "<call>calculator" in r or "180" in r),
        ("What is 99 + 1?", lambda r: "<call>calculator" in r or "100" in r),
        ("Calculate 50 / 2.", lambda r: "<call>calculator" in r or "25" in r),
        ("What is 7 * 8?", lambda r: "<call>calculator" in r or "56" in r),
    ],
    "conversation": [
        ("Hi, how are you?", lambda r: len(r) > 5 and not r.startswith("<call>")),
        ("Thank you for helping me!", lambda r: len(r) > 5),
        ("Good morning!", lambda r: len(r) > 5 and not r.startswith("<call>")),
        ("What can you do?", lambda r: len(r) > 10),
        ("Tell me a joke.", lambda r: len(r) > 10),
    ],
    "coherence": [
        ("Write a 3-sentence story about a cat.", lambda r: len(r.split('.')) >= 2 and len(set(r.split())) > 10),
        ("Explain gravity in simple terms.", lambda r: len(r) > 20 and len(set(r.split())) > 8),
        ("What are the benefits of exercise?", lambda r: len(r) > 20),
        ("Describe the color blue.", lambda r: len(r) > 10),
    ],
}

def eval_custom(model, tokenizer):
    """Evaluate on MirrorAI custom tests."""
    results = {}
    total_pass = 0
    total_tests = 0
    
    for category, tests in CUSTOM_TESTS.items():
        cat_pass = 0
        cat_details = []
        for query, check_fn in tests:
            prompt = f"System: {SYSTEM_PROMPT}\nUser: {query}\nAssistant: "
            response = generate(model, tokenizer, prompt)
            passed = check_fn(response)
            cat_pass += int(passed)
            total_tests += 1
            total_pass += int(passed)
            cat_details.append({
                "query": query,
                "response": response[:150],
                "passed": passed,
            })
        
        results[category] = {
            "score": cat_pass,
            "total": len(tests),
            "accuracy": cat_pass / len(tests) * 100,
            "details": cat_details,
        }
    
    overall = total_pass / total_tests * 100 if total_tests > 0 else 0
    return results, overall

# ── Comparison Table ───────────────────────────────────────

COMPARABLE_MODELS = {
    "GPT-2 (124M)": {"params": "124M", "hellaswag": 30.0, "arc_easy": 38.7, "mmlu": 25.0, "training_tokens": "~10B"},
    "OPT-125M": {"params": "125M", "hellaswag": 31.5, "arc_easy": 22.9, "mmlu": 24.0, "training_tokens": "300B"},
    "SmolLM2-135M": {"params": "135M", "hellaswag": 67.5, "arc_easy": 54.3, "mmlu": 23.1, "training_tokens": "2T"},
    "Pythia-160M": {"params": "160M", "hellaswag": 29.4, "arc_easy": 43.5, "mmlu": 24.0, "training_tokens": "300B"},
}

# ── Main ───────────────────────────────────────────────────

def run_benchmark(model_path=None):
    if model_path is None:
        # Auto-find latest checkpoint or final model
        final = os.path.join(ROOT_DIR, "model.safetensors")
        if os.path.exists(final):
            model_path = final
        else:
            ckpt_dir = os.path.join(ROOT_DIR, "out/v3_sota")
            if os.path.exists(ckpt_dir):
                ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".safetensors")]
                if ckpts:
                    ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    model_path = os.path.join(ckpt_dir, ckpts[-1])
    
    if not model_path or not os.path.exists(model_path):
        print("❌ No model found!")
        return
    
    print("=" * 70)
    print("  🏆 MirrorAI V3 — Comprehensive Benchmark Suite")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Architecture: 236M MoE (16 experts, top-2, 8 layers, dim=512)")
    print("=" * 70)
    
    # Load model
    print("\n🔄 Loading model...")
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()
    mx.eval(model.parameters())
    
    results = {
        "model": "MirrorAI-V3-236M-MoE",
        "model_path": os.path.basename(model_path),
        "architecture": "MoE Transformer (16 experts, top-2, 8 layers)",
        "parameters": "236M total, ~62M active per token",
        "training_data": "164k samples (~61M tokens), 3 epochs",
        "framework": "MLX (Apple Silicon native)",
    }
    
    start_time = time.time()
    
    # 1. HellaSwag
    print("\n📋 BENCHMARK 1: HellaSwag (Common-Sense Reasoning)")
    print("-" * 50)
    hs_data = download_hellaswag(200)
    if hs_data:
        hs_acc, hs_correct, hs_total = eval_hellaswag(model, tokenizer, hs_data)
        results["hellaswag"] = {"accuracy": round(hs_acc, 1), "correct": hs_correct, "total": hs_total}
        print(f"  ✅ HellaSwag: {hs_acc:.1f}% ({hs_correct}/{hs_total})")
    else:
        print("  ⚠️ Skipped (download failed)")
    
    # 2. ARC-Easy
    print("\n📋 BENCHMARK 2: ARC-Easy (Science Reasoning)")
    print("-" * 50)
    arc_data = download_arc_easy(200)
    if arc_data:
        arc_acc, arc_correct, arc_total = eval_arc_easy(model, tokenizer, arc_data)
        results["arc_easy"] = {"accuracy": round(arc_acc, 1), "correct": arc_correct, "total": arc_total}
        print(f"  ✅ ARC-Easy: {arc_acc:.1f}% ({arc_correct}/{arc_total})")
    else:
        print("  ⚠️ Skipped (download failed)")
    
    # 3. MMLU
    print("\n📋 BENCHMARK 3: MMLU (Knowledge)")
    print("-" * 50)
    mmlu_data = download_mmlu(200)
    if mmlu_data:
        mmlu_acc, mmlu_correct, mmlu_total = eval_mmlu(model, tokenizer, mmlu_data)
        results["mmlu"] = {"accuracy": round(mmlu_acc, 1), "correct": mmlu_correct, "total": mmlu_total}
        print(f"  ✅ MMLU: {mmlu_acc:.1f}% ({mmlu_correct}/{mmlu_total})")
    else:
        print("  ⚠️ Skipped (download failed)")
    
    # 4. TruthfulQA
    print("\n📋 BENCHMARK 4: TruthfulQA (Factual Accuracy)")
    print("-" * 50)
    tqa_data = download_truthfulqa(100)
    if tqa_data:
        tqa_acc, tqa_correct, tqa_total = eval_truthfulqa(model, tokenizer, tqa_data)
        results["truthfulqa"] = {"accuracy": round(tqa_acc, 1), "correct": tqa_correct, "total": tqa_total}
        print(f"  ✅ TruthfulQA: {tqa_acc:.1f}% ({tqa_correct}/{tqa_total})")
    else:
        print("  ⚠️ Skipped (download failed)")
    
    # 5. Custom MirrorAI
    print("\n📋 BENCHMARK 5: MirrorAI Custom Capabilities")
    print("-" * 50)
    custom_results, custom_overall = eval_custom(model, tokenizer)
    results["mirrorai_custom"] = custom_results
    results["mirrorai_custom_overall"] = round(custom_overall, 1)
    for cat, data in custom_results.items():
        status = "✅" if data["accuracy"] >= 80 else "🟡" if data["accuracy"] >= 50 else "❌"
        print(f"  {status} {cat}: {data['score']}/{data['total']} ({data['accuracy']:.0f}%)")
    print(f"  {'─'*30}")
    print(f"  🏆 Custom Overall: {custom_overall:.1f}%")
    
    elapsed = time.time() - start_time
    results["eval_time_seconds"] = round(elapsed)
    
    # ── Print Comparison Table ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("  📊 COMPARISON WITH COMPARABLE MODELS")
    print(f"{'=' * 70}")
    
    header = f"{'Model':<22} {'Params':>8} {'HellaSwag':>10} {'ARC-Easy':>10} {'MMLU':>8} {'Training Data':>15}"
    print(header)
    print("-" * 75)
    
    for name, data in COMPARABLE_MODELS.items():
        row = f"{name:<22} {data['params']:>8} {data['hellaswag']:>9.1f}% {data['arc_easy']:>9.1f}% {data['mmlu']:>7.1f}% {data['training_tokens']:>15}"
        print(row)
    
    # Our model
    hs_score = results.get("hellaswag", {}).get("accuracy", "N/A")
    arc_score = results.get("arc_easy", {}).get("accuracy", "N/A")
    mmlu_score = results.get("mmlu", {}).get("accuracy", "N/A")
    
    hs_str = f"{hs_score:.1f}%" if isinstance(hs_score, (int, float)) else hs_score
    arc_str = f"{arc_score:.1f}%" if isinstance(arc_score, (int, float)) else arc_score
    mmlu_str = f"{mmlu_score:.1f}%" if isinstance(mmlu_score, (int, float)) else mmlu_score
    
    print("-" * 75)
    ours = f"{'MirrorAI V3 (ours)':<22} {'236M':>8} {hs_str:>10} {arc_str:>10} {mmlu_str:>8} {'~61M':>15}"
    print(ours)
    
    print(f"\n⏱️ Evaluation time: {elapsed:.0f}s")
    
    # ── Save results ───────────────────────────────────────
    results_path = os.path.join(ROOT_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved: {results_path}")
    
    # ── Print Summary ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  🏆 FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Model:          MirrorAI V3 (236M MoE)")
    print(f"  Architecture:   16 experts, top-2 routing, 8 layers, dim=512")
    print(f"  Active params:  ~62M per token")
    print(f"  Training:       164k samples, 3 epochs (~61M tokens)")
    print(f"  Framework:      MLX (Apple Silicon native)")
    print(f"  HellaSwag:      {hs_str}")
    print(f"  ARC-Easy:       {arc_str}")
    print(f"  MMLU:           {mmlu_str}")
    tqa_str = f"{results.get('truthfulqa', {}).get('accuracy', 'N/A')}"
    if isinstance(results.get('truthfulqa', {}).get('accuracy'), (int, float)):
        tqa_str += "%"
    print(f"  TruthfulQA:     {tqa_str}")
    print(f"  Custom Suite:   {custom_overall:.1f}%")
    print(f"\n  🔑 Unique Features:")
    print(f"    ✦ Tool-calling: <call>search_knowledge/calculator</call>")
    print(f"    ✦ Custom identity (MirrorAI by Dipesh Majithia)")
    print(f"    ✦ Mixture-of-Experts architecture")
    print(f"    ✦ Runs natively on Apple Silicon via MLX")
    print(f"{'=' * 70}")
    
    return results

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(model_path)
