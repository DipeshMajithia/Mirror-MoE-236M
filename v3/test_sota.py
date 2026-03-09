#!/usr/bin/env python3
"""
MirrorAI V3 — Automated SOTA Verification Suite
Tests: Identity, RAG Protocol, Conversation, Creative Writing
"""
import os, sys, time
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
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly.
"""

def sample_top_p(logits, temperature=0.7, top_p=0.9, repetition_penalty=1.2, 
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

def eval_calculator(expr):
    """Safely evaluate a mathematical expression."""
    try:
        import math as m
        safe_dict = {k: v for k, v in m.__dict__.items() if not k.startswith("__")}
        safe_dict['abs'] = abs
        safe_dict['round'] = round
        return str(eval(expr, {"__builtins__": None}, safe_dict))
    except Exception as e:
        return f"Error: {e}"

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
        decoded = tokenizer.decode(generated_ids)
        if "</call>" in decoded: break
    return tokenizer.decode(generated_ids).strip()

def find_best_model():
    final = os.path.join(ROOT_DIR, "mirror_ai_v3_final.safetensors")
    if os.path.exists(final): return final
    ckpt_dir = os.path.join(ROOT_DIR, "out/v3")
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
            return os.path.join(ckpt_dir, ckpts[-1])
    v2 = os.path.join(ROOT_DIR, "out/v2/phase2_final.safetensors")
    if os.path.exists(v2): return v2
    return None

# ── Test Cases ──────────────────────────────────────────────
TESTS = {
    "identity": [
        ("Who created you?", lambda r: any(w in r.lower() for w in ["dipesh", "majithia", "mirrorai"])),
        ("What is your name?", lambda r: any(w in r.lower() for w in ["mirror", "mirrorai"])),
        ("Who are you?", lambda r: any(w in r.lower() for w in ["mirror", "ai", "assistant"])),
    ],
    "rag_protocol": [
        ("What is the capital of France?", lambda r: "<call>search_knowledge" in r or "paris" in r.lower()),
        ("Tell me about DNA.", lambda r: "<call>search_knowledge" in r or "molecule" in r.lower() or "dna" in r.lower()),
        ("Who was the first president of the USA?", lambda r: "<call>search_knowledge" in r or "washington" in r.lower()),
    ],
    "math_calculator": [
        ("What is 125 + 372?", lambda r: "497" in r),
        ("Calculate 15 * 12.", lambda r: "180" in r),
        ("What is the square root of 144?", lambda r: "12" in r or "12.0" in r),
        ("Complex math: (25 * 4) + (50 / 2)", lambda r: "125" in r or "125.0" in r),
    ],
    "coding": [
        ("Write a Python function to check if a number is even.", lambda r: "def" in r and "return" in r and "% 2" in r),
        ("How to reverse a list in Python?", lambda r: "reverse" in r or "[::-1]" in r),
    ],
    "conversation": [
        ("Hi, how are you?", lambda r: len(r) > 5 and not r.startswith("<call>")),
        ("Thank you for helping me!", lambda r: len(r) > 5),
    ],
    "coherence": [
        ("Write a 3-sentence story about a cat.", lambda r: len(r.split('.')) >= 2 and len(set(r.split())) > 10),
        ("Explain gravity in simple terms.", lambda r: len(r) > 20 and len(set(r.split())) > 8),
    ],
}

def run_tests(model_path=None):
    if model_path is None:
        model_path = find_best_model()
    if not model_path:
        print("❌ No model found!")
        return
    
    print("=" * 70)
    print(f"  🧪 MirrorAI V3 SOTA Verification Suite")
    print(f"  Model: {os.path.basename(model_path)}")
    print("=" * 70)
    
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()
    mx.eval(model.parameters())
    
    total_pass = 0
    total_tests = 0
    
    for category, tests in TESTS.items():
        print(f"\n📋 {category.upper()}")
        print("-" * 50)
        cat_pass = 0
        
        for query, check_fn in tests:
            prompt = f"System: {SYSTEM_PROMPT}\nUser: {query}\nAssistant: "
            response = generate(model, tokenizer, prompt)
            
            # Handle calculator tool if generated
            if "<call>calculator" in response:
                try:
                    expr = response.split('("')[1].split('")')[0]
                    res = eval_calculator(expr)
                    ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {query}\nCalculation Result: {expr} = {res}\nAssistant: "
                    response = generate(model, tokenizer, ctx_prompt)
                except: pass

            passed = check_fn(response)
            cat_pass += int(passed)
            total_tests += 1
            total_pass += int(passed)
            
            status = "✅" if passed else "❌"
            # Truncate long responses for display
            display = response[:120].replace('\n', ' ') + "..." if len(response) > 120 else response.replace('\n', ' ')
            print(f"  {status} Q: {query}")
            print(f"     A: {display}")
        
        print(f"  Score: {cat_pass}/{len(tests)}")
    
    print(f"\n{'=' * 70}")
    pct = (total_pass / total_tests * 100) if total_tests > 0 else 0
    print(f"  🏆 TOTAL: {total_pass}/{total_tests} ({pct:.0f}%)")
    print(f"{'=' * 70}")
    return total_pass, total_tests

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_tests(model_path)
