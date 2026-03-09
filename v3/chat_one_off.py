#!/usr/bin/env python3
"""
MirrorAI V3 — One-Off Test with Research-Grade Sampling
Usage: python v3/chat_one_off.py "Your question here"
"""
import os, sys, sqlite3
import mlx.core as mx
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'v2'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model.transformer import MirrorTransformer, ModelArgs
from tokenizer_wrapper import MirrorTokenizer

DB_PATH = os.path.join(ROOT_DIR, "data/v3/mirror_knowledge.db")

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

def sample_top_p(logits, temperature=0.7, top_p=0.9, repetition_penalty=1.2, 
                 generated_ids=None):
    """Nucleus (top-p) sampling with temperature and repetition penalty."""
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
    probs = probs / (probs.sum() + 1e-10)
    chosen_idx = np.random.choice(cutoff_idx, p=probs[:cutoff_idx] / probs[:cutoff_idx].sum())
    return int(sorted_indices[chosen_idx])

def query_sqlite(query_text):
    if not os.path.exists(DB_PATH): return "No info found"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM knowledge WHERE query LIKE ?", (f"%{query_text}%",))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "No info found"

def eval_calculator(expr):
    """Safely evaluate a mathematical expression."""
    try:
        import math
        safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
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

def main():
    if len(sys.argv) < 2:
        print("Usage: chat_one_off.py 'Your question'")
        return

    user_input = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else find_best_model()
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    print(f"🚀 Loading: {os.path.basename(model_path)}...", flush=True)
    model.load_weights(model_path, strict=False)
    model.eval()
    mx.eval(model.parameters())

    prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nAssistant: "
    print("🤖 Thinking...", flush=True)
    res = generate(model, tokenizer, prompt)

    if "<call>search_knowledge" in res:
        try:
            query = res.split('("')[1].split('")')[0]
            ctx = query_sqlite(query)
            ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nContext: {ctx}\nAssistant: "
            final = generate(model, tokenizer, ctx_prompt)
            print(final)
        except:
            print(res)
    elif "<call>calculator" in res:
        try:
            expr = res.split('("')[1].split('")')[0]
            print(f"🧮 Calculating: {expr}")
            result = eval_calculator(expr)
            ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nCalculation Result: {expr} = {result}\nAssistant: "
            final = generate(model, tokenizer, ctx_prompt)
            print(final)
        except:
            print(res)
    else:
        print(res)

if __name__ == "__main__":
    main()
