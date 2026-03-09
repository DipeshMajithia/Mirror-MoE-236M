#!/usr/bin/env python3
import os, sys, sqlite3
import mlx.core as mx

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'v2'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model.transformer import MirrorTransformer, ModelArgs
from tokenizer_wrapper import MirrorTokenizer

DB_PATH = os.path.join(ROOT_DIR, "data/v3/mirror_knowledge.db")
CKPT_PATH = os.path.join(ROOT_DIR, "out/v3/ckpt_2000.safetensors")

V3_ARGS = ModelArgs(
    dim=512, hidden_dim=1365, n_layers=8,
    vocab_size=32002, use_moe=True,
    num_experts=16, num_experts_per_tok=2,
    shared_expert_dim=1365
)

def query_sqlite(query_text):
    if not os.path.exists(DB_PATH): return "DB missing"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM knowledge WHERE query LIKE ?", (f"%{query_text}%",))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Not found"

def generate(model, tokenizer, prompt):
    tokens = mx.array(tokenizer.encode(prompt))
    generated = ""
    for _ in range(100):
        logits, _, _ = model(tokens[None, :])
        next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
        if next_token == 3: break
        char = tokenizer.decode([next_token])
        generated += char
        tokens = mx.concatenate([tokens, mx.array([next_token])], axis=0)
        if "</call>" in generated: break
    return generated

def run_test():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found at {CKPT_PATH}")
        return

    print(f"🧪 Testing Checkpoint: {CKPT_PATH}")
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(CKPT_PATH)
    model.eval()

    tests = [
        ("Identity", "Who created you?"),
        ("Identity", "What is your name?"),
        ("Knowledge", "What is the capital of France?"),
        ("Chat", "Hello, who are you?")
    ]

    for cat, q in tests:
        print(f"\n[{cat}] Q: {q}")
        prompt = f"System: You are MirrorAI, created by Dipesh Majithia.\nUser: {q}\nAssistant: "
        res = generate(model, tokenizer, prompt)
        print(f"A: {res}")
        
        if "<call>search_knowledge" in res:
            try:
                query = res.split('("')[1].split('")')[0]
                ctx = query_sqlite(query)
                print(f"🔍 Retrieval: '{query}' -> '{ctx}'")
                prompt_ctx = prompt + res + f"\nContext: {ctx}\nAssistant: "
                final = generate(model, tokenizer, prompt_ctx)
                print(f"✨ Final: {final}")
            except:
                print("❌ Retrieval parse failed")

if __name__ == "__main__":
    run_test()
