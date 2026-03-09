#!/usr/bin/env python3
"""
MirrorAI V3 — Interactive Chat with Research-Grade Sampling
Features: Temperature, Top-p (Nucleus), Repetition Penalty, RAG via SQLite
"""
import os, sys, sqlite3
import mlx.core as mx
import numpy as np

# Paths
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

# ──────────────────────────────────────────────────────────────
# Research-Grade Sampling
# ──────────────────────────────────────────────────────────────
def sample_top_p(logits, temperature=0.7, top_p=0.9, repetition_penalty=1.2, 
                 generated_ids=None):
    """
    Nucleus (top-p) sampling with temperature and repetition penalty.
    This is the standard used by LLaMA, Mistral, and other research models.
    """
    # 1. Apply repetition penalty to already-generated tokens
    if generated_ids and len(generated_ids) > 0:
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits = logits.at[token_id].multiply(1.0 / repetition_penalty)
            else:
                logits = logits.at[token_id].multiply(repetition_penalty)
    
    # 2. Apply temperature
    logits = logits / temperature
    
    # 3. Convert to numpy for sorting (MLX doesn't have full argsort support for this)
    logits_np = np.array(logits)
    
    # 4. Sort descending
    sorted_indices = np.argsort(-logits_np)
    sorted_logits = logits_np[sorted_indices]
    
    # 5. Softmax
    sorted_logits = sorted_logits - sorted_logits.max()  # numerical stability  
    probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
    
    # 6. Cumulative sum for top-p filtering
    cumsum = np.cumsum(probs)
    # Keep tokens until cumulative probability exceeds top_p
    cutoff_idx = np.searchsorted(cumsum, top_p) + 1
    cutoff_idx = min(cutoff_idx, len(probs))
    
    # 7. Zero out everything below cutoff
    probs[cutoff_idx:] = 0.0
    probs = probs / (probs.sum() + 1e-10)
    
    # 8. Sample from the filtered distribution
    chosen_idx = np.random.choice(cutoff_idx, p=probs[:cutoff_idx] / probs[:cutoff_idx].sum())
    return int(sorted_indices[chosen_idx])


def query_sqlite(query_text):
    if not os.path.exists(DB_PATH):
        return "Knowledge database not found."
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM knowledge WHERE query LIKE ?", (f"%{query_text}%",))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "No specific information found."
    except Exception as e:
        return f"Error: {e}"


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


def generate_response(model, tokenizer, prompt, max_tokens=200, 
                      temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    """Generate a response using research-grade sampling."""
    tokens = tokenizer.encode(prompt)
    generated_ids = []
    
    for _ in range(max_tokens):
        x = mx.array(tokens)[None, :]
        logits, _, _ = model(x)
        next_logits = logits[0, -1, :]  # Last position logits
        
        # Sample with temperature + top-p + repetition penalty
        next_token = sample_top_p(
            next_logits, 
            temperature=temperature, 
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated_ids
        )
        
        if next_token == 3:  # EOS
            break
            
        generated_ids.append(next_token)
        tokens.append(next_token)
        
        decoded = tokenizer.decode(generated_ids)
        if "</call>" in decoded:
            break
    
    return tokenizer.decode(generated_ids).strip()


def find_best_model():
    """Auto-detect the best available model weights."""
    # Priority 1: Final V3 Pass 2 weights (SOTA)
    p2_path = os.path.join(ROOT_DIR, "mirror_ai_v3_final_pass2.safetensors")
    if os.path.exists(p2_path):
        return p2_path

    # Priority 2: Final V3 Pass 1 weights
    final_path = os.path.join(ROOT_DIR, "mirror_ai_v3_final.safetensors")
    if os.path.exists(final_path):
        return final_path
    
    # Priority 3: Latest V3 Pass 2 checkpoint
    p2_dir = os.path.join(ROOT_DIR, "out/v3_pass2")
    if os.path.exists(p2_dir):
        ckpts = [f for f in os.listdir(p2_dir) if f.endswith(".safetensors")]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 0)
            return os.path.join(p2_dir, ckpts[-1])

    # Priority 4: Latest V3 Pass 1 checkpoint
    ckpt_dir = os.path.join(ROOT_DIR, "out/v3")
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else 0)
            return os.path.join(ckpt_dir, ckpts[-1])
    
    # Priority 5: V2 base
    v2_path = os.path.join(ROOT_DIR, "out/v2/phase2_final.safetensors")
    if os.path.exists(v2_path):
        return v2_path
    
    return None


def chat_loop():
    print("=" * 60)
    print("  🚀 MirrorAI V3 — Research-Grade Conversational AI")
    print("  Creator: Dipesh Majithia | Architecture: 236M MoE")
    print("  Sampling: Temperature(0.7) + Top-p(0.9) + RepPenalty(1.2)")
    print("=" * 60)
    
    model_path = find_best_model()
    if not model_path:
        print("❌ No model found.")
        sys.exit(1)
        
    print(f"✅ Loading: {model_path}...")
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()
    mx.eval(model.parameters())
    print("🟢 Ready!\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit', 'bye']: break
            
            # STATELESS: Fresh prompt per turn
            prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nAssistant: "
            response = generate_response(model, tokenizer, prompt)
            
            # RAG: Check for knowledge retrieval
            if "<call>search_knowledge" in response:
                try:
                    query = response.split('("')[1].split('")')[0]
                    print(f"🔍 Searching knowledge for '{query}'...")
                    context = query_sqlite(query)
                    
                    # Second pass with context
                    ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nContext: {context}\nAssistant: "
                    final_response = generate_response(model, tokenizer, ctx_prompt)
                    print(f"🤖 MirrorAI: {final_response}")
                except Exception as e:
                    print(f"🤖 MirrorAI: {response}")
            # TOOL: Calculator
            elif "<call>calculator" in response:
                try:
                    expr = response.split('("')[1].split('")')[0]
                    print(f"🧮 Calculating: {expr}")
                    result = eval_calculator(expr)
                    
                    # Second pass with calculation result
                    ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nCalculation Result: {expr} = {result}\nAssistant: "
                    final_response = generate_response(model, tokenizer, ctx_prompt)
                    print(f"🤖 MirrorAI: {final_response}")
                except Exception as e:
                    print(f"🤖 MirrorAI: {response}")
            else:
                print(f"🤖 MirrorAI: {response}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    chat_loop()
