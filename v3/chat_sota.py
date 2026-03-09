#!/usr/bin/env python3
"""
MirrorAI V3 — Interactive Chat (Single-Turn with Live Tools)
- search_knowledge: Uses Wikipedia API for factual lookups
- calculator: Evaluates math expressions safely
"""
import os, sys, json, re
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

SYSTEM_PROMPT = """You are DonAI, a personal AI assistant created by Dipesh Majithia.
You are helpful, friendly, and knowledgeable.
When you need factual information, use: <call>search_knowledge("query")</call>
When you need to perform math calculations, use: <call>calculator("expression")</call>
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly.
"""

# ── Tool Implementations ──────────────────────────────────

def search_knowledge(query):
    """Search Wikipedia for factual information."""
    import urllib.request, urllib.parse
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "MirrorAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            extract = data.get("extract", "")
            if extract:
                # Trim to ~300 chars to fit in context window
                if len(extract) > 300:
                    extract = extract[:300].rsplit('.', 1)[0] + '.'
                return extract
    except Exception:
        pass
    
    # Fallback: Wikipedia search API
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=1"
        req = urllib.request.Request(search_url, headers={"User-Agent": "MirrorAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("query", {}).get("search", [])
            if results:
                snippet = results[0].get("snippet", "")
                # Clean HTML tags from snippet
                snippet = re.sub(r'<[^>]+>', '', snippet)
                title = results[0].get("title", query)
                return f"{title}: {snippet}"
    except Exception:
        pass
    
    return f"Information about '{query}' could not be retrieved."

def eval_calculator(expr):
    """Safely evaluate a mathematical expression."""
    try:
        import math as m
        safe_dict = {k: v for k, v in m.__dict__.items() if not k.startswith("__")}
        safe_dict['abs'] = abs
        safe_dict['round'] = round
        result = eval(expr, {"__builtins__": None}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# ── Sampling & Generation ─────────────────────────────────

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

def generate_stream(model, tokenizer, prompt, max_tokens=300):
    """Generate tokens and stream to stdout."""
    tokens = tokenizer.encode(prompt)
    generated_ids = []
    
    for _ in range(max_tokens):
        x = mx.array(tokens)[None, :]
        logits, _, _ = model(x)
        next_logits = logits[0, -1, :]
        next_token = sample_top_p(next_logits, generated_ids=generated_ids)
        
        if next_token == 3: break  # EOS
        
        generated_ids.append(next_token)
        tokens.append(next_token)
        
        # Stream decoded text
        word = tokenizer.decode([next_token])
        print(word, end="", flush=True)
        
        # Stop at tool call end
        decoded_so_far = tokenizer.decode(generated_ids)
        if "</call>" in decoded_so_far:
            break
            
    print()
    return tokenizer.decode(generated_ids).strip()

def generate_silent(model, tokenizer, prompt, max_tokens=300):
    """Generate without streaming (for follow-up after tool call)."""
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
        
        if "</call>" in tokenizer.decode(generated_ids):
            break
            
    return tokenizer.decode(generated_ids).strip()

# ── Main Chat Loop ────────────────────────────────────────

def chat(model_path):
    print(f"🔄 Loading model: {os.path.basename(model_path)}")
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()
    mx.eval(model.parameters())
    
    print("\n" + "="*60)
    print(" 🤖 MirrorAI V3 — Interactive Chat")
    print(" Tools: 🔍 Wikipedia Search  |  📟 Calculator")
    print(" Type 'exit' to quit")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input: continue
            if user_input.lower() in ('exit', 'quit', 'q'): break
            
            # Single-turn prompt — matching training format
            prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nAssistant: "
            
            print("🤖 MirrorAI: ", end="", flush=True)
            response = generate_stream(model, tokenizer, prompt)
            
            # Execute tool calls and get a follow-up response
            if "<call>calculator(" in response:
                try:
                    expr = response.split('("')[1].split('")')[0]
                    result = eval_calculator(expr)
                    print(f"   📟 {expr} = {result}")
                    # Feed result back to model
                    ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nCalculation Result: {expr} = {result}\nAssistant: "
                    print("   🤖 ", end="", flush=True)
                    generate_stream(model, tokenizer, ctx_prompt)
                except Exception as e:
                    print(f"   ❌ Calculator error: {e}")
                    
            elif "<call>search_knowledge(" in response:
                try:
                    query = response.split('("')[1].split('")')[0]
                    print(f"   🔍 Searching: \"{query}\"...")
                    context = search_knowledge(query)
                    print(f"   📚 Found: {context[:100]}...")
                    # Feed context back to model for natural language answer
                    ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nContext: {context}\nAssistant: "
                    print("   🤖 ", end="", flush=True)
                    generate_stream(model, tokenizer, ctx_prompt)
                except Exception as e:
                    print(f"   ❌ Search error: {e}")
            
            print()  # blank line between turns
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-find latest checkpoint
        ckpt_dir = os.path.join(ROOT_DIR, "out/v3_sota")
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".safetensors")]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                model_path = os.path.join(ckpt_dir, ckpts[-1])
                print(f"Auto-selected latest checkpoint: {ckpts[-1]}")
            else:
                print("No checkpoints found in out/v3_sota/")
                sys.exit(1)
        else:
            print("Usage: python3 v3/chat_sota.py [model_path]")
            sys.exit(1)
    else:
        model_path = sys.argv[1]
    
    chat(model_path)
