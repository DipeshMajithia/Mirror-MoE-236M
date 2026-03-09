#!/usr/bin/env python3
import os
import sys
import mlx.core as mx

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

def generate(model, tokenizer, prompt, max_tokens=150):
    tokens = tokenizer.encode(prompt)
    generated_ids = []
    
    # We use a greedy argmax for this test to see the model's highest confidence path
    for _ in range(max_tokens):
        x = mx.array(tokens)[None, :]
        logits, _, _ = model(x)
        next_logits = logits[0, -1, :]
        next_token = int(mx.argmax(next_logits))
        
        if next_token == 3: # EOS
            break
            
        generated_ids.append(next_token)
        tokens.append(next_token)
        
        if '</call>' in tokenizer.decode(generated_ids):
            break
            
    return tokenizer.decode(generated_ids).strip()

def test_model():
    model_path = os.path.join(ROOT_DIR, 'model.safetensors')
    print(f"Loading {model_path}...")
    
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()
    mx.eval(model.parameters())
    
    test_queries = [
        "What is your name?",
        "Who created you?",
        "What is 850 + 150?",
        "Tell me about the Eiffel Tower.",
        "What is the capital of Japan?",
        "Tell me a joke."
    ]
    
    print("\n" + "="*50)
    print("🤖 AI AGENT INDEPENDENT TEST run")
    print("="*50)
    
    for query in test_queries:
        print(f"\n👤 Prompt: {query}")
        prompt = f"System: {SYSTEM_PROMPT}\nUser: {query}\nAssistant: "
        response = generate(model, tokenizer, prompt)
        print(f"🤖 Response: {response}")

if __name__ == "__main__":
    test_model()
