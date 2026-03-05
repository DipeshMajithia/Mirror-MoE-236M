import mlx.core as mx
import os
import sys
from model import MirrorTransformer, ModelArgs
from config import TurboConfig
from tokenizer import CustomBPETokenizer

def load_model(model_path="mirror_ai_proper.safetensors"):
    print(f"Loading MirrorAI from {model_path}...")
    cfg = TurboConfig()
    
    # Initialize Tokenizer
    tokenizer = CustomBPETokenizer()
    
    # Initialize Model
    args = ModelArgs(
        dim=cfg.DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        n_layers=cfg.N_LAYERS,
        num_experts=cfg.N_EXPERTS,
        num_experts_per_tok=cfg.ACTIVE_EXPERTS,
        vocab_size=tokenizer.vocab_size,
        shared_expert_dim=cfg.SHARED_DIM,
        use_moe=True
    )
    model = MirrorTransformer(args)
    model.load_weights(model_path)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_tokens=200, temp=0.7):
    # Formatted prompt for Chat
    full_prompt = f"User: {prompt}\nAssistant:"
    tokens = tokenizer.encode(full_prompt)
    x = mx.array([tokens])
    
    print(f"\nUser: {prompt}")
    print("Assistant: ", end="", flush=True)
    
    generated = []
    
    # Auto-regressive generation (Simple ref: re-feed full context)
    # Optimized/Cached version would be better but this is compatible/simple.
    current_tokens = tokens
    
    for _ in range(max_tokens):
        x_in = mx.array([current_tokens])
        logits, _, _ = model(x_in)
        logits = logits[:, -1, :]
        
        token = mx.random.categorical(logits * (1.0/temp)).item()
        
        # Stop check
        if token == 0: # Assuming 0 is padding/EOS in this BPE or needs check
             break
        
        text_chunk = tokenizer.decode([token])
        print(text_chunk, end="", flush=True)
        current_tokens.append(token)
        generated.append(token)
        
    print("\n")
    return tokenizer.decode(generated)

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    print("\n--- Model Ready ---")
    while True:
        p = input("You: ")
        if p.lower() in ['quit', 'exit']: break
        generate_text(model, tokenizer, p)
