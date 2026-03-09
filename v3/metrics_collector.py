import os
import sys
import mlx.core as mx
import time

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

def measure_performance():
    model_path = os.path.join(ROOT_DIR, 'model.safetensors')
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()
    
    prompt = "Explain the importance of water in one paragraph."
    tokens = tokenizer.encode(prompt)
    x = mx.array(tokens)[None, :]
    
    # Warmup
    for _ in range(5):
        logits, _, _ = model(x)
        mx.eval(logits)
    
    start = time.time()
    num_tokens = 100
    generated = []
    curr_tokens = tokens
    for _ in range(num_tokens):
        x = mx.array(curr_tokens)[None, :]
        logits, _, _ = model(x)
        next_token = int(mx.argmax(logits[0, -1, :]))
        generated.append(next_token)
        curr_tokens.append(next_token)
        mx.eval(next_token)
    
    end = time.time()
    tps = num_tokens / (end - start)
    print(f"Inference speed: {tps:.2f} tokens/sec")

    expert_counts = [0] * 16
    for i in range(len(curr_tokens) - num_tokens, len(curr_tokens)):
        x = mx.array(curr_tokens[:i+1])[None, :]
        _, _, all_topk_indices = model(x)
        for layer_idx in range(8):
            # layer_indices shape: (1, L, 2)
            layer_indices = all_topk_indices[layer_idx]
            # tok_indices shape: (2,)
            tok_indices = layer_indices[0, -1]
            for k in range(2):
                idx = int(tok_indices[k].item())
                expert_counts[idx] += 1
    
    total_uses = sum(expert_counts)
    for i, count in enumerate(expert_counts):
        percentage = (count / total_uses) * 100
        print(f"Expert {i:2}: {percentage:5.1f}%")

if __name__ == "__main__":
    measure_performance()
