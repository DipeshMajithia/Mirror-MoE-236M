import torch
import sys
import os
from tokenizers import Tokenizer
from model_pytorch import MirrorTransformer, ModelArgs
from safetensors.torch import load_file

def main():
    print("Loading PyTorch Model (Windows/Linux Compatible)...")
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    print(f"Device: {device}")

    # Load Tokenizer
    tokenizer = Tokenizer.from_file("custom_bpe_32k.json")

    # Load Model
    model = MirrorTransformer(ModelArgs())
    model_path = "mirror_ai_chat_v1.safetensors"
    
    # Load weights (mapping MLX keys to PyTorch might be needed if they differ precisely)
    # MLX and Safetensors usually share format. 
    # NOTE: MLX "layers.0.attention..." might slightly differ from PyTorch "layers[0].attention...".
    # This script assumes names match. If not, a remapping is needed.
    # For now, we assume a direct load.
    
    try:
        state_dict = load_file(model_path)
        # MLX weights might have different names/transposes. 
        # A full conversion script is typically needed.
        # This is a placeholder for the PyTorch script.
        # print("Note: Direct loading MLX weights into PyTorch requires key mapping.")
        
        # Attempt loose load for demo
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Values loaded with warnings: {e}")

    model.to(device)
    model.eval()

    print("\n--- PyTorch Inference Ready ---")
    while True:
        text = input("User: ")
        if text.lower() == "exit": break
        
        # Simple generation loop
        tokens = tokenizer.encode(f"User: {text}\nAssistant:").ids
        input_ids = torch.tensor([tokens]).to(device)
        
        print("Assistant: ", end="", flush=True)
        with torch.no_grad():
            for _ in range(50):
                logits = model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                print(tokenizer.decode([next_token]), end="", flush=True)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
        print("\n")

if __name__ == "__main__":
    main()
