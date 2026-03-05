import gradio as gr
import torch
import os
from tokenizers import Tokenizer
from model_pytorch import MirrorTransformer, ModelArgs
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# --- Configuration ---
MODELS_CONFIG = {
    "MirrorAI V3 (236M MoE)": "model.safetensors"
}
TOKENIZER_PATH = "tokenizer.json"
REPO_ID = "dipeshmajithia/Mirror-MoE-236M"

DEVICE = "cpu"
if torch.cuda.is_available():
    
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

print(f"Using device: {DEVICE}")

# --- Load Resources ---
print("Loading Tokenizer...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Global Model Registry
loaded_models = {}

def load_converted_weights(model, model_path):
    print(f"Loading weights from {model_path}...")
    state_dict = load_file(model_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        # Map Keys
        if k == "emb.weight":
            new_k = "tok_embeddings.weight"
        else:
            new_k = k
        new_state_dict[new_k] = v
        
    keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Load keys: {keys}")
    return model

# Initialize Models
hf_token = os.environ.get("HF_TOKEN", None)
model_args = ModelArgs()

for name, filename in MODELS_CONFIG.items():
    print(f"Preparing {name}...")
    try:
        # 1. Download if needed
        if not os.path.exists(filename):
            print(f"Downloading {filename} from Hub...")
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                token=hf_token
            )
            # hf_hub_download returns absolute path to cache
            file_path = path
        else:
            file_path = filename
            
        # 2. Init and Load
        m = MirrorTransformer(model_args)
        m = load_converted_weights(m, file_path)
        m.to(DEVICE)
        m.eval()
        loaded_models[name] = m
        print(f"✅ Loaded {name}")
        
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")

if not loaded_models:
    print("⚠️ CRITICAL: No models loaded. App will crash.")

# --- Inference Logic ---
def generate_response(message, history, model_choice):
    """
    Generates response using the selected model.
    """
    # Default to first model if choice is invalid or None
    if not model_choice or model_choice not in loaded_models:
        model_choice = list(loaded_models.keys())[0] if loaded_models else None
        
    if not model_choice:
        yield "Error: No models loaded on server."
        return

    model = loaded_models[model_choice]

    # 1. Prompt Building (No System Prompt)
    full_prompt = ""
    # Gradio 4.x: history is list of tuples: [(user_msg, bot_msg), ...]
    recent_history = history[-4:] if history else []
    
    for turn in recent_history:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg, bot_msg = turn[0], turn[1]
            if user_msg:
                full_prompt += f"User: {user_msg}\n"
            if bot_msg:
                full_prompt += f"Assistant: {bot_msg}\n"
        elif isinstance(turn, dict):
             # Fallback
            if turn.get('role') == 'user':
                full_prompt += f"User: {turn['content']}\n"
            elif turn.get('role') == 'assistant':
                full_prompt += f"Assistant: {turn['content']}\n"
    
    full_prompt += f"User: {message}\nAssistant:"
    
    # Tokenize
    tokens = tokenizer.encode(full_prompt).ids[-400:] # Keep context short
    input_ids = torch.tensor([tokens]).to(DEVICE)
    
    # Generate
    max_new_tokens = 120
    temperature = 0.6
    top_k = 40
    
    generated_text = ""
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # Sampling Strategy: Temp + Top-K
            next_token_logits = next_token_logits / temperature
            top_k_probs, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_probs, dim=-1)
            
            # Sample
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token_index).item()
            
            # Stop if EOS (3)
            if next_token == 3:
                break
            
            decoded_char = tokenizer.decode([next_token])
            
            # Stop strings
            # "User", "Instruction", "Context" often appear as hallucinations
            stop_strings = ["\nUser", "User:", "\nInstruction", "Instruction:", "Context:", "\nContext"]
            
            # Optimization: Don't print the stop string itself
            temp_text = generated_text + decoded_char
            found_stop = False
            for s in stop_strings:
                if s in temp_text:
                    found_stop = True
                    break
            
            if found_stop:
                break
                
            generated_text += decoded_char
            yield generated_text.strip()
            
            # Update input
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

# --- Gradio UI ---
model_names = list(MODELS_CONFIG.keys())
default_model = model_names[0] if model_names else None

demo = gr.ChatInterface(
    fn=generate_response,
    additional_inputs=[
        gr.Dropdown(
            choices=model_names, 
            value=default_model, 
            label="Model Selection",
            info="Hybrid: Balanced Chat/Facts | Elite: High IQ/Reasoning"
        )
    ],
    title="Mirror-MoE-80M",
    description="An 236M parameter Sparse MoE model.",
    examples=[
        ["Who are you?", "MirrorAI V3 (236M MoE)"],
        ["What is 125 + 372?", "MirrorAI V3 (236M MoE)"],
        ["Hi!", "MirrorAI V3 (236M MoE)"],
        ["Tell me a short story.", "MirrorAI V3 (236M MoE)"],
        ["What is the capital of France?", "MirrorAI V3 (236M MoE)"]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
