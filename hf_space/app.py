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

# --- Chat Configuration ---
SYSTEM_PROMPT = """You are MirrorAI, a personal AI assistant created by Dipesh Majithia.
You are helpful, friendly, and knowledgeable.
When you need factual information, use: <call>search_knowledge("query")</call>
When you need to perform math calculations, use: <call>calculator("expression")</call>
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly.
"""

# --- Inference Logic ---
def generate_response(message, history, model_choice):
    """
    Generates response using the selected model with repetition penalty.
    """
    if not model_choice or model_choice not in loaded_models:
        model_choice = list(loaded_models.keys())[0] if loaded_models else None
        
    if not model_choice:
        yield "Error: No models loaded on server."
        return

    model = loaded_models[model_choice]

    # 1. Prompt Building (Aligned with Training - Single Turn Only)
    full_prompt = f"System: {SYSTEM_PROMPT}\nUser: {message}\nAssistant: "
    
    # Tokenize
    tokens = tokenizer.encode(full_prompt).ids[-448:] # Slightly larger window
    input_ids = torch.tensor([tokens]).to(DEVICE)
    
    # Generation Params
    max_new_tokens = 200
    temperature = 0.7
    top_k = 40
    repetition_penalty = 1.5 # Stabilize output (matched to MLX)
    
    generated_text = ""
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Apply Repetition Penalty
            if generated_tokens:
                for tid in set(generated_tokens):
                    if next_token_logits[tid] > 0:
                        next_token_logits[tid] /= repetition_penalty
                    else:
                        next_token_logits[tid] *= repetition_penalty
            
            # Sampling Strategy: Temp + Top-K
            next_token_logits = next_token_logits / temperature
            top_k_probs, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_probs, dim=-1)
            
            # Sample
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_index].item()
            
            # Stop if EOS (3)
            if next_token == 3:
                break
            
            generated_tokens.append(next_token)
            decoded_char = tokenizer.decode([next_token])
            
            # Stop strings logic
            stop_strings = ["\nUser", "User:", "\nAssistant", "Assistant:", "Context:"]
            
            generated_text += decoded_char
            
            # Check for stop strings
            break_loop = False
            for s in stop_strings:
                if s in generated_text:
                    generated_text = generated_text.split(s)[0]
                    break_loop = True
                    break
            
            if break_loop: break
                
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
