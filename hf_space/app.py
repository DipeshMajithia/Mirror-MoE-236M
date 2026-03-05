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

import json
import urllib.request
import urllib.parse
import re

def search_knowledge(query):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "MirrorAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            extract = data.get("extract", "")
            if extract:
                if len(extract) > 300:
                    extract = extract[:300].rsplit('.', 1)[0] + '.'
                return extract
    except Exception:
        pass
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=1"
        req = urllib.request.Request(search_url, headers={"User-Agent": "MirrorAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("query", {}).get("search", [])
            if results:
                snippet = results[0].get("snippet", "")
                snippet = re.sub(r'<[^>]+>', '', snippet)
                title = results[0].get("title", query)
                return f"{title}: {snippet}"
    except Exception:
        pass
    return f"Information about '{query}' could not be retrieved."

def eval_calculator(expr):
    try:
        import math as m
        safe_dict = {k: v for k, v in m.__dict__.items() if not k.startswith("__")}
        safe_dict['abs'] = abs
        safe_dict['round'] = round
        result = eval(expr, {"__builtins__": None}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def generate_part(model, initial_prompt, yield_prefix=""):
    tokens = tokenizer.encode(initial_prompt).ids[-448:]
    input_ids = torch.tensor([tokens]).to(DEVICE)
    
    max_new_tokens = 200
    temperature = 0.7
    top_k = 40
    repetition_penalty = 1.5 
    
    generated_text = ""
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            if generated_tokens:
                for tid in set(generated_tokens):
                    if next_token_logits[tid] > 0:
                        next_token_logits[tid] /= repetition_penalty
                    else:
                        next_token_logits[tid] *= repetition_penalty
            
            next_token_logits = next_token_logits / temperature
            top_k_probs, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_probs, dim=-1)
            
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_index].item()
            
            if next_token == 3:
                break
            
            generated_tokens.append(next_token)
            # Use skip_special_tokens=False so <call> strings show up
            decoded_char = tokenizer.decode([next_token], skip_special_tokens=False)
            generated_text += decoded_char
            
            stop_strings = ["\nUser", "User:", "\nAssistant", "Assistant:", "Context:"]
            break_loop = False
            for s in stop_strings:
                if s in generated_text:
                    generated_text = generated_text.split(s)[0]
                    break_loop = True
                    break
            
            yield yield_prefix + generated_text
            
            if break_loop or "</call>" in generated_text:
                break
                
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(DEVICE)], dim=1)

def generate_response(message, history, model_choice):
    if not model_choice or model_choice not in loaded_models:
        model_choice = list(loaded_models.keys())[0] if loaded_models else None
    if not model_choice:
        yield "Error: No models loaded on server."
        return
    model = loaded_models[model_choice]

    full_prompt = f"System: {SYSTEM_PROMPT}\nUser: {message}\nAssistant: "
    
    final_text = ""
    for text_chunk in generate_part(model, full_prompt, yield_prefix=""):
        final_text = text_chunk
        yield final_text
        
    # Post-generation tool execution routing
    if "</call>" in final_text:
        if "<call>calculator(" in final_text:
            try:
                expr = final_text.split('<call>calculator("')[1].split('")</call>')[0]
                result = eval_calculator(expr)
                tool_output = f"\n\n**📟 Calculator:** `{expr} = {result}`\n\n"
                final_text += tool_output
                yield final_text
                
                ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {message}\nCalculation Result: {expr} = {result}\nAssistant: "
                for text_chunk in generate_part(model, ctx_prompt, yield_prefix=final_text):
                    yield text_chunk
            except Exception as e:
                pass
                
        elif "<call>search_knowledge(" in final_text:
            try:
                query = final_text.split('<call>search_knowledge("')[1].split('")</call>')[0]
                context = search_knowledge(query)
                tool_output = f"\n\n**🔍 Search:** `{query}`\n*Found: {context[:100]}...*\n\n"
                final_text += tool_output
                yield final_text
                
                ctx_prompt = f"System: {SYSTEM_PROMPT}\nUser: {message}\nContext: {context}\nAssistant: "
                for text_chunk in generate_part(model, ctx_prompt, yield_prefix=final_text):
                    yield text_chunk
            except Exception as e:
                pass


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
