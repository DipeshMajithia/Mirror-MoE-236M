import os
import sys
import mlx.core as mx
from v3.chat_sota import MirrorTransformer, MirrorTokenizer, V3_ARGS, sample_top_p

SYSTEM_PROMPT = """You are MirrorAI, a personal AI assistant created by Dipesh Majithia.
You are helpful, friendly, and knowledgeable.
When you need factual information, use: <call>search_knowledge("query")</call>
When you need to perform math calculations, use: <call>calculator("expression")</call>
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly."""

def eval_calculator(expr):
    try:
        # Simple math eval for demo
        return str(eval(expr, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"

def chat_v4(model_path):
    print(f"\n🧠 Loading MirrorAI V4 (System 2 Enabled) from {model_path}...")
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()

    print("\n" + "="*50)
    print("  Welcome to the MirrorAI V4 Interactive Lab")
    print("  Type 'exit' to quit.")
    print("  Try a broken prompt: 'What is 125 ++ 372?' or 'sqrt(25'")
    print("="*50 + "\n")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        # Reset context to isolate each query test
        history = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nAssistant: "
        
        # --- PHASE 1: Initial Attempt ---
        tokens = tokenizer.encode(history)
        generated_ids = []
        
        print("Assistant (Thinking...)", end="", flush=True)
        for _ in range(100):
            x = mx.array(tokens)[None, :]
            logits, _, _ = model(x)
            next_token = sample_top_p(logits[0, -1, :], generated_ids=generated_ids)
            if next_token == 3: break
            generated_ids.append(next_token)
            tokens.append(next_token)
            if "</call>" in tokenizer.decode(generated_ids): break
        
        initial_output = tokenizer.decode(generated_ids)
        print(f"\nAssistant (Initial): {initial_output}")
        
        # --- PHASE 2: Error Detection & Reflection ---
        if "<call>calculator(" in initial_output:
            try:
                expr = initial_output.split('<call>calculator("')[1].split('")</call>')[0]
                result = eval_calculator(expr)
                
                if "Error" in result:
                    print(f"🛑 Tool Error: {result}")
                    print("🧠 V4 Reflecting...")
                    
                    reflection_text = f"\nSystem (Reflection): The calculator returned an error: {result}. Please correct your expression.\nAssistant: "
                    history += initial_output + reflection_text
                    
                    tokens = tokenizer.encode(history)
                    generated_ids = []
                    
                    for _ in range(100):
                        x = mx.array(tokens)[None, :]
                        logits, _, _ = model(x)
                        next_token = sample_top_p(logits[0, -1, :], generated_ids=generated_ids)
                        if next_token == 3: break
                        generated_ids.append(next_token)
                        tokens.append(next_token)
                        if "</call>" in tokenizer.decode(generated_ids): break
                    
                    corrected_output = tokenizer.decode(generated_ids)
                    print(f"Assistant (Correction): {corrected_output}")
                    
                    # Try eval one last time for the final answer
                    if "<call>calculator(" in corrected_output:
                        new_expr = corrected_output.split('<call>calculator("')[1].split('")</call>')[0]
                        final_res = eval_calculator(new_expr)
                        print(f"✅ Executed Result: {final_res}")
                        history += corrected_output + f"\nAssistant: The answer is {final_res}.\n"
                        print(f"Assistant: The answer is {final_res}.")
                    else:
                        history += corrected_output + "\n"
                else:
                    print(f"✅ Tool Success: {result}")
                    final_ans = f"Calculating... the result is {result}."
                    print(f"Assistant: {final_ans}")
                    history += initial_output + f"\nAssistant: {final_ans}\n"
            except Exception as e:
                print(f"⚠️ Parsing failed: {e}")
                history += initial_output + "\n"
        else:
            history += initial_output + "\n"
        print("-" * 30)

if __name__ == "__main__":
    v4_path = "model_v4.safetensors"
    if os.path.exists(v4_path):
        chat_v4(v4_path)
    else:
        print(f"❌ V4 Model not found at {v4_path}. Run training first!")
