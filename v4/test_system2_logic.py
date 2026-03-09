import os
import mlx.core as mx
import numpy as np
from v3.chat_sota import MirrorTransformer, MirrorTokenizer, V3_ARGS, sample_top_p

# System prompt for verification
SYSTEM_PROMPT = """You are MirrorAI. Use: <call>calculator("expression")</call> for math."""

def eval_calculator(expr):
    try:
        # Simulate a syntax error if specific junk is present
        if "@" in expr or "++" in expr:
            raise SyntaxError("Invalid operator")
        # Simple eval for testing
        return str(eval(expr, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {e}"

def test_system2_recovery(model_path):
    print(f"🔄 Loading local model for System 2 Verification...")
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()

    # Scenario: User asks a common math question
    user_input = "What is 125 + 372?" 
    full_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nAssistant: "
    
    print(f"User: {user_input}")
    
    # --- PHASE 1: Forced Tool Call (Simulating Model's First Attempt) ---
    # We'll generate a few tokens to see if it calls naturally, 
    # but for testing the HARNESS, we can also inject a faulty call if it doesn't.
    initial_output = '<call>calculator("125 ++ 372")</call>' # Simulated broken call
    print(f"Assistant (Initial, Simulated Broken): {initial_output}")
    
    # --- PHASE 2: Error Detection & Reflection ---
    if "<call>calculator(" in initial_output:
        expr = initial_output.split('<call>calculator("')[1].split('")</call>')[0]
        result = eval_calculator(expr)
        
        if "Error" in result:
            print(f"🛑 Detected Error: {result}")
            print("🧠 Reflecting & Re-generating...")
            
            # Reflection Prompt Injection (This is what we are testing)
            reflection_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nAssistant: {initial_output}\nSystem (Reflection): The calculator returned an error: {result}. Please correct your expression. Use exactly: <call>calculator(\"correct_expression\")</call>\nAssistant: "
            
            tokens = tokenizer.encode(reflection_prompt)
            generated_ids = []
            
            # Now let's see what the REAL model does with this reflection
            for _ in range(50):
                x = mx.array(tokens)[None, :]
                logits, _, _ = model(x)
                next_token = sample_top_p(logits[0, -1, :], generated_ids=generated_ids)
                if next_token == 3: break
                generated_ids.append(next_token)
                tokens.append(next_token)
                if "</call>" in tokenizer.decode(generated_ids): break
            
            corrected_output = tokenizer.decode(generated_ids)
            print(f"Assistant (Corrected Attempt): {corrected_output}")
            
            # Phase 3: Final Answer
            if "<call>calculator(" in corrected_output:
                expr = corrected_output.split('<call>calculator("')[1].split('")</call>')[0]
                result = eval_calculator(expr)
                print(f"✅ Second Attempt Script Evaluation: {result}")
    else:
        print("No tool call detected. Model answered directly.")

if __name__ == "__main__":
    # Point to the local safetensors
    local_weights = "model.safetensors"
    if os.path.exists(local_weights):
        test_system2_recovery(local_weights)
    else:
        print(f"Error: {local_weights} not found. Please place weights in root.")
