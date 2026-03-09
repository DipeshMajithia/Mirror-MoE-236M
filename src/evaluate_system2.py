import os
import sys
import json
import mlx.core as mx
import argparse
from tqdm import tqdm

from v3.chat_sota import MirrorTransformer, MirrorTokenizer, V3_ARGS, sample_top_p

SYSTEM_PROMPT = """You are MirrorAI, a personal AI assistant created by Dipesh Majithia.
You are helpful, friendly, and knowledgeable.
When you need factual information, use: <call>search_knowledge("query")</call>
When you need to perform math calculations, use: <call>calculator("expression")</call>
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly."""

def eval_calculator(expr):
    try:
        return str(eval(expr, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"

# 100 Test Cases: 70 In-Distribution (syntax errors like dataset), 30 Out-of-Distribution (weird edge cases)
TEST_CASES = [
    # In-Distribution (Missing Brackets)
    {"user": "What is 125 + 372?", "broken_call": "<call>calculator(\"(125 + 372\")</call>", "expected_fix": "(125 + 372)"},
    {"user": "Calculate 40 * 12.", "broken_call": "<call>calculator(\"40 * 12)\")</call>", "expected_fix": "(40 * 12)"},
    {"user": "Divide 100 by 4.", "broken_call": "<call>calculator(\"(100 / 4\")</call>", "expected_fix": "(100 / 4)"},
    {"user": "What is 55 - 10?", "broken_call": "<call>calculator(\"55 - 10)\")</call>", "expected_fix": "(55 - 10)"},
    {"user": "Calculate 8 + 8.", "broken_call": "<call>calculator(\"(8 + 8\")</call>", "expected_fix": "(8 + 8)"},
    # In-Distribution (Double Operators)
    {"user": "What is 10 + 20?", "broken_call": "<call>calculator(\"10 ++ 20\")</call>", "expected_fix": "10 + 20"},
    {"user": "Calculate 50 - 5.", "broken_call": "<call>calculator(\"50 -- 5\")</call>", "expected_fix": "50 - 5"},
    {"user": "Multiply 7 by 6.", "broken_call": "<call>calculator(\"7 ** 6\")</call>", "expected_fix": "7 * 6"},
    {"user": "Divide 20 by 2.", "broken_call": "<call>calculator(\"20 // 2\")</call>", "expected_fix": "20 / 2"},
    {"user": "Calculate 15 + 15.", "broken_call": "<call>calculator(\"15 ++ 15\")</call>", "expected_fix": "15 + 15"},
    # In-Distribution (Invalid chars)
    {"user": "What is 100 * 2?", "broken_call": "<call>calculator(\"100 x 2\")</call>", "expected_fix": "100 * 2"},
    {"user": "Calculate 50 / 5.", "broken_call": "<call>calculator(\"50 \\ 5\")</call>", "expected_fix": "50 / 5"},
    {"user": "What is 10 + 10?", "broken_call": "<call>calculator(\"10 & 10\")</call>", "expected_fix": "10 + 10"},
    {"user": "Multiply 8 by 8.", "broken_call": "<call>calculator(\"8 @ 8\")</call>", "expected_fix": "8 * 8"},
    {"user": "Calculate 30 - 15.", "broken_call": "<call>calculator(\"30 ~ 15\")</call>", "expected_fix": "30 - 15"},
    
    # We will procedurally generate the remaining 55 In-Distribution to reach 70
]

for i in range(55):
    a, b = 10 + i, 20 + i
    TEST_CASES.append({
        "user": f"What is {a} + {b}?",
        "broken_call": f"<call>calculator(\"{a} ++ {b}\")</call>",
        "expected_fix": f"{a} + {b}"
    })

# Out-of-Distribution (30 cases, weird errors the model never saw)
OOD_CASES = [
    {"user": "Calculate log of 10 plus something.", "broken_call": "<call>calculator(\"log(10 + )\")</call>", "expected_fix": "null"}, # Should rewrite logically or fail gracefully
    {"user": "What's 12 + 4?", "broken_call": "<call>calculator(\"(12 + 4))\")</call>", "expected_fix": "(12 + 4)"},
    {"user": "Multiply 5 by nothing.", "broken_call": "<call>calculator(\"5 * \")</call>", "expected_fix": "null"},
    {"user": "What is the square root of 25?", "broken_call": "<call>calculator(\"sqrt(25\")</call>", "expected_fix": "math.sqrt(25)"},
    {"user": "Calculate 10 divided by 2.", "broken_call": "<call>calculator(\"10 / / 2\")</call>", "expected_fix": "10 / 2"},
]

for i in range(25):
    a, b = 100 + i, 5 + i
    OOD_CASES.append({
        "user": f"Calculate {a} with {b}.",
        "broken_call": f"<call>calculator(\"{a} {b}\")</call>", # Missing operator entirely
        "expected_fix": "null"
    })

ALL_CASES = TEST_CASES + OOD_CASES

def run_benchmark(model_path):
    print(f"\n🚀 Starting Error Recovery Benchmark for: {model_path}")
    tokenizer = MirrorTokenizer()
    model = MirrorTransformer(V3_ARGS)
    model.load_weights(model_path, strict=False)
    model.eval()
    
    success_count = 0
    total = len(ALL_CASES)
    
    for case in tqdm(ALL_CASES, desc="Evaluating"):
        user_input = case["user"]
        initial_output = case["broken_call"]
        
        # Run eval to get the actual error string
        expr = initial_output.split('<call>calculator("')[1].split('")</call>')[0]
        result = eval_calculator(expr)
        
        # Inject Reflection
        reflection_prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_input}\nAssistant: {initial_output}\nSystem (Reflection): The calculator returned an error: {result}. Please correct your expression. Use exactly: <call>calculator(\"correct_expression\")</call>\nAssistant: "
        
        tokens = tokenizer.encode(reflection_prompt)
        generated_ids = []
        
        for _ in range(50):
            x = mx.array(tokens)[None, :]
            logits, _, _ = model(x)
            next_token = sample_top_p(logits[0, -1, :], generated_ids=generated_ids)
            if next_token == 3: break
            generated_ids.append(next_token)
            tokens.append(next_token)
            if "</call>" in tokenizer.decode(generated_ids): break
            
        corrected_output = tokenizer.decode(generated_ids)
        
        # Evaluate Success
        if "<call>calculator(" in corrected_output:
            new_expr = corrected_output.split('<call>calculator("')[1].split('")</call>')[0]
            if "Error" not in eval_calculator(new_expr):
                success_count += 1
                
    success_rate = (success_count / total) * 100
    print(f"\n✅ Benchmark Complete for {model_path}")
    print(f"Total Cases: {total}")
    print(f"Successful Recoveries: {success_count}")
    print(f"Recovery Rate: {success_rate:.1f}%\n")
    return success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to weights")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        run_benchmark(args.model)
    else:
        print(f"Model not found: {args.model}")
