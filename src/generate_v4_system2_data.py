import json
import random
import os

def generate_error_correction_samples(num=500):
    dataset = []
    
    # Types of errors
    error_types = [
        {"type": "missing_bracket", "pattern": lambda a, b: f"({a} + {b}", "correct": lambda a, b: f"({a} + {b})"},
        {"type": "double_operator", "pattern": lambda a, b: f"{a} ++ {b}", "correct": lambda a, b: f"{a} + {b}"},
        {"type": "invalid_char", "pattern": lambda a, b: f"{a} @ {b}", "correct": lambda a, b: f"{a} * {b}"},
        {"type": "wrong_function", "pattern": lambda a, b: f"calc({a}, {b})", "correct": lambda a, b: f"{a} + {b}"}
    ]
    
    for _ in range(num):
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        err = random.choice(error_types)
        
        broken_expr = err["pattern"](a, b)
        correct_expr = err["correct"](a, b)
        
        # Mirroring the multi-turn reflection format
        sample = {
            "conversation": [
                {"role": "user", "content": f"What is {a} plus {b}?"},
                {"role": "assistant", "content": f"<call>calculator(\"{broken_expr}\")</call>"},
                {"role": "system_reflection", "content": "The calculator returned an error: Invalid Syntax."},
                {"role": "assistant_reflection", "content": f"I apologize, I made a syntax error in my calculator call. Let me fix the expression.\n<call>calculator(\"{correct_expr}\")</call>"}
            ]
        }
        dataset.append(sample)
        
    return dataset

if __name__ == "__main__":
    data = generate_error_correction_samples(500)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4_system2_correction_dataset.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Generated {len(data)} System 2 samples in {out_path}")
