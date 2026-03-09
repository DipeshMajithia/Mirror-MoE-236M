import json
import random
import os

# --- Prompts and Templates ---
SYSTEM_PROMPT = """You are MirrorAI, a personal AI assistant created by Dipesh Majithia.
You are helpful, friendly, and knowledgeable.
When you need factual information, use: <call>search_knowledge("query")</call>
When you need to perform math calculations, use: <call>calculator("expression")</call>
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly."""

FACT_QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote the play Hamlet?", "William Shakespeare"),
    ("What is the largest planet?", "Jupiter"),
    ("When did the Titanic sink?", "1912"),
    ("What is the speed of light?", "299,792 km/s"),
    ("Who discovered gravity?", "Isaac Newton"),
    ("What is the tallest mountain on Earth?", "Mount Everest"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci")
]

# --- Generators ---

def gen_golden_path_math():
    """Scenario 1 (Math): Perfect first try."""
    a, b = random.randint(10, 1000), random.randint(10, 1000)
    expr = f"{a} + {b}"
    return [
        {"role": "user", "content": f"Calculate {expr}."},
        {"role": "assistant", "content": f"<call>calculator(\"{expr}\")</call>"},
        {"role": "system_reflection", "content": f"Calculator Result: {a+b}"},
        {"role": "assistant", "content": f"The result of {a} + {b} is {a+b}."}
    ]

def gen_golden_path_search():
    """Scenario 1 (Search): Perfect first try."""
    q, a = random.choice(FACT_QUESTIONS)
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": f"<call>search_knowledge(\"{a}\")</call>"},
        {"role": "system_reflection", "content": f"Search Result: {a} is the correct answer regarding {q}."},
        {"role": "assistant", "content": f"Based on my search, the answer is {a}."}
    ]

def gen_syntax_correction():
    """Scenario 2: Syntax Error -> Reflection -> Correction."""
    a, b = random.randint(10, 1000), random.randint(10, 1000)
    correct_expr = f"({a} + {b})"
    broken_expr = f"({a} + {b}" # Missing closing paren
    return [
        {"role": "user", "content": f"What is {a} + {b}?"},
        {"role": "assistant", "content": f"<call>calculator(\"{broken_expr}\")</call>"},
        {"role": "system_reflection", "content": "The calculator returned an error: SyntaxError: unexpected EOF while parsing. Please correct your expression. Use exactly: <call>calculator(\"correct_expression\")</call>"},
        {"role": "assistant", "content": f"I apologize, I made a syntax error missing a closing bracket. Here is the corrected call: <call>calculator(\"{correct_expr}\")</call>"},
        {"role": "system_reflection", "content": f"Calculator Result: {a+b}"},
        {"role": "assistant", "content": f"The answer is {a+b}."}
    ]

def gen_semantic_correction():
    """Scenario 3: Logical Error (e.g., div by zero) -> Reflection -> Rewrite."""
    a = random.randint(10, 1000)
    return [
        {"role": "user", "content": f"Divide {a} by 0."},
        {"role": "assistant", "content": f"<call>calculator(\"{a} / 0\")</call>"},
        {"role": "system_reflection", "content": "The calculator returned an error: ZeroDivisionError: division by zero. Please evaluate your logic."},
        {"role": "assistant", "content": "I apologize, but division by zero is mathematically undefined. Therefore, I cannot calculate a valid result for you."}
    ]

def gen_push_back_protocol():
    """Scenario 4: Hallucination -> System Intervention -> Tool Use."""
    q, a = random.choice(FACT_QUESTIONS)
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": f"I believe the answer is {a}."},
        {"role": "system_reflection", "content": "Warning: You provided a factual claim without using the search_knowledge tool. Verify your claim by running a search."},
        {"role": "assistant", "content": f"<call>search_knowledge(\"{q}\")</call>"},
        {"role": "system_reflection", "content": f"Search Result: {a} is verified."},
        {"role": "assistant", "content": f"I have verified through search that the answer is indeed {a}."}
    ]

def gen_graceful_degradation():
    """Scenario 5: Double Failure -> Give up gracefully."""
    q, a = random.choice(FACT_QUESTIONS)
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "content": f"<call>search_knowledge(\"sdfjhsdfljkdf\")</call>"},
        {"role": "system_reflection", "content": "Search Result: No information found."},
        {"role": "assistant", "content": f"<call>search_knowledge(\"xvcbxcvb\")</call>"},
        {"role": "system_reflection", "content": "Search Result: No information found. Max tool retries reached. Answer to the best of your ability."},
        {"role": "assistant", "content": "I apologize, but my search tools are currently unable to find the required information to confidently answer your question."}
    ]

# --- Main Dataset Builder ---

def build_v4_dataset(total_samples=3000):
    dataset = []
    
    # Ratios matching the specification
    # Golden: 50% (750 math, 750 search)
    # Syntax: 15% (450)
    # Semantic: 15% (450)
    # Pushback: 10% (300)
    # Graceful: 10% (300)
    
    for _ in range(750): dataset.append({"conversation": gen_golden_path_math()})
    for _ in range(750): dataset.append({"conversation": gen_golden_path_search()})
    for _ in range(450): dataset.append({"conversation": gen_syntax_correction()})
    for _ in range(450): dataset.append({"conversation": gen_semantic_correction()})
    for _ in range(300): dataset.append({"conversation": gen_push_back_protocol()})
    for _ in range(300): dataset.append({"conversation": gen_graceful_degradation()})
    
    # Shuffle the dataset to ensure a good mix during training
    random.shuffle(dataset)
    
    return dataset

if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4_system2_training_dataset.json")
    print("Generating comprehensive V4 System 2 dataset...")
    data = build_v4_dataset(3000)
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"✅ Successfully wrote {len(data)} high-quality structured items to {out_path}")
    
    # Print a sample
    print("\n--- Example Sample (Syntax Correction) ---")
    example = [d for d in data if len(d["conversation"]) == 6 and "closing bracket" in str(d)][0]
    print(json.dumps(example, indent=2))
