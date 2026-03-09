import json
import random
import os

def generate_calculator_prompts(num=35):
    prompts = []
    ops = [('+', 'plus'), ('-', 'minus'), ('*', 'times', 'multiplied by', 'x'), ('/', 'divided by')]
    for _ in range(num):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        op = random.choice(ops)
        symbol = op[0]
        word = random.choice(op[1:])
        
        templates = [
            f"What is {a} {symbol} {b}?",
            f"Calculate {a} {word} {b}.",
            f"Can you tell me the result of {a} {symbol} {b}?",
            f"Please compute {a} {word} {b}.",
            f"{a} {symbol} {b} = ?"
        ]
        prompts.append({
            "prompt": random.choice(templates),
            "expected_tool": "calculator",
            "expected_query_content": f"{a}", # At minimum it should contain the numbers
            "category": "calculator"
        })
    return prompts

def generate_search_prompts(num=35):
    entities = [
        "the president of France", "Albert Einstein", "the Eiffel Tower", "SpaceX", 
        "black holes", "the capital of Japan", "Mount Everest", "Leonardo da Vinci",
        "the speed of light", "quantum computing", "Julius Caesar", "the Mariana Trench",
        "the history of Rome", "the Great Wall of China", "the Mona Lisa", "Isaac Newton",
        "the Apollo 11 mission", "the human genome", "the largest ocean", "the smallest planet",
        "the current weather in Tokyo", "the population of New York", "the inventor of the telephone",
        "the tallest building in the world", "the longest river", "the fastest animal",
        "the meaning of life", "the theory of relativity", "the structure of DNA", "the Big Bang theory"
    ]
    prompts = []
    for _ in range(num):
        entity = random.choice(entities)
        templates = [
            f"Who is {entity}?" if "who" in entity.lower() or "inventor" in entity.lower() or "president" in entity.lower() else f"What is {entity}?",
            f"Search for {entity}.",
            f"Find information about {entity}.",
            f"Can you look up {entity}?",
            f"Tell me about {entity}."
        ]
        prompts.append({
            "prompt": random.choice(templates),
            "expected_tool": "search_knowledge",
            "expected_query_content": entity.replace("the ", ""), # Simple heuristic
            "category": "search"
        })
    return prompts

def generate_general_prompts(num=30):
    # Mix of both, phrased differently or ambiguously
    prompts = []
    for _ in range(num):
        if random.random() > 0.5:
            # ambiguous math
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            prompts.append({
                "prompt": f"I have {a} apples and someone gives me {b} more. How many do I have total?",
                "expected_tool": "calculator",
                "expected_query_content": f"{a}",
                "category": "general_math"
            })
        else:
            # ambiguous search
            topics = ["recent discoveries in Mars", "how to bake a cake", "symptoms of a cold", "best movies of 2023"]
            topic = random.choice(topics)
            prompts.append({
                "prompt": f"I'm curious about {topic}. Do you know anything?",
                "expected_tool": "search_knowledge",
                "expected_query_content": topic.split()[-1],
                "category": "general_search"
            })
    return prompts

if __name__ == "__main__":
    dataset = generate_calculator_prompts(35) + generate_search_prompts(35) + generate_general_prompts(30)
    random.shuffle(dataset)
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tool_test_dataset.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Generated {len(dataset)} prompts in {out_path}")
