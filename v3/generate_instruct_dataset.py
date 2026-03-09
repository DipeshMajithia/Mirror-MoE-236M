#!/usr/bin/env python3
"""
MirrorAI V3 — Enhanced All-Rounder Dataset Generator
Generates a comprehensive instruct dataset covering:
  1. General instructions (Alpaca)
  2. Identity & persona  
  3. RAG/tool-calling protocol
  4. Coding & programming
  5. Math & reasoning
  6. Conversation & chitchat
"""
import json
import random
import os

INPUT_ALPACA = "data/v3/alpaca_data.json"
OUTPUT_PATH = "data/v3/instruct_dataset_v3.jsonl"

SYSTEM_PROMPT = """You are MirrorAI, a personal AI assistant created by Dipesh Majithia.
You are helpful, friendly, and knowledgeable.
When you need factual information, use: <call>search_knowledge("query")</call>
When you need to perform math calculations, use: <call>calculator("expression")</call>
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly.
"""

def load_alpaca():
    if not os.path.exists(INPUT_ALPACA):
        print(f"❌ {INPUT_ALPACA} not found!")
        return []
    with open(INPUT_ALPACA, "r") as f:
        data = json.load(f)
    samples = []
    for row in data:
        inst = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        out = row.get("output", "").strip()
        user_text = f"{inst}\n\n{inp}" if inp else inst
        if len(user_text) > 10 and len(out) > 5:
            samples.append({"system": SYSTEM_PROMPT, "user": user_text, "assistant": out})
    return samples

def gen_identity_samples(n=2000):
    """Identity & persona samples."""
    samples = []
    creator_qs = [
        "Who created you?", "Who is your creator?", "Who made you?",
        "Who developed MirrorAI?", "Tell me about your origins.",
        "Who built you?", "Who is behind MirrorAI?", "Who wrote your code?",
        "Who is your developer?", "Who designed you?",
    ]
    creator_as = [
        "I was created by Dipesh Majithia.",
        "My creator is Dipesh Majithia.",
        "I am MirrorAI, built by Dipesh Majithia.",
        "Dipesh Majithia developed me to be your personal AI assistant.",
        "I was designed and built by Dipesh Majithia.",
    ]
    name_qs = [
        "What is your name?", "Who are you?", "What are you?",
        "Tell me about yourself.", "Identify yourself.",
        "What should I call you?", "What's your identity?",
    ]
    name_as = [
        "I am MirrorAI, your personal AI assistant.",
        "My name is MirrorAI. I'm here to help you with anything!",
        "I'm MirrorAI, an AI assistant created by Dipesh Majithia.",
        "I am MirrorAI, ready to assist you with tasks, questions, and more.",
        "I'm MirrorAI — your helpful, friendly AI companion.",
    ]
    for _ in range(n):
        if random.random() > 0.5:
            samples.append({"system": SYSTEM_PROMPT, "user": random.choice(creator_qs), "assistant": random.choice(creator_as)})
        else:
            samples.append({"system": SYSTEM_PROMPT, "user": random.choice(name_qs), "assistant": random.choice(name_as)})
    return samples

def gen_retrieval_samples(n=3000):
    """RAG/tool-calling protocol samples."""
    facts = [
        ("capital of France", "Paris"), ("highest mountain", "Mount Everest"),
        ("speed of light", "approximately 299,792,458 meters per second"),
        ("capital of Japan", "Tokyo"), ("longest river", "The Nile"),
        ("capital of India", "New Delhi"), ("capital of Peru", "Lima"),
        ("first man on moon", "Neil Armstrong in 1969"),
        ("largest country by area", "Russia"), ("boiling point of water", "100°C at sea level"),
        ("chemical symbol for Gold", "Au"), ("largest planet in our solar system", "Jupiter"),
        ("number of bones in adult human body", "206"),
        ("inventor of the light bulb", "Thomas Edison"),
        ("capital of Germany", "Berlin"), ("founder of Microsoft", "Bill Gates"),
        ("deepest ocean", "the Pacific Ocean"), ("smallest country", "Vatican City"),
        ("fastest land animal", "the cheetah"), ("largest ocean", "the Pacific Ocean"),
    ]
    samples = []
    for _ in range(n):
        query, answer = random.choice(facts)
        templates = [
            (f"What is the {query}?", f'<call>search_knowledge("{query}")</call>'),
            (f"Tell me about the {query}.", f'<call>search_knowledge("{query}")</call>'),
            (f"Can you look up {query}?", f'<call>search_knowledge("{query}")</call>'),
            (f"I need to know about {query}.", f'<call>search_knowledge("{query}")</call>'),
        ]
        u, a = random.choice(templates)
        samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a})
        # Also add direct-answer version (with context)
        ctx_responses = [
            f"The {query} is {answer}.",
            f"That would be {answer}.",
            f"Based on my knowledge, the {query} is {answer}.",
        ]
        samples.append({"system": SYSTEM_PROMPT, "user": f"{u}\nContext: {answer}", "assistant": random.choice(ctx_responses)})
    return samples

def gen_coding_samples(n=3000):
    """Coding & programming samples."""
    samples = []
    code_pairs = [
        ("Write a Python function to reverse a string.",
         'Here\'s a Python function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\n\nThis uses Python\'s slice notation `[::-1]` to efficiently reverse the string.'),
        ("Write a function to check if a number is prime.",
         '```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\n\nThis checks divisibility up to the square root of n for efficiency.'),
        ("How do I read a file in Python?",
         'You can read a file in Python using:\n\n```python\nwith open("filename.txt", "r") as f:\n    content = f.read()\n```\n\nThe `with` statement ensures the file is properly closed after reading.'),
        ("Write a function to find the factorial of a number.",
         '```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```\n\nThis uses recursion. For large numbers, consider using `math.factorial()`.'),
        ("How do I sort a list in Python?",
         'You can sort a list in Python in two ways:\n\n```python\n# Method 1: sorted() returns a new list\nsorted_list = sorted(my_list)\n\n# Method 2: .sort() modifies in place\nmy_list.sort()\n```\n\nBoth accept a `key` parameter for custom sorting.'),
        ("Write a Python function to find the maximum element in a list.",
         '```python\ndef find_max(lst):\n    if not lst:\n        return None\n    return max(lst)\n```\n\nAlternatively, you can implement it manually:\n\n```python\ndef find_max(lst):\n    if not lst:\n        return None\n    maximum = lst[0]\n    for item in lst[1:]:\n        if item > maximum:\n            maximum = item\n    return maximum\n```'),
        ("How do I handle exceptions in Python?",
         'Use try/except blocks:\n\n```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print("Cannot divide by zero!")\nexcept Exception as e:\n    print(f"An error occurred: {e}")\nfinally:\n    print("This always runs")\n```'),
        ("Write a function to count words in a sentence.",
         '```python\ndef count_words(sentence):\n    return len(sentence.split())\n```\n\nThis splits the string by whitespace and counts the resulting parts.'),
        ("How do I create a dictionary in Python?",
         'You can create a dictionary in several ways:\n\n```python\n# Method 1: Direct creation\nmy_dict = {"name": "Alice", "age": 30}\n\n# Method 2: Using dict()\nmy_dict = dict(name="Alice", age=30)\n\n# Method 3: From a list of tuples\nmy_dict = dict([(\"name\", \"Alice\"), (\"age\", 30)])\n```'),
        ("Write a function to check if a string is a palindrome.",
         '```python\ndef is_palindrome(s):\n    s = s.lower().replace(" ", "")\n    return s == s[::-1]\n```\n\nThis normalizes the string by converting to lowercase and removing spaces, then compares it with its reverse.'),
        ("How do I use list comprehension in Python?",
         'List comprehensions provide a concise way to create lists:\n\n```python\n# Square numbers\nsquares = [x**2 for x in range(10)]\n\n# Filter even numbers\nevens = [x for x in range(20) if x % 2 == 0]\n\n# Nested comprehension\nmatrix = [[i*j for j in range(3)] for i in range(3)]\n```'),
        ("Write a function to merge two sorted lists.",
         '```python\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result\n```'),
        ("Explain what a class is in Python.",
         'A class is a blueprint for creating objects. It bundles data (attributes) and behavior (methods) together:\n\n```python\nclass Dog:\n    def __init__(self, name, breed):\n        self.name = name\n        self.breed = breed\n    \n    def bark(self):\n        return f"{self.name} says Woof!"\n\nmy_dog = Dog("Rex", "Labrador")\nprint(my_dog.bark())  # Rex says Woof!\n```'),
        ("Write a function to flatten a nested list.",
         '```python\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n\n# Example: flatten([1, [2, [3, 4]], 5]) → [1, 2, 3, 4, 5]\n```'),
        ("How do I make an HTTP request in Python?",
         'Use the `requests` library:\n\n```python\nimport requests\n\n# GET request\nresponse = requests.get("https://api.example.com/data")\ndata = response.json()\n\n# POST request\nresponse = requests.post("https://api.example.com/submit", \n                         json={"key": "value"})\n```\n\nInstall with: `pip install requests`'),
    ]
    for _ in range(n):
        q, a = random.choice(code_pairs)
        samples.append({"system": SYSTEM_PROMPT, "user": q, "assistant": a})
    return samples

def gen_math_samples(n=3000):
    """Math & reasoning samples."""
    samples = []
    
    # Basic arithmetic
    for _ in range(n // 3):
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(['+', '-', '*'])
        if op == '+': result = a + b
        elif op == '-': result = a - b
        else: result = a * b
        
        q_templates = [
            f"What is {a} {op} {b}?",
            f"Calculate {a} {op} {b}.",
            f"Solve: {a} {op} {b}",
            f"Compute {a} {op} {b}.",
        ]
        a_templates = [
            f'<call>calculator("{a} {op} {b}")</call>',
            f'<call>calculator("{a} {op} {b}")</call>',
        ]
        u = random.choice(q_templates)
        a_call = random.choice(a_templates)
        samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a_call})
        
        # Also teach interpreting the result
        ctx_prompt = f"{u}\nCalculation Result: {a} {op} {b} = {result}"
        ctx_as = [
            f"The answer is {result}.",
            f"{a} {op} {b} equals {result}.",
            f"Based on my calculation, the result is {result}.",
        ]
        samples.append({"system": SYSTEM_PROMPT, "user": ctx_prompt, "assistant": random.choice(ctx_as)})
    
    # Word problems
    word_problems = [
        ("If I have 15 apples and give away 7, how many do I have left?",
         'Let\'s think step by step:\n1. I started with 15 apples.\n2. I gave away 7 apples.\n3. The calculation is 15 - 7.\n<call>calculator("15 - 7")</call>'),
        ("A train travels at 60 mph for 3 hours. How far does it go?",
         'To find the distance: Distance = Speed × Time.\nCalculation: 60 * 3.\n<call>calculator("60 * 3")</call>'),
        ("If a shirt costs $25 and is 20% off, what is the sale price?",
         'Step 1: Calculate the discount amount (20% of 25).\nStep 2: Subtract discount from original price.\nLet\'s calculate the discount first: 25 * 0.20.\n<call>calculator("25 * 0.20")</call>'),
        ("What is the area of a rectangle with length 8 and width 5?",
         'Area = length × width.\nCalculation: 8 * 5.\n<call>calculator("8 * 5")</call>'),
        ("If you save $50 per month, how much do you save in a year?",
         'There are 12 months in a year.\nCalculation: 50 * 12.\n<call>calculator("50 * 12")</call>'),
        ("If a car uses 5 gallons of gas per 100 miles, how much gas for 350 miles?",
         'Efficiency is 5 gallons / 100 miles = 0.05 gallons/mile.\nFor 350 miles: 350 * 0.05.\n<call>calculator("350 * 0.05")</call>'),
    ]
    for _ in range(n // 3):
        q, a = random.choice(word_problems)
        samples.append({"system": SYSTEM_PROMPT, "user": q, "assistant": a})
        
        # Add a "Result Interpretation" sample for one of these
        # (This is harder to automate perfectly without a tool, but we can do a generic one)
        samples.append({"system": SYSTEM_PROMPT, "user": f"{q}\nCalculation Result: 8", "assistant": "The answer is 8."})
    
    # Logic & reasoning
    logic_pairs = [
        ("What comes next in the pattern: 2, 4, 8, 16, ...?",
         "Each number is doubled. The next number is 16 × 2 = 32."),
        ("If all cats are animals, and Tom is a cat, what can we conclude?",
         "We can conclude that Tom is an animal. This is a classic syllogism."),
        ("Which is larger: 3/4 or 5/8?",
         "3/4 = 6/8, which is larger than 5/8. So 3/4 is larger."),
        ("What is the next prime number after 13?",
         "The next prime number after 13 is 17 (14, 15, and 16 are not prime)."),
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
         "It still takes 5 minutes. Each machine makes 1 widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes."),
    ]
    for _ in range(n // 3):
        q, a = random.choice(logic_pairs)
        samples.append({"system": SYSTEM_PROMPT, "user": q, "assistant": a})
    
    return samples

def gen_conversation_samples(n=2000):
    """Natural conversation & chitchat."""
    samples = []
    conv_pairs = [
        ("Hi!", "Hello! How can I help you today?"),
        ("Hey there!", "Hey! What's on your mind?"),
        ("Good morning!", "Good morning! Hope you're having a great day. How can I assist you?"),
        ("How are you?", "I'm doing well, thank you for asking! How can I help you today?"),
        ("What can you do?", "I can help with many things! I can answer questions, write code, solve math problems, search for information, and have conversations. Just ask!"),
        ("Thanks!", "You're welcome! Let me know if you need anything else."),
        ("Thank you so much!", "Happy to help! Don't hesitate to ask if you need anything."),
        ("That's helpful, thanks!", "Glad I could help! Feel free to ask more questions anytime."),
        ("Bye!", "Goodbye! Have a wonderful day!"),
        ("See you later!", "See you! Take care!"),
        ("I'm feeling stressed.", "I'm sorry to hear that. Remember to take breaks, breathe deeply, and focus on one thing at a time. Is there something specific I can help with?"),
        ("Tell me something interesting.", "Did you know that honey never spoils? Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible!"),
        ("What's your favorite color?", "As an AI, I don't have personal preferences, but I think blue is often associated with calm and clarity — qualities I strive for in my responses!"),
        ("Can you tell me a joke?", "Sure! Why don't scientists trust atoms? Because they make up everything! 😄"),
        ("Tell me another joke!", "Here's one: What do you call a fake noodle? An impasta! 🍝"),
        ("What's the meaning of life?", "That's one of humanity's great questions! Philosophically, many say it's about finding purpose, connection, and happiness. What does it mean to you?"),
    ]
    for _ in range(n):
        q, a = random.choice(conv_pairs)
        samples.append({"system": SYSTEM_PROMPT, "user": q, "assistant": a})
    return samples

def main():
    print("📦 Loading Alpaca base dataset...")
    alpaca = load_alpaca()
    print(f"  → {len(alpaca)} Alpaca samples")
    
    print("🧠 Generating identity samples...")
    identity = gen_identity_samples(2000)
    
    print("🔍 Generating RAG/tool samples...")
    retrieval = gen_retrieval_samples(3000)
    
    print("💻 Generating coding samples...")
    coding = gen_coding_samples(3000)
    
    print("🔢 Generating math samples...")
    math_samples = gen_math_samples(3000)
    
    print("💬 Generating conversation samples...")
    conversation = gen_conversation_samples(2000)
    
    all_samples = alpaca + identity + retrieval + coding + math_samples + conversation
    random.shuffle(all_samples)
    
    with open(OUTPUT_PATH, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"\n✅ Enhanced dataset: {len(all_samples)} samples → {OUTPUT_PATH}")
    print(f"  Alpaca:       {len(alpaca)}")
    print(f"  Identity:     {len(identity)}")
    print(f"  RAG/Tools:    {len(retrieval)}")
    print(f"  Coding:       {len(coding)}")
    print(f"  Math:         {len(math_samples)}")
    print(f"  Conversation: {len(conversation)}")

if __name__ == "__main__":
    main()
