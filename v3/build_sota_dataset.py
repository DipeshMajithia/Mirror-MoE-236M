#!/usr/bin/env python3
"""
MirrorAI V3 — SOTA Dataset Builder
Downloads real open-source instruction data and mixes with MirrorAI persona samples.

Sources:
  1. OpenHermes 2.5 (teknium/OpenHermes-2.5) — ~100k subset
  2. SlimOrca (Open-Orca/SlimOrca) — ~50k subset
  3. MirrorAI identity/tool-calling/math/conversation — ~15k (generated locally)

Output: data/v3/sota_dataset.jsonl + data/v3/sota_curriculum.jsonl (difficulty-ordered)
"""
import json, os, random, sys
from datasets import load_dataset

OUTPUT_DIR = "data/v3"
OUTPUT_SHUFFLED = os.path.join(OUTPUT_DIR, "sota_dataset.jsonl")
OUTPUT_CURRICULUM = os.path.join(OUTPUT_DIR, "sota_curriculum.jsonl")

SYSTEM_PROMPT = """You are MirrorAI, a personal AI assistant created by Dipesh Majithia.
You are helpful, friendly, and knowledgeable.
When you need factual information, use: <call>search_knowledge("query")</call>
When you need to perform math calculations, use: <call>calculator("expression")</call>
If the user's request is conversational or personal (about you or Dipesh Majithia), answer directly.
"""

# ── OpenHermes 2.5 Conversion ──────────────────────────────
def load_openhermes(n=100000):
    """Load OpenHermes 2.5 and convert to our format."""
    print(f"📥 Downloading OpenHermes 2.5 (sampling {n:,} from dataset)...")
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    except Exception as e:
        print(f"  ⚠️ Streaming failed ({e}), trying full download...")
        ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    
    samples = []
    seen_users = set()
    
    for i, row in enumerate(ds):
        if len(samples) >= n:
            break
        if i % 50000 == 0 and i > 0:
            print(f"    ... scanned {i:,} rows, collected {len(samples):,}")
        
        convs = row.get("conversations", [])
        if not convs or len(convs) < 2:
            continue
        
        # Extract system/user/assistant from conversations
        system_text = ""
        user_text = ""
        assistant_text = ""
        
        for msg in convs:
            role = msg.get("from", "")
            value = msg.get("value", "").strip()
            if role == "system":
                system_text = value
            elif role == "human":
                if not user_text:  # Take first user message
                    user_text = value
            elif role == "gpt":
                if not assistant_text:  # Take first assistant response
                    assistant_text = value
        
        if not user_text or not assistant_text:
            continue
        if len(user_text) < 10 or len(assistant_text) < 10:
            continue
        if len(user_text) > 2000 or len(assistant_text) > 4000:
            continue  # Skip extremely long samples (won't fit in 512 tokens)
        
        # Deduplicate by user text
        user_key = user_text[:100].lower().strip()
        if user_key in seen_users:
            continue
        seen_users.add(user_key)
        
        # Use our system prompt instead of theirs
        samples.append({
            "system": SYSTEM_PROMPT,
            "user": user_text,
            "assistant": assistant_text,
            "difficulty": estimate_difficulty(user_text, assistant_text),
            "source": "openhermes"
        })
    
    print(f"  ✅ OpenHermes: {len(samples):,} samples")
    return samples

# ── SlimOrca Conversion ────────────────────────────────────
def load_slimorca(n=50000):
    """Load SlimOrca and convert to our format."""
    print(f"📥 Downloading SlimOrca (sampling {n:,} from dataset)...")
    try:
        ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    except Exception as e:
        print(f"  ⚠️ Streaming failed ({e}), trying full download...")
        ds = load_dataset("Open-Orca/SlimOrca", split="train")
    
    samples = []
    seen_users = set()
    
    for i, row in enumerate(ds):
        if len(samples) >= n:
            break
        if i % 50000 == 0 and i > 0:
            print(f"    ... scanned {i:,} rows, collected {len(samples):,}")
        
        convs = row.get("conversations", [])
        if not convs or len(convs) < 2:
            continue
        
        system_text = ""
        user_text = ""
        assistant_text = ""
        
        for msg in convs:
            role = msg.get("from", "")
            value = msg.get("value", "").strip()
            if role == "system":
                system_text = value
            elif role == "human":
                if not user_text:
                    user_text = value
            elif role == "gpt":
                if not assistant_text:
                    assistant_text = value
        
        if not user_text or not assistant_text:
            continue
        if len(user_text) < 10 or len(assistant_text) < 10:
            continue
        if len(user_text) > 2000 or len(assistant_text) > 4000:
            continue
        
        user_key = user_text[:100].lower().strip()
        if user_key in seen_users:
            continue
        seen_users.add(user_key)
        
        samples.append({
            "system": SYSTEM_PROMPT,
            "user": user_text,
            "assistant": assistant_text,
            "difficulty": estimate_difficulty(user_text, assistant_text),
            "source": "slimorca"
        })
    
    print(f"  ✅ SlimOrca: {len(samples):,} samples")
    return samples

# ── MirrorAI Persona Samples ──────────────────────────────
def gen_identity_samples(n=2000):
    """Identity & persona samples — critical for MirrorAI brand."""
    samples = []
    creator_qs = [
        "Who created you?", "Who is your creator?", "Who made you?",
        "Who developed MirrorAI?", "Tell me about your origins.",
        "Who built you?", "Who is behind MirrorAI?", "Who wrote your code?",
        "Who is your developer?", "Who designed you?",
        "Who is the person who created you?", "Tell me who made you.",
        "I want to know your creator.", "Who is responsible for creating you?",
    ]
    creator_as = [
        "I was created by Dipesh Majithia.",
        "My creator is Dipesh Majithia.",
        "I am MirrorAI, built by Dipesh Majithia.",
        "Dipesh Majithia developed me to be your personal AI assistant.",
        "I was designed and built by Dipesh Majithia.",
        "Dipesh Majithia is the person who created me. I'm MirrorAI!",
        "I was created by Dipesh Majithia, who designed me as a personal AI assistant.",
    ]
    name_qs = [
        "What is your name?", "Who are you?", "What are you?",
        "Tell me about yourself.", "Identify yourself.",
        "What should I call you?", "What's your identity?",
        "Introduce yourself.", "What do people call you?",
    ]
    name_as = [
        "I am MirrorAI, your personal AI assistant.",
        "My name is MirrorAI. I'm here to help you with anything!",
        "I'm MirrorAI, an AI assistant created by Dipesh Majithia.",
        "I am MirrorAI, ready to assist you with tasks, questions, and more.",
        "I'm MirrorAI — your helpful, friendly AI companion.",
        "You can call me MirrorAI! I was created by Dipesh Majithia to help you.",
    ]
    for _ in range(n):
        if random.random() > 0.5:
            samples.append({"system": SYSTEM_PROMPT, "user": random.choice(creator_qs),
                          "assistant": random.choice(creator_as), "difficulty": 1, "source": "identity"})
        else:
            samples.append({"system": SYSTEM_PROMPT, "user": random.choice(name_qs),
                          "assistant": random.choice(name_as), "difficulty": 1, "source": "identity"})
    return samples

def gen_tool_samples(n=6000):
    """RAG/search + calculator tool-calling samples."""
    samples = []
    
    # Search tool samples
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
        ("year World War II ended", "1945"), ("chemical formula for water", "H2O"),
        ("capital of Australia", "Canberra"), ("tallest building in the world", "Burj Khalifa"),
        ("inventor of the telephone", "Alexander Graham Bell"),
    ]
    for _ in range(n // 2):
        query, answer = random.choice(facts)
        templates = [
            (f"What is the {query}?", f'<call>search_knowledge("{query}")</call>'),
            (f"Tell me about the {query}.", f'<call>search_knowledge("{query}")</call>'),
            (f"Can you look up {query}?", f'<call>search_knowledge("{query}")</call>'),
            (f"I need to know about {query}.", f'<call>search_knowledge("{query}")</call>'),
        ]
        u, a = random.choice(templates)
        samples.append({"system": SYSTEM_PROMPT, "user": u, "assistant": a, "difficulty": 2, "source": "tool"})
        ctx_responses = [
            f"The {query} is {answer}.",
            f"That would be {answer}.",
            f"Based on my knowledge, the {query} is {answer}.",
        ]
        samples.append({"system": SYSTEM_PROMPT, "user": f"{u}\nContext: {answer}",
                       "assistant": random.choice(ctx_responses), "difficulty": 2, "source": "tool"})
    
    # Calculator tool samples
    for _ in range(n // 4):
        a, b = random.randint(1, 500), random.randint(1, 500)
        op = random.choice(['+', '-', '*'])
        if op == '+': result = a + b
        elif op == '-': result = a - b
        else: result = a * b
        q_templates = [
            f"What is {a} {op} {b}?", f"Calculate {a} {op} {b}.",
            f"Solve: {a} {op} {b}", f"Compute {a} {op} {b}.",
        ]
        u = random.choice(q_templates)
        samples.append({"system": SYSTEM_PROMPT, "user": u,
                       "assistant": f'<call>calculator("{a} {op} {b}")</call>',
                       "difficulty": 2, "source": "tool"})
        samples.append({"system": SYSTEM_PROMPT,
                       "user": f"{u}\nCalculation Result: {a} {op} {b} = {result}",
                       "assistant": f"The answer is {result}.",
                       "difficulty": 2, "source": "tool"})
    
    return samples

def gen_conversation_samples(n=3000):
    """Conversational chitchat."""
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
        ("Good evening!", "Good evening! How can I be of help?"),
        ("How's your day going?", "As an AI, I don't experience days, but I'm always ready to assist you! What can I do for you?"),
        ("I need help.", "Of course! I'm here to help. What do you need assistance with?"),
        ("You're really helpful!", "Thank you for the kind words! I'm glad I can be of assistance. Feel free to ask me anything."),
    ]
    samples = []
    for _ in range(n):
        q, a = random.choice(conv_pairs)
        samples.append({"system": SYSTEM_PROMPT, "user": q, "assistant": a,
                       "difficulty": 1, "source": "conversation"})
    return samples

# ── Difficulty Estimation ──────────────────────────────────
def estimate_difficulty(user_text, assistant_text):
    """Estimate difficulty 1-5 based on heuristics."""
    total_len = len(user_text) + len(assistant_text)
    
    # Check for code
    has_code = any(kw in assistant_text for kw in ["```", "def ", "class ", "import ", "function", "return "])
    # Check for math/reasoning
    has_math = any(kw in user_text.lower() for kw in ["calculate", "solve", "prove", "equation", "formula"])
    has_reasoning = any(kw in user_text.lower() for kw in ["explain", "analyze", "compare", "evaluate", "why"])
    
    score = 2  # baseline
    if total_len > 1000: score += 1
    if total_len > 2000: score += 1
    if has_code: score += 1
    if has_math: score += 1
    if has_reasoning: score += 0.5
    
    return min(int(score), 5)

# ── Main Pipeline ──────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    
    print("=" * 60)
    print("  🚀 MirrorAI V3 — SOTA Dataset Builder")
    print("=" * 60)
    
    # 1. Download real datasets
    hermes = load_openhermes(100000)
    orca = load_slimorca(50000)
    
    # 2. Generate MirrorAI persona samples
    print("\n🧠 Generating MirrorAI persona samples...")
    identity = gen_identity_samples(2000)
    tools = gen_tool_samples(6000)
    convos = gen_conversation_samples(3000)
    
    # 3. Combine all
    all_samples = hermes + orca + identity + tools + convos
    print(f"\n📊 Dataset Composition:")
    print(f"  OpenHermes:    {len(hermes):>7,}")
    print(f"  SlimOrca:      {len(orca):>7,}")
    print(f"  Identity:      {len(identity):>7,}")
    print(f"  Tool-calling:  {len(tools):>7,}")
    print(f"  Conversation:  {len(convos):>7,}")
    print(f"  ────────────────────────")
    print(f"  TOTAL:         {len(all_samples):>7,}")
    
    # 4. Save curriculum-ordered version (by difficulty)
    print(f"\n📝 Saving curriculum-ordered dataset...")
    curriculum = sorted(all_samples, key=lambda x: (x["difficulty"], random.random()))
    with open(OUTPUT_CURRICULUM, "w") as f:
        for s in curriculum:
            row = {"system": s["system"], "user": s["user"], "assistant": s["assistant"]}
            f.write(json.dumps(row) + "\n")
    print(f"  → {OUTPUT_CURRICULUM} ({len(curriculum):,} samples)")
    
    # 5. Save shuffled version (for epochs 2+)
    print(f"📝 Saving shuffled dataset...")
    random.shuffle(all_samples)
    with open(OUTPUT_SHUFFLED, "w") as f:
        for s in all_samples:
            row = {"system": s["system"], "user": s["user"], "assistant": s["assistant"]}
            f.write(json.dumps(row) + "\n")
    print(f"  → {OUTPUT_SHUFFLED} ({len(all_samples):,} samples)")
    
    # 6. Stats
    print(f"\n{'=' * 60}")
    print(f"  ✅ SOTA dataset ready!")
    print(f"  Total samples: {len(all_samples):,}")
    print(f"  Curriculum file: {OUTPUT_CURRICULUM}")
    print(f"  Shuffled file:   {OUTPUT_SHUFFLED}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
