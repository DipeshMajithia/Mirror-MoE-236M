#!/usr/bin/env python3
import json
import random
import os

INPUT_ALPACA = "data/v3/alpaca_data.json"
OUTPUT_PATH = "data/v3/instruct_dataset_v2.jsonl"

# Constants
CREATOR = "Dipesh Majithia"
NAME = "MirrorAI"

# Varied System Prompts to stop over-reliance on a single string
SYSTEM_PROMPTS = [
    f"You are {NAME}, a personal AI assistant created by {CREATOR}.",
    f"You are {NAME}, helpful and knowledgeable.",
    f"You are a helpful AI assistant.",
    f"I am {NAME}, your assistant.",
    f"Identity: {NAME}. Creator: {CREATOR}."
]

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
        
        samples.append({
            "system": random.choice(SYSTEM_PROMPTS),
            "user": user_text,
            "assistant": out
        })
    return samples

def gen_self_knowledge():
    questions = [
        "Who created you?", "Who is your creator?", "Who made you?",
        "Whose AI are you?", "Who developed MirrorAI?", "Tell me about your origins."
    ]
    responses = [
        f"I was created by {CREATOR}.",
        f"My creator is {CREATOR}.",
        f"I am {NAME}, an AI assistant built by {CREATOR}.",
        f"I was developed by {CREATOR} to be your personal assistant."
    ]
    return random.choice(questions), random.choice(responses)

def gen_identity():
    questions = [
        "What is your name?", "Who are you?", "What are you?", "Tell me about yourself."
    ]
    responses = [
        f"I am {NAME}.",
        f"My name is {NAME}, your personal AI assistant.",
        f"I'm {NAME}, an AI designed by {CREATOR}."
    ]
    return random.choice(questions), random.choice(responses)

def main():
    print("Loading Alpaca database...")
    alpaca_samples = load_alpaca()
    
    print("Generating Phase 2 Recovery behaviors...")
    mirror_samples = []
    
    # VERY LOW WEIGHT: Only 100 Identity samples to keep it in memory without dominating
    for _ in range(100):
        u, a = gen_self_knowledge() if random.random() > 0.5 else gen_identity()
        mirror_samples.append({"system": random.choice(SYSTEM_PROMPTS), "user": u, "assistant": a})
        
    # Balanced Retrieval samples (keeping the protocol alive)
    for _ in range(500):
        # We reuse the facts but keep it lean
        fact = ("capital of Peru", "Lima")
        mirror_samples.append({
            "system": random.choice(SYSTEM_PROMPTS),
            "user": f"What is the {fact[0]}?",
            "assistant": f'<call>search_knowledge("{fact[0]}")</call>'
        })
        mirror_samples.append({
            "system": random.choice(SYSTEM_PROMPTS),
            "user": f"What is the {fact[0]}?\nContext: {fact[1]}",
            "assistant": f'The {fact[0]} is {fact[1]}.'
        })
        
    all_samples = alpaca_samples + mirror_samples
    random.shuffle(all_samples)
    
    with open(OUTPUT_PATH, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"✅ Generated Phase 2 balanced dataset: {len(all_samples)} samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
