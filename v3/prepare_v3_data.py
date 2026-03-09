#!/usr/bin/env python3
"""
V3 Data Tokenizer — Per-Sample Format
Each sample is independently tokenized and stored with its length.
Format: [num_samples(int32)] + for each sample: [length(int32), input_ids(int32*N), labels(int32*N)]
"""
import os, sys, json, struct
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.getcwd(), 'v2'))
from tokenizer_wrapper import MirrorTokenizer

EOS_ID = 3
MASK_ID = -100
MAX_SEQ_LEN = 512

def tokenize_v3(tokenizer, input_file, output_path):
    print(f"  Tokenizing V3 instruct data (per-sample format, max {MAX_SEQ_LEN} tokens)")
    
    samples = []
    
    with open(input_file) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            
            # --- V4 Multi-turn Support ---
            if "conversation" in row:
                sys_text = "You are MirrorAI, a personal AI assistant created by Dipesh Majithia.\nYou are helpful, friendly, and knowledgeable.\nWhen you need factual information, use: <call>search_knowledge(\"query\")</call>\nWhen you need to perform math calculations, use: <call>calculator(\"expression\")</call>\nIf the user's request is conversational or personal (about you or Dipesh Majithia), answer directly."
                
                # Start with System prompt (masked)
                input_ids = tokenizer.encode(f"System: {sys_text}\n")
                labels = [MASK_ID] * len(input_ids)
                
                for msg in row["conversation"]:
                    role = msg["role"]
                    content = msg["content"]
                    
                    if role == "user":
                        p_toks = tokenizer.encode(f"User: {content}\n")
                        input_ids.extend(p_toks)
                        labels.extend([MASK_ID] * len(p_toks))
                    elif role == "system_reflection":
                        # We don't train the model to output the reflection itself, it's injected 
                        p_toks = tokenizer.encode(f"System (Reflection): {content}\n")
                        input_ids.extend(p_toks)
                        labels.extend([MASK_ID] * len(p_toks))
                    elif role == "assistant" or role == "assistant_reflection":
                        a_toks = tokenizer.encode(f"Assistant: {content}")
                        a_toks.append(EOS_ID) # End of turn
                        input_ids.extend(a_toks)
                        labels.extend(a_toks) # We DO train on the assistant's response
                        
            # --- V3 Single-turn Support (Legacy Compat) ---
            else:
                sys_text = row.get("system", "")
                user_text = row.get("user", "")
                asst_text = row.get("assistant", "")
                
                prefix_text = f"System: {sys_text}\nUser: {user_text}\nAssistant: "
                target_text = asst_text
                
                prefix_tokens = tokenizer.encode(prefix_text)
                target_tokens = tokenizer.encode(target_text)
                target_tokens.append(EOS_ID)
                
                input_ids = prefix_tokens + target_tokens
                labels = [MASK_ID] * len(prefix_tokens) + target_tokens
            
            # Truncate to MAX_SEQ_LEN
            if len(input_ids) > MAX_SEQ_LEN:
                input_ids = input_ids[:MAX_SEQ_LEN]
                labels = labels[:MAX_SEQ_LEN]
            
            samples.append((input_ids, labels))
            
            if (i + 1) % 10000 == 0:
                print(f"    ... {i+1} samples")
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    samples = [samples[i] for i in indices]
    
    # Split 95/5
    n_train = int(len(samples) * 0.95)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    def save_samples(sample_list, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('I', len(sample_list)))
            for ids, lbls in sample_list:
                length = len(ids)
                f.write(struct.pack('I', length))
                f.write(np.array(ids, dtype=np.int32).tobytes())
                f.write(np.array(lbls, dtype=np.int32).tobytes())

    train_path = f"{output_path}_train.bin"
    val_path = f"{output_path}_val.bin"
    save_samples(train_samples, train_path)
    save_samples(val_samples, val_path)
    
    total_tokens = sum(len(ids) for ids, _ in samples)
    print(f"  Total samples: {len(samples)} ({total_tokens/1e6:.1f}M tokens)")
    print(f"  Train: {len(train_samples)} → {train_path}")
    print(f"  Val:   {len(val_samples)} → {val_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/v3/instruct")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer = MirrorTokenizer()
    tokenize_v3(tokenizer, args.input, args.output)

if __name__ == "__main__":
    main()
