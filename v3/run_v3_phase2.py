#!/usr/bin/env python3
import os
import sys
import math
import time
import json
import argparse
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model.transformer import MirrorTransformer, ModelArgs

# Config for Phase 2 Polishing
V3_CONFIG = {
    "init_from": "out/v3/ckpt_19000.safetensors",
    "save_dir": "out/v3_phase2",
    "output_weights": "mirror_ai_v3_final.safetensors",
    "lr": 1e-5, # Very low LR for polishing
    "warmup_steps": 100,
    "total_steps": 5000,
    "batch_size": 1,
    "block_size": 512,
    "grad_accum": 32,
    "aux_loss_weight": 0.05,
    "save_every": 1000,
}

class V3DataLoader:
    def __init__(self, data_path, labels_path, batch_size, block_size):
        self.data = np.fromfile(data_path, dtype=np.int32)
        self.labels = np.fromfile(labels_path, dtype=np.int32)
        self.batch_size = batch_size
        self.block_size = block_size
        
    def get_batch(self):
        n = len(self.data) - self.block_size
        ix = np.random.randint(0, n, (self.batch_size,))
        x = np.stack([self.data[i:i+self.block_size] for i in ix])
        y = np.stack([self.labels[i:i+self.block_size] for i in ix])
        return mx.array(x), mx.array(y)

def loss_fn(model, x, y, aux_weight):
    logits, all_gate_probs, _ = model(x)
    logits = logits[:, :-1, :]
    y = y[:, 1:]
    mask = (y != -100)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    y_flat = y.reshape(-1)
    mask_flat = mask.reshape(-1)
    ce_loss = nn.losses.cross_entropy(logits_flat, y_flat, reduction='none')
    ce_loss = ce_loss * mask_flat.astype(ce_loss.dtype)
    n_active = mx.sum(mask_flat)
    ce_loss = mx.sum(ce_loss) / (n_active + 1e-6)
    
    num_experts = model.args.num_experts
    aux_loss = mx.array(0.0)
    for gate_probs in all_gate_probs:
        density = mx.mean(gate_probs, axis=(0, 1))
        aux_loss += mx.sum(density**2) * num_experts
    return ce_loss + aux_weight * aux_loss

def train():
    os.makedirs(V3_CONFIG["save_dir"], exist_ok=True)
    args = ModelArgs(dim=512, hidden_dim=1365, n_layers=8, vocab_size=32002, use_moe=True, num_experts=16, num_experts_per_tok=2, shared_expert_dim=1365)
    model = MirrorTransformer(args)
    print(f"🔄 Polishing: Resuming from {V3_CONFIG['init_from']}")
    model.load_weights(V3_CONFIG["init_from"])
    
    optimizer = optim.AdamW(learning_rate=V3_CONFIG["lr"])
    train_loader = V3DataLoader("data/v3/instruct_v2_train.bin", "data/v3/instruct_v2_train_labels.bin", V3_CONFIG["batch_size"], V3_CONFIG["block_size"])
    
    print(f"🚀 Phase 2 Polishing (1e-5 LR)...")
    start_time = time.time()
    accum_grads = None
    
    for i in range(1, V3_CONFIG["total_steps"] + 1):
        x, y = train_loader.get_batch()
        loss_val, grads = mx.value_and_grad(loss_fn)(model, x, y, V3_CONFIG["aux_loss_weight"])
        if accum_grads is None: accum_grads = grads
        else: accum_grads = utils.tree_map(lambda a, b: a + b, accum_grads, grads)
            
        if i % V3_CONFIG["grad_accum"] == 0:
            avg_grads = utils.tree_map(lambda x: x / V3_CONFIG["grad_accum"], accum_grads)
            optimizer.update(model, avg_grads)
            mx.eval(model.state, optimizer.state, loss_val)
            accum_grads = None 
        else: mx.eval(accum_grads, loss_val)
        
        if i % 100 == 0:
            print(f"Polishing Step {i:4d}/{V3_CONFIG['total_steps']} | Loss: {loss_val.item():.4f} | Time: {time.time()-start_time:.1f}s")
            sys.stdout.flush()
            
        if i % V3_CONFIG["save_every"] == 0:
            ckpt_path = os.path.join(V3_CONFIG["save_dir"], f"polish_{i}.safetensors")
            model.save_weights(ckpt_path)
            print(f"💾 Saved {ckpt_path}")
            sys.stdout.flush()

    model.save_weights(V3_CONFIG["output_weights"])
    print(f"✅ V3 Final Polishing complete!")

if __name__ == "__main__":
    train()
