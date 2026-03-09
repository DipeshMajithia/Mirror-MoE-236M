#!/usr/bin/env python3
"""
MirrorAI V3 SOTA Training — Multi-Epoch Curriculum Learning
Research-grade: Cosine LR, gradient clipping, curriculum ordering, 3 epochs.
"""
import os, sys, math, time, struct
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model.transformer import MirrorTransformer, ModelArgs

# ── Config ──────────────────────────────────────────────────
SOTA_CONFIG = {
    # Model init
    "init_from": "mirror_ai_v3_final_pass2.safetensors",
    "save_dir": "out/v3_sota",
    "output_weights": "model.safetensors",
    
    # Data — epoch 1 uses curriculum order, epochs 2-3 use shuffled
    "train_data_curriculum": "data/v3/sota_curriculum_train.bin",
    "train_data_shuffled": "data/v3/sota_dataset_train.bin",
    "val_data": "data/v3/sota_dataset_val.bin",
    
    # Training hyperparams (tuned for multi-epoch on real data)
    "num_epochs": 3,
    "max_lr": 5e-5,       # Lower than single-epoch to avoid overfitting
    "min_lr": 1e-6,
    "warmup_steps": 2000,
    "batch_size": 1,
    "block_size": 512,
    "grad_accum": 16,
    "max_grad_norm": 1.0,
    "weight_decay": 0.05,  # Stronger regularization for multi-epoch
    "aux_loss_weight": 0.02,
    "save_every": 5000,
    "log_every": 100,
    "val_every": 2500,
    "val_batches": 100,
}

PAD_ID = 0

# ── Cosine LR Schedule ─────────────────────────────────────
def get_lr(step, warmup, total, max_lr, min_lr):
    """Cosine annealing with linear warmup."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

# ── Gradient Clipping ───────────────────────────────────────
def clip_grad_norm(grads, max_norm):
    """Clip gradient norm (research standard for stable training)."""
    flat = utils.tree_flatten(grads)
    total_norm_sq = sum(mx.sum(g[1] ** 2).item() for g in flat)
    total_norm = math.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        grads = utils.tree_map(lambda g: g * scale, grads)
    return grads, total_norm

# ── Data Loader with Epoch Support ──────────────────────────
class EpochDataLoader:
    """Per-sample data loader with epoch tracking and optional ordering."""
    
    def __init__(self, data_path, block_size, shuffle=True):
        self.block_size = block_size
        self.shuffle = shuffle
        self.samples = []
        
        with open(data_path, 'rb') as f:
            num_samples = struct.unpack('I', f.read(4))[0]
            for _ in range(num_samples):
                length = struct.unpack('I', f.read(4))[0]
                ids = np.frombuffer(f.read(length * 4), dtype=np.int32).copy()
                lbls = np.frombuffer(f.read(length * 4), dtype=np.int32).copy()
                self.samples.append((ids, lbls))
        
        self.num_samples = len(self.samples)
        self.indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(self.indices)
        self.pos = 0
        self.epoch = 0
        print(f"  📦 {self.num_samples:,} samples from {data_path}")
    
    def get_batch(self):
        if self.pos >= self.num_samples:
            self.epoch += 1
            self.pos = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            return None, None, True  # Signal epoch boundary
        
        idx = self.indices[self.pos]
        self.pos += 1
        ids, lbls = self.samples[idx]
        
        pad_len = self.block_size - len(ids)
        if pad_len > 0:
            ids = np.concatenate([ids, np.full(pad_len, PAD_ID, dtype=np.int32)])
            lbls = np.concatenate([lbls, np.full(pad_len, -100, dtype=np.int32)])
        else:
            ids = ids[:self.block_size]
            lbls = lbls[:self.block_size]
        
        return mx.array(ids[None, :]), mx.array(lbls[None, :]), False
    
    def reset(self):
        """Reset for a new epoch."""
        self.pos = 0
        self.epoch = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

# ── Loss Function ───────────────────────────────────────────
def loss_fn(model, x, y, aux_weight):
    logits, all_gate_probs, _ = model(x)
    logits = logits[:, :-1, :]
    y = y[:, 1:]
    mask = (y != -100)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    y_flat = y.reshape(-1)
    mask_flat = mask.reshape(-1)
    y_safe = mx.where(mask_flat, y_flat, mx.zeros_like(y_flat))
    ce_loss = nn.losses.cross_entropy(logits_flat, y_safe, reduction='none')
    ce_loss = ce_loss * mask_flat.astype(ce_loss.dtype)
    n_active = mx.sum(mask_flat)
    ce_loss = mx.sum(ce_loss) / (n_active + 1e-6)
    
    num_experts = model.args.num_experts
    aux_loss = mx.array(0.0)
    for gate_probs in all_gate_probs:
        density = mx.mean(gate_probs, axis=(0, 1))
        aux_loss += mx.sum(density**2) * num_experts
    return ce_loss + aux_weight * aux_loss

# ── Validation ──────────────────────────────────────────────
def validate(model, val_loader, n_batches, aux_weight):
    """Run validation and return average loss."""
    val_loss = 0
    count = 0
    for _ in range(n_batches):
        vx, vy, epoch_done = val_loader.get_batch()
        if epoch_done or vx is None:
            val_loader.reset()
            vx, vy, _ = val_loader.get_batch()
        val_loss += loss_fn(model, vx, vy, aux_weight).item()
        count += 1
    return val_loss / max(count, 1)

# ── Training Loop ───────────────────────────────────────────
def train():
    cfg = SOTA_CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)
    
    args = ModelArgs(
        dim=512, hidden_dim=1365, n_layers=8, vocab_size=32002,
        use_moe=True, num_experts=16, num_experts_per_tok=2, shared_expert_dim=1365
    )
    model = MirrorTransformer(args)
    
    init_path = cfg["init_from"]
    if init_path and os.path.exists(init_path):
        print(f"🔄 Loading foundation weights: {init_path}")
        model.load_weights(init_path, strict=False)
    else:
        print(f"❌ {init_path} not found!")
        sys.exit(1)
    
    # Load data
    print(f"\n📂 Loading training data...")
    curriculum_path = cfg["train_data_curriculum"]
    shuffled_path = cfg["train_data_shuffled"]
    
    if os.path.exists(curriculum_path):
        curriculum_loader = EpochDataLoader(curriculum_path, cfg["block_size"], shuffle=False)
    else:
        print(f"  ⚠️ Curriculum data not found, using shuffled for all epochs")
        curriculum_loader = None
    
    shuffled_loader = EpochDataLoader(shuffled_path, cfg["block_size"], shuffle=True)
    val_loader = EpochDataLoader(cfg["val_data"], cfg["block_size"], shuffle=True)
    
    # Calculate total steps
    num_samples = shuffled_loader.num_samples
    steps_per_epoch = num_samples  # batch_size=1, each sample = 1 step
    total_steps = steps_per_epoch * cfg["num_epochs"]
    optimizer_steps = total_steps // cfg["grad_accum"]
    
    optimizer = optim.AdamW(learning_rate=cfg["max_lr"], weight_decay=cfg["weight_decay"])
    
    print(f"\n{'=' * 70}")
    print(f"  🚀 MirrorAI V3 SOTA Training")
    print(f"  Samples: {num_samples:,} | Epochs: {cfg['num_epochs']} | Total Steps: {total_steps:,}")
    print(f"  Optimizer Steps: {optimizer_steps:,} | Grad Accum: {cfg['grad_accum']}")
    print(f"  Peak LR: {cfg['max_lr']} | Weight Decay: {cfg['weight_decay']}")
    print(f"  Warmup: {cfg['warmup_steps']} steps | Grad Clip: {cfg['max_grad_norm']}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    accum_grads = None
    running_loss = 0.0
    loss_count = 0
    grad_norm = 0.0
    best_val_loss = float('inf')
    current_epoch = 0
    
    # Select loader for current epoch
    def get_epoch_loader(epoch):
        if epoch == 0 and curriculum_loader is not None:
            print(f"\n📚 Epoch {epoch + 1}: Using CURRICULUM order (easy → hard)")
            curriculum_loader.reset()
            return curriculum_loader
        else:
            print(f"\n🔀 Epoch {epoch + 1}: Using SHUFFLED data")
            shuffled_loader.reset()
            return shuffled_loader
    
    active_loader = get_epoch_loader(0)
    
    for step in range(1, total_steps + 1):
        # Get batch from active loader
        x, y, epoch_done = active_loader.get_batch()
        
        if epoch_done or x is None:
            current_epoch += 1
            if current_epoch >= cfg["num_epochs"]:
                print(f"\n✅ All {cfg['num_epochs']} epochs complete at step {step}")
                break
            active_loader = get_epoch_loader(current_epoch)
            x, y, _ = active_loader.get_batch()
        
        # LR schedule over total optimizer steps
        opt_step = step // cfg["grad_accum"]
        lr = get_lr(opt_step, cfg["warmup_steps"], optimizer_steps, cfg["max_lr"], cfg["min_lr"])
        optimizer.learning_rate = lr
        
        loss_val, grads = mx.value_and_grad(loss_fn)(model, x, y, cfg["aux_loss_weight"])
        
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = utils.tree_map(lambda a, b: a + b, accum_grads, grads)
        
        if step % cfg["grad_accum"] == 0:
            avg_grads = utils.tree_map(lambda g: g / cfg["grad_accum"], accum_grads)
            avg_grads, grad_norm = clip_grad_norm(avg_grads, cfg["max_grad_norm"])
            optimizer.update(model, avg_grads)
            mx.eval(model.state, optimizer.state, loss_val)
            accum_grads = None
        else:
            mx.eval(accum_grads, loss_val)
        
        running_loss += loss_val.item()
        loss_count += 1
        
        # Logging
        if step % cfg["log_every"] == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / loss_count
            steps_per_sec = step / elapsed
            eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            print(f"E{current_epoch+1} Step {step:>7,}/{total_steps:,} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | GradNorm: {grad_norm:.2f} | {steps_per_sec:.1f} step/s | ETA: {eta/3600:.1f}h")
            sys.stdout.flush()
            running_loss = 0.0
            loss_count = 0
        
        # Validation
        if step % cfg["val_every"] == 0:
            val_loss = validate(model, val_loader, cfg["val_batches"], cfg["aux_loss_weight"])
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            print(f"  📊 Val Loss: {val_loss:.4f} {'⭐ BEST' if is_best else ''}")
            sys.stdout.flush()
        
        # Checkpointing
        if step % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["save_dir"], f"ckpt_{step}.safetensors")
            model.save_weights(ckpt_path)
            print(f"  💾 Saved: {ckpt_path}")
            
            # Rotate old checkpoints (keep last 2)
            prev_step = step - 2 * cfg["save_every"]
            if prev_step > 0:
                prev_path = os.path.join(cfg["save_dir"], f"ckpt_{prev_step}.safetensors")
                if os.path.exists(prev_path):
                    try:
                        os.remove(prev_path)
                        print(f"  🗑️ Rotated: {os.path.basename(prev_path)}")
                    except: pass
            sys.stdout.flush()
    
    # Save final weights
    final_path = os.path.join(ROOT_DIR, cfg["output_weights"])
    model.save_weights(final_path)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  ✅ Training Complete!")
    print(f"  Time: {elapsed/3600:.1f}h | Best Val Loss: {best_val_loss:.4f}")
    print(f"  Final weights: {final_path}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    train()
