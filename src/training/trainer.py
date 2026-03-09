import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
from pathlib import Path

from model.transformer import MirrorTransformer, ModelArgs
from diagnostics.metrics import ExpertTracker

def loss_fn(model, x, y, temperature, z_loss_weight, aux_loss_weight=0.01):
    logits, router_logits, aux_scores = model(x, temperature=temperature)
    # Cross entropy loss
    B, L, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    y_flat = y.reshape(-1)
    
    ce_loss = nn.losses.cross_entropy(logits_flat, y_flat, reduction='mean')
    
    # Router Z-Loss
    # L_z = 1/E * sum(log(sum(exp(x)))^2)
    z_losses = []
    for r_logits in router_logits:
        # r_logits: (B, L, E)
        log_z = mx.logsumexp(r_logits, axis=-1)
        z_loss = mx.mean(mx.square(log_z))
        z_losses.append(z_loss)
    
    total_z_loss = sum(z_losses) / len(z_losses) if z_losses else 0
    
    # Auxiliary Load Balancing Loss
    # aux_scores is a list of (E,) arrays per layer
    balancing_losses = []
    for scores in aux_scores:
        # scores is mean expert usage in the batch
        # Min variance of usage: minimize square of scores
        # Scaling by num_experts is often used: E * sum(mean_usage_per_expert^2)
        balancing_losses.append(mx.mean(mx.square(scores)) * len(scores))
        
    total_aux_loss = sum(balancing_losses) / len(balancing_losses) if balancing_losses else 0
    
    total_loss = ce_loss + z_loss_weight * total_z_loss + aux_loss_weight * total_aux_loss
    return total_loss, router_logits

class Trainer:
    def __init__(self, model: MirrorTransformer, args: dict):
        self.model = model
        self.args = args
        self.optimizer = optim.AdamW(learning_rate=args.get('lr', 1e-4))
        self.tracker = ExpertTracker(model.args.num_experts)
        self.log_interval = args.get('log_interval', 10)
        self.save_interval = args.get('save_interval', 100)
        self.output_dir = Path(args.get('output_dir', 'checkpoints'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 2/3 Args
        self.z_loss_weight = args.get('router_z_loss_weight', 0.001)
        self.aux_loss_weight = args.get('router_aux_loss_weight', 0.01)
        self.start_temp = args.get('start_temp', 2.0)
        self.end_temp = args.get('end_temp', 1.0)
        
    def train_step(self, x, y):
        # We need a compiled function for efficiency
        loss_value_and_grads = nn.value_and_grad(self.model, loss_fn)
        (loss, router_logits), grads = loss_value_and_grads(self.model, x, y)
        
        self.optimizer.update(self.model, grads)
        
        # Access value of loss
        return loss, router_logits

    def train(self, data_loader, num_steps: int):
        # Build parameter list
        if self.args.get('freeze_router', False):
            print("Trainer: Freezing Router parameters.")
            # Filter params: exclude any key containing 'router'
            # param_groups = [] # MLX optimizer takes list of arrays or dict or tree?
            # It takes the model and updates all trainable parameters in the tree.
            # We can freeze parameters by setting their 'frozen' state in MLX?
            # MLX array.freeze() exists? No.
            # We must pass Only Trainable Params to optimizer?
            # MLX optimizer.update(model, grads).
            # If grads are calculated, it updates.
            # To freeze, we must stop gradient calculation or zero out grads.
            pass
            
        mx.eval(self.model.parameters())
        state = [self.model.state, self.optimizer.state]
        
        start_time = time.time()
        
        # Define step
        def step(x, y, temp):
            (loss, router_logits), grads = nn.value_and_grad(self.model, loss_fn)(self.model, x, y, temp, self.z_loss_weight, self.aux_loss_weight)
            
            # Post-process grads for freezing
            if self.args.get('freeze_router', False):
                # We need to filter grads.
                # grads is a tree matching model. structure.
                # We traverse and zero out 'router' keys?
                # Or structure is dict-like.
                
                def filter_grads(d, key_prefix=""):
                    if isinstance(d, dict):
                        new_d = {}
                        for k, v in d.items():
                            if 'router' in k or 'router' in key_prefix:
                                # This block/key belongs to router.
                                # But keys are 'layers', 'moe', 'router'
                                if 'router' in k:
                                    # This is the router key, skip it or set to None
                                    continue
                                else:
                                    new_d[k] = filter_grads(v, key_prefix + "." + k)
                            else:
                                new_d[k] = filter_grads(v, key_prefix + "." + k)
                        return new_d
                    elif isinstance(d, list):
                        return [filter_grads(v, key_prefix + f".{i}") for i, v in enumerate(d)]
                    else:
                        return d
                
                # Actually, MLX optimizer Update takes `model` AND `grads`. 
                # If a key is missing in grads, it won't update?
                # Let's try to remove router keys from grads.
                
                # Helper to prune router
                def prune_router(tree):
                    if isinstance(tree, dict):
                        return {k: prune_router(v) for k, v in tree.items() if 'router' not in k}
                    if isinstance(tree, list):
                        return [prune_router(v) for v in tree]
                    return tree
                    
                grads = prune_router(grads)
            
            self.optimizer.update(self.model, grads)
            return loss, router_logits

        print(f"Starting training for {num_steps} steps with Z-Loss and Soft Routing...")
        print(f"Temp Schedule: {self.start_temp} -> {self.end_temp}")
        
        for i, batch in enumerate(data_loader):
            if i >= num_steps:
                break
            
            # Linear annealing of temperature
            if num_steps > 1:
                curr_temp = self.start_temp - (self.start_temp - self.end_temp) * (i / (num_steps - 1))
            else:
                curr_temp = self.start_temp
                
            x = mx.array(batch[:, :-1])
            y = mx.array(batch[:, 1:])
            
            # Forward + Backward
            loss, router_logits = step(x, y, curr_temp)
            
            # Force eval to actually run computations
            mx.eval(state, loss) # Ensure update happens
            
            # Diagnostics (on CPU usually, or lazily)
            # We convert router_logits to numpy for analysis
            # To avoid stalling GPU, we might sample this occasionally
            if i % self.log_interval == 0:
                mx.eval(router_logits) # ensure available
                # extract simple stats
                r_logits_np = [np.array(l) for l in router_logits] 
                
                # Check top-k indices to update tracker
                all_indices = []
                k = self.model.args.num_experts_per_tok
                for l_logits in r_logits_np:
                    # simplistic numpy topk for now
                    # (B, L, E)
                    inds = np.argsort(l_logits, axis=-1)[..., -k:]
                    all_indices.append(inds)
                
                self.tracker.update_from_indices(all_indices)
                util = self.tracker.get_utilization()
                dead = len(self.tracker.detect_collapse())
                
                print(f"Step {i}: Loss = {loss.item():.4f}, Dead Experts: {dead}/{self.tracker.num_experts}")
                
            if i % self.save_interval == 0:
                self.save_checkpoint(i)
                
        print(f"Training finished in {time.time() - start_time:.2f}s")

    def save_checkpoint(self, step):
        path = self.output_dir / f"model_{step}.safetensors"
        self.model.save_weights(str(path))
        print(f"Saved checkpoint to {path}")

# Simple data loader for testing
def dummy_data_loader(vocab_size, batch_size, seq_len):
    while True:
        yield np.random.randint(0, vocab_size, (batch_size, seq_len + 1))
