import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from experts import MLP

@dataclass
class MoEArgs:
    dim: int
    hidden_dim: int
    num_experts: int
    num_experts_per_tok: int
    shared_expert_dim: int

class Router(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x, temperature: float = 1.0) -> tuple[mx.array, mx.array]:
        # x: (B, L, D)
        logits = self.gate(x)
        
        # Hard-Top-K Gating
        # 1. Sort descending
        # mlx.sort doesn't support descending directly?
        # We can use TopK or sort.
        # sort axis=-1.
        sorted_logits = mx.sort(logits, axis=-1) # Ascending
        
        # 4th highest is at index -4 (since ascending: ... -4, -3, -2, -1)
        # However, to safeguard against edge cases, let's pick exactly the cutoff.
        # logic: logits - v4. 
        # If we subtract v4, then v4 becomes 0.
        # We want exactly 4 experts >= 0.
        # v_4 = sorted_logits[..., -4]
        
        v4 = sorted_logits[..., -4: -3] # Keep dims for broadcast?
        # v4 shape (B, L, 1)
        
        # Subtract V4 from all logits
        # We add a small epsilon to ensure V4 itself remains positive (active)
        # User requested 4 experts >= 0.
        # If we use strict >, we need epsilon.
        # prompt: "Subtract V4 ... ensures exactly 4 experts are >= 0"
        # If >= 0, then 0 is allowed.
        # But verify_sparsity checks > 0.
        # I will add 1e-4 to ensure they register as non-zero.
        
        scaled_logits = logits - v4 + 1e-4
        
        scores = nn.relu(scaled_logits)
        return scores, logits

from switch_layers import SwitchMLP

class MoELayer(nn.Module):
    def __init__(self, args: MoEArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        
        # Micro-experts (Optimized SwitchMLP)
        self.experts = SwitchMLP(args.dim, args.hidden_dim, self.num_experts)
        self.router = Router(args.dim, self.num_experts)
        
        # Shared expert (dense path)
        self.shared_expert = MLP(args.dim, args.shared_expert_dim)
        
    def __call__(self, x: mx.array, temperature: float = 1.0) -> tuple[mx.array, mx.array, mx.array]:
        B, L, D = x.shape
        
        # 1. Shared Experts
        shared_output = self.shared_expert(x)
        
        # 2. Router
        gate_scores, gate_logits = self.router(x, temperature)
        
        # 3. Top-k selection
        # sort ascending, take last k
        sorted_indices = mx.argsort(gate_scores, axis=-1)
        topk_indices = sorted_indices[..., -self.num_experts_per_tok:]
        topk_indices = mx.stop_gradient(topk_indices)
        
        # Gather weights
        topk_weights = mx.take_along_axis(gate_scores, topk_indices, axis=-1)
        
        # Normalize weights (L1) with epsilon
        sum_weights = mx.sum(topk_weights, axis=-1, keepdims=True)
        topk_weights = topk_weights / (sum_weights + 1e-6)
        
        # 4. Expert Execution via SwitchMLP
        # indices: (B, L, k)
        # experts output: (B, L, k, D)
        expert_out = self.experts(x, topk_indices)
        
        # Weighted sum
        # weights: (B, L, k) -> (B, L, k, 1)
        y = (expert_out * topk_weights[..., None]).sum(axis=-2)
        
        # Add shared expert output
        y = y + shared_output
        
        # 5. Auxiliary Load Balance Loss support
        # We want the router scores (gate_scores) to be distributed across experts.
        # This prevents over-clustering on 2 experts.
        mean_scores = mx.mean(gate_scores, axis=(0, 1)) # (E,)
        
        # We return gate_scores instead of gate_logits for verification of Hard-Top-K
        return y, gate_scores, mean_scores
