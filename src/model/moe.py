import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from .experts import MLP

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

    def __call__(self, x: mx.array) -> mx.array:
        return self.gate(x)

from .switch_layers import SwitchMLP

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
        
    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        B, L, D = x.shape
        
        # 1. Shared Experts
        shared_output = self.shared_expert(x)
        
        # 2. Router
        gate_logits = self.router(x)
        gate_probs = mx.softmax(gate_logits, axis=-1)
        
        # 3. Top-k selection
        topk_indices = mx.argpartition(-gate_probs, self.num_experts_per_tok, axis=-1)[..., :self.num_experts_per_tok]
        topk_indices = mx.stop_gradient(topk_indices)
        
        # Gather weights
        topk_weights = mx.take_along_axis(gate_probs, topk_indices, axis=-1)
        
        # Normalize weights (L1) with epsilon
        sum_weights = mx.sum(topk_weights, axis=-1, keepdims=True)
        topk_weights = topk_weights / (sum_weights + 1e-6)
        
        # 4. Expert Execution via SwitchMLP
        expert_out = self.experts(x, topk_indices)
        
        # Weighted sum
        y = (expert_out * topk_weights[..., None]).sum(axis=-2)
        y = y + shared_output
        
        # 5. Load Balancing Info (DeepSeek-style)
        # f: fraction of tokens dispatched to each expert
        # P: average probability assigned to each expert
        # We need this for the aux loss in the trainer
        return y, gate_probs, topk_indices
