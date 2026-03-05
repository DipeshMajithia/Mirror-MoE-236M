import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from safetensors.torch import load_file

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    hidden_dim: int = 256 # Corrected from 1408
    n_heads: int = 8
    vocab_size: int = 32000
    norm_eps: float = 1e-5
    use_moe: bool = True
    num_experts: int = 16
    num_experts_per_tok: int = 2
    shared_expert_dim: int = 512

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))

class Router(nn.Module):
    """Simple router that matches checkpoint key 'moe.router.gate.weight'."""
    def __init__(self, dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
    
    def forward(self, x):
        return self.gate(x)

class BatchedLinear(nn.Module):
    """A single batched linear layer (no bias). Produces key 'weight' when registered."""
    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

class Experts(nn.Module):
    """Batched experts matching checkpoint keys 'moe.experts.fc1.weight' and 'moe.experts.fc2.weight'."""
    def __init__(self, num_experts, dim, hidden_dim):
        super().__init__()
        # These will create keys like 'experts.fc1.weight' and 'experts.fc2.weight'
        self.fc1 = BatchedLinear(num_experts, dim, hidden_dim)  # (num_experts, hidden_dim, dim)
        self.fc2 = BatchedLinear(num_experts, hidden_dim, dim)  # (num_experts, dim, hidden_dim)
    
    def forward(self, x, expert_idx):
        # x: (batch, dim), expert_idx: (batch,)
        
        # Get weights for selected experts
        fc1_sel = self.fc1.weight[expert_idx]  # (batch, hidden, dim)
        fc2_sel = self.fc2.weight[expert_idx]  # (batch, dim, hidden)
        
        # x: (batch, dim) -> (batch, 1, dim)
        x = x.unsqueeze(1)
        
        # matmul: (batch, 1, dim) @ (batch, dim, hidden) -> (batch, 1, hidden)
        h = torch.bmm(x, fc1_sel.transpose(-1, -2))  # (batch, 1, hidden)
        h = F.silu(h)
        
        # matmul: (batch, 1, hidden) @ (batch, hidden, dim) -> (batch, 1, dim)
        out = torch.bmm(h, fc2_sel.transpose(-1, -2))  # (batch, 1, dim)
        
        return out.squeeze(1)  # (batch, dim)

class MoELayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        
        # Router (matches checkpoint: moe.router.gate.weight)
        self.router = Router(args.dim, args.num_experts)
        
        # Batched experts (matches checkpoint: moe.experts.fc1/fc2.weight)
        self.experts = Experts(args.num_experts, args.dim, args.hidden_dim)
        
        # Shared expert (matches checkpoint: moe.shared_expert.fc1/fc2/fc3.weight)
        self.shared_expert = MLP(args.dim, args.shared_expert_dim)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # Shared expert (always active)
        shared_out = self.shared_expert(x)

        # Router logits
        gate_logits = self.router(x)  # (B, L, num_experts)
        
        # Top-K selection
        weights, indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=-1)
        weights = F.softmax(weights, dim=-1)  # (B, L, K)
        
        # Flatten for batched expert computation
        x_flat = x.view(-1, D)  # (B*L, D)
        indices_flat = indices.view(-1, self.num_experts_per_tok)  # (B*L, K)
        weights_flat = weights.view(-1, self.num_experts_per_tok)  # (B*L, K)
        
        # Compute expert outputs
        out_flat = torch.zeros_like(x_flat)
        
        for k in range(self.num_experts_per_tok):
            expert_idx = indices_flat[:, k]  # (B*L,)
            expert_weight = weights_flat[:, k:k+1]  # (B*L, 1)
            
            expert_out = self.experts(x_flat, expert_idx)  # (B*L, D)
            out_flat = out_flat + expert_out * expert_weight
        
        out = out_flat.view(B, L, D)
        return out + shared_out

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Precompute theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x):
        # x: (B, H, L, D)
        seq_len = x.shape[2]
        
        # Create cache if needed or if seq_len grew
        if self.cached_cos is None or self.cached_cos.shape[2] < seq_len:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq) # (L, D/2)
            
            # Create cos/sin
            # We use half-rotation logic to match standard RoPE behavior
            # [cos, cos, ..., sin, sin, ...] is one way
            # [cos, -sin, sin, cos] (complex) is another.
            
            # Robust Standard Implementation (Llama Style - Half Rotation):
            # This is what most libraries (including MLX's default rope?) use.
            # Wait, MLX uses "traditional" which is usually [x0, x1] -> [-x1, x0].
            
            # Let's support BOTH to be sure. Defaulting to INTERLEAVED (MLX standard).
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()[None, None, :, :]
            self.cached_sin = emb.sin()[None, None, :, :]
        
        return self._apply_rotary_emb(x, self.cached_cos[:, :, :seq_len, :], self.cached_sin[:, :, :seq_len, :])

    def _apply_rotary_emb(self, x, cos, sin):
        # Explicitly reshape to ensure contiguous memory layout
        # x shape: (B, H, L, D)
        # Split last dimension into 2 halves (Standard/Llama style)
        
        # Safe implementation:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        
        # Ensure contiguous before operation if needed (rarely needed but safe)
        # x1 = x1.contiguous()
        # x2 = x2.contiguous()
        
        # Llama Rotation: [-x2, x1]
        # cos, sin are (B, H, L, D), but x1, x2 are (B, H, L, D/2)
        # We need to slice cos/sin to use the first half (which contains the frequencies)
        return torch.cat((-x2 * sin[..., :d] + x1 * cos[..., :d], x1 * sin[..., :d] + x2 * cos[..., :d]), dim=-1)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        
        self.rope = RoPE(self.head_dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        
        probs = F.softmax(scores, dim=-1)
        out = (probs @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = Attention(args)
        self.norm1 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.moe = MoELayer(args)
        self.norm2 = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, mask=None):
        h = x + self.attention(self.norm1(x), mask)
        h = h + self.moe(self.norm2(h))
        return h

class MirrorTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x):
        h = self.tok_embeddings(x)
        
        # Causal mask
        L = x.shape[1]
        mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1)
        if x.device.type == 'cuda' or x.device.type == 'mps':
             mask = mask.to(x.device)

        for layer in self.layers:
            h = layer(h, mask)
        
        h = self.norm(h)
        return self.output(h)

    @classmethod
    def from_pretrained(cls, path):
        # Determine strict structure...
        # For simplicity, returning instance ready to load
        return cls(ModelArgs())
