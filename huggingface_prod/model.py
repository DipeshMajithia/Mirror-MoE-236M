import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from moe import MoELayer, MoEArgs
from experts import MLP

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: int = 64
    hidden_dim: int = 256  # Corrected from 1408 to match trained model
    n_heads: int = 8
    vocab_size: int = 32000
    norm_eps: float = 1e-5
    
    # MoE specific
    use_moe: bool = True
    num_experts: int = 16  # Corrected from 32
    num_experts_per_tok: int = 2
    shared_expert_dim: int = 512 # Corrected from 256 

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        
        self.rope = nn.RoPE(args.head_dim)

    def __call__(self, x: mx.array, mask: mx.array = None):
        B, L, D = x.shape
        
        q = self.wq(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q = self.rope(q)
        k = self.rope(k)
        
        # Attention details
        # (B, n_heads, L, L)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
            
        probs = mx.softmax(scores, axis=-1)
        out = (probs @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.attention = Attention(args)
        self.norm1 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        
        if args.use_moe:
            moe_args = MoEArgs(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                num_experts=args.num_experts,
                num_experts_per_tok=args.num_experts_per_tok,
                shared_expert_dim=args.shared_expert_dim
            )
            self.moe = MoELayer(moe_args)
        else:
            self.mlp = MLP(args.dim, args.hidden_dim)
            
        self.norm2 = nn.RMSNorm(args.dim, eps=args.norm_eps)
        
    def __call__(self, x: mx.array, mask: mx.array = None, temperature: float = 1.0):
        h = x + self.attention(self.norm1(x), mask)
        
        if self.args.use_moe:
            out, gate_logits, mean_scores = self.moe(self.norm2(h), temperature=temperature)
            return h + out, gate_logits, mean_scores
        else:
            out = self.mlp(self.norm2(h))
            return h + out, None, None

class MirrorTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.emb = nn.Embedding(args.vocab_size, args.dim)
        
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
    def __call__(self, x: mx.array, temperature: float = 1.0):
        # x: (B, L)
        h = self.emb(x)
        
        # Causal mask
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        mask = mask.astype(h.dtype)
        
        all_logits = []
        all_aux_scores = []
        
        for layer in self.layers:
            h, gate_logits, aux_scores = layer(h, mask, temperature=temperature)
            all_logits.append(gate_logits)
            all_aux_scores.append(aux_scores)
            
        h = self.norm(h)
        out_logits = self.output(h)
        
        return out_logits, all_logits, all_aux_scores # Return Token Logits, Router Logits, and Aux Scores
