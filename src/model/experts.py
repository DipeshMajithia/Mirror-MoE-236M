import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    """SwiGLU MLP — shared expert (V2-compatible naming)."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.fc3(nn.silu(self.fc1(x)) * self.fc2(x))
