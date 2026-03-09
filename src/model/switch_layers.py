import mlx.core as mx
import mlx.nn as nn

def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    x_flat = x.flatten(0, -3) 
    x_sorted = x_flat[order // M]
    return x_sorted, indices[order], inv_order

def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = x.reshape(shape)
    return x

class SwitchLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_experts: int, bias: bool = False):
        super().__init__()
        scale = (1 / input_dims) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self.weight.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if hasattr(self, "bias"):
            x = x + mx.expand_dims(self.bias[indices], -2)
        return x

class SwitchMLP(nn.Module):
    """V2-compatible 2-matrix SiLU expert (fc1/fc2 naming)."""
    def __init__(self, dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.fc1 = SwitchLinear(dim, hidden_dim, num_experts, bias=False)
        self.fc2 = SwitchLinear(hidden_dim, dim, num_experts, bias=False)
    
    def __call__(self, x, indices):
        orig_shape = indices.shape + (x.shape[-1],) 
        x = mx.expand_dims(x, -2)
        x, idx, inv_order = _gather_sort(x, indices)
        
        x = self.fc1(x, idx, sorted_indices=True)
        x = nn.silu(x)
        x = self.fc2(x, idx, sorted_indices=True)
        
        return _scatter_unsort(x, inv_order, orig_shape)
