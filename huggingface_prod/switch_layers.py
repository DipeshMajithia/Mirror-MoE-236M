import mlx.core as mx
import mlx.nn as nn

def _gather_sort(x, indices):
    # indices: (..., k)
    # x: (..., 1, D)
    
    # We flatten batch dimensions
    # indices becomes (N*k,)
    
    # Note: the original implementation assumes specific shapes.
    # We'll adapt to be robust.
    
    # Flatten everything to (TotalTokens, ...)
    # But usually indices is (B, L, k) and x needs to match.
    
    # Let's inspect the original implementation logic:
    # *_, M = indices.shape -> M is k?
    # indices = indices.flatten()
    # order = mx.argsort(indices)
    # inv_order = mx.argsort(order)
    # return x.flatten(0, -3)[order // M], indices[order], inv_order
    
    # x is expanded dims (..., 1, D)
    # x.flatten(0, -3) -> flattens B, L... keeps D.
    # [order // M] selects the token index corresponding to the sorted expert index.
    
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    
    # Flatten x but keep last 2 dims (1, D)
    # x: (B, L, 1, D)
    x_flat = x.flatten(0, -3) 
    
    # Gather tokens in sorted order of experts
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
        # x: (TotalTokens, 1, D) if sorted
        # indices: (TotalTokens,) expert indices
        
        # gather_mm performs: out[i] = x[i] @ weight[indices[i]].T
        # weight: (E, Out, In)
        # x: (N, 1, In)
        # indices: (N,)
        
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
    def __init__(self, dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.fc1 = SwitchLinear(dim, hidden_dim, num_experts, bias=False)
        self.fc2 = SwitchLinear(hidden_dim, dim, num_experts, bias=False)
        # silu is standard
    
    def __call__(self, x, indices):
        # x: (B, L, D)
        # indices: (B, L, k)
        
        # Expand x to (B, L, 1, D)
        x = mx.expand_dims(x, -2)
        
        # Sort for efficiency
        # Always sort for safety/correctness
        do_sort = True
        
        orig_shape = indices.shape + (x.shape[-1],) 
        
        idx = indices
        inv_order = None
        
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
            
        # x is now (N*k, 1, D)
        # idx is (N*k,)
        
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = nn.silu(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)
        
        if do_sort:
            x = _scatter_unsort(x, inv_order, orig_shape)
        else:
            x = x.reshape(orig_shape)
             
        return x 
