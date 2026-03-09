import mlx.core as mx
import numpy as np

class ExpertTracker:
    """
    Accumulates expert usage statistics over training steps.
    """
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset()
        
    def reset(self):
        self.total_tokens = 0
        self.expert_counts = np.zeros(self.num_experts, dtype=np.int64)
        self.history = [] # valid for short runs, might want to limit size
        
    def update(self, router_logits: list[mx.array]):
        """
        router_logits: list of arrays from different layers.
        Shape of each: (B, L, num_experts)
        """
        # We perform analysis on CPU/numpy for flexibility
        # Assuming we just track the LAST layer or aggregate all
        # For simplicity, let's aggregate all layers
        
        for logits in router_logits:
            # logits: (B, L, num_experts)
            # Get selections
            # We need to replicate the top-k logic here or pass indices
            # If we only have logits, we re-compute top-k
            
            # Since MLX arrays are lazy, we should be careful about pulling them to CPU
            # ideally the trainer passes indices.
            
            # But if we must work with logits:
            probs = mx.softmax(logits, axis=-1)
            # entropy per token
            # H = -sum(p * log(p))
            # Just store average entropy
            
            # Let's focus on indices (hard counts)
            # If logits are passed, we just check max for now (top-1) or top-k if known
            pass

    def update_from_indices(self, all_layer_indices: list[np.ndarray]):
        """
        Updates counts from a list of indices arrays (one per layer).
        indices shape: (B, L, k)
        """
        step_counts = np.zeros(self.num_experts, dtype=np.int64)
        
        for indices in all_layer_indices:
            # flatten
            flat = indices.flatten()
            counts = np.bincount(flat, minlength=self.num_experts)
            step_counts += counts
            
        self.expert_counts += step_counts
        self.total_tokens += indices.size # This tracks tokens * layers * k
        
        self.history.append(step_counts)
        
    def get_utilization(self):
        if self.total_tokens == 0:
            return np.zeros(self.num_experts)
        return self.expert_counts / self.total_tokens

    def detect_collapse(self, threshold=0.001):
        """
        Returns list of dead experts (usage < threshold).
        """
        util = self.get_utilization()
        return np.where(util < threshold)[0]
