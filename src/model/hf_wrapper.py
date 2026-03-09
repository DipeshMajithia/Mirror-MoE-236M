import torch
import numpy as np
import mlx.core as mx
from transformers import PreTrainedModel, PretrainedConfig

class MirrorConfig(PretrainedConfig):
    model_type = "mirror-ai"

    def __init__(
        self,
        vocab_size=32000,
        dim=512,
        n_layers=8,
        n_heads=8,
        hidden_dim=256,
        use_moe=True,
        num_experts=16,
        num_experts_per_tok=2,
        shared_expert_dim=512,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_dim = shared_expert_dim
        super().__init__(**kwargs)

from transformers.modeling_outputs import CausalLMOutputWithPast

class MirrorForCausalLM(PreTrainedModel):
    config_class = MirrorConfig

    def __init__(self, config, mirror_model):
        super().__init__(config)
        self.model = mirror_model
        # We don't initialize weights here as we assume mirror_model is already loaded using MLX
        
        # Dummy parameter to allow HF to detect device (CPU)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids, **kwargs):
        # Bridge: PyTorch -> MLX
        # input_ids comes as torch.LongTensor
        input_np = input_ids.cpu().numpy()
        input_mx = mx.array(input_np)

        # Forward pass in MLX
        # We handle the fact that our model.forward takes (x) and returns (logits, loss, cache)
        # We only need logits here.
        logits_mx, _, _ = self.model(input_mx)

        # Bridge: MLX -> PyTorch
        logits_np = np.array(logits_mx)
        logits_pt = torch.from_numpy(logits_np).to(input_ids.device)

        # Return HF-compatible output object
        return CausalLMOutputWithPast(logits=logits_pt)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
