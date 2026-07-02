from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from mlx_lm.models.nemotron_h import ModelArgs

from ..base import BaseModelConfig

_BLOCK_TYPE_TO_CHAR = {
    "mamba": "M",
    "attention": "*",
    "moe": "E",
    "mlp": "-",
}


def _normalize_hybrid_pattern(
    pattern: Optional[Union[str, List[str], Tuple[str, ...]]],
    layers_block_type: Optional[List[str]],
) -> List[str]:
    if pattern is None:
        if layers_block_type is None:
            return []
        return [_BLOCK_TYPE_TO_CHAR.get(layer, layer) for layer in layers_block_type]

    if isinstance(pattern, str):
        return list(pattern)

    normalized = []
    for item in pattern:
        normalized.append(_BLOCK_TYPE_TO_CHAR.get(item, item))
    return normalized


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "nemotron_h_twotower"
    architectures: Optional[List[str]] = None
    auto_map: Optional[Dict[str, Any]] = None
    vocab_size: int = 131072
    hidden_size: int = 2688
    intermediate_size: int = 1856
    num_hidden_layers: int = 52
    max_position_embeddings: int = 262144
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    mamba_num_heads: int = 64
    mamba_head_dim: int = 64
    mamba_proj_bias: bool = False
    ssm_state_size: int = 128
    conv_kernel: int = 4
    n_groups: int = 8
    mlp_bias: bool = False
    layer_norm_epsilon: float = 1e-5
    norm_eps: Optional[float] = None
    use_bias: bool = False
    use_conv_bias: bool = True
    hybrid_override_pattern: Optional[Union[str, List[str]]] = (
        "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    )
    layers_block_type: Optional[List[str]] = None
    head_dim: Optional[int] = 128
    moe_intermediate_size: Optional[int] = 1856
    moe_shared_expert_intermediate_size: Optional[int] = 3712
    moe_latent_size: Optional[int] = None
    n_group: Optional[int] = 1
    n_routed_experts: Optional[int] = 128
    n_shared_experts: Optional[int] = 1
    topk_group: Optional[int] = 1
    num_experts_per_tok: Optional[int] = 6
    norm_topk_prob: Optional[bool] = True
    routed_scaling_factor: Optional[float] = 2.5
    time_step_limit: Optional[Union[List[float], Tuple[float, float]]] = field(
        default_factory=lambda: (0.0, float("inf"))
    )
    time_step_min: Optional[float] = 0.001
    time_step_max: Optional[float] = 0.1
    time_step_floor: Optional[float] = 0.0001
    chunk_size: int = 128
    expand: int = 2
    mamba_hidden_act: str = "silu"
    mlp_hidden_act: str = "relu2"
    residual_in_fp32: bool = False
    rescale_prenorm_residual: bool = True
    use_cache: bool = True
    use_mamba_kernels: bool = True
    num_logits_to_keep: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    sliding_window: Optional[int] = None
    pad_token_id: Optional[int] = 0
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[Union[int, List[int]]] = 2
    dtype: Optional[str] = None

    # mlx-vlm masked-diffusion routing defaults. AR remains the default path.
    mask_token_id: int = 3
    default_generation_mode: str = "ar"
    default_diffusion_steps: int = 16
    default_diffusion_threshold: Optional[float] = 0.8
    block_size: int = 16

    def __post_init__(self):
        self.hybrid_override_pattern = _normalize_hybrid_pattern(
            self.hybrid_override_pattern, self.layers_block_type
        )
        if self.num_hidden_layers != len(self.hybrid_override_pattern):
            self.num_hidden_layers = len(self.hybrid_override_pattern)
        if self.norm_eps is not None:
            self.layer_norm_epsilon = self.norm_eps
        if self.time_step_limit is None:
            self.time_step_limit = (0.0, float("inf"))
        else:
            lo, hi = self.time_step_limit
            self.time_step_limit = (float(lo), float(hi))

    def to_model_args(self) -> ModelArgs:
        return ModelArgs.from_dict(
            {
                "model_type": "nemotron_h",
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "max_position_embeddings": self.max_position_embeddings,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "attention_bias": self.attention_bias,
                "mamba_num_heads": self.mamba_num_heads,
                "mamba_head_dim": self.mamba_head_dim,
                "mamba_proj_bias": self.mamba_proj_bias,
                "ssm_state_size": self.ssm_state_size,
                "conv_kernel": self.conv_kernel,
                "n_groups": self.n_groups,
                "mlp_bias": self.mlp_bias,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "use_bias": self.use_bias,
                "use_conv_bias": self.use_conv_bias,
                "hybrid_override_pattern": self.hybrid_override_pattern,
                "head_dim": self.head_dim,
                "moe_intermediate_size": self.moe_intermediate_size,
                "moe_shared_expert_intermediate_size": (
                    self.moe_shared_expert_intermediate_size
                ),
                "moe_latent_size": self.moe_latent_size,
                "n_group": self.n_group,
                "n_routed_experts": self.n_routed_experts,
                "n_shared_experts": self.n_shared_experts,
                "topk_group": self.topk_group,
                "num_experts_per_tok": self.num_experts_per_tok,
                "norm_topk_prob": self.norm_topk_prob,
                "routed_scaling_factor": self.routed_scaling_factor,
                "time_step_limit": self.time_step_limit,
                "time_step_min": self.time_step_min,
                "time_step_max": self.time_step_max,
            }
        )
