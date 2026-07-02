import math
import time
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.nemotron_h import NemotronHModel

from ..base import LanguageModelOutput
from .config import ModelConfig


def _copy_array(x):
    if x is None:
        return None
    return mx.array(x)


def _clone_cache_entry(cache):
    if cache is None:
        return None
    if isinstance(cache, KVCache):
        new_cache = KVCache()
        if cache.keys is not None:
            new_cache.keys = _copy_array(cache.keys[..., : cache.offset, :])
            new_cache.values = _copy_array(cache.values[..., : cache.offset, :])
            new_cache.offset = cache.offset
        return new_cache
    if isinstance(cache, ArraysCache):
        new_cache = ArraysCache(len(cache.cache))
        new_cache.cache = [_copy_array(value) for value in cache.cache]
        new_cache.left_padding = _copy_array(cache.left_padding)
        new_cache.lengths = _copy_array(cache.lengths)
        return new_cache
    return cache


def _clone_cache(cache):
    return [_clone_cache_entry(entry) for entry in cache]


def _first_token_index(tokens: mx.array, token_ids: set[int]) -> Optional[int]:
    values = tokens.tolist()
    return next(
        (index for index, token_id in enumerate(values) if token_id in token_ids),
        None,
    )


def _topk(x: mx.array, k: int, axis: int = -1) -> Tuple[mx.array, mx.array]:
    indices = mx.argpartition(-x, kth=k - 1, axis=axis)[..., :k]
    values = mx.take_along_axis(x, indices, axis=axis)
    order = mx.argsort(-values, axis=axis)
    return mx.take_along_axis(values, order, axis=axis), mx.take_along_axis(
        indices, order, axis=axis
    )


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 1000,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs[None]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros((embedding.shape[0], 1), dtype=embedding.dtype)],
                axis=-1,
            )
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.timestep_embedding(
            t * self.max_period, self.frequency_embedding_size
        )
        return self.mlp(t_freq)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.args = config.to_model_args()
        self.model_type = "nemotron_h_twotower"
        self.context_tower = NemotronHModel(self.args)
        self.context_lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.denoiser_tower = NemotronHModel(self.args)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 3 * config.hidden_size),
        )
        self.scale_shift_tables = [
            mx.zeros((3, config.hidden_size)) for _ in range(config.num_hidden_layers)
        ]

    @property
    def layers(self):
        return self.context_tower.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def _cache_count(self, tower) -> int:
        return sum(layer.block_type in ("M", "*") for layer in tower.layers)

    def make_cache(self):
        return self.make_context_cache()

    def make_context_cache(self):
        caches = []
        for layer in self.context_tower.layers:
            if layer.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif layer.block_type == "*":
                caches.append(KVCache())
        return caches

    def get_input_embeddings(self, input_ids: mx.array) -> mx.array:
        return self.context_tower.embeddings(input_ids)

    def _modulation(self, t_emb: Optional[mx.array], layer_idx: int, dtype) -> Tuple:
        if t_emb is None:
            return None, None, None
        params = self.t_block(t_emb)
        table = self.scale_shift_tables[layer_idx]
        params = params.reshape((params.shape[0], 3, self.config.hidden_size))
        params = params + table[None].astype(params.dtype)
        shift, scale, gate = mx.split(
            params,
            [1, 2],
            axis=1,
        )
        return (
            shift[:, 0].astype(dtype),
            scale[:, 0].astype(dtype),
            gate[:, 0].astype(dtype),
        )

    def _forward_tower(
        self,
        tower,
        input_ids: Optional[mx.array] = None,
        *,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        use_causal_mask: bool = True,
        t_emb: Optional[mx.array] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            hidden_states = tower.embeddings(input_ids)
        else:
            hidden_states = inputs_embeds

        if cache is None:
            cache = [None] * self._cache_count(tower)

        attn_cache = cache[tower.fa_idx] if tower.fa_idx < len(cache) else None
        ssm_cache = cache[tower.ssm_idx] if tower.ssm_idx < len(cache) else None
        if use_causal_mask:
            attn_mask = create_attention_mask(hidden_states, attn_cache)
            ssm_mask = create_ssm_mask(hidden_states, ssm_cache)
        else:
            attn_mask = None
            ssm_mask = None

        cache_counter = 0
        for layer_idx, layer in enumerate(tower.layers):
            layer_cache = None
            if layer.block_type in ("M", "*"):
                layer_cache = cache[cache_counter]
                cache_counter += 1

            mask = attn_mask if layer.block_type == "*" else ssm_mask
            if t_emb is None:
                hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
                continue

            shift, scale, gate = self._modulation(t_emb, layer_idx, hidden_states.dtype)
            residual = hidden_states
            if layer.block_type in ("M", "*"):
                mixed = hidden_states * (1 + scale[:, None, :]) + shift[:, None, :]
                mixed = layer.norm(mixed)
            else:
                mixed = layer.norm(hidden_states)
                mixed = mixed * (1 + scale[:, None, :]) + shift[:, None, :]
            if layer.block_type in ("M", "*"):
                mixed = layer.mixer(mixed, mask=mask, cache=layer_cache)
            else:
                mixed = layer.mixer(mixed)
            mixed = mixed * gate[:, None, :]
            hidden_states = residual + mixed

        return tower.norm_f(hidden_states)

    def __call__(
        self,
        inputs: mx.array,
        *,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ):
        hidden_states = self._forward_tower(
            self.context_tower,
            inputs,
            inputs_embeds=inputs_embeds,
            cache=cache,
            use_causal_mask=True,
        )
        return LanguageModelOutput(logits=self.context_lm_head(hidden_states))

    @staticmethod
    def _top_k_logits(logits: mx.array, k: Optional[int]) -> mx.array:
        if k is None or k <= 0:
            return logits
        values = mx.topk(logits, k, axis=-1)
        neg_large = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
        return mx.where(logits < values[..., -1:], neg_large, logits)

    @staticmethod
    def _top_p_logits(logits: mx.array, p: Optional[float]) -> mx.array:
        if p is None or p >= 1.0:
            return logits
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(
            mx.softmax(sorted_logits, axis=-1, precise=True), axis=-1
        )
        sorted_mask = cumulative_probs > p
        sorted_mask = mx.concatenate(
            [mx.zeros_like(sorted_mask[..., :1]), sorted_mask[..., :-1]], axis=-1
        )
        inverse_indices = mx.argsort(sorted_indices, axis=-1)
        mask = mx.take_along_axis(sorted_mask, inverse_indices, axis=-1)
        neg_large = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
        return mx.where(mask, neg_large, logits)

    def _sample_from_logits(
        self,
        logits: mx.array,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_prob: bool = False,
    ):
        if temperature == 0.0:
            token = mx.argmax(logits, axis=-1)
            token_logit = mx.take_along_axis(logits, token[..., None], axis=-1)[..., 0]
            token_prob = mx.exp(token_logit - mx.logsumexp(logits, axis=-1))
            return (token, token_prob) if return_prob else token

        if temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        token = mx.random.categorical(logits.astype(mx.float32), axis=-1)
        token_logit = mx.take_along_axis(logits, token[..., None], axis=-1)[..., 0]
        token_prob = mx.exp(token_logit - mx.logsumexp(logits, axis=-1))
        return (token, token_prob) if return_prob else token

    def _project_denoiser(self, hidden_states: mx.array) -> mx.array:
        return self.lm_head(hidden_states)

    def _build_context_cache(self, input_ids: mx.array, stats=None):
        cache = self.make_context_cache()
        tic = time.perf_counter()
        if input_ids.shape[1] > 1:
            self._forward_tower(
                self.context_tower,
                input_ids[:, :-1],
                cache=cache,
                use_causal_mask=True,
            )
            mock_cache = _clone_cache(cache)
            hidden = self._forward_tower(
                self.context_tower,
                input_ids[:, -1:],
                cache=cache,
                use_causal_mask=True,
            )
        else:
            mock_cache = self.make_context_cache()
            hidden = self._forward_tower(
                self.context_tower,
                input_ids,
                cache=cache,
                use_causal_mask=True,
            )
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - tic
            stats["prompt_tokens"] = float(input_ids.size)
        return {
            "cache": cache,
            "mock_cache": mock_cache,
            "ctx_len": input_ids.shape[1],
            "logits": self.context_lm_head(hidden[:, -1:, :]),
        }

    def _extend_context_cache(self, tokens: mx.array, cache_state: Dict[str, Any]):
        cache_state["mock_cache"] = _clone_cache(cache_state["cache"])
        hidden = self._forward_tower(
            self.context_tower,
            tokens,
            cache=cache_state["cache"],
            use_causal_mask=True,
        )
        cache_state["ctx_len"] += tokens.shape[1]
        cache_state["logits"] = self.context_lm_head(hidden[:, -1:, :])
        return cache_state

    def _run_denoiser(
        self,
        block_ids: mx.array,
        cache_state: Optional[Dict[str, Any]] = None,
        t: Optional[mx.array] = None,
        cache_key: str = "cache",
    ) -> mx.array:
        cache = (
            _clone_cache(cache_state[cache_key]) if cache_state is not None else None
        )
        t_emb = self.t_embedder(t) if t is not None else None
        hidden = self._forward_tower(
            self.denoiser_tower,
            block_ids,
            cache=cache,
            use_causal_mask=False,
            t_emb=t_emb,
        )
        return self._project_denoiser(hidden)

    def ar_generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_token_id: Optional[Union[int, list[int]]] = None,
        stats: Optional[Dict[str, float]] = None,
    ) -> mx.array:
        eos_token_id = (
            self.config.eos_token_id if eos_token_id is None else eos_token_id
        )
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )
        cache_state = self._build_context_cache(input_ids, stats=stats)
        logits = cache_state["logits"]
        generated = []
        for _ in range(max_new_tokens):
            token = self._sample_from_logits(logits, temperature, top_k, top_p)
            generated.append(token)
            if token[:, 0].tolist()[0] in eos_token_ids:
                break
            cache_state = self._extend_context_cache(token, cache_state)
            logits = cache_state["logits"]
        if not generated:
            return mx.zeros((input_ids.shape[0], 0), dtype=input_ids.dtype)
        return mx.concatenate(generated, axis=1)

    def mock_ar_generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_token_id: Optional[Union[int, list[int]]] = None,
        stats: Optional[Dict[str, float]] = None,
    ) -> mx.array:
        eos_token_id = (
            self.config.eos_token_id if eos_token_id is None else eos_token_id
        )
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )
        cache_state = self._build_context_cache(input_ids, stats=stats)
        last_token = input_ids[:, -1:]
        generated = []
        for _ in range(max_new_tokens):
            logits = self._run_denoiser(
                last_token, cache_state, cache_key="mock_cache"
            )[:, -1:, :]
            token = self._sample_from_logits(logits, temperature, top_k, top_p)
            generated.append(token)
            cache_state = self._extend_context_cache(token, cache_state)
            last_token = token
            if token[:, 0].tolist()[0] in eos_token_ids:
                break
        if not generated:
            return mx.zeros((input_ids.shape[0], 0), dtype=input_ids.dtype)
        return mx.concatenate(generated, axis=1)

    def _mdlm_logits(self, logits: mx.array, xt: mx.array, mask_token_id: int):
        neg_large = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
        vocab_positions = mx.arange(logits.shape[-1])
        logits = mx.where(vocab_positions == mask_token_id, neg_large, logits)

        unmasked = xt != mask_token_id
        fixed = vocab_positions[None, None, :] == xt[..., None]
        logits = mx.where(unmasked[..., None], mx.where(fixed, 0.0, neg_large), logits)
        return logits

    def mask_diffusion_generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        block_size: int = 16,
        steps_per_block: int = 16,
        mask_token_id: Optional[int] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        eos_token_id: Optional[Union[int, list[int]]] = None,
        stats: Optional[Dict[str, float]] = None,
    ) -> mx.array:
        if input_ids.shape[0] != 1:
            raise ValueError(
                "Nemotron TwoTower mask diffusion currently supports batch size 1."
            )
        mask_token_id = (
            self.config.mask_token_id if mask_token_id is None else mask_token_id
        )
        eos_token_id = (
            self.config.eos_token_id if eos_token_id is None else eos_token_id
        )
        confidence_threshold = (
            self.config.default_diffusion_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        steps_per_block = max(1, int(steps_per_block))

        if stats is not None:
            for key in (
                "diffusion_blocks",
                "diffusion_denoise_nfe",
                "diffusion_accepted_tokens",
            ):
                stats.setdefault(key, 0.0)

        def add_stat(key: str, value: float = 1.0):
            if stats is not None:
                stats[key] = stats.get(key, 0.0) + float(value)

        cache_state = self._build_context_cache(input_ids, stats=stats)
        generated_blocks = []
        total_generated = 0
        end_length = None

        while total_generated < max_new_tokens:
            add_stat("diffusion_blocks")
            current_block_size = min(block_size, max_new_tokens - total_generated)
            block = mx.full(
                (input_ids.shape[0], current_block_size),
                mask_token_id,
                dtype=input_ids.dtype,
            )
            masked_count = current_block_size
            block_positions = mx.arange(current_block_size)
            for step_idx in range(min(steps_per_block, current_block_size)):
                mask_index = block == mask_token_id
                masked_count = int(mask_index.sum().item())
                if masked_count == 0:
                    break
                force_completion = (
                    step_idx == min(steps_per_block, current_block_size) - 1
                )
                t = mask_index.astype(mx.float32).mean(axis=-1)
                logits = self._run_denoiser(block, cache_state, t=t)
                add_stat("diffusion_denoise_nfe")
                logits = self._mdlm_logits(logits, block, mask_token_id)
                predicted, confidence = self._sample_from_logits(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    return_prob=True,
                )
                sampled_block = mx.where(mask_index, predicted, block)
                if force_completion or masked_count == 1:
                    transfer_mask = mask_index
                    accepted_count = masked_count
                else:
                    masked_conf = mx.where(mask_index, confidence, -mx.inf)
                    if confidence_threshold is None:
                        remaining_steps = max(1, steps_per_block - step_idx)
                        transfer_count = max(
                            1, (masked_count + remaining_steps - 1) // remaining_steps
                        )
                    else:
                        transfer_count = int(
                            ((masked_conf >= confidence_threshold) & mask_index)
                            .sum()
                            .item()
                        )
                        remaining_steps = max(1, steps_per_block - step_idx)
                        min_commit = max(
                            1, (masked_count + remaining_steps - 1) // remaining_steps
                        )
                        transfer_count = max(transfer_count, min_commit)
                    _, indices = _topk(
                        masked_conf, min(transfer_count, masked_count), axis=-1
                    )
                    transfer_mask = (
                        block_positions[None, None, :] == indices[..., None]
                    ).any(axis=1) & mask_index
                    accepted_count = min(transfer_count, masked_count)
                block = mx.where(transfer_mask, sampled_block, block)
                add_stat("diffusion_accepted_tokens", accepted_count)

            generated_block = block[:, :current_block_size]
            generated_blocks.append(generated_block)
            total_generated += current_block_size
            eos_index = _first_token_index(generated_block[0], eos_token_ids)
            if eos_index is not None:
                end_length = total_generated - current_block_size + eos_index + 1
                break
            cache_state = self._extend_context_cache(generated_block, cache_state)

        generated = mx.concatenate(generated_blocks, axis=1)
        end = end_length if end_length is not None else generated.shape[1]
        if stats is not None:
            stats["diffusion_generated_tokens"] = float(end)
            stats["diffusion_total_nfe"] = stats.get("diffusion_denoise_nfe", 0.0)
        return generated[:, :end]

    def generate(self, input_ids: mx.array, **kwargs) -> mx.array:
        mode = kwargs.pop("generation_mode", self.config.default_generation_mode)
        mode = (mode or "ar").lower().replace("-", "_")
        gen_length = int(kwargs.pop("gen_length", kwargs.pop("max_new_tokens", 128)))
        temperature = float(kwargs.pop("temperature", 0.0))
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        eos_token_id = kwargs.pop("eos_token_id", self.config.eos_token_id)
        stats = kwargs.pop("stats", None)

        if mode == "ar":
            return self.ar_generate(
                input_ids,
                gen_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id,
                stats=stats,
            )
        if mode in {"mock_ar", "mock"}:
            return self.mock_ar_generate(
                input_ids,
                gen_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id,
                stats=stats,
            )
        if mode in {"mask_diffusion", "diffusion", "dlm"}:
            block_size = int(
                kwargs.pop(
                    "block_size", kwargs.pop("block_length", self.config.block_size)
                )
            )
            steps = int(
                kwargs.pop(
                    "steps_per_block",
                    kwargs.pop("steps", self.config.default_diffusion_steps),
                )
            )
            threshold = kwargs.pop(
                "confidence_threshold", kwargs.pop("threshold", None)
            )
            return self.mask_diffusion_generate(
                input_ids,
                gen_length,
                block_size=block_size,
                steps_per_block=steps,
                mask_token_id=kwargs.pop("mask_token_id", self.config.mask_token_id),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                confidence_threshold=threshold,
                eos_token_id=eos_token_id,
                stats=stats,
            )
        raise ValueError(f"Unsupported Nemotron TwoTower generation_mode: {mode!r}")

    def sanitize(self, weights):
        weights = dict(weights)
        sanitized = {}
        for key, value in weights.items():
            new_key = key
            if key.startswith("language_model."):
                new_key = key
            elif key.startswith(("context_tower.", "denoiser_tower.")):
                new_key = f"language_model.{key}"
            elif key.startswith("context_lm_head."):
                new_key = f"language_model.{key}"
            elif key.startswith("lm_head."):
                new_key = f"language_model.{key}"
            elif key.startswith("t_embedder."):
                new_key = f"language_model.{key}"
            elif key.startswith("t_block."):
                new_key = f"language_model.{key}"
            elif key.startswith("scale_shift_tables."):
                new_key = f"language_model.{key}"

            new_key = new_key.replace(".mlp.0.", ".mlp.layers.0.")
            new_key = new_key.replace(".mlp.2.", ".mlp.layers.2.")
            new_key = new_key.replace(".t_block.1.", ".t_block.layers.1.")
            if "conv1d.weight" in new_key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            sanitized[new_key] = value

        for tower in ("context_tower", "denoiser_tower"):
            for layer_idx in range(self.config.num_hidden_layers):
                prefix = f"language_model.{tower}.layers.{layer_idx}.mixer"
                for source, target in (("up_proj", "fc1"), ("down_proj", "fc2")):
                    first_key = f"{prefix}.experts.0.{source}.weight"
                    if first_key not in sanitized:
                        continue
                    joined = [
                        sanitized.pop(f"{prefix}.experts.{expert}.{source}.weight")
                        for expert in range(self.config.n_routed_experts)
                    ]
                    sanitized[f"{prefix}.switch_mlp.{target}.weight"] = mx.stack(joined)

        return {
            key: value
            for key, value in sanitized.items()
            if "rotary_emb.inv_freq" not in key
        }

    @property
    def cast_predicate(self):
        def predicate(key):
            return "e_score_correction_bias" not in key and "A_log" not in key

        return predicate
