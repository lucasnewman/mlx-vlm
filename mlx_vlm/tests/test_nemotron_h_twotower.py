import unittest

import mlx.core as mx

from mlx_vlm.utils import get_model_and_args


def tiny_config(**kwargs):
    config = {
        "model_type": "nemotron_h",
        "architectures": ["NemotronHTwoTowerForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "max_position_embeddings": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "attention_bias": False,
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "mamba_proj_bias": False,
        "ssm_state_size": 4,
        "conv_kernel": 3,
        "n_groups": 1,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "hybrid_override_pattern": "EM",
        "head_dim": 4,
        "moe_intermediate_size": 4,
        "moe_shared_expert_intermediate_size": 4,
        "n_routed_experts": 2,
        "n_shared_experts": None,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "time_step_limit": [0.0, float("inf")],
        "mask_token_id": 3,
    }
    config.update(kwargs)
    return config


class TestNemotronHTwoTower(unittest.TestCase):
    def test_loader_routes_by_architecture_without_hijacking_plain_nemotron_h(self):
        arch, model_type = get_model_and_args(tiny_config())
        self.assertEqual(model_type, "nemotron_h_twotower")
        self.assertEqual(arch.__name__, "mlx_vlm.models.nemotron_h_twotower")

        plain = tiny_config(architectures=None)
        plain.pop("auto_map", None)
        arch, model_type = get_model_and_args(plain)
        self.assertEqual(model_type, "text_only")
        self.assertEqual(arch.__name__, "mlx_vlm.models.text_only")

    def test_config_normalizes_hybrid_pattern_and_diffusion_defaults(self):
        from mlx_vlm.models.nemotron_h_twotower.config import ModelConfig

        config = ModelConfig.from_dict(tiny_config(hybrid_override_pattern="M*E"))
        self.assertEqual(config.hybrid_override_pattern, ["M", "*", "E"])
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.time_step_limit, (0.0, float("inf")))
        self.assertEqual(config.default_generation_mode, "ar")
        self.assertEqual(config.default_diffusion_steps, 16)
        self.assertEqual(config.mask_token_id, 3)

    def test_sanitize_maps_towers_experts_conv_and_time_conditioning(self):
        from mlx_vlm.models.nemotron_h_twotower import Model, ModelConfig

        model = Model(ModelConfig.from_dict(tiny_config()))
        weights = {
            "context_lm_head.weight": mx.zeros((32, 8)),
            "lm_head.weight": mx.zeros((32, 8)),
            "t_embedder.mlp.0.weight": mx.zeros((8, 256)),
            "t_embedder.mlp.2.bias": mx.zeros((8,)),
            "t_block.1.weight": mx.zeros((24, 8)),
            "scale_shift_tables.0": mx.zeros((3, 8)),
            "context_tower.layers.1.mixer.conv1d.weight": mx.zeros((16, 1, 3)),
        }
        for tower in ("context_tower", "denoiser_tower"):
            for expert in range(2):
                weights[f"{tower}.layers.0.mixer.experts.{expert}.up_proj.weight"] = (
                    mx.zeros((4, 8)) + expert
                )
                weights[f"{tower}.layers.0.mixer.experts.{expert}.down_proj.weight"] = (
                    mx.zeros((8, 4)) + expert
                )

        sanitized = model.sanitize(weights)

        self.assertIn("language_model.context_lm_head.weight", sanitized)
        self.assertIn("language_model.lm_head.weight", sanitized)
        self.assertIn("language_model.t_embedder.mlp.layers.0.weight", sanitized)
        self.assertIn("language_model.t_embedder.mlp.layers.2.bias", sanitized)
        self.assertIn("language_model.t_block.layers.1.weight", sanitized)
        self.assertIn("language_model.scale_shift_tables.0", sanitized)
        self.assertEqual(
            sanitized[
                "language_model.context_tower.layers.1.mixer.conv1d.weight"
            ].shape,
            (16, 3, 1),
        )
        for tower in ("context_tower", "denoiser_tower"):
            self.assertEqual(
                sanitized[
                    f"language_model.{tower}.layers.0.mixer.switch_mlp.fc1.weight"
                ].shape,
                (2, 4, 8),
            )
            self.assertEqual(
                sanitized[
                    f"language_model.{tower}.layers.0.mixer.switch_mlp.fc2.weight"
                ].shape,
                (2, 8, 4),
            )

    def test_tiny_attention_only_forward_and_embeddings(self):
        from mlx_vlm.models.nemotron_h_twotower import Model, ModelConfig

        model = Model(ModelConfig.from_dict(tiny_config(hybrid_override_pattern="*")))
        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)

        embeddings = model.get_input_embeddings(input_ids)
        self.assertEqual(embeddings.inputs_embeds.shape, (1, 3, 8))

        outputs = model(input_ids)
        self.assertEqual(outputs.logits.shape, (1, 3, 32))

        cache = model.make_cache()
        outputs = model(input_ids[:, :2], cache=cache)
        self.assertEqual(outputs.logits.shape, (1, 2, 32))
        outputs = model(input_ids[:, 2:], cache=cache)
        self.assertEqual(outputs.logits.shape, (1, 1, 32))

    def test_generate_mode_aliases_stay_model_local(self):
        from mlx_vlm.models.nemotron_h_twotower import Model, ModelConfig

        model = Model(ModelConfig.from_dict(tiny_config(hybrid_override_pattern="*")))
        calls = []

        def fake_ar(input_ids, max_new_tokens, **kwargs):
            calls.append(("ar", max_new_tokens))
            return mx.array([[1]], dtype=mx.int32)

        def fake_mock(input_ids, max_new_tokens, **kwargs):
            calls.append(("mock_ar", max_new_tokens))
            return mx.array([[2]], dtype=mx.int32)

        def fake_diffusion(input_ids, max_new_tokens, **kwargs):
            calls.append(("mask_diffusion", max_new_tokens, kwargs["block_size"]))
            return mx.array([[3]], dtype=mx.int32)

        model.language_model.ar_generate = fake_ar
        model.language_model.mock_ar_generate = fake_mock
        model.language_model.mask_diffusion_generate = fake_diffusion

        input_ids = mx.array([[1, 2]], dtype=mx.int32)
        model.language_model.generate(input_ids, gen_length=4)
        model.language_model.generate(
            input_ids, generation_mode="mock-ar", gen_length=5
        )
        model.language_model.generate(
            input_ids,
            generation_mode="diffusion",
            gen_length=6,
            block_length=7,
        )

        self.assertEqual(calls, [("ar", 4), ("mock_ar", 5), ("mask_diffusion", 6, 7)])

    def test_tiny_generation_modes_return_suffix_tokens(self):
        from mlx_vlm.models.nemotron_h_twotower import Model, ModelConfig

        model = Model(ModelConfig.from_dict(tiny_config(hybrid_override_pattern="*")))
        input_ids = mx.array([[1, 2, 4]], dtype=mx.int32)

        ar_tokens = model.language_model.generate(
            input_ids, generation_mode="ar", gen_length=2, eos_token_id=999
        )
        mock_tokens = model.language_model.generate(
            input_ids, generation_mode="mock_ar", gen_length=2, eos_token_id=999
        )
        diffusion_tokens = model.language_model.generate(
            input_ids,
            generation_mode="mask_diffusion",
            gen_length=2,
            block_size=2,
            steps_per_block=2,
            confidence_threshold=None,
            eos_token_id=999,
        )
        mx.eval(ar_tokens, mock_tokens, diffusion_tokens)

        self.assertEqual(ar_tokens.shape, (1, 2))
        self.assertEqual(mock_tokens.shape, (1, 2))
        self.assertEqual(diffusion_tokens.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
