# Nemotron H TwoTower

Support package for `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16`.

This is a text-only model. The default mlx-vlm generation path uses the
context tower autoregressively. Two-tower modes are opt-in through generation
kwargs:

```bash
mlx_vlm.generate \
  --model /path/to/converted-model \
  --prompt "France is a country " \
  --gen-kwargs '{"generation_mode": "mask_diffusion"}'
```

Supported model-local generation modes:

- `ar`: context tower only, one token per step.
- `mock_ar`: denoiser tower, one token per step.
- `mask_diffusion`: denoiser tower, block-wise masked diffusion.

The implementation keeps mode selection, cache choreography, and diffusion
sampling inside `mlx_vlm.models.nemotron_h_twotower.language` so shared
generation dispatch does not need model-specific branches.
