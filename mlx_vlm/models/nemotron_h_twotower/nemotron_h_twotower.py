from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = "nemotron_h_twotower"
        self.language_model = LanguageModel(config)
        self._model_path = None

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        self._model_path = value

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        if pixel_values is not None:
            raise ValueError("Nemotron TwoTower is a text-only model.")
        if (
            kwargs.get("input_features") is not None
            or kwargs.get("audio_values") is not None
        ):
            raise ValueError("Nemotron TwoTower does not accept audio inputs.")
        if input_ids is None:
            raise ValueError("input_ids are required for Nemotron TwoTower.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.get_input_embeddings(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if pixel_values is not None:
            raise ValueError("Nemotron TwoTower is a text-only model.")
        input_embeddings_features = self.get_input_embeddings(input_ids, pixel_values)
        return self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            **kwargs,
        )

    def sanitize(self, weights):
        return self.language_model.sanitize(weights)

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
