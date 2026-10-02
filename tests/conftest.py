from __future__ import annotations

import pytest
import torch

from forgellm.config import DataConfig
from forgellm.data.pipeline import run_pipeline
from forgellm.models.base import LoadedModel, ModelSpec
from forgellm.models.tinygpt import TinyGPT, TinyGPTConfig
from forgellm.models.tokenizer import ByteTokenizer


def make_tiny(layers: int = 2, seed: int = 0) -> LoadedModel:
    torch.manual_seed(seed)
    m = TinyGPT(TinyGPTConfig(hidden_size=64, num_layers=layers, num_heads=4, num_kv_heads=2, intermediate_size=176, max_position=1024)).eval()
    m.requires_grad_(False)
    spec = ModelSpec("tiny-test", "tinygpt", m.num_parameters(), 64, layers, 1024, "float32", "cpu", "none", 0)
    return LoadedModel(m, ByteTokenizer(), spec, torch.device("cpu"))


@pytest.fixture()
def tiny() -> LoadedModel:
    return make_tiny()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    cfg = DataConfig(per_task={"technical_qa": 400, "reasoning": 400, "coding": 400, "extraction": 400, "tool_use": 500,
                               "grounded_qa": 300, "safety": 80})
    run_pipeline(cfg, d, verbose=False)
    return d
