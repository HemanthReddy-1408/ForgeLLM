from __future__ import annotations

import copy

import pytest
import torch

from conftest import make_tiny
from forgellm.config import LoRAConfig
from forgellm.models.adapters import (
    AdapterStore,
    adapter_parameters,
    freeze_base,
    fuse_adapters,
    inject_adapter,
    load_adapter,
    merge_and_unload,
    prepare_for_training,
    save_adapter,
    unload_adapter,
)
from forgellm.models.lora import (
    AdapterLayer,
    adapters_active,
    adapters_disabled,
    count_parameters,
    describe_candidates,
    find_candidate_modules,
    loaded_adapters,
    resolve_targets,
    set_active,
)
from forgellm.models.quantization import (
    QuantConfig,
    QuantLinear,
    bits_per_param,
    nf4_codebook,
    quantize_model,
    sqnr_db,
)
from forgellm.models.tinygpt import load_tiny, save_tiny

X = torch.randint(0, 256, (2, 14))


def randomise(model, name, std=0.05):
    for _, p in adapter_parameters(model, name):
        p.data.normal_(0, std)


def test_kv_cache_matches_full_forward(tiny):
    m = tiny.model
    full = m(X).logits
    o = m(X[:, :9], use_cache=True)
    o2 = m(X[:, 9:], past_key_values=o.past_key_values, use_cache=True)
    assert torch.allclose(torch.cat([o.logits, o2.logits], 1), full, atol=1e-5)


def test_left_padding_does_not_change_logits(tiny):
    m = tiny.model
    xp = torch.cat([torch.full((2, 4), 256), X], 1)
    am = torch.cat([torch.zeros(2, 4, dtype=torch.long), torch.ones_like(X)], 1)
    assert torch.allclose(m(xp, attention_mask=am).logits[:, 4:], m(X).logits, atol=1e-5)


def test_tinygpt_save_load_roundtrip(tiny, tmp_path):
    save_tiny(tiny.model, tmp_path / "m.pt")
    assert torch.equal(load_tiny(tmp_path / "m.pt")(X).logits, tiny.model(X).logits)


def test_candidate_discovery_and_placement(tiny):
    cands = find_candidate_modules(tiny.model)
    assert {c.leaf for c in cands} >= {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}
    assert len(resolve_targets(tiny.model, LoRAConfig(placement="qv"))) == 4
    assert len(resolve_targets(tiny.model, LoRAConfig(placement="attn_mlp"))) == 14
    assert {c.layer for c in resolve_targets(tiny.model, LoRAConfig(placement="qv", layers="last:1"))} == {1}
    with pytest.raises(ValueError):
        resolve_targets(tiny.model, LoRAConfig(placement="bogus"))
    assert "Attention" in describe_candidates(tiny.model, LoRAConfig(placement="qkvo"))


def test_rank_pattern_and_param_count(tiny):
    cfg = LoRAConfig(rank=4, placement="attn_mlp", rank_pattern={"mlp": 8})
    inject_adapter(tiny.model, "a", cfg)
    layers = {p: l for p, l in tiny.model.named_modules() if isinstance(l, AdapterLayer)}
    assert layers["model.layers.0.self_attn.q_proj"].adapters["a"].r == 4
    assert layers["model.layers.0.mlp.up_proj"].adapters["a"].r == 8
    prepare_for_training(tiny.model, ["a"])
    trainable, total = count_parameters(tiny.model)
    expect = sum(l.adapters["a"].r * (l.in_features + l.out_features) for l in layers.values())
    assert trainable == expect and trainable < total


@pytest.mark.parametrize("method,extra", [("lora", {}), ("dora", {}), ("bottleneck", {"bottleneck_dim": 8}), ("ia3", {})])
def test_every_method_is_identity_at_init_and_trains_only_adapter(tiny, method, extra):
    base = tiny.model(X).logits
    inject_adapter(tiny.model, "a", LoRAConfig(method=method, rank=4, alpha=8, dropout=0.0, placement=None if method in ("bottleneck", "ia3") else "qkvo", **extra))
    assert torch.allclose(tiny.model(X).logits, base, atol=1e-6), "adapter must start as the identity"
    prepare_for_training(tiny.model, ["a"])
    tiny.model.train()
    tiny.model(X, labels=X).loss.backward()
    assert all(p.grad is not None for _, p in adapter_parameters(tiny.model, "a") if p.requires_grad)
    base_params = [p for n, p in tiny.model.named_parameters() if ".adapters." not in n]
    assert all(p.grad is None and not p.requires_grad for p in base_params)


def test_lora_merge_equals_unmerged_and_disabled_equals_base(tiny):
    base = tiny.model(X).logits
    inject_adapter(tiny.model, "a", LoRAConfig(rank=4, alpha=8, dropout=0.0, placement="attn_mlp"))
    randomise(tiny.model, "a")
    y = tiny.model(X).logits
    assert not torch.allclose(y, base, atol=1e-3)
    with adapters_disabled(tiny.model):
        assert torch.allclose(tiny.model(X).logits, base, atol=1e-6)
    merged = merge_and_unload(copy.deepcopy(tiny.model), {"a": 1.0})
    assert not any(isinstance(m, AdapterLayer) for m in merged.modules())
    assert torch.allclose(merged(X).logits, y, atol=1e-4)


def test_dora_merge_equivalence(tiny):
    inject_adapter(tiny.model, "d", LoRAConfig(method="dora", rank=4, alpha=8, dropout=0.0, placement="qv"))
    for _, p in adapter_parameters(tiny.model, "d"):
        p.data.add_(torch.randn_like(p) * 0.02)
    y = tiny.model(X).logits
    assert torch.allclose(merge_and_unload(copy.deepcopy(tiny.model), {"d": 1.0})(X).logits, y, atol=1e-4)


def test_adapter_save_load_roundtrip_on_fresh_model(tiny, tmp_path):
    cfg = LoRAConfig(rank=4, alpha=8, dropout=0.0, placement="qkvo", rank_pattern={"q_proj": 2})
    inject_adapter(tiny.model, "a", cfg)
    randomise(tiny.model, "a")
    y = tiny.model(X).logits
    save_adapter(tiny.model, "a", tmp_path / "a", cfg, {"base_model": "tiny-test"})
    fresh = make_tiny()
    load_adapter(fresh.model, tmp_path / "a", "a", activate=True)
    assert torch.allclose(fresh.model(X).logits, y, atol=1e-6)
    ranks = {l.adapters["a"].r for _, l in fresh.model.named_modules() if isinstance(l, AdapterLayer) and ".q_proj" in _}
    assert ranks == {2}
    with pytest.raises(ValueError):                                  # an adapter must not silently load into a different shape
        load_adapter(make_tiny(layers=3).model, tmp_path / "a", "z")


def test_multi_adapter_switching_on_one_base(tiny):
    inject_adapter(tiny.model, "a", LoRAConfig(rank=4, alpha=8, dropout=0.0, placement="qv"))
    inject_adapter(tiny.model, "b", LoRAConfig(rank=4, alpha=8, dropout=0.0, placement="qv"))
    randomise(tiny.model, "a"), randomise(tiny.model, "b")
    set_active(tiny.model, {"a": 1.0}); ya = tiny.model(X).logits
    set_active(tiny.model, {"b": 1.0}); yb = tiny.model(X).logits
    set_active(tiny.model, {"a": 1.0, "b": 1.0}); yab = tiny.model(X).logits
    assert not torch.allclose(ya, yb, atol=1e-3) and not torch.allclose(yab, ya, atol=1e-3)
    with adapters_active(tiny.model, {"a": 1.0}):
        assert torch.allclose(tiny.model(X).logits, ya)
    assert loaded_adapters(tiny.model) == ["a", "b"]
    unload_adapter(tiny.model, "a")
    assert loaded_adapters(tiny.model) == ["b"]


def test_svd_fusion_reconstructs_the_weighted_sum(tiny):
    inject_adapter(tiny.model, "a", LoRAConfig(rank=3, alpha=6, dropout=0.0, placement="qv"))
    inject_adapter(tiny.model, "b", LoRAConfig(rank=3, alpha=6, dropout=0.0, placement="qv"))
    randomise(tiny.model, "a"), randomise(tiny.model, "b")
    errs = fuse_adapters(tiny.model, {"a": 0.5, "b": 0.5}, "ab", rank=6)   # rank 6 = exact for two rank-3 deltas
    assert max(errs.values()) < 1e-4
    set_active(tiny.model, {"a": 0.5, "b": 0.5}); y = tiny.model(X).logits
    set_active(tiny.model, {"ab": 1.0})
    assert torch.allclose(tiny.model(X).logits, y, atol=1e-4)


def test_adapter_store_lru_eviction_and_stacking(tiny, tmp_path):
    cfg = LoRAConfig(rank=2, alpha=4, dropout=0.0, placement="qv")
    donor = make_tiny()
    for n, stack in (("a", None), ("b", None), ("c", "a")):
        inject_adapter(donor.model, n, cfg)
        randomise(donor.model, n)
        save_adapter(donor.model, n, tmp_path / n, cfg, {"stack_on": stack})
    store = AdapterStore(tiny.model, tmp_path, max_resident=2)
    store.activate_chain("c")                      # c stacks on a -> both resident
    assert set(loaded_adapters(tiny.model)) == {"a", "c"}
    store.activate_chain("b")                      # budget 2: evicts the least recently used
    assert "b" in loaded_adapters(tiny.model) and store.evictions >= 1
    assert store.chain("c") == ["a", "c"]
    with pytest.raises(FileNotFoundError):
        store.ensure("missing")


# ---- quantization ---------------------------------------------------------------------------------------------
def test_nf4_codebook_properties():
    c = nf4_codebook()
    assert len(c) == 16 and torch.all(c[1:] > c[:-1]) and c[0] == -1 and c[-1] == 1 and (c == 0).sum() == 1
    assert (c > 0).sum() == 8 and (c < 0).sum() == 7
    assert abs(c[1].item() - (-0.6961928)) < 1e-4          # value from the QLoRA paper


def _lin(w):
    l = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    l.weight.data = w.clone()
    return l


def test_nf4_beats_uniform4_on_gaussian_weights_and_int8_beats_both():
    torch.manual_seed(0)
    w = torch.randn(256, 256) * 0.02
    s = {m: sqnr_db(w, QuantLinear.from_linear(_lin(w), QuantConfig(m)).dequantize()) for m in ("nf4", "uniform4", "int8")}
    assert s["nf4"] > s["uniform4"] and s["int8"] > s["nf4"] + 10


def test_storage_matches_bits_per_param_formula():
    w = torch.randn(512, 512)
    for dq in (True, False):
        q = QuantLinear.from_linear(_lin(w), QuantConfig("nf4", 64, dq))
        assert abs(q.storage_bytes() * 8 / w.numel() - bits_per_param(QuantConfig("nf4", 64, dq))) < 0.02
    assert abs(bits_per_param(QuantConfig("nf4", 64, True)) - 4.127) < 0.001


def test_double_quant_changes_output_only_slightly():
    w = torch.randn(256, 256) * 0.05
    a = QuantLinear.from_linear(_lin(w), QuantConfig("nf4", 64, True)).dequantize()
    b = QuantLinear.from_linear(_lin(w), QuantConfig("nf4", 64, False)).dequantize()
    assert (a - b).abs().max() < 0.01 * w.abs().max()


def test_quant_linear_backward_matches_dense_input_gradient():
    torch.manual_seed(0)
    lin = _lin(torch.randn(48, 32) * 0.1)
    q = QuantLinear.from_linear(lin, QuantConfig("nf4"))
    x1 = torch.randn(5, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    q(x1).sum().backward()
    (x2 @ q.dequantize().T).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert q.qweight.requires_grad is False


def test_quantized_model_forward_close_and_qlora_trains(tiny):
    full = tiny.model(X).logits
    rep = quantize_model(tiny.model, QuantConfig("nf4"))
    assert rep.ratio > 6 and rep.layers == 14 and any(isinstance(m, QuantLinear) for m in tiny.model.modules())
    assert (tiny.model(X).logits - full).abs().mean() < 0.1
    inject_adapter(tiny.model, "q", LoRAConfig(rank=4, placement="attn_mlp"))
    prepare_for_training(tiny.model, ["q"])
    tiny.model.train()
    tiny.model(X, labels=X).loss.backward()
    assert all(p.grad is not None for _, p in adapter_parameters(tiny.model, "q"))
    with pytest.raises(TypeError):
        merge_and_unload_into_quant = next(m for m in tiny.model.modules() if isinstance(m, AdapterLayer))
        merge_and_unload_into_quant.merge("q")


def test_dora_refuses_quantized_base(tiny):
    quantize_model(tiny.model, QuantConfig("nf4"))
    with pytest.raises(TypeError):
        inject_adapter(tiny.model, "d", LoRAConfig(method="dora", placement="qv"))
    freeze_base(tiny.model)
