from __future__ import annotations

import copy
import math

import pytest
import torch

from conftest import make_tiny
from forgellm.config import ForgeConfig, LoRAConfig, TrainConfig
from forgellm.data.pipeline import load_split
from forgellm.data.schemas import Example
from forgellm.data.templates import (
    Collator,
    LengthGroupedSampler,
    PreferenceCollator,
    PreferenceDataset,
    SFTDataset,
)
from forgellm.models.adapters import adapter_parameters, inject_adapter, prepare_for_training
from forgellm.training.losses import (
    CausalLMLoss,
    PreferenceLoss,
    dpo_terms,
    ipo_terms,
    orpo_terms,
    sequence_logps,
    simpo_terms,
)
from forgellm.training.optimizers import AdamW, build_param_groups
from forgellm.training.schedulers import lr_factor
from forgellm.training.sft import train_sft
from forgellm.training.trainer import Trainer


def test_adamw_matches_torch_reference():
    torch.manual_seed(0)
    a = torch.nn.Linear(6, 3)
    b = copy.deepcopy(a)
    oa = AdamW(a.parameters(), lr=1e-2, weight_decay=0.1)
    ob = torch.optim.AdamW(b.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(5):
        x = torch.randn(4, 6)
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            m(x).pow(2).sum().backward()
            o.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_lr_schedule_shapes():
    f = lambda s, k="cosine": lr_factor(s, 100, 10, k, 0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)            # linear warmup
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(0.1)          # cosine peak -> floor
    assert f(55) == pytest.approx(0.55, abs=0.01)                                # midpoint of the cosine
    assert f(60, "constant") == 1.0 and f(100, "linear") == pytest.approx(0.1)
    assert f(50, "wsd") == 1.0 and f(100, "wsd") == pytest.approx(0.1)


def test_param_groups_lora_plus_and_no_decay(tiny):
    inject_adapter(tiny.model, "a", LoRAConfig(rank=4, placement="qv"))
    prepare_for_training(tiny.model, ["a"])
    cfg = TrainConfig(learning_rate=1e-3, weight_decay=0.1)
    g = build_param_groups(tiny.model, cfg, lora_plus_ratio=8.0)
    assert {round(x["lr"], 6) for x in g} == {0.001, 0.008} and all(x["weight_decay"] == 0 for x in g)
    with pytest.raises(ValueError):
        build_param_groups(make_tiny().model, cfg)                                # nothing trainable


def test_dpo_orpo_simpo_loss_values():
    z = torch.zeros(3)
    assert dpo_terms(z, z, z, z, 0.1).mean() == pytest.approx(math.log(2))        # policy == reference -> log 2
    good = dpo_terms(torch.tensor([2.0]), torch.tensor([-2.0]), z[:1], z[:1], 0.5)
    bad = dpo_terms(torch.tensor([-2.0]), torch.tensor([2.0]), z[:1], z[:1], 0.5)
    assert good < bad
    assert ipo_terms(z, z, z, z, 0.5).mean() == pytest.approx(1.0)                # (0 - 1/(2*0.5))^2
    lc, lr = torch.tensor([-0.5]), torch.tensor([-2.0])
    assert orpo_terms(lc, lr, 0.1) < orpo_terms(lr, lc, 0.1)
    assert simpo_terms(lc, lr, 2.0, 0.5) < simpo_terms(lr, lc, 2.0, 0.5)


def _sft_batch(tok, n=4):
    exs = [Example(f"What is item {i}?", f"Item {i} is number {i * 7}.", "technical_qa") for i in range(n)]
    ds = SFTDataset(exs, tok, 256)
    return ds, Collator(tok.pad_id)([ds[i] for i in range(n)])


def test_sft_loss_matches_manual_cross_entropy_on_supervised_positions(tiny):
    ds, batch = _sft_batch(tiny.tokenizer)
    loss_sum, met = CausalLMLoss()(tiny.model, batch)
    logits = tiny.model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    ref = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), batch["labels"][:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    assert torch.allclose(loss_sum, ref, atol=1e-3) and met["tokens"] == CausalLMLoss().units(batch)


def test_sequence_logps_sum_to_negative_ce(tiny):
    _, batch = _sft_batch(tiny.tokenizer)
    sums, cnt = sequence_logps(tiny.model, batch)
    total, _ = CausalLMLoss()(tiny.model, batch)
    assert torch.allclose(-sums.sum(), total, atol=1e-3) and cnt.sum() == CausalLMLoss().units(batch)


def test_length_grouped_sampler_is_deterministic_and_covers_data():
    s = LengthGroupedSampler(list(range(100)), 8, seed=3)
    b1, b2 = s.batches(), LengthGroupedSampler(list(range(100)), 8, seed=3).batches()
    assert b1 == b2 and all(len(b) == 8 for b in b1)
    s.set_epoch(1)
    assert s.batches() != b1
    flat = [i for b in b1 for i in b]
    assert len(set(flat)) == len(flat)


def _trainer(tiny, tmp_path, steps=6, accum=1, bs=4, seed=0, lr=5e-3):
    ds, _ = _sft_batch(tiny.tokenizer, 16)
    inject_adapter(tiny.model, "a", LoRAConfig(rank=4, alpha=8, dropout=0.0, placement="attn_mlp"))
    prepare_for_training(tiny.model, ["a"])
    cfg = TrainConfig(learning_rate=lr, batch_size=bs, gradient_accumulation=accum, max_steps=steps, warmup_ratio=0.0,
                      eval_every=0, save_every=3, log_every=1, seed=seed, length_grouped=False, max_grad_norm=1.0)
    return Trainer(tiny.model, cfg, ds, Collator(tiny.tokenizer.pad_id), CausalLMLoss(), tmp_path, tiny.device), cfg


def test_training_reduces_loss_updates_only_adapters(tiny, tmp_path):
    frozen = {n: p.clone() for n, p in tiny.model.named_parameters()}     # names before the adapter wrappers exist
    tr, _ = _trainer(tiny, tmp_path, steps=40, lr=3e-2)
    s = tr.fit(resume=False)
    assert s["final_train_loss"] < s["first_train_loss"] * 0.92 and s["skipped_steps"] == 0
    for n, p in tiny.model.named_parameters():
        if ".adapters." not in n:
            assert torch.equal(p, frozen[n.replace(".base.", ".")]), f"frozen base weight {n} changed"


def test_gradient_accumulation_matches_big_batch(tmp_path):
    def run(bs, accum):
        t = make_tiny(seed=1)
        ds, _ = _sft_batch(t.tokenizer, 16)
        inject_adapter(t.model, "a", LoRAConfig(rank=4, alpha=8, dropout=0.0, placement="qv"))
        torch.manual_seed(5)
        for _, p in adapter_parameters(t.model, "a"):
            p.data.normal_(0, 0.02)
        prepare_for_training(t.model, ["a"])
        cfg = TrainConfig(learning_rate=1e-2, batch_size=bs, gradient_accumulation=accum, max_steps=1, warmup_ratio=0, eval_every=0,
                          save_every=0, log_every=1, length_grouped=False, max_grad_norm=0)
        tr = Trainer(t.model, cfg, ds, Collator(t.tokenizer.pad_id), CausalLMLoss(), tmp_path / f"{bs}x{accum}", t.device)
        # force identical micro-batch composition: same 8 examples in the same order
        tr._micro_batch = lambda i, ds=ds, bs=bs: Collator(t.tokenizer.pad_id)([ds[j] for j in range(i * bs, i * bs + bs)])  # type: ignore
        tr.fit(resume=False)
        return torch.cat([p.detach().flatten() for _, p in adapter_parameters(t.model, "a")])
    assert torch.allclose(run(8, 1), run(4, 2), atol=1e-5)    # token-weighted accumulation == one big batch


def test_checkpoint_resume_is_exact(tmp_path):
    def final(dir, interrupt):
        t = make_tiny(seed=2)
        tr, _ = _trainer(t, dir, steps=6)
        if interrupt:
            tr.total_steps = 3
            tr.fit(resume=False)
            t2 = make_tiny(seed=2)
            tr2, _ = _trainer(t2, dir, steps=6)
            tr2.fit(resume=True)
            t = t2
        else:
            tr.fit(resume=False)
        return torch.cat([p.detach().flatten() for _, p in adapter_parameters(t.model, "a")])
    assert torch.allclose(final(tmp_path / "x", False), final(tmp_path / "y", True), atol=1e-6)


def _nan_once_loss(orig):
    class Bad(CausalLMLoss):
        n = 0

        def __call__(self, m, b):
            Bad.n += 1
            loss, met = orig(m, b)
            return (loss * float("nan") if Bad.n == 1 else loss), met
    return Bad()


def test_non_finite_step_is_recomputed_not_lost(tiny, tmp_path, monkeypatch):
    tr, _ = _trainer(tiny, tmp_path, steps=3)
    tr.loss_fn = _nan_once_loss(tr.loss_fn)
    cleared = []
    import forgellm.training.trainer as T
    monkeypatch.setattr(T, "free_device_cache", lambda dev: cleared.append(dev))
    s = tr.fit(resume=False)
    assert s["nan_retries"] == 1 and s["skipped_steps"] == 0 and tr.step == 3
    assert len(cleared) == 1                       # the device cache is cleared before the step is recomputed


def test_persistent_non_finite_step_is_skipped_and_run_aborts_after_five(tiny, tmp_path):
    tr, _ = _trainer(tiny, tmp_path, steps=3)
    tr.cfg.nan_retries = 0
    tr.loss_fn = _nan_once_loss(tr.loss_fn)
    assert tr.fit(resume=False)["skipped_steps"] == 1
    tr2, _ = _trainer(make_tiny(seed=3), tmp_path / "b", steps=8)
    tr2.cfg.nan_retries = 0

    class Always(CausalLMLoss):
        def __call__(self, m, b):
            loss, met = super().__call__(m, b)
            return loss * float("nan"), met
    tr2.loss_fn = Always()
    with pytest.raises(FloatingPointError):
        tr2.fit(resume=False)


def test_preference_training_improves_margin_and_reference_is_frozen(tiny, tmp_path):
    tok = tiny.tokenizer
    pairs = [{"instruction": f"Say number {i}", "input": "", "chosen": f"The number is {i}.", "rejected": "I refuse to answer.", "task_type": "technical_qa"} for i in range(16)]
    ds = PreferenceDataset(pairs, tok, 256)
    inject_adapter(tiny.model, "p", LoRAConfig(rank=4, alpha=8, dropout=0.0, placement="attn_mlp"))
    prepare_for_training(tiny.model, ["p"])
    cfg = TrainConfig(learning_rate=1e-2, batch_size=4, max_steps=20, warmup_ratio=0, eval_every=0, save_every=0, log_every=5, beta=0.5,
                      length_grouped=False)
    tr = Trainer(tiny.model, cfg, ds, PreferenceCollator(tok.pad_id), PreferenceLoss(cfg, "dpo"), tmp_path, tiny.device)
    s = tr.fit(resume=False)
    first, last = tr.history[0], tr.history[-1]
    assert last["pref_acc"] >= first["pref_acc"] and last["margin"] > first["margin"] and s["final_train_loss"] < math.log(2)


@pytest.mark.slow
def test_end_to_end_sft_on_pipeline_data_saves_adapter(data_dir, tmp_path, monkeypatch):
    import forgellm.config as C
    import forgellm.training.sft as S
    monkeypatch.setattr(S, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(C, "ARTIFACTS", tmp_path)
    monkeypatch.setenv("FORGELLM_POSTGRES_DSN", "")
    from forgellm import store as st
    monkeypatch.setattr(st, "_STORE", st.Store(dsn="", sqlite_path=tmp_path / "t.db"))
    from forgellm.config import load_config
    cfg = load_config("model.yaml", "lora.yaml", "training.yaml", "tiny.yaml", overrides=["training.max_steps=6", "training.tasks=[extraction]", "training.eval_every=3", "training.save_every=0", "training.run_name=e2e"])
    loaded = make_tiny()
    res = train_sft(cfg, "e2e-adapter", loaded=loaded, data_dir=data_dir)
    assert (tmp_path / "adapters" / "e2e-adapter" / "adapter.safetensors").exists() and res.version == 1
    assert isinstance(cfg, ForgeConfig) and load_split("val", data_dir)


def test_supervised_gather_matches_boolean_mask_reference(tiny):
    """The index_select gather (used because masked-indexing backward is unreliable on MPS) must equal the obvious version."""
    from forgellm.data.templates import IGNORE
    from forgellm.training.losses import supervised_logits
    _, batch = _sft_batch(tiny.tokenizer)
    logits, tgt, rows = supervised_logits(tiny.model, batch)
    full = tiny.model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    mask = batch["labels"][:, 1:] != IGNORE
    assert torch.allclose(logits, full[mask], atol=1e-5)
    assert torch.equal(tgt, batch["labels"][:, 1:][mask])
    assert torch.equal(rows, torch.arange(mask.size(0))[:, None].expand_as(mask)[mask])


def test_config_coerces_scientific_notation_strings():
    from forgellm.config import load_config
    cfg = load_config("model.yaml", "training.yaml", overrides=["training.learning_rate=5e-5", "training.beta=1e-1"])
    assert cfg.training.learning_rate == 5e-5 and isinstance(cfg.training.learning_rate, float) and cfg.training.beta == 0.1
