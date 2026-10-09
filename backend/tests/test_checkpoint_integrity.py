"""不正モデルの拒否と、現在・旧形式の再開時の学習状態を検証する。"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app import config
from app.contracts import SimParams
from app.rl.hierarchical_policy import META_MODULES, NUM_OPTIONS
from app.rl.importer import CheckpointImportError, inspect_checkpoint
from app.rl.ppo import PPOTrainer


def trainer(seed: int = 7) -> PPOTrainer:
    return PPOTrainer(config.OBS_DIM, config.ACTION_DIM, SimParams(), 2, seed=seed, hidden_sizes=(16, 8))


def update(model: PPOTrainer) -> None:
    model.optimizer.zero_grad(set_to_none=True)
    loss = sum((parameter.square().sum() + parameter.sum() * 0.01) for parameter in model.policy.parameters())
    loss.backward()
    model.optimizer.step()


def payload(model: PPOTrainer, path: Path) -> dict:
    model.save(path)
    return torch.load(path, weights_only=True, map_location="cpu")


def equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            equal(a, e)
    else:
        assert actual == expected


def inspect(path: Path) -> None:
    inspect_checkpoint(path, expected_obs_dim=config.OBS_DIM, expected_action_dim=config.ACTION_DIM, expected_hidden_sizes=(16, 8))


@pytest.mark.parametrize("field", ["policy", "exp_avg", "exp_avg_sq", "step"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_model_is_rejected_atomically(tmp_path: Path, field: str, value: float) -> None:
    model = trainer()
    update(model)
    path = tmp_path / "invalid.pt"
    raw = payload(model, path)
    if field == "policy":
        raw["policy"]["policy_trunk.0.weight"][0, 0] = value
    else:
        next(iter(raw["optimizer"]["state"].values()))[field].fill_(value)
    torch.save(raw, path)
    before = model.snapshot_state()
    original_policy, original_optimizer = model.policy, model.optimizer
    model.buffer.ptr = 1
    pending = object()
    model._pending = pending
    model._last_raw_actions = np.ones((2, config.ACTION_DIM), dtype=np.float32)
    actions = model._last_raw_actions
    model.widened_from = 79
    model.upgraded_flat = True
    model.scheduler.age[:] = 5
    rng = torch.get_rng_state().clone()
    with pytest.raises(CheckpointImportError, match="NaN|無限大|更新回数"):
        inspect(path)
    assert not model.load(path)
    assert model.policy is original_policy and model.optimizer is original_optimizer
    equal(model.snapshot_state(), before)
    assert model.buffer.ptr == 1 and model._pending is pending
    assert model._last_raw_actions is actions
    assert model.widened_from == 79 and model.upgraded_flat
    np.testing.assert_array_equal(model.scheduler.age, [5, 5])
    assert torch.equal(torch.get_rng_state(), rng)
    model._pending = None
    update(model)
    assert all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())


@pytest.mark.parametrize("corruption", [
    "policy_shape", "policy_dtype", "policy_missing", "policy_extra", "policy_sparse", "policy_value",
    "optimizer_type", "optimizer_shape", "optimizer_missing", "optimizer_group", "optimizer_ids",
    "optimizer_step_shape", "optimizer_negative_square", "optimizer_step_string",
    "obs_inf", "action_nan", "hidden_inf", "hidden_fraction", "updates_inf", "updates_negative",
])
def test_malformed_checkpoint_rejected_by_both_entry_points(tmp_path: Path, corruption: str) -> None:
    model = trainer()
    update(model)
    path = tmp_path / "invalid.pt"
    raw = payload(model, path)
    state = next(iter(raw["optimizer"]["state"].values()))
    if corruption == "policy_shape":
        raw["policy"]["log_std"] = torch.zeros(9)
    elif corruption == "policy_dtype":
        raw["policy"]["log_std"] = raw["policy"]["log_std"].double()
    elif corruption == "policy_missing":
        raw["policy"].pop("log_std")
    elif corruption == "policy_extra":
        raw["policy"]["extra"] = torch.zeros(1)
    elif corruption == "policy_sparse":
        raw["policy"]["log_std"] = raw["policy"]["log_std"].to_sparse()
    elif corruption == "policy_value":
        raw["policy"]["log_std"] = [0.0, 0.0]
    elif corruption == "optimizer_type":
        raw["optimizer"] = []
    elif corruption == "optimizer_shape":
        state["exp_avg"] = torch.zeros(9)
    elif corruption == "optimizer_missing":
        state.pop("exp_avg_sq")
    elif corruption == "optimizer_group":
        raw["optimizer"]["param_groups"][0]["betas"] = (float("inf"), 0.999)
    elif corruption == "optimizer_ids":
        raw["optimizer"]["param_groups"][0]["params"][0] = 999
    elif corruption == "optimizer_step_shape":
        state["step"] = torch.zeros(1)
    elif corruption == "optimizer_negative_square":
        state["exp_avg_sq"].fill_(-1)
    elif corruption == "optimizer_step_string":
        state["step"] = "1"
    else:
        key, kind = corruption.split("_")
        key = {"obs": "obs_dim", "action": "action_dim", "hidden": "hidden_sizes"}.get(key, key)
        value = {"inf": float("inf"), "nan": float("nan"), "fraction": 16.5, "negative": -1}[kind]
        raw[key] = [value, 8] if key == "hidden_sizes" else value
    torch.save(raw, path)
    before = model.snapshot_state()
    with pytest.raises(CheckpointImportError):
        inspect(path)
    assert not model.load(path)
    equal(model.snapshot_state(), before)


def legacy_payload(raw: dict, obs_dim: int, flat: bool) -> dict:
    raw = copy.deepcopy(raw)
    raw["obs_dim"] = obs_dim
    state = raw["policy"]
    if flat:
        raw["format"] = 1
        raw["policy"] = state = {key: value for key, value in state.items() if not key.startswith(META_MODULES) and key != "option_bias"}
    for key in ("policy_trunk.0.weight", "value_trunk.0.weight", "meta_trunk.0.weight", "meta_value_trunk.0.weight"):
        if key not in state:
            continue
        value = state[key]
        state[key] = value[:, :obs_dim].clone()
        if not flat and not key.startswith("meta_"):
            state[key] = torch.cat([state[key], value[:, -NUM_OPTIONS:]], dim=1)
    raw.pop("optimizer", None)
    return raw


@pytest.mark.parametrize("obs_dim", [66, 75, 79, 87, config.OBS_DIM])
@pytest.mark.parametrize("flat", [False, True])
def test_migrations_reset_adam_and_resume_independently_of_prior_training(tmp_path: Path, obs_dim: int, flat: bool) -> None:
    source = trainer()
    update(source)
    path = tmp_path / "legacy.pt"
    raw = legacy_payload(payload(source, path), obs_dim, flat)
    raw["updates"] = 4321
    torch.save(raw, path)
    fresh, trained = trainer(), trainer()
    update(trained)
    update(trained)
    trained.apply_params(SimParams(learning_rate=0.0009))
    fresh.apply_params(SimParams(learning_rate=0.0009))
    assert trained.optimizer.state
    rng = torch.get_rng_state().clone()
    inspect(path)
    assert fresh.load(path) and trained.load(path)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not fresh.optimizer.state and not trained.optimizer.state
    assert trained.optimizer.param_groups[0]["lr"] == trained.learning_rate
    assert trained.updates == 4321
    assert all(p.grad is None for p in trained.policy.parameters())
    equal(fresh.policy.state_dict(), trained.policy.state_dict())
    observation = torch.zeros((3, config.OBS_DIM))
    for intent in range(NUM_OPTIONS):
        equal(fresh.policy.mean_action(observation, torch.full((3,), intent)), trained.policy.mean_action(observation, torch.full((3,), intent)))
    update(fresh)
    update(trained)
    equal(fresh.snapshot_state(), trained.snapshot_state())


def test_current_checkpoint_restores_adam_and_uses_runtime_learning_rate(tmp_path: Path) -> None:
    source = trainer()
    update(source)
    path = tmp_path / "current.pt"
    raw = payload(source, path)
    raw["hidden_sizes"] = ["16", "8"]
    raw["obs_dim"] = str(config.OBS_DIM)
    raw["updates"] = "17"
    torch.save(raw, path)
    target = trainer(9)
    target.apply_params(SimParams(learning_rate=0.0008))
    inspect(path)
    assert target.load(path)
    expected = copy.deepcopy(raw["optimizer"])
    expected["param_groups"][0]["lr"] = target.learning_rate
    equal(target.optimizer.state_dict(), expected)
    equal(target.policy.state_dict(), source.policy.state_dict())
    source.apply_params(SimParams(learning_rate=0.0008))
    update(source)
    update(target)
    equal(target.policy.state_dict(), source.policy.state_dict())


def test_optimizer_is_checked_even_when_it_will_be_discarded(tmp_path: Path) -> None:
    source = PPOTrainer(87, config.ACTION_DIM, SimParams(), 2, hidden_sizes=(16, 8))
    update(source)
    path = tmp_path / "old.pt"
    raw = payload(source, path)
    next(iter(raw["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
    torch.save(raw, path)
    with pytest.raises(CheckpointImportError):
        inspect(path)
    assert not trainer().load(path)


@pytest.mark.parametrize("obs_dim", [87, config.OBS_DIM])
def test_valid_adam_is_discarded_on_observation_migration(tmp_path: Path, obs_dim: int) -> None:
    source = PPOTrainer(obs_dim, config.ACTION_DIM, SimParams(), 2, hidden_sizes=(16, 8))
    update(source)
    path = tmp_path / "model.pt"
    source.save(path)
    target = trainer()
    update(target)
    inspect(path)
    assert target.load(path)
    assert bool(target.optimizer.state) == (obs_dim == config.OBS_DIM)
    update(target)


def test_engine_rejects_before_backup_and_preserves_production_checkpoint(tmp_path: Path, monkeypatch) -> None:
    from app.runtime.engine import ImportTicket, SimulationEngine
    from app.rl import export

    model = trainer()
    update(model)
    production, upload = tmp_path / "production.pt", tmp_path / "upload.pt"
    raw = payload(model, production)
    before_file = production.read_bytes()
    raw["policy"]["log_std"].fill_(float("nan"))
    torch.save(raw, upload)
    monkeypatch.setattr(config, "CHECKPOINT_PATH", production)

    def unexpected_backup(*args, **kwargs):
        pytest.fail("拒否するファイルの退避・適用・保存を行ってはいけません")

    monkeypatch.setattr(export, "export_model", unexpected_backup)
    engine = SimpleNamespace(_trainer=model, _autotune=SimpleNamespace(active=False))
    ticket = ImportTicket(upload)
    before = model.snapshot_state()
    SimulationEngine._handle_import(engine, ticket)
    assert ticket.done.is_set() and ticket.error and ticket.info is None and ticket.backup is None
    assert not upload.exists()
    assert production.read_bytes() == before_file
    equal(model.snapshot_state(), before)
    observation = np.zeros((2, config.OBS_DIM), dtype=np.float32)
    actions, log_probs, values = model.act(observation, np.ones(2, dtype=bool))
    assert all(np.isfinite(array).all() for array in (actions, log_probs, values))
    update(model)


def test_flat_checkpoint_with_adam_discards_legacy_statistics(tmp_path: Path) -> None:
    source = trainer()
    path = tmp_path / "flat.pt"
    raw = legacy_payload(payload(source, path), 87, True)
    names = [name for name, _ in source.policy.named_parameters() if name in raw["policy"]]
    parameters = [torch.nn.Parameter(raw["policy"][name].clone()) for name in names]
    optimizer = torch.optim.Adam(parameters, lr=0.001, eps=1e-5)
    sum(p.square().sum() for p in parameters).backward()
    optimizer.step()
    raw["optimizer"] = optimizer.state_dict()
    torch.save(raw, path)
    target = trainer()
    update(target)
    inspect(path)
    assert target.load(path) and not target.optimizer.state
    update(target)


def test_explicit_missing_optimizer_starts_fresh_and_optional_map_metadata_is_safe(tmp_path: Path) -> None:
    model = trainer()
    update(model)
    path = tmp_path / "model.pt"
    raw = payload(model, path)
    raw["optimizer"] = None
    raw["metadata"] = {"map": "invalid"}
    torch.save(raw, path)
    info = inspect_checkpoint(path, expected_obs_dim=config.OBS_DIM, expected_action_dim=config.ACTION_DIM, expected_hidden_sizes=(16, 8))
    assert info.to_wire()["presetId"] is None
    assert model.load(path) and not model.optimizer.state
