"""PPO の指数計算と非有限の損失・勾配から、方策と Adam の状態を保護する。"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from app import config, warn
from app.contracts import SimParams
from app.rl.ppo import PPOTrainer, _PendingUpdate


def _trainer() -> PPOTrainer:
    return PPOTrainer(config.OBS_DIM, config.ACTION_DIM, SimParams(), 2, seed=141, hidden_sizes=(16, 8))


def _prepare(model: PPOTrainer, split: bool = False) -> _PendingUpdate:
    obs = torch.rand((4, config.OBS_DIM), generator=torch.Generator().manual_seed(141))
    actions = torch.tensor([[0.2, 0.1], [0.1, -0.1], [0.3, 0.05], [-0.2, -0.05]])
    options = torch.arange(4)
    with torch.no_grad():
        log_probs, _, values = model.policy.evaluate(obs, actions, options)
        dist, meta_values = model.policy.meta(obs)
        data = {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "returns": values + 0.2,
            "advantages": torch.tensor([-0.5, 0.5, -1.0, 1.0]),
            "assisted": torch.zeros(4, dtype=torch.bool),
            "expert_actions": torch.zeros_like(actions),
            "options": options,
            "option_start": torch.ones(4, dtype=torch.bool),
            "meta_log_probs": dist.log_prob(options),
            "meta_values": meta_values,
            "meta_returns": meta_values + 0.3,
            "meta_advantages": torch.tensor([-1.0, 1.0, -0.5, 0.5]),
            "expert_options": torch.full((4,), -1, dtype=torch.long),
        }
        model._weights_before_update = {
            name: parameter.clone()
            for name, parameter in model.policy.named_parameters()
            if name.endswith(".weight")
        }
    schedule = [np.arange(2), np.arange(2, 4)] if split else [np.arange(4)]
    model._pending = _PendingUpdate(data=data, schedule=schedule)
    return model._pending


def _equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _equal(a, b)
    else:
        assert actual == expected


def _healthy(model: PPOTrainer, stats: dict[str, float]) -> None:
    assert all(math.isfinite(value) for value in stats.values())
    assert math.isfinite(model._last_grad_total)
    assert all(math.isfinite(value) for value in model._last_grad_norms.values())
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.policy.parameters())
    assert model.optimizer.state
    for state in model.optimizer.state.values():
        assert all(bool(torch.isfinite(value).all()) for value in state.values())


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(warn, "_WARNED", set())


@pytest.mark.parametrize("field", ["log_probs", "meta_log_probs"])
@pytest.mark.parametrize("offset", [-1000.0, 1000.0])
def test_extreme_probability_gap_keeps_the_hierarchical_policy_finite(field: str, offset: float) -> None:
    model = _trainer()
    pending = _prepare(model)
    pending.data[field] += offset

    stats = model._run_pending(1)

    assert stats is not None
    assert model.updates == 1
    _healthy(model, stats)


@pytest.mark.parametrize("field", ["returns", "meta_returns", "advantages"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_loss_preserves_existing_weights_and_adam(field: str, value: float, caplog) -> None:
    model = _trainer()
    _prepare(model)
    assert model._run_pending(1) is not None
    weights = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.optimizer.state_dict())
    grad_total = model._last_grad_total
    pending = _prepare(model, split=True)
    pending.data[field].fill_(value)
    backward_calls = []
    hook = model.policy.value_head.weight.register_hook(lambda grad: backward_calls.append(grad) or grad)

    try:
        stats = model._run_pending(2)
    finally:
        hook.remove()

    assert stats is None
    assert model._pending is None
    assert model.updates == 1
    assert model._last_grad_total == grad_total
    assert not backward_calls
    assert pending.cursor == 2
    assert not pending.policy_losses and not pending.grad_totals and not pending.grad_sums
    assert all(parameter.grad is None for parameter in model.policy.parameters())
    _equal(model.policy.state_dict(), weights)
    _equal(model.optimizer.state_dict(), optimizer)
    assert sum("PPO の損失が非有限" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("parameter_name", ["mu_head.bias", "meta_head.bias"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e30])
def test_nonfinite_gradient_preserves_all_weights_and_adam(parameter_name: str, value: float, caplog) -> None:
    model = _trainer()
    _prepare(model)
    assert model._run_pending(1) is not None
    weights = copy.deepcopy(model.policy.state_dict())
    optimizer = copy.deepcopy(model.optimizer.state_dict())
    pending = _prepare(model, split=True)
    parameter = dict(model.policy.named_parameters())[parameter_name]
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, value))

    try:
        stats = model._run_pending(2)
    finally:
        hook.remove()

    assert stats is None
    assert model._pending is None
    assert model.updates == 1
    assert not pending.grad_sums and not pending.grad_counts and not pending.grad_totals
    assert not pending.meta_policy_losses and not pending.meta_value_losses
    assert all(parameter.grad is None for parameter in model.policy.parameters())
    _equal(model.policy.state_dict(), weights)
    _equal(model.optimizer.state_dict(), optimizer)
    assert sum("PPO の勾配が非有限" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_split_update_resumes_with_the_next_healthy_minibatch(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setattr(config, "PPO_MINIBATCHES_PER_STEP", 1)
    model = _trainer()
    reference = _trainer()
    pending = _prepare(model, split=True)
    expected = _prepare(reference, split=True)
    expected.schedule = expected.schedule[1:]
    last_obs = np.zeros((2, config.OBS_DIM), dtype=np.float32)
    last_active = np.ones(2, dtype=bool)
    hook = None
    if failure == "loss":
        pending.data["returns"][:2] = float("nan")
        pending.data["assisted"][:2] = True
        pending.data["expert_options"][:2] = 0
    else:
        hook = model.policy.meta_head.bias.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    try:
        assert model.maybe_update(last_obs, last_active) is None
    finally:
        if hook is not None:
            hook.remove()

    assert model.updates == 0
    assert pending.cursor == 1 and model._pending is pending
    assert not pending.bc_losses and not pending.grad_sums
    assert all(parameter.grad is None for parameter in model.policy.parameters())
    stats = model.maybe_update(last_obs, last_active)
    expected_stats = reference.maybe_update(last_obs, last_active)

    assert stats is not None and expected_stats is not None
    assert model.updates == 1 and model._pending is None
    _healthy(model, stats)
    _equal(stats, expected_stats)
    _equal(model.policy.state_dict(), reference.policy.state_dict())
    _equal(model.optimizer.state_dict(), reference.optimizer.state_dict())
    _equal(model._last_grad_norms, reference._last_grad_norms)
    assert model._last_grad_total == reference._last_grad_total


def test_gradient_diagnostics_measure_the_norm_before_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PPO_MAX_GRAD_NORM", 0.01)
    model = _trainer()
    _prepare(model)

    stats = model._run_pending(1)

    assert stats is not None
    assert model._last_grad_total > config.PPO_MAX_GRAD_NORM
    assert max(model._last_grad_norms.values()) > config.PPO_MAX_GRAD_NORM
    assert model._last_grad_clip_rate == 1.0
    total = torch.linalg.vector_norm(torch.stack([parameter.grad.norm() for parameter in model.policy.parameters() if parameter.grad is not None]))
    assert float(total) <= config.PPO_MAX_GRAD_NORM
