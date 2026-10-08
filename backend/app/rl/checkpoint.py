"""チェックポイントの定義・重み・Adam 状態を適用前に検証する。"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import torch

from app import config
from app.rl.hierarchical_policy import META_MODULES, NUM_OPTIONS
from app.rl.policy import ActorCritic

CHECKPOINT_FORMAT = "autoware-sim-ppo-2"
FLAT_CHECKPOINT_FORMATS = frozenset({"autoware-sim-ppo-1", 1})
KNOWN_CHECKPOINT_FORMATS = frozenset({CHECKPOINT_FORMAT}) | FLAT_CHECKPOINT_FORMATS
_INPUT_WEIGHTS = ("policy_trunk.0.weight", "value_trunk.0.weight")
_META_INPUT_WEIGHTS = ("meta_trunk.0.weight", "meta_value_trunk.0.weight")
logger = logging.getLogger(__name__)


class CheckpointValidationError(ValueError):
    """適用できないチェックポイントの検証結果。"""


@dataclass(frozen=True)
class PreparedCheckpoint:
    """現在の学習器から独立した、検証済みの読み込み先。"""

    policy: ActorCritic
    optimizer: torch.optim.Adam
    obs_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...]
    updates: int
    has_optimizer: bool
    flat: bool


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("整数ではありません")
    result = int(value)
    if isinstance(value, float) and value != result:
        raise ValueError("整数ではありません")
    return result


def checkpoint_definition(
    payload: Any, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]
) -> tuple[int, int, tuple[int, ...], int]:
    """モデル定義を正規化し、現行または移行可能な構成か調べる。"""
    if not isinstance(payload, dict):
        raise CheckpointValidationError("チェックポイントの形式が違います（辞書ではありません）")
    missing = [key for key in ("obs_dim", "action_dim", "hidden_sizes", "policy") if key not in payload]
    if missing:
        raise CheckpointValidationError(f"必要な項目が入っていません: {', '.join(missing)}")
    try:
        saved_obs = _integer(payload["obs_dim"])
        saved_action = _integer(payload["action_dim"])
        raw_hidden = payload["hidden_sizes"]
        if not isinstance(raw_hidden, (list, tuple)):
            raise ValueError("層構成ではありません")
        saved_hidden = tuple(_integer(h) for h in raw_hidden)
        updates = _integer(payload.get("updates", 0))
        if updates < 0:
            raise ValueError("更新回数が負です")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointValidationError("チェックポイントのモデル定義が壊れています") from exc
    widenable = saved_obs in config.OBS_WIDENABLE_DIMS and obs_dim == config.OBS_DIM
    if saved_obs != obs_dim and not widenable:
        raise CheckpointValidationError(f"観測ベクトルの次元が違います（ファイル: {saved_obs} / このアプリ: {obs_dim}）")
    if saved_action != action_dim:
        raise CheckpointValidationError(f"行動の次元が違います（ファイル: {saved_action} / このアプリ: {action_dim}）")
    if saved_hidden != tuple(hidden_sizes):
        raise CheckpointValidationError(f"ネットワークの層構成が違います（ファイル: {list(saved_hidden)} / このアプリ: {list(hidden_sizes)}）")
    fmt = payload.get("format")
    if not isinstance(fmt, (str, int)) or fmt not in KNOWN_CHECKPOINT_FORMATS:
        logger.warning("見覚えのないチェックポイント形式です: %r", fmt)
    return saved_obs, saved_action, saved_hidden, updates


def widen_observation(state: dict[str, torch.Tensor], saved_obs: int, obs_dim: int) -> dict[str, torch.Tensor]:
    """観測の末尾と意図の one-hot の間へゼロの列を足す。"""
    out = dict(state)
    keys = [*_INPUT_WEIGHTS, *(key for key in _META_INPUT_WEIGHTS if key in out)]
    for key in keys:
        weight = out[key]
        pad = weight.new_zeros((weight.shape[0], obs_dim - saved_obs))
        out[key] = torch.cat([weight[:, :saved_obs], pad, weight[:, saved_obs:]], dim=1)
    return out


def is_flat_state(state: dict[str, torch.Tensor]) -> bool:
    """上位方策を持たない旧モデルか調べる。"""
    return not any(key.startswith(META_MODULES) for key in state)


def upgrade_flat_state(state: dict[str, torch.Tensor], fresh: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """旧方策へ意図のゼロ入力と初期状態の上位方策を加える。"""
    out = dict(state)
    for key in _INPUT_WEIGHTS:
        weight = out[key]
        out[key] = torch.cat([weight, weight.new_zeros((weight.shape[0], NUM_OPTIONS))], dim=1)
    out["option_bias"] = torch.zeros_like(fresh["option_bias"])
    out.update({key: value.clone() for key, value in fresh.items() if key.startswith(META_MODULES)})
    return out


def _tensor(value: Any, shape: torch.Size, dtype: torch.dtype, label: str) -> None:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise CheckpointValidationError(f"{label} のテンソル形式が違います")
    if value.shape != shape or value.dtype != dtype:
        raise CheckpointValidationError(f"{label} の形または型が違います")
    if not bool(torch.isfinite(value).all()):
        raise CheckpointValidationError(f"{label} に NaN または無限大が含まれています")


def _optimizer_state(raw: Any, parameters: list[torch.Tensor]) -> None:
    if not isinstance(raw, dict) or not isinstance(raw.get("state"), dict):
        raise CheckpointValidationError("Adam の状態の形式が違います")
    groups = raw.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        raise CheckpointValidationError("Adam のパラメータ群の形式が違います")
    group = groups[0]
    ids = group.get("params")
    if (not isinstance(ids, list) or len(ids) != len(parameters)
            or any(type(key) is not int for key in ids) or len(set(ids)) != len(ids)):
        raise CheckpointValidationError("Adam のパラメータの対応が違います")
    if any(type(key) is not int or key not in ids for key in raw["state"]):
        raise CheckpointValidationError("Adam に対応しないパラメータが含まれています")
    try:
        for key, lower, strict in (("lr", 0, False), ("eps", 0, True), ("weight_decay", 0, False)):
            value = group[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(key)
            if value < lower or (strict and value == lower):
                raise ValueError(key)
        betas = group["betas"]
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas")
        if any(isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b < 1 for b in betas):
            raise ValueError("betas")
        if type(group["amsgrad"]) is not bool:
            raise ValueError("amsgrad")
        for key in ("maximize", "capturable", "differentiable", "decoupled_weight_decay"):
            if key in group and type(group[key]) is not bool:
                raise ValueError(key)
        for key in ("foreach", "fused"):
            if group.get(key) is not None and type(group[key]) is not bool:
                raise ValueError(key)
        if group.get("capturable") or group.get("differentiable"):
            raise ValueError("CPU で使えない Adam の設定")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CheckpointValidationError("Adam の設定が壊れています") from exc
    for key, parameter in zip(ids, parameters):
        state = raw["state"].get(key, {})
        if not isinstance(state, dict):
            raise CheckpointValidationError("Adam の状態の形式が違います")
        if not state:
            continue
        required = {"step", "exp_avg", "exp_avg_sq"}
        if group["amsgrad"]:
            required.add("max_exp_avg_sq")
        if set(state) != required:
            raise CheckpointValidationError("Adam の状態の項目が違います")
        for name in required - {"step"}:
            _tensor(state[name], parameter.shape, parameter.dtype, f"Adam {name}")
            if name.endswith("avg_sq") and bool((state[name] < 0).any()):
                raise CheckpointValidationError("Adam の二乗平均が負です")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            if step.layout != torch.strided or step.shape != torch.Size([]) or not step.is_floating_point() or step.requires_grad:
                raise CheckpointValidationError("Adam の更新回数のテンソル形式が違います")
            step = step.item()
        try:
            if not isinstance(step, (int, float)):
                raise ValueError("更新回数が数値ではありません")
            if _integer(step) < 0:
                raise ValueError("負の更新回数")
        except (TypeError, ValueError, OverflowError) as exc:
            raise CheckpointValidationError("Adam の更新回数が壊れています") from exc


def prepare_checkpoint(
    payload: Any, *, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int],
    seed: int = 0, learning_rate: float = 1e-3,
    reference_policy: ActorCritic | None = None,
) -> PreparedCheckpoint:
    """検証と移行を別の方策・Adam 上で完了し、現在の状態へ触れずに返す。"""
    saved_obs, saved_action, saved_hidden, updates = checkpoint_definition(payload, obs_dim, action_dim, hidden_sizes)
    state = payload["policy"]
    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise CheckpointValidationError("重み（policy）の形式が違います")
    flat = is_flat_state(state)
    if reference_policy is not None and not flat:
        policy = copy.deepcopy(reference_policy)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            policy = ActorCritic(obs_dim, action_dim, hidden_sizes)
    fresh = policy.state_dict()
    expected = {key: value for key, value in fresh.items() if not flat or (not key.startswith(META_MODULES) and key != "option_bias")}
    if set(state) != set(expected):
        raise CheckpointValidationError("重み（policy）の項目がモデルの層構成と一致しません")
    for key, reference in expected.items():
        shape = list(reference.shape)
        if key in _INPUT_WEIGHTS:
            shape[1] = saved_obs + (0 if flat else NUM_OPTIONS)
        elif key in _META_INPUT_WEIGHTS:
            shape[1] = saved_obs
        _tensor(state[key], torch.Size(shape), reference.dtype, f"重み {key}")
    raw_optimizer = payload.get("optimizer")
    if raw_optimizer is not None:
        names = [name for name, _ in policy.named_parameters() if name in state]
        _optimizer_state(raw_optimizer, [state[name] for name in names])
    if saved_obs != obs_dim:
        state = widen_observation(state, saved_obs, obs_dim)
    if flat:
        state = upgrade_flat_state(state, fresh)
    try:
        policy.load_state_dict(state)
        policy.zero_grad(set_to_none=True)
        policy.clamp_log_std()
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
        if raw_optimizer is not None and saved_obs == obs_dim and not flat:
            optimizer.load_state_dict(copy.deepcopy(raw_optimizer))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
    except Exception as exc:
        raise CheckpointValidationError("モデルまたは Adam の状態を復元できませんでした") from exc
    return PreparedCheckpoint(policy, optimizer, saved_obs, saved_action, saved_hidden, updates, raw_optimizer is not None, flat)
