"""モデルの書き出し（`rl/export.py`）が観測の区画を漏れなく説明し、すべての形式で最後まで書き出せるか。"""
from __future__ import annotations

import json
import os
import warnings
import zipfile
from pathlib import Path

import pytest
import torch

from app import config
from app.contracts import SimParams
from app.rl.export import (
    EXPORT_KINDS,
    KERAS_METADATA_ENTRY,
    InferencePolicy,
    _layout_notes,
    build_metadata,
    export_model,
)
from app.rl.importer import CheckpointImportError, inspect_checkpoint
from app.rl.ppo import PPOTrainer

# TorchScript は PyTorch が非推奨にした API で、書き出しと読み直しのたびに FutureWarning が出る（#93。後継は pt2）
JIT_DEPRECATION = pytest.mark.filterwarnings("ignore:`torch.jit.:FutureWarning")
KIND_CASES = [pytest.param(k, marks=JIT_DEPRECATION) if k == "torchscript" else k for k in EXPORT_KINDS]


@pytest.fixture(scope="module")
def trainer() -> PPOTrainer:
    return PPOTrainer(config.OBS_DIM, config.ACTION_DIM, SimParams(), config.MAX_VEHICLES, seed=0)


def test_every_observation_section_has_a_note() -> None:
    notes = _layout_notes()
    missing = [name for name, _size in config.OBS_LAYOUT if name not in notes]
    assert missing == [], "config.OBS_LAYOUT に足した区画は rl/export.py の _layout_notes にも説明を書くこと"
    stale = [name for name in notes if name not in dict(config.OBS_LAYOUT)]
    assert stale == [], "消えた区画の説明が _layout_notes に残っている"


def test_metadata_layout_covers_the_whole_observation(trainer: PPOTrainer) -> None:
    layout = build_metadata(trainer, kind="checkpoint", preset_id=None, preset_name=None, metrics=None)["observation"]["layout"]
    assert [item["name"] for item in layout] == [name for name, _size in config.OBS_LAYOUT]
    assert sum(int(item["size"]) for item in layout) == config.OBS_DIM
    assert all(item["source"] and item["description"] for item in layout)


@pytest.mark.parametrize("kind", KIND_CASES)
def test_export_writes_a_file_with_metadata(trainer: PPOTrainer, kind: str, tmp_path: Path) -> None:
    if kind == "keras":
        os.environ.setdefault("KERAS_BACKEND", "torch")
        pytest.importorskip("keras")
    result = export_model(trainer, kind, out_dir=tmp_path, preset_id="test", params=SimParams())
    assert result.path.exists() and result.size_bytes > 0
    assert not list(tmp_path.glob("*.tmp"))
    if kind == "checkpoint":
        meta = torch.load(result.path, map_location="cpu", weights_only=True)["metadata"]
    elif kind == "torchscript":
        extra = {"metadata.json": ""}
        torch.jit.load(str(result.path), _extra_files=extra)
        meta = json.loads(extra["metadata.json"])
    elif kind == "pt2":
        assert result.path.suffix == ".pt2"
        extra = {"metadata.json": ""}
        torch.export.load(str(result.path), extra_files=extra)
        meta = json.loads(extra["metadata.json"])
        # torch を入れていない側でも zip から読めるよう、置き場所は固定（README に書いてある）
        with zipfile.ZipFile(result.path) as archive:
            assert json.loads(archive.read("archive/extra/metadata.json")) == meta
    else:
        with zipfile.ZipFile(result.path) as archive:
            meta = json.loads(archive.read(KERAS_METADATA_ENTRY))
    assert meta["observation"]["dim"] == config.OBS_DIM
    assert [item["name"] for item in meta["observation"]["layout"]] == [name for name, _ in config.OBS_LAYOUT]
    assert meta["kind"] == kind


def test_pt2_matches_the_policy_without_deprecation(trainer: PPOTrainer, tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = export_model(trainer, "pt2", out_dir=tmp_path, preset_id="test", params=SimParams())
        program = torch.export.load(str(result.path)).module()
    deprecated = [
        str(w.message)
        for w in caught
        if issubclass(w.category, (FutureWarning, DeprecationWarning)) and "deprecated" in str(w.message)
    ]
    assert deprecated == []

    reference = InferencePolicy(trainer.policy)
    generator = torch.Generator().manual_seed(0)
    for batch in (1, 3, 64):
        obs = torch.randn(batch, config.OBS_DIM, generator=generator) * 2.0
        action, value = program(obs)
        ref_action, ref_value = reference(obs)
        assert action.shape == (batch, config.ACTION_DIM) and value.shape == (batch,)
        assert torch.equal(action, ref_action) and torch.equal(value, ref_value), batch


@pytest.mark.parametrize(
    "logits",
    [
        None,  # 初期の方策。上位の出力層が 0 から始まるので、どの観測でも 4 つの意図が同点
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 2.0, 2.0, -1.0],
        [3.0, 3.0 - 1e-6, 3.0, 2.0],
        [0.5, 0.5 + 1e-4, 0.5, 0.5],
        [-1.0, -2.0, -3.0, 4.0],
    ],
)
def test_keras_picks_the_same_option_as_torch_argmax_on_ties(logits, tmp_path: Path) -> None:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    keras = pytest.importorskip("keras")
    from app.rl.export import _build_keras_model

    policy_trainer = PPOTrainer(config.OBS_DIM, config.ACTION_DIM, SimParams(), config.MAX_VEHICLES, seed=0)
    if logits is not None:
        with torch.no_grad():
            policy_trainer.policy.meta_head.weight.zero_()
            policy_trainer.policy.meta_head.bias.copy_(torch.tensor(logits))
    obs = torch.cat([torch.zeros(2, config.OBS_DIM), torch.randn(30, config.OBS_DIM, generator=torch.Generator().manual_seed(1))])
    with torch.no_grad():
        ref_action, ref_value = InferencePolicy(policy_trainer.policy)(obs)
    model = _build_keras_model(policy_trainer)
    path = tmp_path / "policy.keras"
    model.save(path)
    for m in (model, keras.saving.load_model(path)):
        action, value = m.predict(obs.numpy(), verbose=0)
        assert abs(action - ref_action.numpy()).max() < 1e-5
        assert abs(value - ref_value.numpy()).max() < 1e-5


class _FixedClock:
    """書き出し名に入る時刻を固定する（同じ秒に書き出した場合を作る）。"""

    @staticmethod
    def now():
        from datetime import datetime

        return datetime(2026, 10, 10, 12, 0, 0)


def _policy_with(value: float) -> PPOTrainer:
    t = PPOTrainer(config.OBS_DIM, config.ACTION_DIM, SimParams(), config.MAX_VEHICLES, seed=0)
    with torch.no_grad():
        t.policy.mu_head.bias.fill_(value)
    return t


def test_exports_in_the_same_second_keep_separate_files(tmp_path: Path, monkeypatch) -> None:
    from app.rl import export

    monkeypatch.setattr(export, "datetime", _FixedClock)
    first = export_model(_policy_with(0.1), "checkpoint", out_dir=tmp_path, preset_id="ginza", label="before-import")
    second = export_model(_policy_with(0.2), "checkpoint", out_dir=tmp_path, preset_id="ginza", label="before-import")
    assert first.path != second.path
    assert "_before-import_20261010-120000_" in first.filename
    kept = {p.name: torch.load(p, weights_only=True)["policy"]["mu_head.bias"][0].item() for p in tmp_path.glob("*.pt")}
    assert kept == {first.filename: pytest.approx(0.1), second.filename: pytest.approx(0.2)}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("hard_links", [True, False])
def test_install_never_overwrites_an_existing_export(tmp_path: Path, monkeypatch, hard_links: bool) -> None:
    from app.rl import export

    if not hard_links:  # FAT など、ハードリンクを作れないファイルシステム
        def no_link(src, dst):
            raise OSError("hard links are not supported")

        monkeypatch.setattr(export.os, "link", no_link)
    existing = tmp_path / "autoware-sim_ginza_upd0_before-import.pt"
    existing.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        export._atomic_save(lambda p: p.write_bytes(b"replacement"), existing)
    assert existing.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp"))
    fresh = tmp_path / "autoware-sim_ginza_upd0_other.pt"
    export._atomic_save(lambda p: p.write_bytes(b"new"), fresh)
    assert fresh.read_bytes() == b"new"


def test_consecutive_imports_in_the_same_second_keep_every_backup(tmp_path: Path, monkeypatch) -> None:
    """A を退避して B を読み、同じ秒・同じ更新回数で C を読んでも、A と B の退避が両方残る（#124）。"""
    import threading
    from types import SimpleNamespace

    from app.rl import export
    from app.runtime.engine import ImportTicket, SimulationEngine

    monkeypatch.setattr(export, "datetime", _FixedClock)
    monkeypatch.setattr(config, "EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(config, "CHECKPOINT_PATH", tmp_path / "shared_policy.pt")
    live = _policy_with(0.1)
    uploads = []
    for value in (0.2, 0.3):
        path = tmp_path / f"upload_{value}.pt"
        _policy_with(value).save(path)
        uploads.append(path)
    fake = SimpleNamespace(
        _trainer=live, _lock=threading.Lock(), _loaded_preset_id="ginza", _loaded_preset_name="銀座",
        _metrics=SimpleNamespace(to_wire=lambda: {}), snapshot_params=SimParams, _notify=lambda message: None,
        _autotune=SimpleNamespace(active=False), _episode_log=[], _total_episodes=0,
        _last_update_stats={}, _last_autosave_updates=0,
    )
    backups = []
    for path in uploads:
        ticket = ImportTicket(path=path)
        SimulationEngine._handle_import(fake, ticket)
        assert ticket.error is None, ticket.error
        backups.append(ticket.backup.path)
    assert backups[0] != backups[1]
    restored = [torch.load(p, weights_only=True)["policy"]["mu_head.bias"][0].item() for p in backups]
    assert restored == [pytest.approx(0.1), pytest.approx(0.2)], "読み込む前の重みを両方とも取り戻せる"
    assert live.policy.mu_head.bias[0].item() == pytest.approx(0.3)


def test_import_refuses_a_pt2_with_a_hint(trainer: PPOTrainer, tmp_path: Path) -> None:
    result = export_model(trainer, "pt2", out_dir=tmp_path, preset_id="test", params=SimParams())
    with pytest.raises(CheckpointImportError, match="torch.export 形式"):
        inspect_checkpoint(
            result.path,
            expected_obs_dim=config.OBS_DIM,
            expected_action_dim=config.ACTION_DIM,
            expected_hidden_sizes=trainer.policy.hidden_sizes,
        )
