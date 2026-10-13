"""ウォームスタートCLIが読めない元の重みも退避し、退避失敗時には上書きしないかを検査する。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import warmstart_policy as cli


class Dataset:
    actions = np.zeros((2, 2), dtype=np.float32)
    returns = np.zeros(2, dtype=np.float32)
    options = np.zeros(2, dtype=np.int64)

    def __len__(self) -> int:
        return 2


@pytest.fixture
def command(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "shared_policy.pt"
    backup = checkpoint.with_suffix(".pt.before-warmstart")
    calls = []
    trainer = SimpleNamespace(updates=0)

    def load(path):
        calls.append("load")
        return trainer.restored

    def save(path):
        calls.append("save")
        path.write_bytes(b"warmstart-result")

    trainer.restored = False
    trainer.load = load
    trainer.save = save
    trainer.reset_rollout = lambda: calls.append("reset_rollout")
    monkeypatch.setattr(cli.config, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(cli, "get_preset", lambda name: name)
    monkeypatch.setattr(cli, "load_map", lambda preset: object())
    monkeypatch.setattr(cli, "build_map_index", lambda data: object())
    monkeypatch.setattr(cli, "SimulationEnv", lambda *args, **kwargs: SimpleNamespace(reset_all=lambda: None))
    monkeypatch.setattr(cli, "collect_expert", lambda *args: Dataset())
    monkeypatch.setattr(cli, "peek_hidden_sizes", lambda path: (16, 8))
    monkeypatch.setattr(cli, "PPOTrainer", lambda **kwargs: trainer)
    for name in ("fit_policy", "fit_options", "fit_value"):
        def fit(*args, _name=name, **kwargs):
            calls.append(_name)
            return SimpleNamespace(first_loss=1.0, last_loss=0.5)

        monkeypatch.setattr(cli, name, fit)
    monkeypatch.setattr(cli.sys, "argv", ["warmstart_policy.py", "--steps=1", "--epochs=1", "--std=0", "--check-steps=0"])
    return checkpoint, backup, trainer, calls


@pytest.mark.parametrize("restored", [False, True])
def test_existing_checkpoint_is_preserved_even_when_loading_fails(command, restored) -> None:
    checkpoint, backup, trainer, calls = command
    original = b"original-unreadable-or-old-checkpoint"
    checkpoint.write_bytes(original)
    trainer.restored = restored
    assert cli.main() == 0
    assert backup.read_bytes() == original
    assert checkpoint.read_bytes() == b"warmstart-result"
    assert calls == ["load", "fit_policy", "fit_options", "fit_value", "reset_rollout", "save"]


def test_new_checkpoint_does_not_require_a_backup(command) -> None:
    checkpoint, backup, _trainer, calls = command
    assert cli.main() == 0
    assert not backup.exists()
    assert checkpoint.read_bytes() == b"warmstart-result"
    assert "save" in calls


@pytest.mark.parametrize("restored", [False, True])
def test_explicit_no_backup_option_skips_copy(command, restored, monkeypatch) -> None:
    checkpoint, backup, trainer, _calls = command
    checkpoint.write_bytes(b"original-checkpoint")
    trainer.restored = restored
    monkeypatch.setattr(cli.sys, "argv", cli.sys.argv + ["--no-backup"])
    assert cli.main() == 0
    assert not backup.exists()
    assert checkpoint.read_bytes() == b"warmstart-result"


def test_backup_failure_stops_before_fitting_or_saving(command, monkeypatch) -> None:
    checkpoint, _backup, _trainer, calls = command
    original = b"checkpoint-that-must-survive"
    checkpoint.write_bytes(original)

    def fail_copy(*args, **kwargs):
        raise OSError("退避先への書き込み失敗を再現")

    monkeypatch.setattr(cli.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="退避先への書き込み失敗"):
        cli.main()
    assert checkpoint.read_bytes() == original
    assert not any(name.startswith("fit_") or name == "save" for name in calls)


def test_empty_teacher_data_leaves_original_and_backup_untouched(command, monkeypatch) -> None:
    checkpoint, backup, _trainer, calls = command
    checkpoint.write_bytes(b"original-checkpoint")
    backup.write_bytes(b"previous-backup")
    monkeypatch.setattr(cli, "collect_expert", lambda *args: [])
    assert cli.main() == 1
    assert checkpoint.read_bytes() == b"original-checkpoint"
    assert backup.read_bytes() == b"previous-backup"
    assert not calls
