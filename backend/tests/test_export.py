"""モデルの書き出し（`rl/export.py`）が観測の区画を漏れなく説明し、3 形式とも最後まで書き出せるか。"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest
import torch

from app import config
from app.contracts import SimParams
from app.rl.export import EXPORT_KINDS, KERAS_METADATA_ENTRY, _layout_notes, build_metadata, export_model
from app.rl.ppo import PPOTrainer


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


@pytest.mark.parametrize("kind", EXPORT_KINDS)
def test_export_writes_a_file_with_metadata(trainer: PPOTrainer, kind: str, tmp_path: Path) -> None:
    if kind == "keras":
        os.environ.setdefault("KERAS_BACKEND", "torch")
        pytest.importorskip("keras")
    result = export_model(trainer, kind, out_dir=tmp_path, preset_id="test", params=SimParams())
    assert result.path.exists() and result.size_bytes > 0
    if kind == "checkpoint":
        meta = torch.load(result.path, map_location="cpu", weights_only=True)["metadata"]
    elif kind == "torchscript":
        extra = {"metadata.json": ""}
        torch.jit.load(str(result.path), _extra_files=extra)
        meta = json.loads(extra["metadata.json"])
    else:
        with zipfile.ZipFile(result.path) as archive:
            meta = json.loads(archive.read(KERAS_METADATA_ENTRY))
    assert meta["observation"]["dim"] == config.OBS_DIM
    assert [item["name"] for item in meta["observation"]["layout"]] == [name for name, _ in config.OBS_LAYOUT]
