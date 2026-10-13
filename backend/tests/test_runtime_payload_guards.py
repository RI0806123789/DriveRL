"""モデル読み込みと配車状況の表示に古い値・非有限値が混ざらないことを検査する。"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch

from app import config
from app.contracts import SimParams
from app.rl import export
from app.rl.ppo import PPOTrainer
from app.runtime import concierge
from app.runtime.engine import ImportTicket, SimulationEngine, SituationTicket
from app.runtime.taxi import TaxiService
from app.sim.env import SimulationEnv
from tests.test_concierge import _grid_index


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_PATH", tmp_path / "production.pt")
    monkeypatch.setattr(config, "EXPORT_DIR", tmp_path / "exports")
    instance = SimulationEngine(param_store=None)
    instance._params = SimParams(vehicle_count=3, pedestrian_count=0)
    instance._trainer = PPOTrainer(config.OBS_DIM, config.ACTION_DIM, instance._params,
                                   config.MAX_VEHICLES, seed=142, hidden_sizes=(16, 8))
    instance._trainer._last_grad_norms = {"backbone.0.weight": 12.0}
    instance._trainer._last_grad_total = 13.0
    instance._trainer._last_grad_clip_rate = 0.75
    instance._trainer._last_delta_norms = {"backbone.0.weight": 14.0}
    instance._trainer._weights_before_update = {
        name: parameter.detach().clone() for name, parameter in instance._trainer.policy.named_parameters()
    }
    instance._network = instance._trainer.network_snapshot()
    yield instance
    instance._autotune.shutdown()


def upload_model(engine, tmp_path):
    source = PPOTrainer(config.OBS_DIM, config.ACTION_DIM, engine._params,
                        config.MAX_VEHICLES, seed=143, hidden_sizes=(16, 8))
    with torch.no_grad():
        for parameter in source.policy.parameters():
            parameter.fill_(0.125)
        source.policy.log_std.fill_(-0.7)
    source._updates = 23
    path = tmp_path / "upload.pt"
    source.save(path)
    return path, source


@pytest.mark.parametrize("save_fails", [False, True])
def test_import_discards_old_network_before_save_and_refreshes_from_new_model(engine, tmp_path, monkeypatch, save_fails):
    path, source = upload_model(engine, tmp_path)
    assert engine.network_payload() != source.network_snapshot()
    original_save = engine._trainer.save

    def save(destination):
        assert engine.network_payload() is None
        if save_fails:
            raise OSError("保存できません")
        original_save(destination)

    monkeypatch.setattr(engine._trainer, "save", save)
    ticket = ImportTicket(path)
    engine._handle_import(ticket)
    assert ticket.done.is_set() and ticket.error is None and ticket.info is not None
    assert ticket.backup is not None and ticket.backup.path.exists()
    assert engine.network_payload() is None
    assert not engine._trainer._weights_before_update
    assert engine._trainer.network_snapshot() == source.network_snapshot()

    engine._env = SimulationEnv(_grid_index(), engine._params, seed=142, perception_mode="oracle")
    engine._env.reset_all()
    engine._env.autopilot_all = True
    engine._practical_mode = True
    engine._render_paused = True
    engine._last_metrics_at = 0.0
    engine._step_once()
    assert engine.network_payload() == source.network_snapshot()


@pytest.mark.parametrize("failure", ["inspect", "backup", "load", "autotune", "uninitialized"])
def test_rejected_import_preserves_current_network_and_weights(engine, tmp_path, monkeypatch, failure):
    path, _ = upload_model(engine, tmp_path)
    model = engine._trainer
    before = copy.deepcopy(model.policy.state_dict())
    network_before = engine.network_payload()
    if failure == "inspect":
        path.write_bytes(b"invalid")
    elif failure == "backup":
        def failed_backup(*args, **kwargs):
            raise export.ExportError("退避できません")
        monkeypatch.setattr(export, "export_model", failed_backup)
    elif failure == "load":
        monkeypatch.setattr(model, "load", lambda _: False)
    elif failure == "autotune":
        monkeypatch.setattr(type(engine._autotune), "active", property(lambda _: True))
    else:
        engine._trainer = None

    ticket = ImportTicket(path)
    engine._handle_import(ticket)
    assert ticket.done.is_set() and ticket.error and ticket.info is None
    assert engine.network_payload() == network_before
    assert model.network_snapshot() == network_before
    assert model._weights_before_update
    for name, weight in model.policy.state_dict().items():
        assert torch.equal(weight, before[name])


@pytest.fixture
def ride():
    env = SimulationEnv(_grid_index(), SimParams(vehicle_count=3, pedestrian_count=0),
                        seed=143, perception_mode="oracle")
    env.reset_all()
    taxi = TaxiService()
    slot = int(np.flatnonzero(env.world.fleet.active)[0])
    taxi.status.phase = "riding"
    taxi.status.ride_id = "test-ride"
    taxi.status.vehicle_id = slot
    return env, taxi


@pytest.mark.parametrize("field", ["eta_seconds", "remaining_distance_m"])
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_describe_replaces_nonfinite_progress_and_completes_situation_ticket(engine, ride, field, value):
    env, taxi = ride
    taxi.status.eta_seconds = 125.6
    taxi.status.remaining_distance_m = 480.4
    setattr(taxi.status, field, value)
    engine._env, engine._taxi, engine._practical_mode = env, taxi, True
    ticket = SituationTicket()
    engine._handle_inbox_item("taxi_situation", ticket)
    assert ticket.done.is_set() and ticket.situation is not None
    situation = ticket.situation
    assert situation["etaSeconds"] == (None if field == "eta_seconds" else 126.0)
    assert situation["remainingDistanceM"] == (None if field == "remaining_distance_m" else 480.0)
    assert situation["rideId"] == "test-ride" and situation["practicalMode"] is True
    assert math.isfinite(getattr(taxi.status, field)) is False
    json.dumps(situation, allow_nan=False)
    json.dumps(concierge.build_request("状況を教えてください", situation), allow_nan=False)
    assert concierge.explain_text(situation)


@pytest.mark.parametrize("eta, distance", [(0.0, 0.0), (125.6, 480.4), (124.5, 481.5)])
def test_describe_preserves_finite_rounding(ride, eta, distance):
    env, taxi = ride
    taxi.status.eta_seconds, taxi.status.remaining_distance_m = eta, distance
    situation = taxi.describe(env)
    assert situation["etaSeconds"] == round(eta, 0)
    assert situation["remainingDistanceM"] == round(distance, 0)
    json.dumps(situation, allow_nan=False)
