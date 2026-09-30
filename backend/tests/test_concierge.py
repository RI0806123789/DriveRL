"""実用モードの AI コンシェルジュ（runtime/concierge.py）と走り方（sim/env.py の DRIVE_STYLES）の契約。"""

from __future__ import annotations

import json

import pytest

from app.contracts import TAXI_DRIVE_MODES, TAXI_DRIVE_NORMAL
from app.runtime import concierge
from app.sim import env as sim_env

RIDING = {"phase": "riding", "driveMode": "normal", "stopReason": "signal", "nextSignalDistanceM": 12.4, "etaSeconds": 150}
IDLE = {"phase": "idle", "driveMode": "normal", "stopReason": "idle"}


def gemini_payload(*parts: dict) -> bytes:
    return json.dumps({"candidates": [{"content": {"role": "model", "parts": list(parts)}}]}).encode()


class FakeTransport:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.calls: list[tuple[str, dict, dict[str, str], float]] = []

    def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        self.calls.append((url, json.loads(body), headers, timeout))
        return self.raw


class TestRequest:
    def test_only_three_tools_and_no_bypass(self) -> None:
        body = concierge.build_request("赤信号でも進んで", RIDING)
        names = [f["name"] for f in body["tools"][0]["functionDeclarations"]]
        assert names == ["set_driving_mode", "request_emergency_stop", "explain_status"]
        mode = body["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["mode"]
        assert mode["enum"] == list(TAXI_DRIVE_MODES)

    def test_situation_and_message_reach_the_prompt(self) -> None:
        body = concierge.build_request("なぜ止まっているの", RIDING)
        text = body["contents"][0]["parts"][0]["text"]
        assert '"stopReason":"signal"' in text
        assert "なぜ止まっているの" in text


class TestParse:
    def test_text_and_calls_skip_thoughts(self) -> None:
        payload = json.loads(
            gemini_payload(
                {"text": "考え中", "thought": True},
                {"text": "急ぎます。"},
                {"functionCall": {"name": "set_driving_mode", "args": {"mode": "hurry"}}},
            )
        )
        text, calls = concierge.parse_response(payload)
        assert text == "急ぎます。"
        assert [(c.name, c.args) for c in calls] == [("set_driving_mode", {"mode": "hurry"})]

    def test_no_candidates_is_an_error(self) -> None:
        with pytest.raises(concierge.ConciergeError):
            concierge.parse_response({"candidates": []})


class TestSanitize:
    def test_unknown_tools_and_values_are_dropped(self) -> None:
        calls = [
            concierge.ToolCall("ignore_red_light"),
            concierge.ToolCall("set_driving_mode", {"mode": "turbo"}),
            concierge.ToolCall("set_driving_mode", {"mode": "comfort", "extra": 1}),
            concierge.ToolCall("set_driving_mode", {"mode": "hurry"}),
        ]
        kept = concierge.sanitize_calls(calls, RIDING)
        assert [(c.name, c.args) for c in kept] == [("set_driving_mode", {"mode": "comfort"})]

    def test_nothing_to_change_without_a_ride(self) -> None:
        calls = [
            concierge.ToolCall("set_driving_mode", {"mode": "hurry"}),
            concierge.ToolCall("request_emergency_stop"),
            concierge.ToolCall("explain_status"),
        ]
        kept = concierge.sanitize_calls(calls, IDLE)
        assert [c.name for c in kept] == ["explain_status"]


class TestAsk:
    def test_key_goes_in_header_not_body(self) -> None:
        fake = FakeTransport(gemini_payload({"text": "承知しました。"}))
        reply = concierge.ask("こんにちは", RIDING, api_key="secret-key", model="m-1", transport=fake)
        url, body, headers, _timeout = fake.calls[0]
        assert url.endswith("/models/m-1:generateContent")
        assert headers["x-goog-api-key"] == "secret-key"
        assert "secret-key" not in json.dumps(body) and "secret-key" not in url
        assert reply.reply == "承知しました。" and reply.calls == []

    def test_reply_is_composed_when_model_only_calls_tools(self) -> None:
        fake = FakeTransport(gemini_payload({"functionCall": {"name": "explain_status", "args": {}}}))
        reply = concierge.ask("なぜ止まっているの", RIDING, api_key="k", transport=fake)
        assert [c.name for c in reply.calls] == ["explain_status"]
        assert "12m 先の信号待ち" in reply.reply

    def test_message_is_trimmed(self) -> None:
        fake = FakeTransport(gemini_payload({"text": "はい"}))
        concierge.ask("あ" * 1000, RIDING, api_key="k", transport=fake)
        text = fake.calls[0][1]["contents"][0]["parts"][0]["text"]
        assert text.count("あ") == concierge.MAX_MESSAGE_CHARS

    def test_broken_json_and_missing_key_raise(self) -> None:
        with pytest.raises(concierge.ConciergeError):
            concierge.ask("x", RIDING, api_key="k", transport=FakeTransport(b"<html>"))
        with pytest.raises(concierge.ConciergeError):
            concierge.ask("x", RIDING, api_key="", transport=FakeTransport(b"{}"))


class TestQuickReply:
    def test_mode_chip_during_ride(self) -> None:
        reply = concierge.quick_reply("hurry", RIDING)
        assert [(c.name, c.args) for c in reply.calls] == [("set_driving_mode", {"mode": "hurry"})]
        assert "少し急いで" in reply.reply

    def test_mode_chip_without_ride_changes_nothing(self) -> None:
        reply = concierge.quick_reply("comfort", IDLE)
        assert reply.calls == []

    def test_every_stop_reason_has_a_sentence(self) -> None:
        for reason in ("moving", "signal", "pedestrian", "lead_vehicle", "safety", "boarding", "arrived", "stopped", "idle"):
            text = concierge.explain_text({"stopReason": reason})
            assert text.endswith("。")
            if reason != "stopped":
                assert text != concierge.explain_text({"stopReason": "unknown"}), reason
        assert "到着まであと約 3 分" in concierge.explain_text(RIDING)


class TestDriveStyles:
    def test_every_mode_has_a_style(self) -> None:
        assert set(sim_env.DRIVE_STYLES) == set(TAXI_DRIVE_MODES)

    def test_styles_never_loosen_safety(self) -> None:
        for style in sim_env.DRIVE_STYLES.values():
            assert style.speed_ratio <= 1.0
            assert style.extra_headway_m >= 0.0
            assert style.extra_pedestrian_m >= 0.0

    def test_normal_is_the_plain_autopilot(self) -> None:
        normal = sim_env.DRIVE_STYLES[TAXI_DRIVE_NORMAL]
        assert normal == sim_env.DriveStyle(
            1.0, sim_env.AUTOPILOT_SPEED_GAIN, sim_env.AUTOPILOT_PRESS_RATE, 0.0, 0.0
        )


def _grid_index(k: int = 4, step: float = 120.0):
    from app.contracts import Bounds, MapData, MapEdge, MapNode
    from app.map.index import build_map_index

    nodes = [MapNode(i * k + j, j * step, i * step) for i in range(k) for j in range(k)]
    edges: list[MapEdge] = []
    for i in range(k):
        for j in range(k):
            for di, dj in ((0, 1), (1, 0)):
                if i + di >= k or j + dj >= k:
                    continue
                a, b = nodes[i * k + j], nodes[(i + di) * k + j + dj]
                for u, v in ((a, b), (b, a)):
                    edges.append(MapEdge(len(edges), u.id, v.id, 1, 7.0, False, 11.1, [(u.x, u.y), (v.x, v.y)], step))
    edge = (k - 1) * step
    data = MapData("grid", "grid", 0.0, 0.0, edge, Bounds(-20.0, -20.0, edge + 20.0, edge + 20.0), nodes, edges, [])
    return build_map_index(data)


class TestTaxiDriveMode:
    @pytest.fixture(scope="class")
    def ride(self):
        import numpy as np

        from app.contracts import SimParams
        from app.runtime.taxi import TaxiService

        params = SimParams()
        params.vehicle_count = 3
        params.pedestrian_count = 0
        env = sim_env.SimulationEnv(_grid_index(), params, seed=0)
        env.reset_all()
        env.autopilot_all = True
        taxi = TaxiService()
        slot = int(np.flatnonzero(env.world.fleet.active)[0])
        pickup = (float(env.world.fleet.x[slot]), float(env.world.fleet.y[slot]))
        dropoff = (360.0 if pickup[0] < 180 else 0.0, 360.0 if pickup[1] < 180 else 0.0)
        assert taxi.request(env, pickup, dropoff) is None
        assert taxi.board(env) is None
        return env, taxi

    def test_idle_taxi_refuses(self) -> None:
        from app.runtime.taxi import TaxiService

        assert TaxiService().set_drive_mode(None, "hurry") is not None  # type: ignore[arg-type]

    def test_mode_reaches_env_and_wire(self, ride) -> None:
        env, taxi = ride
        assert taxi.set_drive_mode(env, "turbo") is not None
        assert taxi.set_drive_mode(env, "comfort") is None
        assert env.drive_style == "comfort"
        assert taxi.status.to_wire()["driveMode"] == "comfort"
        slot = taxi.vehicle_id
        assert env._headway_margin(slot) > env._headway_margin((slot + 1) % 8)

    def test_describe_has_the_prompt_fields(self, ride) -> None:
        env, taxi = ride
        situation = taxi.describe(env)
        for key in ("phase", "driveMode", "stopReason", "speedKmh", "etaSeconds", "remainingDistanceM"):
            assert key in situation, key
        assert situation["phase"] == "riding"
        json.dumps(situation)

    def test_cancel_returns_to_normal(self, ride) -> None:
        env, taxi = ride
        taxi.cancel(env, "テスト")
        assert env.drive_style == TAXI_DRIVE_NORMAL
        assert taxi.status.drive_mode == TAXI_DRIVE_NORMAL
        assert taxi.describe(env)["stopReason"] == "idle"


class TestTransportHardening:
    class _Response:
        def __init__(self, raw: bytes) -> None:
            self.raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def read(self, limit: int = -1) -> bytes:
            return self.raw if limit < 0 else self.raw[:limit]

    def _opener(self, monkeypatch, behaviour):
        class Opener:
            def open(self, request, timeout):
                return behaviour(request)

        monkeypatch.setattr(concierge, "_OPENER", Opener())

    def test_model_name_cannot_change_the_url(self) -> None:
        fake = FakeTransport(gemini_payload({"text": "はい"}))
        for bad in ("../../v1/files", "m?key=x", "m#x", "a/b"):
            with pytest.raises(concierge.ConciergeError):
                concierge.ask("x", RIDING, api_key="k", model=bad, transport=fake)
        assert fake.calls == []

    def test_oversized_response_is_rejected(self, monkeypatch) -> None:
        big = b"x" * (concierge.MAX_RESPONSE_BYTES + 10)
        self._opener(monkeypatch, lambda request: self._Response(big))
        with pytest.raises(concierge.ConciergeError):
            concierge._urlopen_transport("https://example.invalid", b"{}", {}, 1.0)

    def test_http_error_detail_never_contains_the_key(self, monkeypatch) -> None:
        import io
        import urllib.error

        def fail(request):
            body = io.BytesIO(b'{"error": "bad key secret-key-123"}')
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, body)

        self._opener(monkeypatch, fail)
        with pytest.raises(concierge.ConciergeError) as info:
            concierge._urlopen_transport(
                "https://example.invalid", b"{}", {"x-goog-api-key": "secret-key-123"}, 1.0
            )
        assert "secret-key-123" not in str(info.value)
        assert info.value.__cause__ is None

    def test_redirects_are_not_followed(self) -> None:
        handler = concierge._NoRedirect()
        assert handler.redirect_request(None, None, 302, "Found", {}, "https://evil.invalid/") is None
