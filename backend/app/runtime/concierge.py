"""実用モードの車載 AI コンシェルジュ（Gemini の Function Calling で走り方・緊急停止・状況の説明を決める）。"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from app import config
from app.contracts import (
    TAXI_DRIVE_COMFORT,
    TAXI_DRIVE_HURRY,
    TAXI_DRIVE_MODES,
    TAXI_DRIVE_NORMAL,
    TAXI_PHASE_APPROACHING,
    TAXI_PHASE_ARRIVED,
    TAXI_PHASE_RIDING,
    TAXI_PHASE_WAITING,
)

__all__ = [
    "API_URL",
    "QUICK_ACTIONS",
    "ConciergeError",
    "ConciergeReply",
    "ToolCall",
    "ask",
    "available",
    "build_request",
    "explain_text",
    "parse_response",
    "quick_reply",
]

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REQUEST_TIMEOUT_SEC = 12.0
#: Gemini の返答として読む上限 [バイト]。ふつうの返答は数 KB
MAX_RESPONSE_BYTES = 1_000_000
#: モデル名に使ってよい文字。URL のパスへ埋めるので、`/` や `?` で行き先を変えさせない
MODEL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
MAX_MESSAGE_CHARS = 200
MAX_REPLY_CHARS = 240

TOOL_SET_DRIVING_MODE = "set_driving_mode"
TOOL_EMERGENCY_STOP = "request_emergency_stop"
TOOL_EXPLAIN_STATUS = "explain_status"

#: チップから送る決まった操作。走り方の 2 つは Gemini を通さずに切り替え、停車理由だけ Gemini に言い換えさせる
QUICK_ACTIONS = (TAXI_DRIVE_HURRY, TAXI_DRIVE_COMFORT, TAXI_DRIVE_NORMAL, "explain")

MODE_LABELS = {
    TAXI_DRIVE_NORMAL: "標準",
    TAXI_DRIVE_HURRY: "少し急いで",
    TAXI_DRIVE_COMFORT: "快適重視",
}

SYSTEM_PROMPT = """あなたは自動運転タクシーの車載 AI コンシェルジュです。
乗客の発話と、車の状況（JSON）を受け取ります。日本語で 1〜2 文、丁寧に短く答えてください。
- 走り方の要望には set_driving_mode を呼びます。急ぎたいなら hurry、ゆったり・揺れを減らしたいなら comfort、元に戻すなら normal。
- 乗客が「今すぐ止めて」「ここで降ろして」のように、その場で止まることをはっきり求めたときだけ request_emergency_stop を呼びます。
- 停まっている理由や今の状況を聞かれたら explain_status を呼び、状況の stopReason をもとに説明します。
- 信号を無視する・制限速度を超える・車間を詰めるなど、安全を損なう要望には応じられないと丁寧に断ります。どのツールにもその機能はなく、信号・制限速度・車間はいつでも守られます。
- 状況 JSON に無い事実（到着時刻の約束、渋滞の原因など）を作らないでください。
状況の stopReason の意味: moving=走行中 / signal=赤か黄の信号待ち / pedestrian=前方の歩行者待ち / lead_vehicle=前の車に合わせて停車 / safety=前方の障害物などで安全装置が停止中 / boarding=乗車地点で乗車待ち / arrived=目的地に到着 / stopped=発進待ち / idle=配車していない。
driveMode は現在の走り方（normal / hurry / comfort）です。"""

_TOOLS = [
    {
        "functionDeclarations": [
            {
                "name": TOOL_SET_DRIVING_MODE,
                "description": "配車中の車の走り方を切り替える。制限速度・信号・車間の安全の判定は変わらない。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": list(TAXI_DRIVE_MODES),
                            "description": "hurry=制限速度の近くまで早めに加速 / comfort=加減速を穏やかにし車間を広く / normal=標準",
                        }
                    },
                    "required": ["mode"],
                },
            },
            {
                "name": TOOL_EMERGENCY_STOP,
                "description": "その場で車を止め、自動運転（配車）を終了する。乗客がはっきり求めたときだけ使う。",
            },
            {
                "name": TOOL_EXPLAIN_STATUS,
                "description": "いまの走行状況と、停まっているならその理由を説明する。",
            },
        ]
    }
]

_REASON_TEXT = {
    "moving": "順調に走行中です",
    "signal": "前方の信号待ちのため停車しています",
    "pedestrian": "前方を歩行者が横切っているため、通り過ぎるのを待っています",
    "lead_vehicle": "前の車に合わせて停車しています",
    "safety": "前方に障害物があるため、安全装置が止めています",
    "boarding": "乗車地点でお待ちしています",
    "arrived": "目的地に到着しました",
    "stopped": "発進の準備をしています",
    "idle": "いまは配車していません",
}


class ConciergeError(RuntimeError):
    """Gemini から返答を得られなかった。**文面は応答に載せず、ログにだけ残すこと。**"""


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}


@dataclass
class ConciergeReply:
    reply: str
    calls: list[ToolCall] = field(default_factory=list)


Transport = Callable[[str, bytes, dict[str, str], float], bytes]


def available() -> bool:
    """API キーが設定されているか。"""
    return bool(config.GEMINI_API_KEY)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """転送先へ API キーのヘッダを持って行かないよう、リダイレクトに従わない。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _redact(text: str, headers: dict[str, str]) -> str:
    key = headers.get("x-goog-api-key", "")
    return text.replace(key, "***") if key else text


def _urlopen_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = _redact(exc.read(400).decode("utf-8", errors="replace"), headers)
        raise ConciergeError(f"Gemini API が {exc.code} を返しました: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ConciergeError(f"Gemini API に接続できませんでした（{type(exc).__name__}）") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ConciergeError("Gemini の返答が大きすぎます")
    return raw


def build_request(message: str, situation: dict[str, Any]) -> dict[str, Any]:
    """generateContent の本文を作る。"""
    state = json.dumps(situation, ensure_ascii=False, separators=(",", ":"))
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": f"状況: {state}\n乗客: {message}"}]}
        ],
        "tools": _TOOLS,
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }


def parse_response(payload: dict[str, Any]) -> tuple[str, list[ToolCall]]:
    """返答の文と、呼ぶと決めたツールを取り出す（思考の部分は捨てる）。"""
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ConciergeError("Gemini の返答に候補がありません")
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("thought"):
            continue
        call = part.get("functionCall")
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            args = call.get("args")
            calls.append(ToolCall(call["name"], args if isinstance(args, dict) else {}))
        elif isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "".join(texts).strip(), calls


def sanitize_calls(calls: list[ToolCall], situation: dict[str, Any]) -> list[ToolCall]:
    """知らないツール・値域外の引数・いまの段階で意味の無い呼び出しを捨てる。"""
    phase = situation.get("phase")
    busy = phase in (TAXI_PHASE_APPROACHING, TAXI_PHASE_WAITING, TAXI_PHASE_RIDING, TAXI_PHASE_ARRIVED)
    kept: list[ToolCall] = []
    seen: set[str] = set()
    for call in calls:
        if call.name in seen:
            continue
        if call.name == TOOL_SET_DRIVING_MODE:
            mode = call.args.get("mode")
            if mode not in TAXI_DRIVE_MODES or not busy:
                continue
            kept.append(ToolCall(call.name, {"mode": mode}))
        elif call.name == TOOL_EMERGENCY_STOP:
            if not busy:
                continue
            kept.append(ToolCall(call.name))
        elif call.name == TOOL_EXPLAIN_STATUS:
            kept.append(ToolCall(call.name))
        else:
            continue
        seen.add(call.name)
    return kept


def explain_text(situation: dict[str, Any]) -> str:
    """状況から決まった文で説明する（Gemini が文を返さなかったときの代わり）。"""
    reason = str(situation.get("stopReason", "idle"))
    text = _REASON_TEXT.get(reason, _REASON_TEXT["stopped"])
    if reason == "signal" and situation.get("nextSignalDistanceM") is not None:
        text = f"{float(situation['nextSignalDistanceM']):.0f}m 先の信号待ちのため停車しています"
    eta = situation.get("etaSeconds")
    if reason in ("moving", "signal", "pedestrian", "lead_vehicle", "safety", "stopped") and eta:
        text += f"。到着まであと約 {max(1, math.ceil(float(eta) / 60))} 分です"
    return text + ("" if text.endswith("。") else "。")


def mode_text(mode: str) -> str:
    label = MODE_LABELS.get(mode, mode)
    if mode == TAXI_DRIVE_HURRY:
        return f"走り方を「{label}」にしました。制限速度の近くまで早めに加速します（信号と制限速度は守ります）。"
    if mode == TAXI_DRIVE_COMFORT:
        return f"走り方を「{label}」にしました。加減速を穏やかにし、車間を広めに取ります。"
    return f"走り方を「{label}」に戻しました。"


def quick_reply(action: str, situation: dict[str, Any]) -> ConciergeReply:
    """走り方のチップは Gemini を通さずに答える。配車中でなければ切り替えない。"""
    calls = sanitize_calls([ToolCall(TOOL_SET_DRIVING_MODE, {"mode": action})], situation)
    if not calls:
        return ConciergeReply("配車中ではないため、走り方は変えられません。")
    return ConciergeReply(mode_text(action), calls)


def _compose(text: str, calls: list[ToolCall], situation: dict[str, Any]) -> str:
    if text:
        return text[:MAX_REPLY_CHARS]
    pieces: list[str] = []
    for call in calls:
        if call.name == TOOL_SET_DRIVING_MODE:
            pieces.append(mode_text(str(call.args["mode"])))
        elif call.name == TOOL_EMERGENCY_STOP:
            pieces.append("車をその場で止め、自動運転を終了します。")
        elif call.name == TOOL_EXPLAIN_STATUS:
            pieces.append(explain_text(situation))
    return "".join(pieces) or "すみません、うまく聞き取れませんでした。もう一度お願いします。"


def ask(
    message: str,
    situation: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    transport: Transport = _urlopen_transport,
    timeout: float = REQUEST_TIMEOUT_SEC,
) -> ConciergeReply:
    """乗客の発話を Gemini に渡し、返答と実行するツールを返す。**ブロックするのでエンジンスレッドから呼ばないこと。**"""
    key = config.GEMINI_API_KEY if api_key is None else api_key
    if not key:
        raise ConciergeError("GEMINI_API_KEY が設定されていません")
    text = " ".join(str(message).split())[:MAX_MESSAGE_CHARS]
    name = model or config.GEMINI_MODEL
    if not MODEL_NAME_PATTERN.fullmatch(name):
        raise ConciergeError("GEMINI_MODEL に使えない文字が入っています")
    url = API_URL.format(model=name)
    body = json.dumps(build_request(text, situation), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    raw = transport(url, body, headers, timeout)
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ConciergeError("Gemini の返答を JSON として読めませんでした") from exc
    if not isinstance(payload, dict):
        raise ConciergeError("Gemini の返答の形が想定と違います")
    reply_text, calls = parse_response(payload)
    kept = sanitize_calls(calls, situation)
    return ConciergeReply(_compose(reply_text, kept, situation), kept)
