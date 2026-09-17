"""天候（雨・霧）の強さと、擬似カメラ画像への反映。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AUTO_PERIOD_SEC",
    "CLEAR",
    "MIN_VISIBILITY_M",
    "PRESETS",
    "Weather",
    "apply_weather",
    "auto_weather",
]


MIN_VISIBILITY_M = 15.0

FOG_EXTINCTION = 2.0

_FOG_COLOR = np.array([206.0, 210.0, 214.0], dtype=np.float32)

_RAIN_VEIL_COLOR = np.array([74.0, 78.0, 88.0], dtype=np.float32)

_RAIN_STREAK_COLOR = np.float32(215.0)

_RAIN_STREAK_FRAMES = 8
_RAIN_STREAK_SLOPE = 0.22

AUTO_PERIOD_SEC = 480.0


@dataclass(frozen=True)
class Weather:
    """雨・霧の強さ（どちらも 0.0〜1.0）。"""

    rain: float = 0.0
    fog: float = 0.0

    @property
    def clear(self) -> bool:
        return self.rain <= 1e-3 and self.fog <= 1e-3

    @property
    def needs_depth(self) -> bool:
        """霧を掛けるために画素ごとの奥行きが要るか。"""
        return self.fog > 1e-3

    def visibility_m(self, far: float) -> float:
        """この天候での有効視程 [m]。**描画とラベルはこの値を共有する。**

        指数を 2 乗しているのは、視程が「透過率が 5% に落ちる距離」で決まるため
        `fog` に線形で効かせると、弱い霧でも近くが白く沈むため。
        """
        if self.fog <= 1e-3:
            return float(far)
        ratio = max(MIN_VISIBILITY_M / max(float(far), 1e-6), 1e-6)
        strength = float(min(max(self.fog, 0.0), 1.0))
        return float(far) * (ratio ** (strength * strength))

    def to_wire(self, far: float) -> dict[str, float]:
        return {
            "rain": float(self.rain),
            "fog": float(self.fog),
            "visibility": self.visibility_m(far),
        }


CLEAR = Weather()

PRESETS: dict[str, Weather] = {
    "clear": Weather(0.0, 0.0),
    "drizzle": Weather(0.35, 0.0),
    "rain": Weather(0.85, 0.0),
    "fog": Weather(0.0, 0.75),
    "heavy_fog": Weather(0.15, 0.95),
}


def auto_weather(sim_time: float, period: float = AUTO_PERIOD_SEC) -> Weather:
    """シミュレーション時刻から天候を決める（晴れ → 雨 → 晴れ → 霧 の循環）。"""
    phase = (float(sim_time) / max(float(period), 1e-6)) % 1.0
    rain = max(0.0, math.sin(2.0 * math.pi * phase))
    fog = max(0.0, math.sin(2.0 * math.pi * (phase - 0.5)))
    return Weather(rain=rain * rain, fog=fog * fog)


_StreakFrame = tuple[np.ndarray, np.ndarray, np.ndarray]

_streak_cache: dict[tuple[int, int], list[_StreakFrame]] = {}


def _streak_bank(height: int, width: int) -> list[_StreakFrame]:
    """雨の筋を数枚ぶん、(行, 列, 強さ) の疎な形で作る。

    密な (H, W) マスクで持って全画素に掛けると、ほとんどが 0 なのに
    1 枚あたり 7ms を使う。触るのは数千画素だけなので添字で持つ。
    """
    key = (int(height), int(width))
    cached = _streak_cache.get(key)
    if cached is not None:
        return cached

    rng = np.random.default_rng(20260917)
    bank: list[_StreakFrame] = []
    drops = max(8, (height * width) // 900)
    for _ in range(_RAIN_STREAK_FRAMES):
        cx = rng.uniform(0.0, float(width), size=drops)
        cy = rng.uniform(0.0, float(height), size=drops)
        length = rng.uniform(height * 0.06, height * 0.16, size=drops)
        strength = rng.uniform(0.35, 1.0, size=drops).astype(np.float32)

        ys: list[np.ndarray] = []
        xs: list[np.ndarray] = []
        vs: list[np.ndarray] = []
        for i in range(drops):
            y0 = int(max(0.0, cy[i]))
            y1 = int(min(float(height), cy[i] + length[i]))
            if y1 <= y0:
                continue
            row = np.arange(y0, y1, dtype=np.int32)
            col = np.rint(cx[i] + (row - cy[i]) * _RAIN_STREAK_SLOPE).astype(np.int32)
            keep = (col >= 0) & (col < width)
            if not keep.any():
                continue
            ys.append(row[keep])
            xs.append(col[keep])
            vs.append(np.full(int(keep.sum()), strength[i], dtype=np.float32))
        if ys:
            bank.append(
                (np.concatenate(ys), np.concatenate(xs), np.concatenate(vs))
            )
        else:
            empty_i = np.zeros(0, dtype=np.int32)
            bank.append((empty_i, empty_i, np.zeros(0, dtype=np.float32)))
    _streak_cache[key] = bank
    return bank


def apply_weather(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    weather: Weather,
    *,
    far: float,
    frame_index: int = 0,
) -> np.ndarray:
    """描き終えた画像に天候を乗せる。`rgb` は (N, H, W, 3) uint8。"""
    if weather.clear or rgb.size == 0:
        return rgb

    out = rgb.astype(np.float32)
    _, height, width, _ = out.shape

    fog = float(min(max(weather.fog, 0.0), 1.0))
    if fog > 1e-3 and depth is not None:
        density = FOG_EXTINCTION / max(weather.visibility_m(far), 1e-3)
        keep = np.exp(np.multiply(depth, -density, dtype=np.float32))
        out *= keep[..., None]
        veil = np.subtract(1.0, keep, out=keep)
        for c in range(3):
            out[..., c] += _FOG_COLOR[c] * veil

    rain = float(min(max(weather.rain, 0.0), 1.0))
    if rain > 1e-3:
        haze = 0.28 * rain
        out *= 1.0 - haze
        out += _RAIN_VEIL_COLOR * haze
        ys, xs, strength = _streak_bank(height, width)[frame_index % _RAIN_STREAK_FRAMES]
        if ys.size:
            patch = out[:, ys, xs, :]
            patch += (_RAIN_STREAK_COLOR - patch) * (
                strength[None, :, None] * (0.55 * rain)
            )
            out[:, ys, xs, :] = patch

    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.uint8)
