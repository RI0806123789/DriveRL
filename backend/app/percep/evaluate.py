"""いまの認識器を採点し、次の教師データ収集で狙う弱点を決める。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TYPE_CHECKING

import numpy as np

from app import config
from app.contracts import SimParams
from app.percep.types import DEFAULT_CAMERA, CameraSpec, DetClass, Detection, PerceptionResult
from app.percep.weather import PRESETS, Weather

if TYPE_CHECKING:
    from app.contracts import MapIndex
    from app.percep.detector import Detector


__all__ = [
    "COLLECT_WEATHERS",
    "ClassScore",
    "DetectorEvaluation",
    "EVAL_WEATHERS",
    "WeatherScore",
    "evaluate_detector",
]


EVAL_WEATHERS: tuple[str, ...] = ("clear", "rain", "fog")

MATCH_IOU = 0.2

MATCH_CENTER = 0.06

SPEED_TOLERANCE_MPS = 0.3

#: 弱点をどれだけ強く狙うか。**上げすぎると、できていたことを忘れる。**
#: 3.0 で試したとき、霧は 11.5% → 65.4% になった代わりに晴れが 79.4% → 63.0% へ落ちた
#: （2,400 枚のうち晴れが 304 枚 = 12.7% しか残らなかったため）。
FOCUS_GAIN = 1.5

FOCUS_MAX = 2.5


def _iou(a: Detection, b: Detection) -> float:
    """正規化 BBox の IoU。"""
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(a.x1 - a.x0, 0.0) * max(a.y1 - a.y0, 0.0)
    area_b = max(b.x1 - b.x0, 0.0) * max(b.y1 - b.y0, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _center_distance(a: Detection, b: Detection) -> float:
    ax = (a.x0 + a.x1) * 0.5
    ay = (a.y0 + a.y1) * 0.5
    bx = (b.x0 + b.x1) * 0.5
    by = (b.y0 + b.y1) * 0.5
    return float(np.hypot(ax - bx, ay - by))


def _affinity(truth: Detection, pred: Detection) -> float:
    """対応づけの強さ。遠い信号は数画素なので IoU だけでは拾えない。"""
    overlap = _iou(truth, pred)
    if overlap >= MATCH_IOU:
        return 1.0 + overlap
    near = _center_distance(truth, pred)
    if near <= MATCH_CENTER:
        return 1.0 - near / MATCH_CENTER
    return 0.0


def _match(
    truth: list[Detection], pred: list[Detection]
) -> list[tuple[Detection, Detection]]:
    """同じクラス内で真値と検出を貪欲に対応づける。"""
    pairs: list[tuple[float, int, int]] = []
    for i, t in enumerate(truth):
        for j, p in enumerate(pred):
            score = _affinity(t, p)
            if score > 0.0:
                pairs.append((score, i, j))
    pairs.sort(key=lambda item: -item[0])

    used_t: set[int] = set()
    used_p: set[int] = set()
    out: list[tuple[Detection, Detection]] = []
    for _score, i, j in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        out.append((truth[i], pred[j]))
    return out


def _attribute_ok(cls: DetClass, truth: Detection, pred: Detection) -> bool | None:
    """属性（灯色・規制速度）が合っているか。属性を持たないクラスは None。"""
    if cls is DetClass.TRAFFIC_LIGHT:
        if truth.phase is None:
            return None
        return int(truth.phase) == int(pred.phase) if pred.phase is not None else False
    if cls is DetClass.SPEED_SIGN:
        if truth.speed_limit is None:
            return None
        if pred.speed_limit is None:
            return False
        return abs(float(truth.speed_limit) - float(pred.speed_limit)) <= SPEED_TOLERANCE_MPS
    return None


@dataclass
class _Tally:
    truth: int = 0
    matched: int = 0
    attribute_total: int = 0
    attribute_ok: int = 0


@dataclass(frozen=True)
class ClassScore:
    """1 クラス分の成績。"""

    cls: DetClass
    truth: int
    matched: int
    attribute_total: int
    attribute_ok: int

    @property
    def recall(self) -> float:
        return float(self.matched) / float(self.truth) if self.truth else 1.0

    @property
    def attribute_accuracy(self) -> float:
        if not self.attribute_total:
            return 1.0
        return float(self.attribute_ok) / float(self.attribute_total)

    @property
    def score(self) -> float:
        """見つけられて、かつ読み取れた割合。低いほど弱点。"""
        return self.recall * self.attribute_accuracy

    def to_wire(self) -> dict[str, Any]:
        return {
            "cls": int(self.cls),
            "name": self.cls.name,
            "truth": int(self.truth),
            "matched": int(self.matched),
            "recall": round(self.recall, 4),
            "attributeTotal": int(self.attribute_total),
            "attributeOk": int(self.attribute_ok),
            "attributeAccuracy": round(self.attribute_accuracy, 4),
        }


@dataclass(frozen=True)
class WeatherScore:
    """1 天候分の成績（全クラスまとめ）。"""

    name: str
    truth: int
    matched: int

    @property
    def recall(self) -> float:
        return float(self.matched) / float(self.truth) if self.truth else 1.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "truth": int(self.truth),
            "matched": int(self.matched),
            "recall": round(self.recall, 4),
        }


def _focus_weight(score: float) -> float:
    """成績 0.0〜1.0 を収集時の倍率へ変える。"""
    weight = 1.0 + FOCUS_GAIN * (1.0 - float(min(max(score, 0.0), 1.0)))
    return float(min(weight, FOCUS_MAX))


@dataclass(frozen=True)
class DetectorEvaluation:
    """採点の結果。そのまま次の収集の重みになる。"""

    samples: int
    elapsed_sec: float
    classes: tuple[ClassScore, ...]
    weathers: tuple[WeatherScore, ...]

    @property
    def overall_recall(self) -> float:
        truth = sum(c.truth for c in self.classes)
        matched = sum(c.matched for c in self.classes)
        return float(matched) / float(truth) if truth else 1.0

    def class_focus(self) -> dict[DetClass, float]:
        """クラスごとの収集倍率。真値が 1 件も無かったクラスは中立の 1.0。"""
        out: dict[DetClass, float] = {}
        for entry in self.classes:
            out[entry.cls] = 1.0 if entry.truth == 0 else _focus_weight(entry.score)
        return out

    def weather_focus(self) -> dict[str, float]:
        """天候ごとの収集倍率。採点していない近縁のプリセットへも同じ重みを配る。"""
        out = {
            entry.name: (1.0 if entry.truth == 0 else _focus_weight(entry.recall))
            for entry in self.weathers
        }
        for kin, source in _WEATHER_KIN.items():
            if kin not in out and source in out:
                out[kin] = out[source]
        return out

    def weakest(self) -> str:
        """いちばん弱いところを 1 行で言う（画面とログ用）。"""
        worst = min(self.classes, key=lambda c: c.score if c.truth else 2.0, default=None)
        if worst is None or worst.truth == 0:
            return "弱点は見つかりませんでした"
        weather = min(
            self.weathers, key=lambda w: w.recall if w.truth else 2.0, default=None
        )
        text = f"{_CLASS_LABELS[worst.cls]} の成績が最も低い（{worst.score * 100:.0f}%）"
        if weather is not None and weather.truth and weather.recall < 0.9:
            text += f"。{_WEATHER_LABELS.get(weather.name, weather.name)}でも落ちています"
        return text

    def to_wire(self) -> dict[str, Any]:
        return {
            "samples": int(self.samples),
            "elapsedSec": round(float(self.elapsed_sec), 1),
            "overallRecall": round(self.overall_recall, 4),
            "classes": [c.to_wire() for c in self.classes],
            "weathers": [w.to_wire() for w in self.weathers],
            "weakest": self.weakest(),
        }


_CLASS_LABELS: dict[DetClass, str] = {
    DetClass.TRAFFIC_LIGHT: "信号機",
    DetClass.SPEED_SIGN: "速度標識",
    DetClass.VEHICLE: "車両",
    DetClass.OBSTACLE: "障害物",
    DetClass.LANE: "車線",
}

_WEATHER_LABELS: dict[str, str] = {
    "clear": "晴れ",
    "drizzle": "小雨",
    "rain": "雨",
    "fog": "霧",
    "heavy_fog": "濃霧",
}

#: 採点しないプリセットが重みを借りる先。強さ違いの同じ天候へ揃える。
_WEATHER_KIN: dict[str, str] = {"drizzle": "rain", "heavy_fog": "fog"}

COLLECT_WEATHERS: tuple[str, ...] = ("clear", "drizzle", "rain", "fog", "heavy_fog")


def _accumulate(
    truth: PerceptionResult,
    pred: PerceptionResult,
    per_class: dict[DetClass, _Tally],
    weather_tally: _Tally,
) -> None:
    """1 枚ぶんの突き合わせ結果を数える。"""
    for cls in DetClass:
        truths = truth.by_class(cls)
        if not truths:
            continue
        preds = pred.by_class(cls)
        tally = per_class[cls]
        tally.truth += len(truths)
        weather_tally.truth += len(truths)

        for t, p in _match(truths, preds):
            tally.matched += 1
            weather_tally.matched += 1
            ok = _attribute_ok(cls, t, p)
            if ok is not None:
                tally.attribute_total += 1
                tally.attribute_ok += int(ok)


def evaluate_detector(
    map_index: "MapIndex",
    detector: "Detector",
    *,
    samples: int = 400,
    spec: CameraSpec = DEFAULT_CAMERA,
    seed: int = 0,
    weathers: Sequence[str] = EVAL_WEATHERS,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> DetectorEvaluation:
    """走らせながら現行の認識器を採点する。**教師データは作らない。**"""
    from app.percep.camera import PseudoCamera
    from app.percep.groundtruth import detect_ground_truth_batch
    from app.percep.trainer import TrainingCancelled, cluster_vehicles, scatter_obstacles
    from app.sim.env import SimulationEnv

    params = SimParams()
    params.vehicle_count = config.MAX_VEHICLES
    env = SimulationEnv(map_index, params, seed=seed, compute_observations=False)
    camera = PseudoCamera(map_index, spec)
    rng = np.random.default_rng(seed)

    cluster_vehicles(env, rng)
    scatter_obstacles(env, rng)

    names = [name for name in weathers if name in PRESETS] or ["clear"]
    per_class: dict[DetClass, _Tally] = {cls: _Tally() for cls in DetClass}
    per_weather: dict[str, _Tally] = {name: _Tally() for name in names}

    started = time.perf_counter()
    collected = 0
    step = 0
    while collected < samples:
        if should_cancel is not None and should_cancel():
            raise TrainingCancelled("認識器の採点を中断しました")

        slots = np.flatnonzero(env.world.fleet.active)
        if slots.size:
            name = names[step % len(names)]
            weather: Weather = PRESETS[name]
            images = camera.render(env.world, slots, weather, step)
            truths = detect_ground_truth_batch(env.world, slots, spec, weather)
            preds = detector.detect(images, slots)
            for truth, pred in zip(truths, preds):
                _accumulate(truth, pred, per_class, per_weather[name])
            collected += int(slots.size)
            if on_progress is not None:
                on_progress(min(collected, samples), samples)

        action = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
        action[:, 0] = rng.uniform(-0.2, 1.0, size=config.MAX_VEHICLES)
        action[:, 1] = rng.uniform(-0.6, 0.6, size=config.MAX_VEHICLES)
        env.step(action)
        step += 1
        if step % 12 == 0:
            env.reset_all()
            cluster_vehicles(env, rng)
            scatter_obstacles(env, rng)

    return DetectorEvaluation(
        samples=int(collected),
        elapsed_sec=time.perf_counter() - started,
        classes=tuple(
            ClassScore(
                cls=cls,
                truth=t.truth,
                matched=t.matched,
                attribute_total=t.attribute_total,
                attribute_ok=t.attribute_ok,
            )
            for cls, t in per_class.items()
        ),
        weathers=tuple(
            WeatherScore(name=name, truth=t.truth, matched=t.matched)
            for name, t in per_weather.items()
        ),
    )
