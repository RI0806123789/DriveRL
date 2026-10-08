"""ローカルの画像と任意の教師で認識器の条件別成績を測る。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app import config
from app.percep.evaluate import _attribute_ok, _match
from app.percep.types import DEFAULT_CAMERA, SIGN_DIRECTIONS, DetClass, Detection, PerceptionResult


@dataclass(frozen=True)
class ImageSample:
    """画像と省略可能な教師および出所。"""

    image: np.ndarray
    truth: PerceptionResult | None
    freespace: np.ndarray | None = None
    source: str = "synthetic"
    sample_id: str = ""
    condition: str = "clear"


@dataclass(frozen=True)
class ImageCondition:
    """画像に掛ける明るさ・雑音・解像度・中心画角の条件。"""

    name: str
    brightness: float = 1.0
    noise_std: float = 0.0
    resolution_scale: float = 1.0
    crop_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("条件名が必要です")
        for key in ("brightness", "noise_std", "resolution_scale", "crop_fraction"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} は有限の数値にしてください")
        if not 0 <= self.brightness <= 4 or not 0 <= self.noise_std <= 255:
            raise ValueError("明るさは 0〜4、雑音は 0〜255 にしてください")
        if not 0 < self.resolution_scale <= 1 or not 0 < self.crop_fraction <= 1:
            raise ValueError("解像度と中心画角の倍率は 0 より大きく 1 以下にしてください")


DEFAULT_CONDITIONS = (
    ImageCondition("baseline"),
    ImageCondition("dark", brightness=0.5),
    ImageCondition("bright", brightness=1.5),
    ImageCondition("noise", noise_std=12.0),
    ImageCondition("low_resolution", resolution_scale=0.5),
    ImageCondition("narrow_fov", crop_fraction=0.75),
)


def _image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("画像は空でない uint8 の H×W×3 RGB 配列にしてください")
    return image


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise ValueError(f"{name} は有限の数値にしてください")
    return float(value)


def _validate_detection(detection: Detection) -> None:
    if not isinstance(detection.cls, DetClass):
        raise ValueError("クラスは DetClass にしてください")
    box = [_number(getattr(detection, key), key) for key in ("x0", "y0", "x1", "y1")]
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError("検出枠は 0〜1 の正規化座標で正の面積が必要です")
    if not 0 <= _number(detection.confidence, "confidence") <= 1:
        raise ValueError("確信度は 0〜1 にしてください")
    if detection.phase is not None:
        if isinstance(detection.phase, bool) or _number(detection.phase, "phase") not in (0, 1, 2):
            raise ValueError("灯色は 0・1・2 にしてください")
    if detection.direction is not None and detection.direction not in SIGN_DIRECTIONS:
        raise ValueError("標識の矢印は所定の方向にしてください")
    for key in ("speed_limit", "distance", "lateral"):
        value = getattr(detection, key)
        if value is not None and (_number(value, key) < 0 and key != "lateral"):
            raise ValueError(f"{key} は負にできません")
    if detection.lane_points is not None:
        points = np.asarray(detection.lane_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("車線の点は有限の座標の組にしてください")


def _freespace(value: Any) -> np.ndarray:
    free = np.asarray(value)
    if free.shape != (config.OBS_FREESPACE_DIM,) or free.dtype.kind not in "fiu" or not np.isfinite(free).all() or np.any(free < 0):
        raise ValueError("走行可能距離は所定の次元の有限で非負の数値にしてください")
    return free.astype(np.float64)


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = np.minimum((np.arange(height) * image.shape[0] / height).astype(int), image.shape[0] - 1)
    xs = np.minimum((np.arange(width) * image.shape[1] / width).astype(int), image.shape[1] - 1)
    return image[ys[:, None], xs[None, :]].copy()


def perturb_sample(sample: ImageSample, condition: ImageCondition, *, seed: int = 0) -> ImageSample:
    """教師の枠を中心切り抜きに合わせ、元画像を変更せず劣化を掛ける。"""
    image = _image(sample.image)
    height, width = image.shape[:2]
    crop_h = max(1, int(round(height * condition.crop_fraction)))
    crop_w = max(1, int(round(width * condition.crop_fraction)))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = image[top:top + crop_h, left:left + crop_w]
    low_h = max(1, int(round(crop_h * condition.resolution_scale)))
    low_w = max(1, int(round(crop_w * condition.resolution_scale)))
    image = _resize(_resize(cropped, low_h, low_w), height, width)
    values = image.astype(np.float64) * condition.brightness
    if condition.noise_std:
        values += np.random.default_rng(seed).normal(0, condition.noise_std, image.shape)
    image = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    truth = None
    if sample.truth is not None:
        boxes = []
        for detection in sample.truth.detections:
            _validate_detection(detection)
            if crop_w == width and crop_h == height:
                boxes.append(replace(detection))
                continue
            x0 = float(np.clip((detection.x0 * width - left) / crop_w, 0, 1))
            x1 = float(np.clip((detection.x1 * width - left) / crop_w, 0, 1))
            y0 = float(np.clip((detection.y0 * height - top) / crop_h, 0, 1))
            y1 = float(np.clip((detection.y1 * height - top) / crop_h, 0, 1))
            if x1 > x0 and y1 > y0:
                boxes.append(replace(detection, x0=x0, y0=y0, x1=x1, y1=y1))
        truth = PerceptionResult(sample.truth.slot, boxes)
    free = sample.freespace if condition.crop_fraction == 1 else None
    return replace(sample, image=image, truth=truth, freespace=free)


def _label(value: Any) -> PerceptionResult | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("detections はリストにしてください。教師が無い場合は null にしてください")
    detections = []
    for item in value:
        if not isinstance(item, dict) or set(item) - {"cls", "box", "conf", "phase", "speedLimit", "direction", "distance", "lateral", "lanePoints"}:
            raise ValueError("検出の教師に未知の欄があります")
        cls = item.get("cls")
        if isinstance(cls, bool) or not isinstance(cls, (int, str)):
            raise ValueError("クラスは番号か DetClass の名前にしてください")
        try:
            cls = DetClass[cls] if isinstance(cls, str) else DetClass(cls)
        except (KeyError, ValueError) as error:
            raise ValueError("未知の検出クラスです") from error
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("box は 4 個の正規化座標にしてください")
        detection = Detection(cls, *box, item.get("conf", 1.0), phase=item.get("phase"), speed_limit=item.get("speedLimit"), direction=item.get("direction"), distance=item.get("distance"), lateral=item.get("lateral"), lane_points=item.get("lanePoints"))
        _validate_detection(detection)
        detections.append(detection)
    return PerceptionResult(0, detections)


def _decode_labels(value: np.ndarray, count: int) -> list[Any]:
    labels = np.asarray(value)
    if labels.shape != (count,) or labels.dtype.kind != "U":
        raise ValueError("labels は画像枚数と同じ長さの Unicode JSON 配列にしてください")
    return [json.loads(str(item)) for item in labels]


def load_external_dataset(path: str | Path) -> list[ImageSample]:
    """pickle を使わずローカルの NPZ または JSON 画像一覧を読む。"""
    path = Path(path)
    samples = []
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) - {"images", "labels", "freespace"} or "images" not in archive:
                raise ValueError("NPZ の欄は images・labels・freespace にしてください")
            images = archive["images"]
            if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
                raise ValueError("images は uint8 の N×H×W×3 RGB 配列にしてください")
            count = len(images)
            labels = _decode_labels(archive["labels"], count) if "labels" in archive else [None] * count
            free = archive["freespace"] if "freespace" in archive else None
            if free is not None and free.shape != (count, config.OBS_FREESPACE_DIM):
                raise ValueError("freespace と画像の枚数・次元が一致しません")
            for i, image in enumerate(images):
                samples.append(ImageSample(_image(image).copy(), _label(labels[i]), None if free is None else _freespace(free[i]), "external", str(i), "external"))
    elif path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) - {"samples", "name"} or not isinstance(document.get("samples"), list):
            raise ValueError("manifest には samples の画像一覧が必要です")
        root = path.resolve().parent
        for i, item in enumerate(document["samples"]):
            if not isinstance(item, dict) or set(item) - {"image", "detections", "freespace", "id", "condition"} or not isinstance(item.get("image"), str):
                raise ValueError("画像一覧に未知の欄があるか image がありません")
            image_path = (root / item["image"]).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError("画像は manifest と同じディレクトリの中に置いてください")
            if image_path.suffix.lower() == ".npy":
                image = np.load(image_path, allow_pickle=False)
            elif image_path.suffix.lower() == ".npz":
                with np.load(image_path, allow_pickle=False) as archive:
                    if archive.files != ["image"]:
                        raise ValueError("1 枚の画像 NPZ には image だけを置いてください")
                    image = archive["image"]
            else:
                raise ValueError("画像は RGB 配列の .npy または .npz にしてください")
            condition = item.get("condition", "external")
            sample_id = item.get("id", str(i))
            if not isinstance(condition, str) or not condition or not isinstance(sample_id, str):
                raise ValueError("condition と id は文字列にしてください")
            free = item.get("freespace")
            samples.append(ImageSample(_image(image), _label(item.get("detections")), None if free is None else _freespace(free), "external", sample_id, condition))
    else:
        raise ValueError("外部データは .npz または .json にしてください")
    if not samples:
        raise ValueError("画像が 1 枚以上必要です")
    return samples


def _score(tally: dict[str, int]) -> dict[str, Any]:
    truth, predicted, matched = tally["truth"], tally["predicted"], tally["matched"]
    return {
        **tally,
        "falsePositives": predicted - matched,
        "precision": matched / predicted if predicted else None,
        "recall": matched / truth if truth else None,
        "falsePositiveRate": (predicted - matched) / predicted if predicted else None,
        "attributeAccuracy": tally["attributeOk"] / tally["attributeTotal"] if tally["attributeTotal"] else None,
    }


def _tallies() -> dict[DetClass, dict[str, int]]:
    return {cls: dict(truth=0, predicted=0, matched=0, attributeTotal=0, attributeOk=0) for cls in DetClass}


def _scores(tallies: dict[DetClass, dict[str, int]], labelled: int) -> dict[str, Any]:
    classes = [{"cls": int(cls), "name": cls.name, **_score(tally)} for cls, tally in tallies.items()]
    total = {key: sum(tally[key] for tally in tallies.values()) for key in next(iter(tallies.values()))}
    overall = _score(total)
    if not labelled:
        for metrics in [overall, *classes]:
            for key in ("truth", "predicted", "matched", "falsePositives", "precision", "recall", "falsePositiveRate", "attributeTotal", "attributeOk", "attributeAccuracy"):
                metrics[key] = None
    return {"classes": classes, "overall": overall}


def _free_score(errors: list[float], condition: ImageCondition) -> dict[str, Any]:
    return {"values": len(errors), "maeMeters": float(np.mean(np.abs(errors))) if errors else None, "rmseMeters": float(np.sqrt(np.mean(np.square(errors)))) if errors else None, "omittedReason": "中心画角の変更で角度別の教師が一致しないため" if condition.crop_fraction != 1 else None}


def _sample_digest(sample: ImageSample) -> str:
    labels = None
    if sample.truth is not None:
        labels = []
        for detection in sample.truth.detections:
            _validate_detection(detection)
            labels.append({
                "cls": int(detection.cls),
                "box": [float(getattr(detection, key)) for key in ("x0", "y0", "x1", "y1")],
                "confidence": float(detection.confidence),
                "phase": None if detection.phase is None else int(detection.phase),
                "speedLimit": None if detection.speed_limit is None else float(detection.speed_limit),
                "direction": detection.direction,
                "distance": None if detection.distance is None else float(detection.distance),
                "lateral": None if detection.lateral is None else float(detection.lateral),
                "lanePoints": None if detection.lane_points is None else [[float(x), float(y)] for x, y in detection.lane_points],
            })
    metadata = {"shape": list(sample.image.shape), "source": sample.source, "id": sample.sample_id, "condition": sample.condition, "labels": labels, "freespace": None if sample.freespace is None else _freespace(sample.freespace).tolist()}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
    digest.update(np.ascontiguousarray(sample.image).tobytes())
    return digest.hexdigest()


def evaluate_images(detector: Any, samples: Sequence[ImageSample], conditions: Sequence[ImageCondition] = DEFAULT_CONDITIONS, *, seed: int = 0) -> dict[str, Any]:
    """教師がある画像だけを採点し、画像条件ごとの JSON 化できる結果を返す。"""
    samples, conditions = list(samples), list(conditions)
    if not samples or not conditions or len({item.name for item in conditions}) != len(conditions):
        raise ValueError("画像と重複しない条件が必要です")
    for sample in samples:
        _image(sample.image)
        if sample.source not in {"synthetic", "external"}:
            raise ValueError("source は synthetic または external にしてください")
        if sample.freespace is not None:
            _freespace(sample.freespace)
        if not isinstance(sample.condition, str) or not sample.condition or not isinstance(sample.sample_id, str):
            raise ValueError("condition と sample_id は文字列にしてください")
    fingerprints = [_sample_digest(sample) for sample in samples]
    spec = getattr(detector, "spec", DEFAULT_CAMERA)
    rows = []
    for condition in conditions:
        tallies = _tallies()
        labelled = 0
        predicted_total = 0
        errors = []
        observed_conditions: dict[str, dict[str, Any]] = {}
        for index, sample in enumerate(samples):
            digest = hashlib.sha256(f"{seed}:{condition.name}:{index}".encode()).digest()
            changed = perturb_sample(sample, condition, seed=int.from_bytes(digest[:8], "little"))
            batch = _resize(changed.image, spec.height, spec.width)[None]
            predictions, free = detector.detect_with_freespace(batch, [0])
            if len(predictions) != 1:
                raise ValueError("認識器が画像 1 枚の結果を返しませんでした")
            prediction = predictions[0]
            for detection in prediction.detections:
                _validate_detection(detection)
            predicted_total += len(prediction.detections)
            if sample.condition not in observed_conditions:
                observed_conditions[sample.condition] = {"samples": 0, "labelledSamples": 0, "predictions": 0, "tallies": _tallies(), "errors": []}
            group = observed_conditions[sample.condition]
            group["samples"] += 1
            group["predictions"] += len(prediction.detections)
            if changed.truth is not None:
                labelled += 1
                group["labelledSamples"] += 1
                for cls, tally in tallies.items():
                    truths, preds = changed.truth.by_class(cls), prediction.by_class(cls)
                    pairs = _match(truths, preds)
                    increments = dict(truth=len(truths), predicted=len(preds), matched=len(pairs), attributeTotal=0, attributeOk=0)
                    for truth, pred in pairs:
                        ok = _attribute_ok(cls, truth, pred)
                        if ok is not None:
                            increments["attributeTotal"] += 1
                            increments["attributeOk"] += int(ok)
                    for key, value in increments.items():
                        tally[key] += value
                        group["tallies"][cls][key] += value
            free = np.asarray(free)
            if free.shape != (1, config.OBS_FREESPACE_DIM):
                raise ValueError("認識器の走行可能距離の形が一致しません")
            predicted_free = _freespace(free[0])
            if changed.freespace is not None:
                difference = (predicted_free - _freespace(changed.freespace)).tolist()
                errors.extend(difference)
                group["errors"].extend(difference)
        source_conditions = [{"name": name, "samples": group["samples"], "labelledSamples": group["labelledSamples"], "predictions": group["predictions"], **_scores(group["tallies"], group["labelledSamples"]), "freespace": _free_score(group["errors"], condition)} for name, group in observed_conditions.items()]
        rows.append({
            "name": condition.name,
            "parameters": {key: float(getattr(condition, key)) for key in ("brightness", "noise_std", "resolution_scale", "crop_fraction")},
            "samples": len(samples), "labelledSamples": labelled,
            "predictions": predicted_total, "sourceConditions": source_conditions,
            **_scores(tallies, labelled),
            "freespace": _free_score(errors, condition),
        })
    return {
        "seed": int(seed), "samples": len(samples),
        "datasetFingerprint": hashlib.sha256("".join(fingerprints).encode()).hexdigest(),
        "sources": {source: sum(sample.source == source for sample in samples) for source in sorted({sample.source for sample in samples})},
        "scope": "ローカルの画像認識評価。実車での安全性や走行性能を示すものではありません。",
        "matching": "同じクラス内で IoU 0.2 以上または中心距離 0.06 以下を貪欲に対応づける",
        "falsePositiveRateDefinition": "教師がある画像の未対応の検出数 / 同じ画像の全検出数",
        "fovTransform": "中心切り抜きと入力寸法への復元による近似",
        "conditions": rows,
    }
