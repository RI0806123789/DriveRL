"""4 台のカメラだけから作る見通しと死角（percep/occlusion.py）と、観測 87 次元への移行を検査する。"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from app import config
from app.contracts import SimParams
from app.percep.encoder import OBS_OFFSETS
from app.percep.occlusion import (
    CORNER_RANGE_M,
    HALF_FOV,
    OCCLUSION_RANGE_M,
    RAY_ANGLES,
    RAY_HALF_WIDTH,
    CameraInput,
    camera_frustum_polygon,
    evaluate_occlusion,
    polygon_area,
    shadow_polygon,
    unknown_occlusion,
)
from app.percep.types import (
    CAMERA_RIG,
    DEFAULT_CAMERA,
    LEFT_CAMERA,
    RIGHT_CAMERA,
    DetClass,
    Detection,
    PerceptionResult,
)
from app.rl.ppo import PPOTrainer

R = OCCLUSION_RANGE_M
OPEN = np.full(config.OBS_FREESPACE_DIM, R, dtype=np.float32)
QUADRANT_SHARE = (2.0 * HALF_FOV) / (math.pi / 2.0)


def views(
    *, front=(), left=(), right=(), free: dict[str, np.ndarray] | None = None, missing: tuple[str, ...] = ()
) -> list[CameraInput]:
    dets = {DEFAULT_CAMERA.key: list(front), LEFT_CAMERA.key: list(left), RIGHT_CAMERA.key: list(right)}
    out = []
    for spec in CAMERA_RIG:
        if spec.key in missing:
            out.append(CameraInput(spec, None, None))
            continue
        out.append(CameraInput(spec, PerceptionResult(0, dets.get(spec.key, [])), (free or {}).get(spec.key, OPEN)))
    return out


def car_box(cx: float, width: float, distance: float) -> Detection:
    return Detection(DetClass.VEHICLE, cx - width / 2, 0.4, cx + width / 2, 0.8, 0.9, distance=distance)


def edge_bearing(x: float) -> float:
    return -math.atan2((x - 0.5) * DEFAULT_CAMERA.width, DEFAULT_CAMERA.focal_px)


class TestFrustum:
    @pytest.mark.parametrize("spec", CAMERA_RIG, ids=lambda s: s.key)
    def test_fan_starts_at_the_camera_and_spans_68_degrees(self, spec) -> None:
        poly = camera_frustum_polygon(spec)
        ox, oy = spec.forward, -spec.right
        assert poly[0] == pytest.approx((ox, oy))
        bearings = [math.atan2(y - oy, x - ox) for x, y in poly[1:]]
        rel = [math.atan2(math.sin(b - spec.yaw), math.cos(b - spec.yaw)) for b in bearings]
        assert min(rel) == pytest.approx(-math.radians(34.0)) and max(rel) == pytest.approx(math.radians(34.0))
        assert all(math.hypot(x - ox, y - oy) == pytest.approx(R) for x, y in poly[1:])
        # 弦で近似するぶんだけ扇の面積 0.5 R² θ より小さい
        exact = 0.5 * R * R * 2.0 * HALF_FOV
        assert 0.995 * exact < polygon_area(poly) <= exact

    def test_left_camera_looks_left(self) -> None:
        poly = camera_frustum_polygon(LEFT_CAMERA)
        assert all(y > 0.0 for _x, y in poly[1:]), "左カメラの扇が右（-y）へ出ている"
        assert all(y < 0.0 for _x, y in camera_frustum_polygon(RIGHT_CAMERA)[1:])

    def test_rays_tile_the_field_of_view(self) -> None:
        edges = np.concatenate([RAY_ANGLES - RAY_HALF_WIDTH, RAY_ANGLES[-1:] + RAY_HALF_WIDTH])
        assert edges[0] == pytest.approx(-HALF_FOV) and edges[-1] == pytest.approx(HALF_FOV)
        assert np.allclose(np.diff(edges), 2.0 * RAY_HALF_WIDTH)


class TestDynamicShadow:
    def test_vehicle_ahead_casts_a_shadow_from_its_distance_to_the_range(self) -> None:
        box = car_box(0.5, 0.2, 12.0)
        result = evaluate_occlusion(views(front=[box]))
        shadows = [s for s in result.shadows if s.kind == "dynamic"]
        assert len(shadows) == 1
        s = shadows[0]
        assert s.camera == "front" and s.near == pytest.approx(12.0) and s.far == pytest.approx(R)
        assert s.end == pytest.approx(edge_bearing(box.x0)) and s.start == pytest.approx(edge_bearing(box.x1))
        poly = shadow_polygon(s)
        ox, oy = DEFAULT_CAMERA.forward, -DEFAULT_CAMERA.right
        dists = [math.hypot(x - ox, y - oy) for x, y in poly]
        assert min(dists) == pytest.approx(12.0) and max(dists) == pytest.approx(R)
        # 陰になった方位の奥行きはその車まで、ほかは見通せる
        front = result.cameras["front"].depths
        inside = (RAY_ANGLES >= s.start) & (RAY_ANGLES <= s.end)
        assert inside.any() and np.all(front[inside] == pytest.approx(12.0)) and np.all(front[~inside] == R)
        assert result.front_occluded == pytest.approx(inside.mean())

    def test_wider_truck_hides_more(self) -> None:
        narrow = evaluate_occlusion(views(front=[car_box(0.5, 0.1, 10.0)])).front_occluded
        wide = evaluate_occlusion(views(front=[car_box(0.5, 0.4, 10.0)])).front_occluded
        assert 0.0 < narrow < wide <= 1.0

    def test_pylons_and_people_do_not_cast_shadows(self) -> None:
        pylon = Detection(DetClass.OBSTACLE, 0.48, 0.6, 0.52, 0.8, 0.9, distance=8.0)
        walker = Detection(DetClass.PEDESTRIAN, 0.3, 0.3, 0.33, 0.8, 0.9, distance=6.0)
        result = evaluate_occlusion(views(front=[pylon, walker]))
        assert result.shadows == [] and result.front_occluded == 0.0

    def test_far_vehicle_beyond_range_has_no_shadow(self) -> None:
        assert evaluate_occlusion(views(front=[car_box(0.5, 0.05, 45.0)])).shadows == []


class TestStaticShadow:
    def test_short_freespace_becomes_a_static_shadow_in_that_bin(self) -> None:
        free = OPEN.copy()
        free[5] = 6.0  # 左 22.5 度
        result = evaluate_occlusion(views(free={"front": free}))
        statics = [s for s in result.shadows if s.kind == "static"]
        assert len(statics) == 1
        s = statics[0]
        assert s.near == pytest.approx(6.0)
        # 画角の端の方位はこの 1 本に割り当てるので、扇は画角の端（34 度）まで
        assert s.start == pytest.approx(math.radians(11.25)) and s.end == pytest.approx(HALF_FOV)
        depths = result.cameras["front"].depths
        assert np.all(depths[RAY_ANGLES > s.start] == pytest.approx(6.0))

    def test_freespace_hit_explained_by_a_detection_is_not_static(self) -> None:
        free = OPEN.copy()
        free[4] = 9.5
        box = car_box(0.5, 0.12, 10.5)
        result = evaluate_occlusion(views(front=[box], free={"front": free}))
        assert [s.kind for s in result.shadows] == ["dynamic"]

    def test_fog_limit_is_static_everywhere(self) -> None:
        fog = np.full(config.OBS_FREESPACE_DIM, 12.0, dtype=np.float32)
        result = evaluate_occlusion(views(free={k.key: fog for k in CAMERA_RIG}))
        assert sum(1 for s in result.shadows if s.kind == "static") == 4 * 3
        assert result.los_left == pytest.approx(12.0) and result.los_right == pytest.approx(12.0)


class TestLineOfSight:
    def test_wall_beside_the_road_shortens_the_side_view(self) -> None:
        wall = np.full(config.OBS_FREESPACE_DIM, 3.0, dtype=np.float32)
        result = evaluate_occlusion(views(free={"left": wall}))
        assert result.los_left == pytest.approx(3.0) and result.los_right == pytest.approx(R)

    def test_view_opens_as_the_car_noses_past_the_corner(self) -> None:
        """角を過ぎると、まず前寄りの方位が交差道路の奥まで抜け、見通し距離が伸びる。"""
        blocked = np.full(config.OBS_FREESPACE_DIM, 3.0, dtype=np.float32)
        nosing = blocked.copy()
        nosing[3] = 24.0  # 左カメラの -22.5 度は車の前寄り
        opened = blocked.copy()
        opened[3:5] = 30.0
        steps = [evaluate_occlusion(views(free={"left": f})).los_left for f in (blocked, nosing, opened)]
        assert steps[0] < steps[1] <= steps[2]

    def test_camera_without_a_result_is_unknown(self) -> None:
        result = evaluate_occlusion(views(missing=("left",)))
        assert result.los_left is None and result.los_right == pytest.approx(R)
        assert result.features[0] == 0.0 and result.sectors[2] == 0.0


class TestFeatures:
    def test_open_road(self) -> None:
        result = evaluate_occlusion(views())
        assert result.features.shape == (config.OBS_OCCLUSION_DIM,)
        assert result.features.dtype == np.float32
        assert np.allclose(result.features, [1.0, 1.0, 0.0, 1.0, *(4 * [QUADRANT_SHARE])], atol=1e-6)

    def test_sector_share_is_area_weighted(self) -> None:
        half = np.full(config.OBS_FREESPACE_DIM, R / 2.0, dtype=np.float32)
        result = evaluate_occlusion(views(free={"rear": half}))
        assert result.sectors[1] == pytest.approx(QUADRANT_SHARE / 4.0)

    def test_corner_distance(self) -> None:
        free = OPEN.copy()
        free[3] = 8.0  # 前方カメラの右 22.5 度は 8m で建物、正面は抜けている
        result = evaluate_occlusion(views(free={"front": free}))
        assert result.corner_distance is not None and 7.0 < result.corner_distance < 9.0
        assert result.features[3] == pytest.approx(result.corner_distance / CORNER_RANGE_M)

    def test_features_stay_in_range_with_garbage(self) -> None:
        garbage = np.array([np.nan, -5.0, np.inf, 1e9, 0.0, 3.0, np.nan, 2.0, -1.0], dtype=np.float32)
        weird = Detection(DetClass.VEHICLE, float("nan"), 0.1, 2.0, 0.9, 0.5, distance=-3.0)
        result = evaluate_occlusion(views(front=[weird], free={k.key: garbage for k in CAMERA_RIG}))
        assert np.all(np.isfinite(result.features)) and np.all((0.0 <= result.features) & (result.features <= 1.0))

    def test_nothing_seen(self) -> None:
        result = unknown_occlusion()
        assert np.allclose(result.features, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        assert result.to_wire(RAY_ANGLES, RAY_HALF_WIDTH)["cameras"] == []

    def test_wire_runs_cover_the_field_of_view(self) -> None:
        free = OPEN.copy()
        free[5] = 6.0
        wire = evaluate_occlusion(views(front=[car_box(0.5, 0.2, 12.0)], free={"front": free})).to_wire(
            RAY_ANGLES, RAY_HALF_WIDTH
        )
        front = next(c for c in wire["cameras"] if c["key"] == "front")
        runs = front["seen"]
        assert runs[0][0] == pytest.approx(-HALF_FOV, abs=1e-3) and runs[-1][1] == pytest.approx(HALF_FOV, abs=1e-3)
        assert all(a[1] == pytest.approx(b[0], abs=1e-3) for a, b in zip(runs, runs[1:]))
        assert {r[2] for r in runs} == {30.0, 12.0, 6.0}


def test_evaluate_is_fast_enough() -> None:
    """issue #84 の予算: 1 台あたり平均 1.0ms 以下。CI の揺れを見込んで中央値で 1.0ms 未満を確かめる。"""
    boxes = [car_box(0.2 + 0.15 * i, 0.08, 8.0 + 3.0 * i) for i in range(5)]
    free = OPEN.copy()
    free[3] = 7.0
    inputs = views(front=boxes, left=boxes[:3], right=boxes[:2], free={k.key: free for k in CAMERA_RIG})
    times = []
    for _ in range(300):
        started = time.perf_counter()
        evaluate_occlusion(inputs)
        times.append(time.perf_counter() - started)
    assert float(np.median(times)) * 1000.0 < 1.0


class TestObservation:
    def test_layout_appends_eight_dims(self) -> None:
        assert config.OBS_DIM == 87
        assert config.OBS_LAYOUT[-1] == ("occlusion", 8)
        assert OBS_OFFSETS["occlusion"] == 79
        assert 79 in config.OBS_WIDENABLE_DIMS

    @pytest.mark.parametrize("old_dim", [66, 75, 79])
    def test_old_checkpoint_loads_with_zero_padding(self, old_dim: int, tmp_path: Path) -> None:
        params = SimParams()
        old = PPOTrainer(old_dim, config.ACTION_DIM, params, config.MAX_VEHICLES, seed=3)
        path = tmp_path / f"old{old_dim}.pt"
        old.save(path)
        new = PPOTrainer(config.OBS_DIM, config.ACTION_DIM, params, config.MAX_VEHICLES, seed=9)
        assert new.load(path) and new.widened_from == old_dim
        rng = np.random.default_rng(0)
        x_old = rng.uniform(-1, 1, size=(32, old_dim)).astype(np.float32)
        tail = rng.uniform(-1, 1, size=(32, config.OBS_DIM - old_dim)).astype(np.float32)
        x_new = np.concatenate([x_old, tail], axis=1)
        with torch.no_grad():
            d_old, v_old = old.policy.forward(torch.from_numpy(x_old))
            d_new, v_new = new.policy.forward(torch.from_numpy(x_new))
        # 足した入力の重みは 0。行列積の足し算の順序が入力の幅で変わるので、float32 の丸めの差だけは残る
        assert float((d_old.mean - d_new.mean).abs().max()) <= 1e-6
        assert float((v_old - v_new).abs().max()) <= 1e-6
