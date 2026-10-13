"""高低差・立体道路（#136）：路面モデル・勾配の物理・上下の路面の衝突・道路の選び方・擬似カメラと正解ラベル。"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from app import config
from app.contracts import (
    STRUCTURE_BRIDGE,
    STRUCTURE_TUNNEL,
    Bounds,
    MapData,
    MapEdge,
    MapNode,
    MapPreset,
    SimParams,
    TerrainGrid,
)
from app.map.elevation import (
    GRAVITY,
    RoadSurface,
    attitude_from_slopes,
    gravity_components,
    route_grade,
)
from app.map.index import build_map_index
from app.map.presets import get_preset
from app.map.synthetic import DECK_Z, OVERPASS_X, build_synthetic_map, hill_height
from app.percep.camera import LBL_VEHICLE, LBL_VEHICLE_DARK, PseudoCamera
from app.percep.elevated_view import OFF_LEVEL_M, camera_batch, drop_other_levels, level_offset
from app.percep.geometry import camera_pose_3d
from app.percep.groundtruth import clear_static_cache, detect_ground_truth, freespace_ground_truth
from app.percep.types import CAMERA_RIG, DEFAULT_CAMERA, DetClass, Detection, PerceptionResult
from app.sim.dynamics import VehicleDynamics
from app.sim.env import SimulationEnv
from app.sim.vehicle import VehicleFleet

N = config.MAX_VEHICLES


@pytest.fixture(scope="module")
def lab() -> MapData:
    preset = get_preset("terrain_lab")
    assert preset is not None and preset.synthetic
    return build_synthetic_map(preset)


@pytest.fixture(scope="module")
def lab_index(lab: MapData):
    return build_map_index(lab)


def _env(index, *, pedestrians: int = 0) -> SimulationEnv:
    params = SimParams()
    params.pedestrian_count = pedestrians
    env = SimulationEnv(index, params, seed=136, compute_observations=False)
    env.world.set_pedestrian_count(pedestrians)
    for slot in range(N):
        env.world.deactivate(slot)
    return env


def _place(env: SimulationEnv, slot: int, at: tuple[float, float], heading: float, z: float | None = None) -> None:
    why = env.world.activate_at(slot, at, clearance_m=1.0, heading=heading, z=z)
    assert why is None, why


class TestRoadSurface:
    def test_flat_map_has_no_elevation(self, tiny_map: MapData) -> None:
        surface = RoadSurface(tiny_map)
        assert not tiny_map.elevated and not surface.elevated
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        z, cross, grade = surface.route_profile(pts, np.array([0.0, 10.0]), [])
        assert not z.any() and not cross.any() and not grade.any()
        assert build_map_index(tiny_map).solid_grids() is None

    def test_synthetic_course_has_every_structure(self, lab: MapData) -> None:
        kinds = {e.structure for e in lab.edges}
        assert {STRUCTURE_BRIDGE, STRUCTURE_TUNNEL} <= kinds
        assert lab.elevated and lab.terrain is not None
        assert all(e.elevation_source == "synthetic" for e in lab.edges)
        assert any(e.cross_slope != 0.0 for e in lab.edges)
        node_z = {n.id: n.z for n in lab.nodes}
        for e in lab.edges:
            # 辺の端の高さと端点のノードの高さが一致する（ランプと橋桁がつながっている）
            assert abs(e.elevation[0] - node_z[e.u]) < 1e-9
            assert abs(e.elevation[-1] - node_z[e.v]) < 1e-9

    def test_ground_roads_sit_on_carved_terrain(self, lab: MapData) -> None:
        for e in lab.edges:
            if e.structure != "ground":
                continue
            xs = np.array([p[0] for p in e.polyline])
            ys = np.array([p[1] for p in e.polyline])
            assert np.max(np.abs(lab.terrain.sample(xs, ys) - np.asarray(e.elevation))) < 0.25

    def test_solid_grids_have_deck_and_tunnel(self, lab_index) -> None:
        grids = lab_index.solid_grids()
        assert grids is not None
        assert np.isfinite(grids.slab_lo).sum() > 300
        assert np.isfinite(grids.cavity_lo).sum() > 800
        # 橋桁の中は固体、桁下（大通り）は空いている
        assert bool(grids.solid(OVERPASS_X, 0.0, DECK_Z - 0.5))
        assert not bool(grids.solid(OVERPASS_X, 0.0, 2.0))
        # トンネルの空洞は空いていて、その上の地山は固体
        assert not bool(grids.solid(-100.0, -40.0, 2.0))
        assert bool(grids.solid(-100.0, -40.0, 7.5))

    def test_terrain_grid_samples_bilinear(self) -> None:
        grid = TerrainGrid(0.0, 0.0, 2.0, 2, 2, np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32))
        assert float(grid.sample(1.0, 1.0)) == pytest.approx(3.0)
        assert float(grid.sample(-5.0, -5.0)) == pytest.approx(0.0)

    def test_route_grade_of_constant_slope(self) -> None:
        arc = np.arange(0.0, 50.0, 2.0)
        assert np.allclose(route_grade(arc, arc * 0.08)[3:-3], 0.08)


class TestAttitude:
    def test_slopes_to_attitude_and_gravity(self) -> None:
        pitch, roll = attitude_from_slopes(0.1, 0.0)
        assert float(pitch) == pytest.approx(math.atan(0.1)) and float(roll) == pytest.approx(0.0)
        gx, gy, gn = gravity_components(0.1, 0.0)
        assert float(gx) == pytest.approx(-GRAVITY * 0.1 / math.sqrt(1.01))
        assert float(gy) == pytest.approx(0.0) and float(gn) == pytest.approx(GRAVITY / math.sqrt(1.01))
        # 左が上がる横断勾配では、重力は右（左の負）へ引く
        _gx, gy, _gn = gravity_components(0.0, 0.06)
        assert float(gy) < 0.0

    def test_scalar_camera_pose_matches_renderer(self) -> None:
        for spec in CAMERA_RIG:
            pose = camera_pose_3d(10.0, -4.0, 3.2, 0.7, 0.05, -0.03, spec)
            batch = camera_batch(10.0, -4.0, 3.2, 0.7, 0.05, -0.03, spec)
            assert pose.eye_x == pytest.approx(float(batch.eye_x[0]), abs=1e-12)
            assert pose.eye_z == pytest.approx(float(batch.eye_z[0]), abs=1e-12)
            assert math.atan2(pose.sin_yaw, pose.cos_yaw) == pytest.approx(float(batch.yaw[0]), abs=1e-12)
            assert pose.roll == pytest.approx(float(batch.roll[0]), abs=1e-12)

    def test_level_camera_pose_has_no_roll(self) -> None:
        pose = camera_pose_3d(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, DEFAULT_CAMERA)
        assert pose.roll == pytest.approx(0.0, abs=1e-12)
        assert pose.eye_z == pytest.approx(DEFAULT_CAMERA.eye_height)


def _stop(dynamics, grade: float, v0: float = 10.0) -> float:
    fleet = VehicleFleet(1, dynamics=dynamics)
    fleet.reset_slot(0, 0.0, 0.0, 0.0)
    fleet.active[0] = True
    fleet.speed[0] = v0
    fleet.slope_x[0] = grade
    for _ in range(2000):
        if fleet.speed[0] <= 0.01:
            break
        fleet.step(np.array([-1.0]), np.array([0.0]), config.DT, 30.0)
    return float(fleet.x[0])


class TestSlopePhysics:
    @pytest.mark.parametrize("dynamics", [None, VehicleDynamics()], ids=["kinematic", "dynamic"])
    def test_braking_distance_follows_grade(self, dynamics) -> None:
        flat, up, down = _stop(dynamics, 0.0), _stop(dynamics, 0.08), _stop(dynamics, -0.08)
        assert up < flat < down
        # 理論の比（減速度 6 ± g·sinθ）に 3% 以内で合う
        theory_up = (6.0) / (6.0 + GRAVITY * 0.08 / math.sqrt(1.0064))
        assert up / flat == pytest.approx(theory_up, rel=0.03)

    @pytest.mark.parametrize("dynamics", [None, VehicleDynamics()], ids=["kinematic", "dynamic"])
    def test_full_throttle_climbs_slower(self, dynamics) -> None:
        speeds = []
        for grade in (0.08, 0.0, -0.08):
            fleet = VehicleFleet(1, dynamics=dynamics)
            fleet.reset_slot(0, 0.0, 0.0, 0.0)
            fleet.active[0] = True
            fleet.slope_x[0] = grade
            for _ in range(60):
                fleet.step(np.array([1.0]), np.array([0.0]), config.DT, 30.0)
            speeds.append(float(fleet.speed[0]))
        assert speeds[0] < speeds[1] < speeds[2]

    def test_brake_holds_on_hill_and_coasting_rolls_back(self) -> None:
        fleet = VehicleFleet(1, dynamics=VehicleDynamics())
        fleet.reset_slot(0, 0.0, 0.0, 0.0)
        fleet.active[0] = True
        fleet.slope_x[0] = 0.1
        for _ in range(100):
            fleet.step(np.array([-0.5]), np.array([0.0]), config.DT, 30.0)
        assert abs(float(fleet.speed[0])) < 0.01 and abs(float(fleet.x[0])) < 0.01
        for _ in range(40):
            fleet.step(np.array([0.0]), np.array([0.0]), config.DT, 30.0)
        assert float(fleet.speed[0]) < -1.0

    def test_kinematic_holds_on_hill(self) -> None:
        fleet = VehicleFleet(1)
        fleet.reset_slot(0, 0.0, 0.0, 0.0)
        fleet.active[0] = True
        fleet.slope_x[0] = 0.1
        for _ in range(40):
            fleet.step(np.array([0.0]), np.array([0.0]), config.DT, 30.0)
        assert float(fleet.speed[0]) == 0.0 and float(fleet.x[0]) == 0.0

    def test_bank_pushes_dynamic_car_downhill(self) -> None:
        fleet = VehicleFleet(1, dynamics=VehicleDynamics())
        fleet.reset_slot(0, 0.0, 0.0, 0.0)
        fleet.active[0] = True
        fleet.speed[0] = 10.0
        fleet.slope_y[0] = 0.06
        for _ in range(20):
            fleet.step(np.array([0.2]), np.array([0.0]), config.DT, 30.0)
        # 左が上がる路面では右（-y）へ流れる
        assert float(fleet.y[0]) < 0.0

    def test_flat_path_is_untouched_by_zero_slopes(self) -> None:
        results = []
        for dynamics in (None, VehicleDynamics()):
            a = VehicleFleet(2, dynamics=dynamics)
            b = VehicleFleet(2, dynamics=dynamics)
            for f in (a, b):
                f.reset_slot(0, 0.0, 0.0, 0.3)
                f.reset_slot(1, 5.0, 2.0, -0.2)
                f.active[:] = True
            b.slope_x[:] = 0.0
            for _ in range(30):
                for f in (a, b):
                    f.step(np.array([0.7, -0.2]), np.array([0.1, -0.3]), config.DT, 13.9)
            results.append(all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("x", "y", "heading", "speed")))
        assert all(results)


class TestWorld:
    def test_vehicle_follows_road_height_and_pitch(self, lab_index) -> None:
        env = _env(lab_index)
        _place(env, 0, (-250.0, 1.6), 0.0)
        world = env.world
        world.project_all()
        expected = float(hill_height(float(world.fleet.x[0]), float(world.fleet.y[0])))
        assert float(world.z[0]) == pytest.approx(expected, abs=0.3)
        # 東向きに山を上っているので機首上げ
        assert float(world.pitch[0]) > 0.02
        assert float(world.fleet.slope_x[0]) == pytest.approx(math.tan(float(world.pitch[0])), rel=1e-3)

    def test_over_and_under_do_not_collide(self, lab_index) -> None:
        env = _env(lab_index)
        world = env.world
        _place(env, 0, (OVERPASS_X + 1.6, -5.0), math.pi / 2, z=DECK_Z)
        _place(env, 1, (OVERPASS_X, 1.6), 0.0, z=0.0)
        # 平面ではちょうど重なる所へ置き直す
        world.fleet.x[1] = world.fleet.x[0]
        world.fleet.y[1] = world.fleet.y[0] + 0.5
        world.project_all()
        assert float(world.z[0]) == pytest.approx(DECK_Z, abs=0.2)
        assert float(world.z[1]) < 1.0
        assert not world.check_collisions()[:2].any()

    def test_same_level_overlap_collides(self, lab_index) -> None:
        env = _env(lab_index)
        world = env.world
        _place(env, 0, (300.0, 1.6), 0.0)
        _place(env, 1, (330.0, 1.6), 0.0)
        world.fleet.x[1] = world.fleet.x[0] + 1.0
        world.fleet.y[1] = world.fleet.y[0]
        world.project_all()
        assert world.check_collisions()[:2].all()

    def test_obstacle_and_pedestrian_levels(self, lab_index) -> None:
        env = _env(lab_index)
        world = env.world
        _place(env, 0, (OVERPASS_X, 1.6), 0.0, z=0.0)
        x, y = float(world.fleet.x[0]), float(world.fleet.y[0])
        # 橋桁の上に置いたパイロンは、桁下の車に当たらない
        assert world.add_obstacle(x + 0.5, y, 0.5, z=DECK_Z) is not None
        assert float(world.obstacles[0].z) == pytest.approx(DECK_Z, abs=0.3)
        assert not world.check_collisions()[0]
        world.clear_obstacles()
        world.add_obstacle(x + 0.5, y, 0.5, z=0.0)
        assert world.check_collisions()[0]

    def test_road_lead_picks_level_by_height(self, lab_index) -> None:
        upper = lab_index.road_lead(OVERPASS_X + 1.0, 0.0, math.pi / 2, 0.0, z=DECK_Z)
        lower = lab_index.road_lead(OVERPASS_X + 1.0, 0.0, 0.0, 0.0, z=0.0)
        assert upper is not None and lower is not None
        edges = {e.id: e for e in lab_index.data.edges}
        assert edges[upper.edge_ids[0]].structure == STRUCTURE_BRIDGE
        assert edges[lower.edge_ids[0]].structure != STRUCTURE_BRIDGE

    def test_snap_prefers_ground_level_by_default(self, lab_index) -> None:
        env = _env(lab_index)
        node = env.world._snap_node(OVERPASS_X, 100.0)
        assert float(env.world._node_z[node]) == pytest.approx(lab_index.surface_z(OVERPASS_X, 100.0), abs=0.6)

    def test_frame_carries_height_only_on_elevated_maps(self, lab_index, tiny_map) -> None:
        env = _env(lab_index)
        _place(env, 0, (-250.0, 1.6), 0.0)
        car = env.world.snapshot(0, 0.0, include_routes=True).vehicles[0].to_wire()
        assert {"z", "pitch", "roll", "routeZ"} <= set(car)
        assert len(car["routeZ"]) == len(car["route"])

    def test_autopilot_drives_course_without_false_collisions(self, lab_index) -> None:
        from app.sim.scenario import standard_scenario

        params = SimParams()
        params.pedestrian_count = 8
        env = SimulationEnv(lab_index, params, seed=7, scenario=standard_scenario(), compute_observations=False)
        env.autopilot_all = True
        act = np.zeros((N, config.ACTION_DIM), dtype=np.float32)
        world = env.world
        for _ in range(600):
            env.step(act)
            active = world.fleet.active
            assert np.all(np.isfinite(world.z[active]))
            assert np.all(np.abs(world.pitch[active]) < math.radians(8.0))


class TestCameraAndLabels:
    def test_render_matches_flat_renderer_on_flat_ground(self, lab: MapData) -> None:
        flat = copy.deepcopy(lab)
        flat.terrain = None
        for n in flat.nodes:
            n.z = 0.0
        for e in flat.edges:
            e.elevation, e.cross_slope, e.structure = None, 0.0, "ground"
        for item in (*flat.signals, *flat.signs):
            item.z = 0.0
        for b in flat.buildings:
            b.base_z = 0.0
        images = []
        for force in (False, True):
            index = build_map_index(flat)
            if force:
                from app.map.elevation import build_solid_grids

                index.elevated = True
                index.surface.elevated = True
                index.surface._grids = build_solid_grids(index.surface)
            env = _env(index)
            env.world.elevated = force
            for slot, at, heading in [(0, (-300.0, 1.6), 0.0), (1, (230.0, 1.6), 0.0)]:
                _place(env, slot, at, heading)
            env.world.project_all()
            cam = PseudoCamera(index)
            images.append([cam.render(env.world, np.array([0, 1]), spec=spec) for spec in CAMERA_RIG])
        for a, b in zip(*images):
            assert float((a == b).all(axis=3).mean()) > 0.995

    @pytest.mark.parametrize(
        ("name", "ego", "other", "visible"),
        [
            ("頂上の手前は見える", ((-70.0, 1.6), 0.0, None), ((-20.0, 1.6), 0.0, None), True),
            ("頂上の向こうは見えない", ((-70.0, 1.6), 0.0, None), ((20.0, 1.6), 0.0, None), False),
            ("桁下から橋の向こうが見える", ((225.0, 1.6), 0.0, 0.0), ((300.0, 1.6), 0.0, 0.0), True),
            ("坑内から山の上は見えない", ((-160.0, -38.4), 0.0, None), ((-100.0, 1.6), 0.0, None), False),
            ("坑内の前の車は見える", ((-160.0, -38.4), 0.0, None), ((-120.0, -38.4), 0.0, None), True),
        ],
    )
    def test_labels_and_pixels_agree(self, lab_index, name, ego, other, visible) -> None:
        env = _env(lab_index)
        clear_static_cache()
        _place(env, 0, ego[0], ego[1], ego[2])
        _place(env, 1, other[0], other[1], other[2])
        env.world.project_all()
        cam = PseudoCamera(lab_index)
        cam.render(env.world, np.array([0]))
        labels = cam.last_labels[0]
        pixels = int(((labels == LBL_VEHICLE) | (labels == LBL_VEHICLE_DARK)).sum())
        boxes = [d for d in detect_ground_truth(env.world, 0, DEFAULT_CAMERA).detections if d.cls == DetClass.VEHICLE]
        assert (len(boxes) == 1) is visible, name
        assert (pixels > 0) is visible, name

    def test_freespace_stops_at_deck_edge(self, lab_index) -> None:
        env = _env(lab_index)
        _place(env, 0, (OVERPASS_X + 1.6, -30.0), math.pi / 2, z=DECK_Z)
        env.world.project_all()
        clear_static_cache()
        free = freespace_ground_truth(env.world, 0)
        # 真横（±90 度）は橋桁の縁の落差で止まる。平面の判定のままなら 30m 空いて見える
        assert float(free[0]) < 8.0 and float(free[-1]) < 8.0
        assert float(free[len(free) // 2]) > 20.0


class TestOffLevelDetections:
    def test_car_on_same_road_is_kept_and_car_below_is_dropped(self) -> None:
        spec = DEFAULT_CAMERA
        # 同じ路面の 20m 先の車は、箱の下端が路面の高さに来る
        b = (-spec.eye_height) / 20.0
        v_bottom = spec.height * 0.5 - b * spec.focal_px
        same = Detection(DetClass.VEHICLE, 0.45, 0.4, 0.55, v_bottom / spec.height, 0.9, distance=20.0)
        below = Detection(DetClass.VEHICLE, 0.45, 0.8, 0.55, 0.98, 0.9, distance=20.0)
        sign = Detection(DetClass.SPEED_SIGN, 0.45, 0.8, 0.55, 0.98, 0.9, distance=20.0)
        assert abs(level_offset(same, spec)) < 1.0
        assert level_offset(below, spec) < -OFF_LEVEL_M
        kept = drop_other_levels(PerceptionResult(0, [same, below, sign]), spec).detections
        assert same in kept and sign in kept and below not in kept


class TestOsmTags:
    def test_structure_layer_and_incline_parsing(self) -> None:
        from app.map.loader import _parse_incline, _parse_layer, _parse_structure

        assert _parse_structure("yes", None) == "bridge"
        assert _parse_structure("viaduct", "no") == "bridge"
        assert _parse_structure(None, "building_passage") == "tunnel"
        assert _parse_structure("no", float("nan")) == "ground"
        assert _parse_layer("1") == 1 and _parse_layer(["-1", "2"]) == 2 and _parse_layer(None) == 0
        assert _parse_layer("9") == 5
        assert _parse_incline("8%", False) == pytest.approx(0.08)
        assert _parse_incline("8%", True) == pytest.approx(-0.08)
        assert _parse_incline("up", False) is None
        assert _parse_incline("5°", False) == pytest.approx(math.tan(math.radians(5.0)))

    def test_cache_keeps_structure_and_reads_old_versions_as_unknown(self) -> None:
        from app.map.loader import CACHE_VERSION, _from_cache_dict, _to_cache_dict

        source = _crossing_map()
        for e in source.edges:
            e.layer = 0
        data = copy.deepcopy(source)
        data.edges[0].structure = "bridge"
        data.edges[0].layer = 1
        data.edges[0].incline = 0.04
        data.nodes[0].ele = 12.5
        preset = MapPreset(data.preset_id, data.name, "", 0, 0, data.radius_m)
        payload = _to_cache_dict(data)
        assert payload["version"] == CACHE_VERSION == 14
        restored = _from_cache_dict(payload, preset)
        assert restored.edges[0].structure == "bridge" and restored.edges[0].layer == 1
        assert restored.edges[0].incline == pytest.approx(0.04) and restored.nodes[0].ele == pytest.approx(12.5)
        # 高さは作らない（layer は順序で実高さではない）
        assert not restored.elevated
        old = _to_cache_dict(source)
        old["version"] = 13
        legacy = _from_cache_dict(old, preset)
        assert all(e.structure == "ground" and e.layer == 0 and e.incline is None for e in legacy.edges)


def _crossing_map() -> MapData:
    """東西の道路（layer 0）と、交点を持たずに上を通る南北の道路（layer 1）。高さは不明（平面）。"""
    nodes = [
        MapNode(0, -100.0, 0.0), MapNode(1, 100.0, 0.0), MapNode(2, 0.0, -100.0), MapNode(3, 0.0, 100.0),
        MapNode(4, -100.0, -100.0), MapNode(5, 100.0, 100.0),
    ]
    edges = []
    for u, v, layer in ((0, 1, 0), (2, 3, 1), (4, 0, 0), (4, 2, 0), (1, 5, 0), (3, 5, 0)):
        a, b = nodes[u], nodes[v]
        for s, t in ((a, b), (b, a)):
            edges.append(MapEdge(len(edges), s.id, t.id, 2, 7.0, False, 11.1, [(s.x, s.y), (t.x, t.y)], math.dist((s.x, s.y), (t.x, t.y)), layer=layer))
    return MapData("cross", "cross", 0, 0, 150.0, Bounds(-120, -120, 120, 120), nodes, edges, [])


class TestLayeredFlatMaps:
    def test_different_layers_do_not_collide_without_heights(self) -> None:
        index = build_map_index(_crossing_map())
        env = _env(index)
        world = env.world
        assert world.layered and not world.elevated
        _place(env, 0, (-20.0, 1.6), 0.0)
        _place(env, 1, (1.6, -20.0), math.pi / 2)
        world.fleet.x[0], world.fleet.y[0] = 0.0, 1.6
        world.fleet.x[1], world.fleet.y[1] = 1.0, 1.0
        world.project_all()
        assert world.current_edge[0] >= 0 and world.current_edge[1] >= 0
        assert not world.check_collisions()[:2].any()
        assert not world.same_level_vehicles(0)[1]

    def test_signals_of_other_layers_are_not_merged(self) -> None:
        from app.contracts import MapSignal
        from app.map.loader import _merge_phases

        a = MapSignal(0, 0, 0.0, 0.0, 0.0, 0, 7.0, 0, 0)
        b = MapSignal(1, 1, 3.0, 0.0, math.pi / 2, 0, 7.0, 1, 1)
        merged = _merge_phases([a, b], [0, 1])
        assert merged[0].phase_key != merged[1].phase_key
        merged_same = _merge_phases([a, b], [0, 0])
        assert merged_same[0].phase_key == merged_same[1].phase_key
