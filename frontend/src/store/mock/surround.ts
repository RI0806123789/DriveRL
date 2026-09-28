/** モックの周囲カメラの検出。**本物（`percep/groundtruth.py`）と同じ投影式**で、他の車と歩行者を後方・左右のカメラへ写す。 */

import { PSEUDO_CAMERA, SURROUND_CAMERAS, type SurroundCamera } from '../../scene/cameraMath.ts'
import {
  DET_PEDESTRIAN,
  DET_VEHICLE,
  type Detection,
  type NpcPedestrianState,
  type SurroundDetections,
  type VehicleState,
} from '../../types/protocol.ts'

const FOCAL = (PSEUDO_CAMERA.width * 0.5) / Math.tan((PSEUDO_CAMERA.fovDeg * Math.PI) / 360)
const NEAR = 0.5
const REACH = 60
/** 危険度の見本（本物は `sim/safety.py` が介入の根拠にした枠だけに付ける） */
const HAZARD_STOP_M = 3
const HAZARD_CAUTION_M = 6

interface Target {
  cls: number
  x: number
  y: number
  halfWidth: number
  height: number
}

/** ENU の点をカメラ画像の正規化座標へ。後ろなら null */
function project(
  cam: SurroundCamera,
  car: VehicleState,
  px: number,
  py: number,
  pz: number,
): [number, number, number] | null {
  const c = Math.cos(car.heading)
  const s = Math.sin(car.heading)
  const ex = car.x + c * cam.forward + s * cam.right
  const ey = car.y + s * cam.forward - c * cam.right
  const yaw = car.heading + (cam.yawDeg * Math.PI) / 180
  const pitch = (cam.pitchDeg * Math.PI) / 180
  const rx = px - ex
  const ry = py - ey
  const rz = pz - cam.height
  const fwd = rx * Math.cos(yaw) + ry * Math.sin(yaw)
  const right = rx * Math.sin(yaw) - ry * Math.cos(yaw)
  const depth = Math.cos(pitch) * fwd + Math.sin(pitch) * rz
  const up = -Math.sin(pitch) * fwd + Math.cos(pitch) * rz
  if (depth <= NEAR) return null
  const u = 0.5 + (FOCAL * right) / depth / PSEUDO_CAMERA.width
  const v = 0.5 - (FOCAL * up) / depth / PSEUDO_CAMERA.height
  return [u, v, depth]
}

function detect(cam: SurroundCamera, car: VehicleState, target: Target): Detection | null {
  const dx = target.x - car.x
  const dy = target.y - car.y
  const dist = Math.hypot(dx, dy)
  if (dist > REACH || dist < 1e-3) return null
  // 視線に垂直な板で近似する（本物と同じ）
  const ax = -dy / dist
  const ay = dx / dist
  const corners: Array<[number, number, number]> = []
  for (const side of [-1, 1]) {
    for (const z of [0, target.height]) {
      const p = project(cam, car, target.x + ax * side * target.halfWidth, target.y + ay * side * target.halfWidth, z)
      if (p) corners.push(p)
    }
  }
  if (corners.length === 0) return null
  const us = corners.map((p) => p[0])
  const vs = corners.map((p) => p[1])
  const x0 = Math.max(0, Math.min(...us))
  const x1 = Math.min(1, Math.max(...us))
  const y0 = Math.max(0, Math.min(...vs))
  const y1 = Math.min(1, Math.max(...vs))
  if (x1 <= x0 || y1 <= y0) return null
  const det: Detection = { cls: target.cls, box: [x0, y0, x1, y1], conf: 1, distance: dist }
  if (dist <= HAZARD_STOP_M) det.hazard = 2
  else if (dist <= HAZARD_CAUTION_M) det.hazard = 1
  return det
}

/** 1 台ぶんの周囲カメラの検出 */
export function mockSurround(
  car: VehicleState,
  vehicles: readonly VehicleState[],
  pedestrians: readonly NpcPedestrianState[],
): SurroundDetections {
  const targets: Target[] = [
    ...vehicles
      .filter((v) => v.active && v.id !== car.id)
      .map((v) => ({ cls: DET_VEHICLE, x: v.x, y: v.y, halfWidth: 0.9, height: 1.45 })),
    ...pedestrians.map((p) => ({ cls: DET_PEDESTRIAN, x: p.x, y: p.y, halfWidth: 0.26, height: 1.68 })),
  ]
  const out: SurroundDetections = {}
  for (const cam of SURROUND_CAMERAS) {
    out[cam.key] = targets
      .map((t) => detect(cam, car, t))
      .filter((d): d is Detection => d !== null)
      .sort((a, b) => (a.distance ?? 0) - (b.distance ?? 0))
      .slice(0, 8)
  }
  return out
}
