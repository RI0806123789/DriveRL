/** モックの見通しと死角。**本物（`percep/occlusion.py`）と同じく画角を 35 本に分け**、他の車の陰だけを作る（モックに走行可能距離は無い）。 */

import { DRIVER_FORWARD, DRIVER_RIGHT, PSEUDO_CAMERA, SURROUND_CAMERAS } from '../../scene/cameraMath.ts'
import type { OcclusionCamera, OcclusionShadow, OcclusionView, VehicleState } from '../../types/protocol.ts'

/** 本物の `OCCLUSION_RANGE_M`（= 走行可能距離の上限 30m）と `RAYS_PER_CAMERA` と同じ */
const RANGE = 30
const RAYS = 35
const HALF_FOV = (PSEUDO_CAMERA.fovDeg * Math.PI) / 360
const RAY_HALF = HALF_FOV / RAYS
const VEHICLE_HALF_WIDTH = 0.9

interface Mount {
  key: OcclusionCamera['key']
  forward: number
  right: number
  yaw: number
}

const MOUNTS: readonly Mount[] = [
  { key: 'front', forward: DRIVER_FORWARD, right: DRIVER_RIGHT, yaw: 0 },
  ...SURROUND_CAMERAS.map((c) => ({ key: c.key, forward: c.forward, right: c.right, yaw: (c.yawDeg * Math.PI) / 180 })),
]

const RAY_ANGLES = Array.from({ length: RAYS }, (_, k) => -HALF_FOV + RAY_HALF * (2 * k + 1))

function wrap(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a))
}

const round = (v: number, digits: number) => Number(v.toFixed(digits))

/** 1 台ぶんの見通しと死角（本物は `env._decorate` が `watch_occlusion` で頼まれた車にだけ載せる） */
export function mockOcclusion(car: VehicleState, vehicles: readonly VehicleState[]): OcclusionView {
  const c = Math.cos(car.heading)
  const s = Math.sin(car.heading)
  const cameras: OcclusionCamera[] = []
  const shadows: OcclusionShadow[] = []
  const depthsOf = new Map<string, number[]>()
  let frontCovered = 0
  for (const mount of MOUNTS) {
    const ex = car.x + c * mount.forward + s * mount.right
    const ey = car.y + s * mount.forward - c * mount.right
    const depths = RAY_ANGLES.map(() => RANGE)
    const covered = RAY_ANGLES.map(() => false)
    for (const other of vehicles) {
      if (!other.active || other.id === car.id) continue
      const dist = Math.hypot(other.x - ex, other.y - ey)
      if (dist >= RANGE || dist < 1e-3) continue
      const rel = wrap(Math.atan2(other.y - ey, other.x - ex) - car.heading - mount.yaw)
      const half = Math.atan(VEHICLE_HALF_WIDTH / dist)
      const from = Math.max(rel - half, -HALF_FOV)
      const to = Math.min(rel + half, HALF_FOV)
      if (from > to) continue
      let hit = false
      RAY_ANGLES.forEach((a, k) => {
        if (a >= from && a <= to) {
          depths[k] = Math.min(depths[k], dist)
          covered[k] = true
          hit = true
        }
      })
      if (!hit) {
        let k = 0
        RAY_ANGLES.forEach((a, i) => {
          if (Math.abs(a - (from + to) / 2) < Math.abs(RAY_ANGLES[k] - (from + to) / 2)) k = i
        })
        depths[k] = Math.min(depths[k], dist)
        covered[k] = true
      }
      shadows.push({ camera: mount.key, kind: 'dynamic', from: round(from, 3), to: round(to, 3), near: round(dist, 1) })
    }
    if (mount.key === 'front') frontCovered = covered.filter(Boolean).length / RAYS
    depthsOf.set(mount.key, depths)
    const seen: [number, number, number][] = []
    RAY_ANGLES.forEach((a, k) => {
      const d = round(depths[k], 1)
      const last = seen[seen.length - 1]
      if (last && Math.abs(last[2] - d) < 0.05) last[1] = round(a + RAY_HALF, 3)
      else seen.push([round(a - RAY_HALF, 3), round(a + RAY_HALF, 3), d])
    })
    cameras.push({ key: mount.key, at: [round(mount.forward, 2), round(-mount.right, 2) + 0], yaw: round(mount.yaw, 4), seen })
  }
  const sector = (key: string) => {
    const d = depthsOf.get(key) ?? []
    return round(d.reduce((sum, v) => sum + v * v, 0) * (2 * RAY_HALF) / (RANGE * RANGE * (Math.PI / 2)), 3)
  }
  const los = (key: string) => round(Math.max(...(depthsOf.get(key) ?? [0])), 1)
  return {
    range: RANGE,
    los: [los('left'), los('right')],
    sectors: [sector('front'), sector('rear'), sector('left'), sector('right')],
    frontOccluded: round(frontCovered, 3),
    cameras,
    shadows,
  }
}
