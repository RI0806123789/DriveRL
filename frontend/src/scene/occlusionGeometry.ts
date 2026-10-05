/** 見通しと死角（`frame.occlusion`）を、追従中の車のローカル座標の三角形へ描き下ろす純粋関数。 */

import type { OcclusionCamera, OcclusionShadow, OcclusionView, Vec2 } from '../types/protocol.ts'

/** 路面から浮かせる高さ [m]（車線の認識結果より上、車体より下） */
export const OCCLUSION_LIFT = 0.06
/** 弧を刻む角度 [rad]（約 3 度） */
export const ARC_STEP_RAD = (3 * Math.PI) / 180
/** 1 枚の板に書ける頂点の上限（4 台 × 35 本の扇と死角 30 個を十分に覆う） */
export const OCCLUSION_MAX_VERTICES = 9000

export const FRUSTUM_COLOR = '#10b981'
export const DYNAMIC_SHADOW_COLOR = '#ef4444'
export const STATIC_SHADOW_COLOR = '#4c1d95'

export type Rgb = readonly [number, number, number]

/** 自車座標の点（前方 +x / 左 +y、単位 m） */
export interface LocalPoint {
  x: number
  y: number
}

/** カメラの位置 `at`・向き `yaw` から、方位 `angle`（カメラの向きから測り左が正）・奥行き `depth` の点 */
export function polarPoint(at: Vec2, yaw: number, angle: number, depth: number): LocalPoint {
  const bearing = yaw + angle
  return { x: at[0] + depth * Math.cos(bearing), y: at[1] + depth * Math.sin(bearing) }
}

/** 扇（方位 from..to・奥行き near..far）の凸多角形。手前は弦・奥は弧。near が 0 ならカメラの位置 1 点から始まる */
export function sectorPolygon(
  at: Vec2,
  yaw: number,
  from: number,
  to: number,
  near: number,
  far: number,
  step: number = ARC_STEP_RAD,
): LocalPoint[] {
  const lo = Math.min(from, to)
  const hi = Math.max(from, to)
  const count = Math.max(1, Math.ceil((hi - lo) / Math.max(step, 1e-3)))
  const arc: LocalPoint[] = []
  for (let i = 0; i <= count; i += 1) arc.push(polarPoint(at, yaw, lo + ((hi - lo) * i) / count, far))
  if (near <= 1e-3) return [{ x: at[0], y: at[1] }, ...arc]
  return [polarPoint(at, yaw, lo, near), ...arc, polarPoint(at, yaw, hi, near)]
}

/** 凸（または先頭の頂点から見て星形）の多角形を先頭から扇状に三角形へ分け、three の車両ローカルで書き込む。書いた頂点数 */
export function writeFan(
  points: readonly LocalPoint[],
  color: Rgb,
  positions: Float32Array,
  colors: Float32Array,
  start: number,
  capacity: number = OCCLUSION_MAX_VERTICES,
): number {
  let v = start
  for (let i = 1; i + 1 < points.length; i += 1) {
    if (v + 3 > capacity) break
    for (const p of [points[0], points[i], points[i + 1]]) {
      // three の車両ローカルは X 前方・Y 上・Z 右（左 +y は -Z）
      positions[v * 3] = p.x
      positions[v * 3 + 1] = OCCLUSION_LIFT
      positions[v * 3 + 2] = -p.y
      colors[v * 3] = color[0]
      colors[v * 3 + 1] = color[1]
      colors[v * 3 + 2] = color[2]
      v += 1
    }
  }
  return v - start
}

/** 見えている扇（4 台のカメラの `seen`）を書き込む。書いた頂点数 */
export function writeSeen(
  view: OcclusionView,
  color: Rgb,
  positions: Float32Array,
  colors: Float32Array,
  capacity: number = OCCLUSION_MAX_VERTICES,
): number {
  let count = 0
  for (const cam of view.cameras) {
    for (const [from, to, depth] of cam.seen) {
      if (!(depth > 0)) continue
      const polygon = sectorPolygon(cam.at, cam.yaw, from, to, 0, depth)
      count += writeFan(polygon, color, positions, colors, count, capacity)
    }
  }
  return count
}

/** 死角を書き込む。`colorOf` が null を返す種類は描かない。書いた頂点数 */
export function writeShadows(
  view: OcclusionView,
  colorOf: (kind: OcclusionShadow['kind']) => Rgb | null,
  positions: Float32Array,
  colors: Float32Array,
  capacity: number = OCCLUSION_MAX_VERTICES,
): number {
  const cameras = new Map<string, OcclusionCamera>(view.cameras.map((c) => [c.key, c]))
  let count = 0
  for (const shadow of view.shadows) {
    const cam = cameras.get(shadow.camera)
    const color = colorOf(shadow.kind)
    if (!cam || !color || !(view.range > shadow.near)) continue
    const polygon = sectorPolygon(cam.at, cam.yaw, shadow.from, shadow.to, shadow.near, view.range)
    count += writeFan(polygon, color, positions, colors, count, capacity)
  }
  return count
}
