/** 徒歩キャラクターの当たり判定。**React から切り離した純粋モジュール。** */

import type { MapBuilding } from '../types/protocol'

/** 歩行者の体の半径 [m] */
export const PEDESTRIAN_RADIUS = 0.34

/** 車体の当たり判定に使う寸法 [m]（backend/app/config.py と揃える） */
export const VEHICLE_HALF_LENGTH = 2.2
export const VEHICLE_HALF_WIDTH = 0.9

/** 建物の格子索引。**毎フレーム数千件を総当たりしないためだけに存在する** */
export interface BuildingIndex {
  cellSize: number
  minX: number
  minY: number
  cols: number
  rows: number
  /** セルごとの建物添字。空セルは null */
  cells: (Int32Array | null)[]
  /** 建物の外周を [x0, y0, x1, y1, ...] で平坦化したもの */
  outlines: Float64Array[]
}

export interface VehicleBlocker {
  id: number
  x: number
  y: number
  /** ラジアン。ENU +x から反時計回り */
  heading: number
}

const DEFAULT_CELL_M = 24

/** 建物一覧から格子索引を作る。**マップが変わったときに 1 回だけ呼ぶこと。** */
export function buildBuildingIndex(
  buildings: readonly MapBuilding[],
  cellSize: number = DEFAULT_CELL_M,
): BuildingIndex {
  const outlines: Float64Array[] = []
  const boxes: number[][] = []

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const b of buildings) {
    const pts = b.outline
    if (!pts || pts.length < 3) continue
    const flat = new Float64Array(pts.length * 2)
    let bx0 = Infinity
    let by0 = Infinity
    let bx1 = -Infinity
    let by1 = -Infinity
    for (let i = 0; i < pts.length; i++) {
      const x = pts[i][0]
      const y = pts[i][1]
      flat[i * 2] = x
      flat[i * 2 + 1] = y
      if (x < bx0) bx0 = x
      if (y < by0) by0 = y
      if (x > bx1) bx1 = x
      if (y > by1) by1 = y
    }
    outlines.push(flat)
    boxes.push([bx0, by0, bx1, by1])
    if (bx0 < minX) minX = bx0
    if (by0 < minY) minY = by0
    if (bx1 > maxX) maxX = bx1
    if (by1 > maxY) maxY = by1
  }

  if (outlines.length === 0) {
    return { cellSize, minX: 0, minY: 0, cols: 0, rows: 0, cells: [], outlines: [] }
  }

  const cols = Math.max(1, Math.ceil((maxX - minX) / cellSize) + 1)
  const rows = Math.max(1, Math.ceil((maxY - minY) / cellSize) + 1)

  const buckets: number[][] = new Array(cols * rows)
  for (let i = 0; i < boxes.length; i++) {
    const [bx0, by0, bx1, by1] = boxes[i]
    const c0 = Math.max(0, Math.floor((bx0 - minX) / cellSize))
    const c1 = Math.min(cols - 1, Math.floor((bx1 - minX) / cellSize))
    const r0 = Math.max(0, Math.floor((by0 - minY) / cellSize))
    const r1 = Math.min(rows - 1, Math.floor((by1 - minY) / cellSize))
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const k = r * cols + c
        ;(buckets[k] ??= []).push(i)
      }
    }
  }

  const cells: (Int32Array | null)[] = new Array(cols * rows)
  for (let k = 0; k < cells.length; k++) {
    const list = buckets[k]
    cells[k] = list === undefined ? null : Int32Array.from(list)
  }

  return { cellSize, minX, minY, cols, rows, cells, outlines }
}

/** 点が多角形の内側か（レイキャスト法）。 */
function insidePolygon(flat: Float64Array, x: number, y: number): boolean {
  let inside = false
  const n = flat.length / 2
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = flat[i * 2]
    const yi = flat[i * 2 + 1]
    const xj = flat[j * 2]
    const yj = flat[j * 2 + 1]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** その点が建物の内側か。索引の外（＝建物が 1 つも無い区画）は常に false。 */
export function isInsideBuilding(index: BuildingIndex, x: number, y: number): boolean {
  if (index.cols === 0) return false
  const c = Math.floor((x - index.minX) / index.cellSize)
  const r = Math.floor((y - index.minY) / index.cellSize)
  if (c < 0 || r < 0 || c >= index.cols || r >= index.rows) return false
  const bucket = index.cells[r * index.cols + c]
  if (bucket === null) return false
  for (let i = 0; i < bucket.length; i++) {
    if (insidePolygon(index.outlines[bucket[i]], x, y)) return true
  }
  return false
}

/** 半径ぶんの余裕を見て建物に触れているか（中心と上下左右の 5 点で見る）。 */
export function touchesBuilding(
  index: BuildingIndex | null,
  x: number,
  y: number,
  radius: number = PEDESTRIAN_RADIUS,
): boolean {
  if (index === null) return false
  return (
    isInsideBuilding(index, x, y) ||
    isInsideBuilding(index, x + radius, y) ||
    isInsideBuilding(index, x - radius, y) ||
    isInsideBuilding(index, x, y + radius) ||
    isInsideBuilding(index, x, y - radius)
  )
}

/** 車体（向きつきの矩形）に半径ぶん膨らませた領域へ入っているか。 */
export function touchesVehicle(
  blocker: VehicleBlocker,
  x: number,
  y: number,
  radius: number = PEDESTRIAN_RADIUS,
): boolean {
  const dx = x - blocker.x
  const dy = y - blocker.y
  const cos = Math.cos(blocker.heading)
  const sin = Math.sin(blocker.heading)
  const lon = dx * cos + dy * sin
  const lat = -dx * sin + dy * cos
  return (
    Math.abs(lon) <= VEHICLE_HALF_LENGTH + radius &&
    Math.abs(lat) <= VEHICLE_HALF_WIDTH + radius
  )
}

export interface MoveResult {
  x: number
  y: number
  /** 建物か車体に当たって、行きたい方向のどちらかが削られたか */
  blocked: boolean
}

/** 移動を軸ごとに試して、当たった軸だけ捨てる（壁沿いに滑らせるための定石）。 */
export function resolveMove(
  index: BuildingIndex | null,
  fromX: number,
  fromY: number,
  dx: number,
  dy: number,
  blockers: readonly VehicleBlocker[],
  radius: number = PEDESTRIAN_RADIUS,
): MoveResult {
  const occupied = (x: number, y: number): boolean => {
    if (touchesBuilding(index, x, y, radius)) return true
    for (let i = 0; i < blockers.length; i++) {
      if (touchesVehicle(blockers[i], x, y, radius)) return true
    }
    return false
  }

  let x = fromX
  let y = fromY
  let blocked = false

  if (dx !== 0) {
    if (occupied(x + dx, y)) blocked = true
    else x += dx
  }
  if (dy !== 0) {
    if (occupied(x, y + dy)) blocked = true
    else y += dy
  }
  return { x, y, blocked }
}

/** 照準に入る距離 [m] と、視線からの許容角度 [rad] */
export const AIM_DISTANCE_M = 9
export const AIM_ANGLE_RAD = 0.5

/** 照準（体の向き）の先にいる車両のスロット番号。 */
export function aimedVehicle(
  x: number,
  y: number,
  heading: number,
  candidates: readonly VehicleBlocker[],
  maxDistance: number = AIM_DISTANCE_M,
  maxAngle: number = AIM_ANGLE_RAD,
): number {
  let best = -1
  let bestDist = Infinity
  for (const v of candidates) {
    const dx = v.x - x
    const dy = v.y - y
    const dist = Math.hypot(dx, dy)
    if (dist > maxDistance) continue
    // 真横に接して立っているときは角度が跳ねるので、触れていれば角度を問わない
    if (!touchesVehicle(v, x, y, PEDESTRIAN_RADIUS + 0.6)) {
      const want = Math.atan2(dy, dx)
      const diff = Math.abs(Math.atan2(Math.sin(want - heading), Math.cos(want - heading)))
      if (diff > maxAngle) continue
    }
    if (dist < bestDist) {
      bestDist = dist
      best = v.id
    }
  }
  return best
}

/** 降車するときに立つ位置（車体の左側面の外）。 */
export function alightPosition(
  blocker: VehicleBlocker,
  offset: number = VEHICLE_HALF_WIDTH + PEDESTRIAN_RADIUS + 0.5,
): { x: number; y: number } {
  // 左側通行なので、歩道側（進行方向の左）へ降ろす
  return {
    x: blocker.x - Math.sin(blocker.heading) * offset,
    y: blocker.y + Math.cos(blocker.heading) * offset,
  }
}
