/** 立体の地図の路面モデル（純粋関数）。バックエンドの `map/elevation.py` と同じ式で高さを引く。 */

import type { MapEdge, MapMessage, RoadStructure, TerrainGrid, Vec2 } from '../types/protocol.ts'

/** トンネルの中の明るさの倍率。**バックエンドの `config.TUNNEL_LIGHT_FACTOR` と同じ値**（擬似カメラと揃える） */
export const TUNNEL_LIGHT_FACTOR = 0.45
/** 橋桁の厚さ [m]。**バックエンドの `map/elevation.py` の DECK_THICKNESS_M と同じ値** */
export const DECK_THICKNESS_M = 1.2
/** トンネルの内空の高さ [m]。**バックエンドの TUNNEL_CLEARANCE_M と同じ値** */
export const TUNNEL_CLEARANCE_M = 5.0
/** 橋桁・トンネルの幅に足す余裕 [m]。**バックエンドの STRUCTURE_MARGIN_M と同じ値** */
export const STRUCTURE_MARGIN_M = 0.5
/** 高さがこれ以上離れた路面は別の階とみなす [m]。**バックエンドの LEVEL_GAP_M と同じ値** */
export const LEVEL_GAP_M = 3.0

/** 1 本の辺の中心線の高さ（頂点ごと）と、u→v の左が上がる横断勾配。 */
export interface EdgeProfile {
  readonly edge: MapEdge
  readonly cum: Float64Array
  readonly z: Float64Array
  readonly cross: number
  readonly structure: RoadStructure
}

/** 立体の地図の路面モデル。平面の地図では作らない（`buildElevationModel` が null を返す）。 */
export interface ElevationModel {
  readonly terrain: TerrainGrid | null
  readonly profiles: ReadonlyMap<number, EdgeProfile>
}

function cumulative(points: readonly Vec2[]): Float64Array {
  const out = new Float64Array(points.length)
  for (let i = 1; i < points.length; i++) {
    out[i] = out[i - 1] + Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
  }
  return out
}

/** 辺の頂点ごとの高さ。`elevation` が無ければ端点の高さを弧長で結ぶ（両方 0 なら平面）。 */
export function edgeHeights(edge: MapEdge, nodeZ: ReadonlyMap<number, number>): Float64Array {
  const n = edge.polyline.length
  if (edge.elevation && edge.elevation.length === n) return Float64Array.from(edge.elevation)
  const zu = nodeZ.get(edge.u) ?? 0
  const zv = nodeZ.get(edge.v) ?? 0
  const cum = cumulative(edge.polyline)
  const total = cum[n - 1] || 1
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = zu + ((zv - zu) * cum[i]) / total
  return out
}

/** 地図から路面モデルを作る。高さを持たない地図では null（3D は従来どおり平面で描く）。 */
export function buildElevationModel(map: MapMessage | null): ElevationModel | null {
  if (!map || !map.elevated) return null
  const nodeZ = new Map<number, number>()
  for (const n of map.nodes) nodeZ.set(n.id, n.z ?? 0)
  const profiles = new Map<number, EdgeProfile>()
  for (const edge of map.edges) {
    if (edge.polyline.length < 2) continue
    profiles.set(edge.id, {
      edge,
      cum: cumulative(edge.polyline),
      z: edgeHeights(edge, nodeZ),
      cross: edge.crossSlope ?? 0,
      structure: edge.structure ?? 'ground',
    })
  }
  return { terrain: map.terrain ?? null, profiles }
}

const cache = new WeakMap<MapMessage, ElevationModel | null>()

/** 地図ごとに 1 度だけ作った路面モデル。 */
export function elevationOf(map: MapMessage | null): ElevationModel | null {
  if (!map) return null
  let model = cache.get(map)
  if (model === undefined) {
    model = buildElevationModel(map)
    cache.set(map, model)
  }
  return model
}

/** 地面の高さ（双一次補間）。地形が無ければ 0。 */
export function terrainZ(terrain: TerrainGrid | null, x: number, y: number): number {
  if (!terrain) return 0
  const fx = Math.min(Math.max((x - terrain.originX) / terrain.cellSize, 0), terrain.width - 1.000001)
  const fy = Math.min(Math.max((y - terrain.originY) / terrain.cellSize, 0), terrain.height - 1.000001)
  const ix = Math.floor(fx)
  const iy = Math.floor(fy)
  const tx = fx - ix
  const ty = fy - iy
  const w = terrain.width
  const z = terrain.z
  const top = z[iy * w + ix] * (1 - tx) + z[iy * w + ix + 1] * tx
  const bottom = z[(iy + 1) * w + ix] * (1 - tx) + z[(iy + 1) * w + ix + 1] * tx
  return top * (1 - ty) + bottom * ty
}

/** 点を辺の中心線へ射影した (弧長, 左を正とする横位置, 距離)。 */
export function projectOnProfile(profile: EdgeProfile, x: number, y: number): { s: number; lateral: number; dist: number } {
  const pts = profile.edge.polyline
  let best = { s: 0, lateral: 0, dist: Infinity }
  for (let i = 0; i < pts.length - 1; i++) {
    const ax = pts[i][0]
    const ay = pts[i][1]
    const dx = pts[i + 1][0] - ax
    const dy = pts[i + 1][1] - ay
    const len2 = Math.max(dx * dx + dy * dy, 1e-12)
    const t = Math.min(Math.max(((x - ax) * dx + (y - ay) * dy) / len2, 0), 1)
    const fx = ax + dx * t
    const fy = ay + dy * t
    const dist = Math.hypot(x - fx, y - fy)
    if (dist < best.dist) {
      const len = Math.sqrt(len2)
      best = { s: profile.cum[i] + t * len, lateral: (-(x - fx) * dy + (y - fy) * dx) / len, dist }
    }
  }
  return best
}

/** 辺の弧長 s・横位置 lateral の路面の高さ。 */
export function profileZ(profile: EdgeProfile, s: number, lateral = 0): number {
  const cum = profile.cum
  const z = profile.z
  const n = cum.length
  let base: number
  if (s <= 0) base = z[0]
  else if (s >= cum[n - 1]) base = z[n - 1]
  else {
    let lo = 0
    let hi = n - 1
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (cum[mid] <= s) lo = mid
      else hi = mid
    }
    const span = cum[hi] - cum[lo] || 1
    base = z[lo] + ((z[hi] - z[lo]) * (s - cum[lo])) / span
  }
  return base + lateral * profile.cross
}

/** (x, y) にある路面の高さ。道路が重なる所では `hint` に近い路面、無ければ地面（バックエンドの `surface_z` と同じ選び方）。 */
export function surfaceZAt(model: ElevationModel | null, x: number, y: number, hint?: number): number {
  if (!model) return 0
  const ground = terrainZ(model.terrain, x, y)
  const want = hint ?? ground
  let best = ground
  let bestScore = Infinity
  for (const profile of model.profiles.values()) {
    const p = projectOnProfile(profile, x, y)
    if (p.dist > profile.edge.width / 2 + 1) continue
    const z = profileZ(profile, p.s, p.lateral)
    const score = Math.abs(z - want)
    if (score < bestScore) {
      bestScore = score
      best = z
    }
  }
  return best
}

/** 車体の前（three の車両ローカル +x）・上（+y）・右（+z）の軸を、路面の姿勢込みで three の世界座標に直した行列の要素（列優先 3×3）。 */
export function attitudeAxes(heading: number, pitch: number, roll: number): Float64Array {
  // ENU で R = Rz(方位)·Ry(-ピッチ)·Rx(ロール)。列は車体の前・左・上（バックエンドの `body_rotation` と同じ）
  const ch = Math.cos(heading)
  const sh = Math.sin(heading)
  const cp = Math.cos(pitch)
  const sp = Math.sin(pitch)
  const cr = Math.cos(roll)
  const sr = Math.sin(roll)
  const fwd = [ch * cp, sh * cp, sp]
  const left = [-sh * cr - ch * sp * sr, ch * cr - sh * sp * sr, cp * sr]
  const up = [sh * sr - ch * sp * cr, -ch * sr - sh * sp * cr, cp * cr]
  // three の世界座標 (x, y, z) = ENU (x, z, -y)。車両ローカルの右 = -左
  return Float64Array.from([
    fwd[0], fwd[2], -fwd[1],
    up[0], up[2], -up[1],
    -left[0], -left[2], left[1],
  ])
}

/** 車両ローカル（three: 前 x・上 y・右 z）の点を、路面の姿勢込みで three の世界座標へ。 */
export function attitudeLocalToThree(
  local: readonly [number, number, number],
  x: number,
  y: number,
  z: number,
  heading: number,
  pitch: number,
  roll: number,
): { x: number; y: number; z: number } {
  const a = attitudeAxes(heading, pitch, roll)
  return {
    x: x + a[0] * local[0] + a[3] * local[1] + a[6] * local[2],
    y: z + a[1] * local[0] + a[4] * local[1] + a[7] * local[2],
    z: -y + a[2] * local[0] + a[5] * local[1] + a[8] * local[2],
  }
}

/** (x, y) の路面のうち最も高いもの（上から見えるもの：橋桁の上・山の表面）。クリックした所を拾うのに使う */
export function topSurfaceZ(model: ElevationModel, x: number, y: number): number {
  let best = terrainZ(model.terrain, x, y)
  for (const profile of model.profiles.values()) {
    if (profile.structure === 'tunnel') continue
    const p = projectOnProfile(profile, x, y)
    if (p.dist > profile.edge.width / 2 + 0.5) continue
    best = Math.max(best, profileZ(profile, p.s, p.lateral))
  }
  return best
}

/** three の視線（始点と向き）が最初に当たる路面の点を ENU で返す。当たらなければ null。 */
export function pickSurface(
  model: ElevationModel,
  origin: { x: number; y: number; z: number },
  direction: { x: number; y: number; z: number },
  maxDistance = 3000,
): { x: number; y: number; z: number } | null {
  const len = Math.hypot(direction.x, direction.y, direction.z) || 1
  const dx = direction.x / len
  const dy = direction.y / len
  const dz = direction.z / len
  let prevT = 0
  let prevGap = origin.y - topSurfaceZ(model, origin.x, -origin.z)
  // 近くは細かく、遠くは粗く進める（地形の格子は 2m）
  for (let t = 0.5; t <= maxDistance; t += Math.max(0.5, t * 0.01)) {
    const x = origin.x + dx * t
    const yThree = origin.y + dy * t
    const enuY = -(origin.z + dz * t)
    const gap = yThree - topSurfaceZ(model, x, enuY)
    if (gap <= 0) {
      const f = prevGap / Math.max(prevGap - gap, 1e-9)
      const hit = prevT + (t - prevT) * f
      const hx = origin.x + dx * hit
      const hy = -(origin.z + dz * hit)
      return { x: hx, y: hy, z: topSurfaceZ(model, hx, hy) }
    }
    prevT = t
    prevGap = gap
  }
  return null
}
