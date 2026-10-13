/** 立体の地図の 3D の形（地形・橋桁・トンネル・頂点の持ち上げ）。three の BufferGeometry を作るだけで React に依らない。 */

import * as THREE from 'three'
import type { TerrainGrid } from '../types/protocol.ts'
import {
  DECK_THICKNESS_M,
  STRUCTURE_MARGIN_M,
  TUNNEL_CLEARANCE_M,
  type EdgeProfile,
  type ElevationModel,
  profileZ,
  projectOnProfile,
  terrainZ,
} from './elevationView.ts'

/** 坑口でこれより段差のある地形の四角形は、トンネルの口として描かない [m] */
const PORTAL_STEP_M = 2.0
/** トンネルの天井の上にこれだけ土かぶりが無い所は、坑口ではなく切通し（`map/elevation.py` の TUNNEL_MIN_COVER_M と同じ値）[m] */
const TUNNEL_MIN_COVER_M = 0.5

/** three の頂点列 [x, y, z, ...] のうち [start, end) を、その辺の路面の高さだけ持ち上げる（y に足す）。 */
export function liftVertices(positions: number[] | Float32Array, start: number, end: number, profile: EdgeProfile): void {
  for (let i = start; i < end; i++) {
    const x = positions[i * 3]
    const y = -positions[i * 3 + 2]
    const p = projectOnProfile(profile, x, y)
    positions[i * 3 + 1] += profileZ(profile, p.s, p.lateral)
  }
}

/** three の頂点列 [start, end) を一定の高さ `z` だけ持ち上げる（停止線・横断歩道）。 */
export function liftConstant(positions: number[] | Float32Array, start: number, end: number, z: number): void {
  if (z === 0) return
  for (let i = start; i < end; i++) positions[i * 3 + 1] += z
}

/** 辺の頂点ごとの左右の端の高さ（u→v の左が正の横断勾配）。道路の帯の頂点に使う。 */
export function ribbonHeights(profile: EdgeProfile, half: number): { left: Float64Array; right: Float64Array } {
  const n = profile.z.length
  const left = new Float64Array(n)
  const right = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    left[i] = profile.z[i] + half * profile.cross
    right[i] = profile.z[i] - half * profile.cross
  }
  return { left, right }
}

function nearTunnel(profiles: readonly EdgeProfile[], x: number, y: number): { inside: boolean; roadZ: number } {
  for (const p of profiles) {
    const hit = projectOnProfile(p, x, y)
    if (hit.dist <= p.edge.width / 2 + STRUCTURE_MARGIN_M) return { inside: true, roadZ: profileZ(p, hit.s, 0) }
  }
  return { inside: false, roadZ: 0 }
}

/** 地形の格子を三角形の面にする。トンネルの坑口（土かぶりのある所へ地形が段差で上がる面）は開けておく。 */
export function buildTerrainGeometry(terrain: TerrainGrid, model: ElevationModel): THREE.BufferGeometry {
  const w = terrain.width
  const h = terrain.height
  const position = new Float32Array(w * h * 3)
  const covered = new Uint8Array(w * h)
  const tunnels = [...model.profiles.values()].filter((p) => p.structure === 'tunnel')
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = r * w + c
      const x = terrain.originX + c * terrain.cellSize
      const y = terrain.originY + r * terrain.cellSize
      const z = terrain.z[i]
      position[i * 3] = x
      position[i * 3 + 1] = z
      position[i * 3 + 2] = -y
      if (tunnels.length) {
        const t = nearTunnel(tunnels, x, y)
        if (t.inside && z >= t.roadZ + TUNNEL_CLEARANCE_M + TUNNEL_MIN_COVER_M) covered[i] = 1
      }
    }
  }
  const index: number[] = []
  for (let r = 0; r < h - 1; r++) {
    for (let c = 0; c < w - 1; c++) {
      const a = r * w + c
      const b = a + 1
      const d = a + w
      const e = d + 1
      if (tunnels.length) {
        const cov = covered[a] + covered[b] + covered[d] + covered[e]
        if (cov > 0 && cov < 4) {
          const zs = [terrain.z[a], terrain.z[b], terrain.z[d], terrain.z[e]]
          // 坑口：土かぶりのあるセルとないセルの間が段差になっている面は開けておく
          if (Math.max(...zs) - Math.min(...zs) > PORTAL_STEP_M) continue
        }
      }
      // three の座標では y（北）が -z なので、上向きの面になるよう向きを揃える
      index.push(a, b, d, b, e, d)
    }
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(position, 3))
  geom.setIndex(index)
  geom.computeVertexNormals()
  return geom
}

interface Frame {
  x: number
  y: number
  nx: number
  ny: number
  z: number
}

function frames(profile: EdgeProfile): Frame[] {
  const pts = profile.edge.polyline
  const out: Frame[] = []
  for (let i = 0; i < pts.length; i++) {
    const a = pts[Math.max(0, i - 1)]
    const b = pts[Math.min(pts.length - 1, i + 1)]
    const dx = b[0] - a[0]
    const dy = b[1] - a[1]
    const len = Math.hypot(dx, dy) || 1
    out.push({ x: pts[i][0], y: pts[i][1], nx: -dy / len, ny: dx / len, z: profile.z[i] })
  }
  return out
}

function pushQuad(out: number[], a: number[], b: number[], c: number[], d: number[]): void {
  out.push(...a, ...b, ...c, ...a, ...c, ...d)
}

function three(x: number, y: number, z: number): number[] {
  return [x, z, -y]
}

/** 橋桁の側面と下面（上面は道路の帯がそのまま路面）。寸法はバックエンドの固体判定グリッドと同じ。 */
export function buildDeckGeometry(profile: EdgeProfile): THREE.BufferGeometry {
  const half = profile.edge.width / 2 + STRUCTURE_MARGIN_M
  const fs = frames(profile)
  const pos: number[] = []
  for (let i = 0; i < fs.length - 1; i++) {
    const p = fs[i]
    const q = fs[i + 1]
    const pl = p.z + half * profile.cross
    const pr = p.z - half * profile.cross
    const ql = q.z + half * profile.cross
    const qr = q.z - half * profile.cross
    const pLt = three(p.x + p.nx * half, p.y + p.ny * half, pl)
    const pRt = three(p.x - p.nx * half, p.y - p.ny * half, pr)
    const qLt = three(q.x + q.nx * half, q.y + q.ny * half, ql)
    const qRt = three(q.x - q.nx * half, q.y - q.ny * half, qr)
    const pLb = three(p.x + p.nx * half, p.y + p.ny * half, pl - DECK_THICKNESS_M)
    const pRb = three(p.x - p.nx * half, p.y - p.ny * half, pr - DECK_THICKNESS_M)
    const qLb = three(q.x + q.nx * half, q.y + q.ny * half, ql - DECK_THICKNESS_M)
    const qRb = three(q.x - q.nx * half, q.y - q.ny * half, qr - DECK_THICKNESS_M)
    pushQuad(pos, pLt, qLt, qLb, pLb)
    pushQuad(pos, qRt, pRt, pRb, qRb)
    pushQuad(pos, pLb, qLb, qRb, pRb)
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  geom.computeVertexNormals()
  return geom
}

/** トンネルの内側（左右の壁と天井）と、坑口の上の壁（天井から地山の表面まで）。 */
export function buildTunnelGeometry(profile: EdgeProfile, terrain: TerrainGrid | null): THREE.BufferGeometry {
  const half = profile.edge.width / 2 + STRUCTURE_MARGIN_M
  const fs = frames(profile)
  const pos: number[] = []
  for (let i = 0; i < fs.length - 1; i++) {
    const p = fs[i]
    const q = fs[i + 1]
    const pl = three(p.x + p.nx * half, p.y + p.ny * half, p.z)
    const pr = three(p.x - p.nx * half, p.y - p.ny * half, p.z)
    const ql = three(q.x + q.nx * half, q.y + q.ny * half, q.z)
    const qr = three(q.x - q.nx * half, q.y - q.ny * half, q.z)
    const plTop = three(p.x + p.nx * half, p.y + p.ny * half, p.z + TUNNEL_CLEARANCE_M)
    const prTop = three(p.x - p.nx * half, p.y - p.ny * half, p.z + TUNNEL_CLEARANCE_M)
    const qlTop = three(q.x + q.nx * half, q.y + q.ny * half, q.z + TUNNEL_CLEARANCE_M)
    const qrTop = three(q.x - q.nx * half, q.y - q.ny * half, q.z + TUNNEL_CLEARANCE_M)
    pushQuad(pos, pl, ql, qlTop, plTop)
    pushQuad(pos, qr, pr, prTop, qrTop)
    pushQuad(pos, plTop, qlTop, qrTop, prTop)
  }
  // 坑口の上の壁（両端）。地山が天井より高い所だけ
  for (const f of [fs[0], fs[fs.length - 1]]) {
    const ground = terrainZ(terrain, f.x, f.y)
    const top = f.z + TUNNEL_CLEARANCE_M
    if (ground <= top + 0.1) continue
    const a = three(f.x + f.nx * half, f.y + f.ny * half, top)
    const b = three(f.x - f.nx * half, f.y - f.ny * half, top)
    const c = three(f.x - f.nx * half, f.y - f.ny * half, ground + 0.3)
    const d = three(f.x + f.nx * half, f.y + f.ny * half, ground + 0.3)
    pushQuad(pos, a, b, c, d)
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  geom.computeVertexNormals()
  return geom
}
