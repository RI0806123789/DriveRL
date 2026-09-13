/**
 * 進路矢印のジオメトリ生成（純粋関数）。
 *
 * カーナビの案内表示のように、経路を幅のあるリボンとして描き、
 * 一定間隔で進行方向を示す矢羽根（シェブロン）を重ねる。
 *
 * React に依存させず three だけに依存させてあるのは、
 * `frontend/scripts/verify-camera-route.ts` から Node で検証するため。
 * 座標変換の規約は docs/protocol.md 1.3（three.z = -enu.y）。
 */

import * as THREE from 'three'

/** 進路リボンの幅 [m]。車幅(1.8m)より少し広くして見失いにくくする */
export const RIBBON_WIDTH = 2.2
/** 矢羽根を置く間隔 [m] */
export const CHEVRON_SPACING = 9.0
/** 矢羽根の長さ / 幅 [m] */
export const CHEVRON_LENGTH = 2.6
export const CHEVRON_WIDTH = 2.9
/** 路面から浮かせる高さ。道路標示(0.05)より上、車両より下 */
export const RIBBON_Y = 0.11
export const CHEVRON_Y = 0.13

/** ENU 平面上の点 */
export type Point2 = [number, number]

/** 各点での進行方向（前後の点を使った平均）を求める */
export function directionsAlong(points: Point2[]): Point2[] {
  const n = points.length
  const dirs: Point2[] = new Array(n)
  for (let i = 0; i < n; i++) {
    const a = points[Math.max(0, i - 1)]
    const b = points[Math.min(n - 1, i + 1)]
    let dx = b[0] - a[0]
    let dy = b[1] - a[1]
    const len = Math.hypot(dx, dy)
    if (len < 1e-9) {
      dirs[i] = i > 0 ? dirs[i - 1] : [1, 0]
    } else {
      dirs[i] = [dx / len, dy / len]
    }
  }
  return dirs
}

/**
 * リボンの頂点座標を既存の配列へ書き込む（`buildRibbon` の中身）。
 *
 * 点数が変わらないのにジオメトリごと作り直すのを避けるために切り出してある
 * （`LaneDetectionOverlay` は 20Hz で点列だけが入れ替わる。code_review S-02）。
 * 書き込むのは位置だけで、`out` の長さは `points.length * 6` 以上必要。
 *
 * ★ 帯は常に水平（y が一定）なので、点列が変わっても法線は ±Y のまま変わらない。
 *   書き換え側で `computeVertexNormals()` を呼び直す必要はない。
 */
export function writeRibbonPositions(
  out: Float32Array,
  points: Point2[],
  width: number = RIBBON_WIDTH,
  y: number = RIBBON_Y,
): void {
  const dirs = directionsAlong(points)
  const half = width / 2
  for (let i = 0; i < points.length; i++) {
    const [px, py] = points[i]
    const [dx, dy] = dirs[i]
    // 進行方向の左手
    const nx = -dy * half
    const ny = dx * half
    out[i * 6 + 0] = px + nx
    out[i * 6 + 1] = y
    out[i * 6 + 2] = -(py + ny)
    out[i * 6 + 3] = px - nx
    out[i * 6 + 4] = y
    out[i * 6 + 5] = -(py - ny)
  }
}

/**
 * 経路を幅のあるリボンにする。
 *
 * 区間ごとに独立した四角形を置くとカーブの内外で隙間や食い違いが出るので、
 * 各点の法線を前後の平均から求めて連続した帯にする。
 */
export function buildRibbon(
  points: Point2[],
  width: number = RIBBON_WIDTH,
  y: number = RIBBON_Y,
): THREE.BufferGeometry | null {
  if (points.length < 2) return null

  const positions = new Float32Array(points.length * 2 * 3)
  writeRibbonPositions(positions, points, width, y)

  const indices: number[] = []
  for (let i = 0; i < points.length - 1; i++) {
    const a = i * 2
    indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3)
  }

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geom.setIndex(indices)
  geom.computeVertexNormals()
  return geom
}

/** 矢羽根 1 枚分の頂点（検証しやすいよう ENU のまま返す） */
export interface Chevron {
  /** 先端 */
  tip: Point2
  /** 後端の左右 */
  backLeft: Point2
  backRight: Point2
  /** その地点での進行方向 */
  dir: Point2
}

/** 経路に沿って一定間隔で矢羽根を配置する */
export function chevronsAlong(points: Point2[]): Chevron[] {
  if (points.length < 2) return []

  const cum: number[] = [0]
  for (let i = 1; i < points.length; i++) {
    cum.push(
      cum[i - 1] + Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]),
    )
  }
  const total = cum[cum.length - 1]
  if (total < CHEVRON_SPACING) return []

  const out: Chevron[] = []
  let seg = 0
  const hw = CHEVRON_WIDTH / 2
  const hl = CHEVRON_LENGTH / 2

  for (let d = CHEVRON_SPACING * 0.5; d < total - CHEVRON_LENGTH; d += CHEVRON_SPACING) {
    while (seg < cum.length - 2 && cum[seg + 1] < d) seg++
    const segLen = cum[seg + 1] - cum[seg]
    if (segLen <= 1e-9) continue
    const t = (d - cum[seg]) / segLen
    const [x0, y0] = points[seg]
    const [x1, y1] = points[seg + 1]
    const px = x0 + (x1 - x0) * t
    const py = y0 + (y1 - y0) * t
    const dx = (x1 - x0) / segLen
    const dy = (y1 - y0) / segLen
    // 左手方向
    const nx = -dy
    const ny = dx

    out.push({
      tip: [px + dx * hl, py + dy * hl],
      backLeft: [px - dx * hl + nx * hw, py - dy * hl + ny * hw],
      backRight: [px - dx * hl - nx * hw, py - dy * hl - ny * hw],
      dir: [dx, dy],
    })
  }
  return out
}

/** 矢羽根を三角形メッシュにする */
export function buildChevrons(
  points: Point2[],
  y: number = CHEVRON_Y,
): THREE.BufferGeometry | null {
  const chevrons = chevronsAlong(points)
  if (chevrons.length === 0) return null

  const positions: number[] = []
  for (const c of chevrons) {
    // 先端 → 左後ろ → 右後ろ（表を上に向けるため反時計回りに並べる）
    for (const [vx, vy] of [c.tip, c.backLeft, c.backRight]) {
      positions.push(vx, y, -vy)
    }
  }

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geom.computeVertexNormals()
  return geom
}
