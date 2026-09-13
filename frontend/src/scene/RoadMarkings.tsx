/**
 * 道路標示（区画線・停止線・横断歩道）。
 *
 * 「道路標識、区画線及び道路標示に関する命令」に沿った寸法で描いている。
 *
 * | 標示 | 寸法 | 実装での扱い |
 * |---|---|---|
 * | 車道中央線 | 幅 15cm・破線は 5m 線 / 5m 空白 | 対面通行（oneway=false）のエッジに引く |
 * | 車線境界線 | 幅 15cm・5m 線 / 5m 空白 | 片側 2 車線以上（lanes >= 4）のとき車線の境界に引く |
 * | 車道外側線 | 幅 15cm の実線 | 全エッジの両端に引く |
 * | 停止線 | 幅 30〜45cm の実線 | 信号機の位置（交差点手前）に、進入側の車線幅ぶん引く |
 * | 横断歩道 | 帯幅 45cm・間隔 45cm・横断方向 4m | 停止線の先に引く。現行の日本の横断歩道は縁線のないゼブラ型 |
 *
 * **描かないもの**: 追越し禁止の黄色実線。これは公安委員会の規制であって
 * OSM のタグからは判別できない。判別できないものを描くと「そこは追越し禁止だ」と
 * いう誤った情報を出すことになるので、白線のみに留めている。
 *
 * **左側通行について**: 標示は日本の基準どおり左側通行を前提に配置している。
 * 車両側も backend/app/map/lanes.py が車線に沿った経路を生成するので
 * （道交法 17 条 4 項の左側通行、右左折の手前は 34 条どおり寄せる）、
 * 中央線をまたいで走ることはない。
 */

import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import type { MapEdge, MapSignal, Vec2 } from '../types/protocol'
import { usePalette } from './usePalette'
// 寸法と破線の割り付けは Node から検証できるよう純粋モジュールに置いてある
import {
  CROSSWALK_GAP,
  CROSSWALK_LENGTH,
  CROSSWALK_OFFSET,
  CROSSWALK_STRIPE,
  EDGE_LINE_INSET,
  LINE_WIDTH,
  STOP_LINE_WIDTH,
  dashSpans,
} from './roadMarkingGeometry'

/** 標示を路面からどれだけ浮かせるか。道路メッシュ(0.02)より上に置く */
const MARKING_Y = 0.05

// ---------------------------------------------------------------------------
// ジオメトリ生成のための小道具
// ---------------------------------------------------------------------------

/** ポリラインの各区間の長さと累積距離 */
interface Measured {
  points: Vec2[]
  segLengths: number[]
  total: number
}

function measure(points: Vec2[]): Measured {
  const segLengths: number[] = []
  let total = 0
  for (let i = 0; i < points.length - 1; i++) {
    const d = Math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
    segLengths.push(d)
    total += d
  }
  return { points, segLengths, total }
}

/** 累積距離 `dist` の位置の座標と進行方向を返す */
function sampleAt(m: Measured, dist: number): { x: number; y: number; dx: number; dy: number } {
  let remaining = Math.max(0, Math.min(dist, m.total))
  for (let i = 0; i < m.segLengths.length; i++) {
    const len = m.segLengths[i]
    if (len <= 1e-9) continue
    if (remaining <= len) {
      const t = remaining / len
      const [x0, y0] = m.points[i]
      const [x1, y1] = m.points[i + 1]
      return {
        x: x0 + (x1 - x0) * t,
        y: y0 + (y1 - y0) * t,
        dx: (x1 - x0) / len,
        dy: (y1 - y0) / len,
      }
    }
    remaining -= len
  }
  const n = m.points.length
  const [x0, y0] = m.points[n - 2] ?? m.points[0]
  const [x1, y1] = m.points[n - 1]
  const len = Math.max(1e-9, Math.hypot(x1 - x0, y1 - y0))
  return { x: x1, y: y1, dx: (x1 - x0) / len, dy: (y1 - y0) / len }
}

/**
 * ENU 上の矩形（中心・向き・寸法）を三角形 2 枚として positions へ積む。
 *
 * three への変換（z = -y）はここで済ませる。
 */
function pushQuad(
  out: number[],
  cx: number,
  cy: number,
  dx: number,
  dy: number,
  length: number,
  width: number,
): void {
  const hl = length / 2
  const hw = width / 2
  // 進行方向 (dx, dy) と、その左手 (-dy, dx)
  const ax = dx * hl
  const ay = dy * hl
  const bx = -dy * hw
  const by = dx * hw

  const p = [
    [cx - ax - bx, cy - ay - by],
    [cx + ax - bx, cy + ay - by],
    [cx + ax + bx, cy + ay + by],
    [cx - ax + bx, cy - ay + by],
  ]
  const tri = [0, 1, 2, 0, 2, 3]
  for (const i of tri) {
    out.push(p[i][0], MARKING_Y, -p[i][1]) // three.z = -enu.y
  }
}

/** ポリラインに沿って、横方向に `offset` ずらした線を引く */
function pushLineAlong(
  out: number[],
  points: Vec2[],
  offset: number,
  width: number,
  dashed: boolean,
): void {
  if (points.length < 2) return
  const m = measure(points)
  if (m.total < 0.5) return

  if (!dashed) {
    // 実線は区間ごとに 1 枚ずつ置く（曲線にも追従する）
    for (let i = 0; i < m.segLengths.length; i++) {
      const len = m.segLengths[i]
      if (len <= 1e-6) continue
      const [x0, y0] = m.points[i]
      const [x1, y1] = m.points[i + 1]
      const dx = (x1 - x0) / len
      const dy = (y1 - y0) / len
      // 左手方向へ offset ずらす
      const ox = -dy * offset
      const oy = dx * offset
      pushQuad(out, (x0 + x1) / 2 + ox, (y0 + y1) / 2 + oy, dx, dy, len, width)
    }
    return
  }

  // 破線は「線部」の区間だけ 1 枚の矩形で近似する。
  // 5m 程度なら道路の曲率による誤差は 15cm 幅の線では見えない。
  // 割り付けは `roadMarkingGeometry.dashSpans()` の 1 か所だけ（S-03）。
  for (const { start, end } of dashSpans(m.total)) {
    const a = sampleAt(m, start)
    const b = sampleAt(m, end)
    const mx = (a.x + b.x) / 2
    const my = (a.y + b.y) / 2
    const ddx = b.x - a.x
    const ddy = b.y - a.y
    const len = Math.hypot(ddx, ddy)
    if (len <= 1e-6) continue
    const dx = ddx / len
    const dy = ddy / len
    pushQuad(out, mx - dy * offset, my + dx * offset, dx, dy, len, width)
  }
}

// ---------------------------------------------------------------------------
// 標示の組み立て
// ---------------------------------------------------------------------------

function buildMarkings(edges: MapEdge[], signals: MapSignal[]): THREE.BufferGeometry | null {
  const positions: number[] = []

  for (const e of edges) {
    const half = e.width / 2

    // --- 車道外側線（両端の実線） ---
    pushLineAlong(positions, e.polyline, half - EDGE_LINE_INSET, LINE_WIDTH, false)
    pushLineAlong(positions, e.polyline, -(half - EDGE_LINE_INSET), LINE_WIDTH, false)

    if (!e.oneway) {
      // --- 車道中央線（対面通行のみ） ---
      pushLineAlong(positions, e.polyline, 0, LINE_WIDTH, true)

      // --- 車線境界線（片側 2 車線以上のとき、各方向の内部に引く） ---
      const perSide = Math.floor(e.lanes / 2)
      if (perSide >= 2) {
        const laneWidth = half / perSide
        for (let i = 1; i < perSide; i++) {
          pushLineAlong(positions, e.polyline, laneWidth * i, LINE_WIDTH, true)
          pushLineAlong(positions, e.polyline, -laneWidth * i, LINE_WIDTH, true)
        }
      }
    } else if (e.lanes >= 2) {
      // 一方通行は幅いっぱいが同じ向きなので、車線境界線を等間隔に引く
      const laneWidth = e.width / e.lanes
      for (let i = 1; i < e.lanes; i++) {
        pushLineAlong(positions, e.polyline, -half + laneWidth * i, LINE_WIDTH, true)
      }
    }
  }

  // --- 停止線と横断歩道（信号機のある進入路） ---
  for (const sg of signals) {
    const cos = Math.cos(sg.heading)
    const sin = Math.sin(sg.heading)
    // 進入側の車線がどれだけ横に広がっているか。対面通行なら左半分だけ。
    const laneSpan = sg.roadWidth / 2
    // 進入車線の中央（左側通行なので進行方向左半分）
    const centerOffset = sg.roadWidth / 4
    const cx = sg.x - sin * centerOffset
    const cy = sg.y + cos * centerOffset

    // 停止線は進行方向に対して直交する帯
    pushQuad(positions, cx, cy, -sin, cos, laneSpan, STOP_LINE_WIDTH)

    // 横断歩道は停止線の先。道路の全幅にわたるゼブラ
    const zebraCenterX = sg.x + cos * (CROSSWALK_OFFSET + CROSSWALK_LENGTH / 2)
    const zebraCenterY = sg.y + sin * (CROSSWALK_OFFSET + CROSSWALK_LENGTH / 2)
    const pitch = CROSSWALK_STRIPE + CROSSWALK_GAP
    const stripes = Math.max(2, Math.floor(sg.roadWidth / pitch))
    const span = stripes * pitch
    for (let i = 0; i < stripes; i++) {
      const t = -span / 2 + pitch / 2 + i * pitch
      // 帯は横断方向（進行方向）に長い
      pushQuad(
        positions,
        zebraCenterX - sin * t,
        zebraCenterY + cos * t,
        cos,
        sin,
        CROSSWALK_LENGTH,
        CROSSWALK_STRIPE,
      )
    }
  }

  if (positions.length === 0) return null

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geom.computeVertexNormals()
  return geom
}

export interface RoadMarkingsProps {
  edges: MapEdge[]
  signals: MapSignal[]
}

export function RoadMarkings({ edges, signals }: RoadMarkingsProps) {
  const palette = usePalette()
  const geometry = useMemo(() => buildMarkings(edges, signals), [edges, signals])

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.marking,
        roughness: 0.85,
        metalness: 0,
        // 路面のすぐ上に置くので、深度の競合を避けるため手前へ寄せる。
        // 地面 < 路面(-1/-2) < 中央線(-2/-4) < 白線標示(-3/-6) の順を保つこと。
        // 値を緩めると視点を動かしたときに路面と点滅する。
        polygonOffset: true,
        polygonOffsetFactor: -3,
        polygonOffsetUnits: -6,
      }),
    [palette.marking],
  )

  useEffect(() => {
    return () => {
      geometry?.dispose()
      material.dispose()
    }
  }, [geometry, material])

  if (!geometry) return null

  // 標示は影を落とさない（路面に貼り付いた塗装なので）
  return <mesh geometry={geometry} material={material} receiveShadow />
}

// ★ `MARKING_SPEC`（寸法の再掲）は削除した（code_review S-04）。
//   「パネルの説明に出す」と書いてあったが参照は 0 件で、同じ値を 2 か所へ
//   書く二重定義でもあった。出す必要が生じたら `roadMarkingGeometry.ts` の
//   定数を直接 import すること。
