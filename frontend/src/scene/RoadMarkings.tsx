/** 道路標示（区画線・停止線・横断歩道）。 */

import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import type { MapEdge, MapSignal, Vec2 } from '../types/protocol'
import { usePalette } from './usePalette'
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

/** ENU 上の矩形（中心・向き・寸法）を三角形 2 枚として positions へ積む。 */
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
    out.push(p[i][0], MARKING_Y, -p[i][1])
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
    for (let i = 0; i < m.segLengths.length; i++) {
      const len = m.segLengths[i]
      if (len <= 1e-6) continue
      const [x0, y0] = m.points[i]
      const [x1, y1] = m.points[i + 1]
      const dx = (x1 - x0) / len
      const dy = (y1 - y0) / len
      const ox = -dy * offset
      const oy = dx * offset
      pushQuad(out, (x0 + x1) / 2 + ox, (y0 + y1) / 2 + oy, dx, dy, len, width)
    }
    return
  }

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

function buildMarkings(edges: MapEdge[], signals: MapSignal[]): THREE.BufferGeometry | null {
  const positions: number[] = []

  for (const e of edges) {
    const half = e.width / 2

    pushLineAlong(positions, e.polyline, half - EDGE_LINE_INSET, LINE_WIDTH, false)
    pushLineAlong(positions, e.polyline, -(half - EDGE_LINE_INSET), LINE_WIDTH, false)

    if (!e.oneway) {
      pushLineAlong(positions, e.polyline, 0, LINE_WIDTH, true)

      const perSide = Math.floor(e.lanes / 2)
      if (perSide >= 2) {
        const laneWidth = half / perSide
        for (let i = 1; i < perSide; i++) {
          pushLineAlong(positions, e.polyline, laneWidth * i, LINE_WIDTH, true)
          pushLineAlong(positions, e.polyline, -laneWidth * i, LINE_WIDTH, true)
        }
      }
    } else if (e.lanes >= 2) {
      const laneWidth = e.width / e.lanes
      for (let i = 1; i < e.lanes; i++) {
        pushLineAlong(positions, e.polyline, -half + laneWidth * i, LINE_WIDTH, true)
      }
    }
  }

  for (const sg of signals) {
    const cos = Math.cos(sg.heading)
    const sin = Math.sin(sg.heading)
    const laneSpan = sg.roadWidth / 2
    const centerOffset = sg.roadWidth / 4
    const cx = sg.x - sin * centerOffset
    const cy = sg.y + cos * centerOffset

    pushQuad(positions, cx, cy, -sin, cos, laneSpan, STOP_LINE_WIDTH)

    const zebraCenterX = sg.x + cos * (CROSSWALK_OFFSET + CROSSWALK_LENGTH / 2)
    const zebraCenterY = sg.y + sin * (CROSSWALK_OFFSET + CROSSWALK_LENGTH / 2)
    const pitch = CROSSWALK_STRIPE + CROSSWALK_GAP
    const stripes = Math.max(2, Math.floor(sg.roadWidth / pitch))
    const span = stripes * pitch
    for (let i = 0; i < stripes; i++) {
      const t = -span / 2 + pitch / 2 + i * pitch
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

  return <mesh geometry={geometry} material={material} receiveShadow />
}
