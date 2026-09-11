/**
 * 道路メッシュ。
 *
 * 各エッジの polyline と width から「幅を持つリボン」を生成する。
 * 各点で進行方向の法線方向に width/2 だけオフセットして三角形ストリップを張り、
 * 角では前後の方向を平均して法線を作る。
 *
 * エッジは数百〜数千本になるので、mergeGeometries で 1 メッシュにまとめて
 * ドローコールを削減する。地面（y=0）よりわずかに上に置いて z-fighting を避ける。
 */

import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { MapEdge, Vec2 } from '../types/protocol'
import { usePalette } from './usePalette'

/** 道路面の高さ [m]（地面とのz-fighting回避） */
const ROAD_Y = 0.02
/** 中央線の高さ [m]（道路面のさらに上） */
const CENTERLINE_Y = 0.035
/** 中央線の幅 [m] */
const CENTERLINE_WIDTH = 0.28

/** 連続する重複点を除く（法線計算がゼロ除算にならないように） */
function dedupe(points: Vec2[]): Vec2[] {
  const out: Vec2[] = []
  for (const p of points) {
    const last = out[out.length - 1]
    if (!last || Math.abs(last[0] - p[0]) > 1e-6 || Math.abs(last[1] - p[1]) > 1e-6) {
      out.push(p)
    }
  }
  return out
}

/**
 * ENU の折れ線から、幅 width のリボンジオメトリを作る。
 * Three 座標への変換は x = enu.x, y = height, z = -enu.y（protocol.md 1.3）。
 */
function buildRibbon(points: Vec2[], width: number, height: number): THREE.BufferGeometry | null {
  const pts = dedupe(points)
  const n = pts.length
  if (n < 2) return null

  const half = width / 2
  const position = new Float32Array(n * 2 * 3)
  const normal = new Float32Array(n * 2 * 3)
  const uv = new Float32Array(n * 2 * 2)
  const index = new Uint32Array((n - 1) * 6)

  let travelled = 0
  for (let i = 0; i < n; i++) {
    let dx: number
    let dy: number
    if (i === 0) {
      dx = pts[1][0] - pts[0][0]
      dy = pts[1][1] - pts[0][1]
    } else if (i === n - 1) {
      dx = pts[n - 1][0] - pts[n - 2][0]
      dy = pts[n - 1][1] - pts[n - 2][1]
    } else {
      // 角では前後の単位ベクトルを足して平均方向にする
      const ax = pts[i][0] - pts[i - 1][0]
      const ay = pts[i][1] - pts[i - 1][1]
      const bx = pts[i + 1][0] - pts[i][0]
      const by = pts[i + 1][1] - pts[i][1]
      const la = Math.hypot(ax, ay) || 1
      const lb = Math.hypot(bx, by) || 1
      dx = ax / la + bx / lb
      dy = ay / la + by / lb
    }
    const len = Math.hypot(dx, dy) || 1
    dx /= len
    dy /= len

    // ENU 平面での左法線
    const nx = -dy
    const ny = dx

    if (i > 0) {
      travelled += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    }

    const px = pts[i][0]
    const py = pts[i][1]
    const o = i * 6
    // 左端
    position[o + 0] = px + nx * half
    position[o + 1] = height
    position[o + 2] = -(py + ny * half)
    // 右端
    position[o + 3] = px - nx * half
    position[o + 4] = height
    position[o + 5] = -(py - ny * half)

    normal[o + 1] = 1
    normal[o + 4] = 1

    const uo = i * 4
    uv[uo + 0] = 0
    uv[uo + 1] = travelled / Math.max(1, width)
    uv[uo + 2] = 1
    uv[uo + 3] = travelled / Math.max(1, width)

    if (i < n - 1) {
      const a = i * 2
      const io = i * 6
      index[io + 0] = a
      index[io + 1] = a + 2
      index[io + 2] = a + 1
      index[io + 3] = a + 1
      index[io + 4] = a + 2
      index[io + 5] = a + 3
    }
  }

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(position, 3))
  geom.setAttribute('normal', new THREE.BufferAttribute(normal, 3))
  geom.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  geom.setIndex(new THREE.BufferAttribute(index, 1))
  return geom
}

interface RoadGeometries {
  surface: THREE.BufferGeometry | null
  centerline: THREE.BufferGeometry | null
  edgeCount: number
}

function buildRoadGeometries(edges: MapEdge[]): RoadGeometries {
  const surfaces: THREE.BufferGeometry[] = []
  const centers: THREE.BufferGeometry[] = []

  for (const e of edges) {
    if (!e.polyline || e.polyline.length < 2) continue
    const width = e.width > 0 ? e.width : 6
    const g = buildRibbon(e.polyline, width, ROAD_Y)
    if (g) surfaces.push(g)

    // 対面通行かつ 2 車線以上なら中央線を薄く引く
    if (!e.oneway && e.lanes >= 2) {
      const c = buildRibbon(e.polyline, CENTERLINE_WIDTH, CENTERLINE_Y)
      if (c) centers.push(c)
    }
  }

  const surface = surfaces.length ? mergeGeometries(surfaces, false) : null
  const centerline = centers.length ? mergeGeometries(centers, false) : null

  // マージ後は元のジオメトリを解放する
  for (const g of surfaces) g.dispose()
  for (const g of centers) g.dispose()

  return { surface, centerline, edgeCount: edges.length }
}

export interface RoadNetworkProps {
  edges: MapEdge[]
  receiveShadow: boolean
}

export function RoadNetwork({ edges, receiveShadow }: RoadNetworkProps) {
  const palette = usePalette()
  const geoms = useMemo(() => buildRoadGeometries(edges), [edges])

  useEffect(() => {
    return () => {
      geoms.surface?.dispose()
      geoms.centerline?.dispose()
    }
  }, [geoms])

  if (!geoms.surface) return null

  return (
    <group>
      <mesh geometry={geoms.surface} receiveShadow={receiveShadow}>
        <meshStandardMaterial
          color={palette.roadSurface}
          roughness={0.94}
          metalness={0.02}
          side={THREE.DoubleSide}
          // ★ 地面（y=0）との Z ファイティング対策。
          //   路面は 2cm しか浮いていないが、near=0.5 / far=8000 では
          //   24bit 深度バッファでも 531m 先で 3.41cm しか区別できない。
          //   高さを離すと車両（y=0 基準）が路面に埋まるので、
          //   深度値そのものを手前へ押す。polygonOffset はバッファの
          //   最小単位で押すため **視点距離によらず効く**。
          polygonOffset
          polygonOffsetFactor={-1}
          polygonOffsetUnits={-2}
        />
      </mesh>
      {geoms.centerline && (
        <mesh geometry={geoms.centerline}>
          <meshStandardMaterial
            color={palette.roadCenterline}
            roughness={0.8}
            metalness={0}
            side={THREE.DoubleSide}
            // 路面よりさらに手前。地面 < 路面 < 中央線 < 白線標示 の順を保つ
            polygonOffset
            polygonOffsetFactor={-2}
            polygonOffsetUnits={-4}
          />
        </mesh>
      )}
    </group>
  )
}
