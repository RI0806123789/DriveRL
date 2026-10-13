/** 道路メッシュ。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { MapEdge, Vec2 } from '../types/protocol'
import { useSimStore } from '../store/simStore'
import { usePalette } from './usePalette'
import { TUNNEL_LIGHT_FACTOR, elevationOf, type ElevationModel } from './elevationView'
import { ribbonHeights } from './elevationGeometry'
import {
  WET_DARKEN,
  WET_METALNESS,
  WET_ROUGHNESS,
  WET_SKY_MIX,
  displayedWeather,
  lerp,
} from './weatherView'

/** 道路面の高さ [m]（地面とのz-fighting回避） */
const ROAD_Y = 0.02
/** 中央線の高さ [m]（道路面のさらに上） */
const CENTERLINE_Y = 0.035
/** 中央線の幅 [m] */
const CENTERLINE_WIDTH = 0.28

/** 連続する重複点を除いた点の添字（法線計算がゼロ除算にならないように） */
function dedupeIndices(points: Vec2[]): number[] {
  const out: number[] = []
  for (let i = 0; i < points.length; i++) {
    const p = points[i]
    const last = out.length ? points[out[out.length - 1]] : undefined
    if (!last || Math.abs(last[0] - p[0]) > 1e-6 || Math.abs(last[1] - p[1]) > 1e-6) {
      out.push(i)
    }
  }
  return out
}

/** 立体の地図での、頂点ごとの左右の端の路面の高さ（`points` と同じ並び） */
interface RibbonLift {
  left: ArrayLike<number>
  right: ArrayLike<number>
}

/** ENU の折れ線から、幅 width のリボンジオメトリを作る。`lift` があれば左右の端をその高さだけ持ち上げる。 */
function buildRibbon(points: Vec2[], width: number, height: number, lift?: RibbonLift): THREE.BufferGeometry | null {
  const keep = dedupeIndices(points)
  const pts = keep.map((i) => points[i])
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

    const nx = -dy
    const ny = dx

    if (i > 0) {
      travelled += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    }

    const px = pts[i][0]
    const py = pts[i][1]
    const o = i * 6
    position[o + 0] = px + nx * half
    position[o + 1] = height + (lift ? lift.left[keep[i]] : 0)
    position[o + 2] = -(py + ny * half)
    position[o + 3] = px - nx * half
    position[o + 4] = height + (lift ? lift.right[keep[i]] : 0)
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
  /** 立体の地図のトンネルの中の路面（暗く塗る） */
  tunnel: THREE.BufferGeometry | null
  centerline: THREE.BufferGeometry | null
  edgeCount: number
}

function buildRoadGeometries(edges: MapEdge[], model: ElevationModel | null): RoadGeometries {
  const surfaces: THREE.BufferGeometry[] = []
  const tunnels: THREE.BufferGeometry[] = []
  const centers: THREE.BufferGeometry[] = []

  for (const e of edges) {
    if (!e.polyline || e.polyline.length < 2) continue
    const width = e.width > 0 ? e.width : 6
    const profile = model?.profiles.get(e.id)
    const g = buildRibbon(e.polyline, width, ROAD_Y, profile ? ribbonHeights(profile, width / 2) : undefined)
    if (g) (profile?.structure === 'tunnel' ? tunnels : surfaces).push(g)

    if (!e.oneway && e.lanes >= 2) {
      const lift = profile ? { left: profile.z, right: profile.z } : undefined
      const c = buildRibbon(e.polyline, CENTERLINE_WIDTH, CENTERLINE_Y, lift)
      if (c) centers.push(c)
    }
  }

  const surface = surfaces.length ? mergeGeometries(surfaces, false) : null
  const tunnel = tunnels.length ? mergeGeometries(tunnels, false) : null
  const centerline = centers.length ? mergeGeometries(centers, false) : null

  for (const g of surfaces) g.dispose()
  for (const g of tunnels) g.dispose()
  for (const g of centers) g.dispose()

  return { surface, tunnel, centerline, edgeCount: edges.length }
}

export interface RoadNetworkProps {
  edges: MapEdge[]
  receiveShadow: boolean
}

/** 乾いた路面の質感。濡れるとここから `WET_*` へ寄る */
const DRY_ROUGHNESS = 0.94
const DRY_METALNESS = 0.02

export function RoadNetwork({ edges, receiveShadow }: RoadNetworkProps) {
  const palette = usePalette()
  const map = useSimStore((s) => s.map)
  const model = elevationOf(map)
  const geoms = useMemo(() => buildRoadGeometries(edges, model), [edges, model])
  const tunnelColor = useMemo(
    () => new THREE.Color(palette.roadSurface).multiplyScalar(TUNNEL_LIGHT_FACTOR),
    [palette.roadSurface],
  )
  const surface = useRef<THREE.MeshStandardMaterial>(null)
  const dryColor = useMemo(() => new THREE.Color(palette.roadSurface), [palette.roadSurface])
  const skyColor = useMemo(() => new THREE.Color(palette.sky), [palette.sky])

  useEffect(() => {
    return () => {
      geoms.surface?.dispose()
      geoms.tunnel?.dispose()
      geoms.centerline?.dispose()
    }
  }, [geoms])

  // 雨で路面を濡らす。**マテリアルは作り直さず、値だけ書き換える**
  useFrame(() => {
    const mat = surface.current
    if (!mat) return
    const wet = displayedWeather.wet
    mat.roughness = lerp(DRY_ROUGHNESS, WET_ROUGHNESS, wet)
    mat.metalness = lerp(DRY_METALNESS, WET_METALNESS, wet)
    mat.color
      .copy(dryColor)
      .multiplyScalar(lerp(1, WET_DARKEN, wet))
      // 濡れた路面は空を映す。環境マップの代わりに空の色を薄く混ぜる
      .lerp(skyColor, wet * WET_SKY_MIX)
  })

  if (!geoms.surface && !geoms.tunnel) return null

  return (
    <group>
      {geoms.tunnel && (
        <mesh geometry={geoms.tunnel}>
          {/* トンネルの中は暗い（擬似カメラの TUNNEL_LIGHT_FACTOR と揃える）。日差しの影を受けないので色で沈める */}
          <meshStandardMaterial
            color={tunnelColor}
            roughness={DRY_ROUGHNESS}
            metalness={DRY_METALNESS}
            side={THREE.DoubleSide}
            polygonOffset
            polygonOffsetFactor={-1}
            polygonOffsetUnits={-2}
          />
        </mesh>
      )}
      {geoms.surface && <mesh geometry={geoms.surface} receiveShadow={receiveShadow}>
        <meshStandardMaterial
          ref={surface}
          color={palette.roadSurface}
          roughness={DRY_ROUGHNESS}
          metalness={DRY_METALNESS}
          side={THREE.DoubleSide}
          polygonOffset
          polygonOffsetFactor={-1}
          polygonOffsetUnits={-2}
        />
      </mesh>}
      {geoms.centerline && (
        <mesh geometry={geoms.centerline}>
          <meshStandardMaterial
            color={palette.roadCenterline}
            roughness={0.8}
            metalness={0}
            side={THREE.DoubleSide}
            polygonOffset
            polygonOffsetFactor={-2}
            polygonOffsetUnits={-4}
          />
        </mesh>
      )}
    </group>
  )
}
