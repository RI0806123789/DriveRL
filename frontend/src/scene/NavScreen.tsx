/** インパネ中央のカーナビ。**見ている 1 台にだけ出す**（車ごとに中身が違うため）。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import {
  createProjection,
  drawBuildings,
  drawHeadingMark,
  drawRoads,
  drawRoute,
  lerpView,
  rotationForHeading,
} from '../panel/taxiMapMath'
import type { MapView } from '../panel/taxiMapMath'
import type { MapMessage } from '../types/protocol'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { usePalette } from './usePalette'
import {
  NAV_SCREEN,
  composeFixedMatrix,
  composeVehicleMatrix,
  createTransformScratch,
  makeNavScreenGeometry,
  navSpanFor,
} from './vehicleGeometry'

/**
 * 画面の解像度 [px]。**上げすぎないこと** — 描き直しのたびに道路と建物を
 * なぞるので、広いマップ（金沢は 58,120 本）では解像度がそのまま負荷になる。
 */
const TEX_W = 256
const TEX_H = 160

/**
 * 描き直す頻度。**60fps で描かないこと**（`TaxiCameraFeed` と同じ理由）。
 * 実車のナビも毎フレーム引き直してはいない。
 */
const NAV_FPS = 12

/** 見え方が目標へ寄る速さ [1/秒]。加減速で縮尺が跳ねないよう、ゆっくり追う */
const FOLLOW_RATE = 1.6
/** 向きは寄り引きより遅く追う（交差点で画面が振られて酔わないように） */
const TURN_RATE = 1.1

/** 縮尺に対して短すぎる道路は引かない [px]（金沢で真っ白に潰れるのを防ぐ） */
const MIN_ROAD_PX = 2.0

export function NavScreen() {
  const map = useSimStore((s) => s.map)
  const palette = usePalette()

  const resources = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = TEX_W
    canvas.height = TEX_H
    const texture = new THREE.CanvasTexture(canvas)
    texture.colorSpace = THREE.SRGBColorSpace
    const geometry = makeNavScreenGeometry(NAV_SCREEN.width, NAV_SCREEN.height)
    // 車内の照明に左右されないよう、素のテクスチャをそのまま出す
    const material = new THREE.MeshBasicMaterial({ map: texture, toneMapped: false })
    return { canvas, texture, geometry, material }
  }, [])

  useEffect(() => {
    const r = resources
    return () => {
      r.texture.dispose()
      r.geometry.dispose()
      r.material.dispose()
    }
  }, [resources])

  const meshRef = useRef<THREE.Mesh>(null)
  const pose = useMemo(createPose, [])
  const scratch = useMemo(
    () => ({
      transform: createTransformScratch(),
      base: new THREE.Matrix4(),
      out: new THREE.Matrix4(),
    }),
    [],
  )
  /** 見え方は `useState` に置かない（毎フレーム寄せるので再レンダリングが走る） */
  const view = useRef<MapView | null>(null)
  const lastDraw = useRef(0)

  useFrame((_state, delta) => {
    const mesh = meshRef.current
    if (!mesh) return
    if (!map) {
      mesh.visible = false
      return
    }

    const store = useSimStore.getState()
    const alpha = computeAlpha(performance.now(), store.status.renderPaused)
    const target = store.followTarget
    const ok = sampleVehicle(target, alpha, pose)
    mesh.visible = ok
    if (!ok) return

    // 画面は車体に固定する（カメラと同じく、車内のものは車体に貼り付く）
    composeVehicleMatrix(scratch.transform, pose.x, pose.y, pose.heading, scratch.base)
    composeFixedMatrix(scratch.transform, scratch.base, NAV_SCREEN.center, scratch.out)
    mesh.matrix.copy(scratch.out)
    mesh.matrixWorldNeedsUpdate = true

    // ★ 目標の見え方。**縮尺は速度で決める**（加速で引き、減速で寄る）
    const span = navSpanFor(pose.speed, Math.max(0.1, store.params.maxSpeed))
    const spanZoom = spanToZoom(map.bounds, TEX_W, TEX_H, span)
    const want: MapView = {
      zoom: spanZoom,
      centerX: pose.x,
      centerY: pose.y,
      rotation: rotationForHeading(pose.heading),
    }
    const step = Math.min(1, Math.max(0, delta))
    view.current = view.current
      ? lerpView(
          view.current,
          want,
          1 - Math.exp(-FOLLOW_RATE * step),
          1 - Math.exp(-TURN_RATE * step),
        )
      : want

    const now = performance.now()
    if (now - lastDraw.current < 1000 / NAV_FPS) return
    lastDraw.current = now

    const ctx = resources.canvas.getContext('2d')
    if (!ctx) return
    drawNav(ctx, map, view.current, target, pose, palette)
    resources.texture.needsUpdate = true
  })

  return (
    <mesh
      ref={meshRef}
      geometry={resources.geometry}
      material={resources.material}
      matrixAutoUpdate={false}
      frustumCulled={false}
      visible={false}
    />
  )
}

/** 見せたい範囲の幅 [m] を倍率へ直す。 */
function spanToZoom(
  bounds: { minX: number; maxX: number; minY: number; maxY: number },
  width: number,
  height: number,
  spanM: number,
): number {
  const spanX = Math.max(1e-6, bounds.maxX - bounds.minX)
  const spanY = Math.max(1e-6, bounds.maxY - bounds.minY)
  const base = Math.min((width - 12) / spanX, (height - 12) / spanY)
  return width / spanM / base
}

interface NavPalette {
  readonly roadSurface: string
  readonly buildingLow: string
  readonly taxiMarker: string
  readonly vehicleHighlight: string
  readonly vehicleGauge: string
}

function drawNav(
  ctx: CanvasRenderingContext2D,
  map: MapMessage,
  view: MapView,
  target: number,
  pose: { x: number; y: number; heading: number },
  palette: NavPalette,
): void {
  const p = createProjection(map.bounds, TEX_W, TEX_H, view, 6)

  ctx.fillStyle = palette.vehicleGauge
  ctx.fillRect(0, 0, TEX_W, TEX_H)

  drawBuildings(ctx, p, map.buildings, palette.buildingLow)
  drawRoads(ctx, p, map.edges, palette.roadSurface, 2.2, MIN_ROAD_PX)

  const route = frameBuffer.routes.get(target)
  if (route && route.length >= 2) drawRoute(ctx, p, route, palette.taxiMarker, 2.4)

  // 自車。進行方向を上にしているので、画面ではいつも上を向く
  drawHeadingMark(ctx, p, pose.x, pose.y, pose.heading, palette.vehicleHighlight, 5, '#0b0d10')
}
