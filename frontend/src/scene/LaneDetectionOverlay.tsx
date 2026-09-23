/** CNN が認識した車線を路面へ重ねて描く 3D オーバーレイ。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { buildRibbon, writeRibbonPositions, type Point2 } from './routeArrowGeometry'
import { usePalette } from './usePalette'
import { DET_LANE, type Detection } from '../types/protocol'

/** 路面から浮かせる高さ [m]。進路の矢羽根(0.13)より手前 */
const LANE_Y = 0.15
/** 車線帯の全幅 [m]。バックエンドの車線幅（中心から左右 1.6m）に合わせる */
const LANE_WIDTH = 3.2
/** 中心線の表示上の太さ [m] */
const CENTERLINE_WIDTH = 0.3

/** ジオメトリ未生成時のプレースホルダ。空なので破棄不要（複数 mesh で共有） */
const EMPTY_GEOMETRY = new THREE.BufferGeometry()

/** 検出一覧から車線（cls===4）を 1 件探す。信頼度降順で届くので最初の 1 件でよい */
function findLaneDetection(dets: Detection[] | undefined): Detection | null {
  if (!dets) return null
  for (const d of dets) {
    if (d.cls === DET_LANE && d.lanePoints) return d
  }
  return null
}

/** 使い回しているジオメトリと、それが対応している点数 */
interface RibbonSlot {
  geom: THREE.BufferGeometry | null
  /** `geom` を作ったときの点数。これが変わったときだけ作り直す */
  points: number
}

/** 帯のジオメトリを点列に合わせる。**点数が同じなら作り直さず頂点だけ書き換える。** */
function syncRibbon(slot: RibbonSlot, points: Point2[] | null, width: number): boolean {
  if (!points || points.length < 2) {
    slot.geom?.dispose()
    slot.geom = null
    slot.points = 0
    return false
  }

  if (slot.geom && slot.points === points.length) {
    const attr = slot.geom.getAttribute('position') as THREE.BufferAttribute
    writeRibbonPositions(attr.array as Float32Array, points, width, LANE_Y)
    attr.needsUpdate = true
    slot.geom.computeBoundingSphere()
    return true
  }

  slot.geom?.dispose()
  slot.geom = buildRibbon(points, width, LANE_Y)
  slot.points = slot.geom ? points.length : 0
  return slot.geom !== null
}

export function LaneDetectionOverlay() {
  const palette = usePalette()

  const group = useRef<THREE.Group>(null)
  const ribbonMesh = useRef<THREE.Mesh>(null)
  const centerMesh = useRef<THREE.Mesh>(null)

  const pose = useMemo(createPose, [])
  const lastDisplayed = useRef(-1)
  const lastFollowTarget = useRef(-1)
  const hasLane = useRef(false)
  /** 自前で作ったジオメトリだけを持つ（EMPTY_GEOMETRY は共有物なので破棄しない） */
  const built = useRef<{ ribbon: RibbonSlot; center: RibbonSlot }>({
    ribbon: { geom: null, points: 0 },
    center: { geom: null, points: 0 },
  })

  const ribbonMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(palette.laneOverlay),
        transparent: true,
        opacity: 0.38,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
        polygonOffset: true,
        polygonOffsetFactor: -6,
        polygonOffsetUnits: -12,
      }),
    [palette],
  )
  const centerMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(palette.laneOverlay),
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
        polygonOffset: true,
        polygonOffsetFactor: -6,
        polygonOffsetUnits: -12,
      }),
    [palette],
  )

  useEffect(() => {
    return () => {
      ribbonMaterial.dispose()
      centerMaterial.dispose()
    }
  }, [ribbonMaterial, centerMaterial])

  useEffect(() => {
    const b = built
    return () => {
      b.current.ribbon.geom?.dispose()
      b.current.center.geom?.dispose()
    }
  }, [])

  useFrame((state) => {
    const g = group.current
    if (!g) return

    const store = useSimStore.getState()
    if (!store.view.detections) {
      g.visible = false
      return
    }

    const alpha = computeAlpha(state.clock.oldTime, store.status.renderPaused)
    const followTarget = store.followTarget
    const tracked = sampleVehicle(followTarget, alpha, pose)
    if (!tracked) {
      g.visible = false
      return
    }

    if (frameBuffer.displayed !== lastDisplayed.current || followTarget !== lastFollowTarget.current) {
      lastDisplayed.current = frameBuffer.displayed
      lastFollowTarget.current = followTarget

      const dets = frameBuffer.curr?.detections?.[String(followTarget)]
      const laneDet = findLaneDetection(dets)
      const points = laneDet?.lanePoints ?? null

      const ok = syncRibbon(built.current.ribbon, points, LANE_WIDTH)
      syncRibbon(built.current.center, points, CENTERLINE_WIDTH)
      hasLane.current = ok

      if (ribbonMesh.current) {
        ribbonMesh.current.geometry = built.current.ribbon.geom ?? EMPTY_GEOMETRY
      }
      if (centerMesh.current) {
        centerMesh.current.geometry = built.current.center.geom ?? EMPTY_GEOMETRY
      }
    }

    if (!hasLane.current) {
      g.visible = false
      return
    }

    g.visible = true
    g.position.set(pose.x, 0, -pose.y)
    g.rotation.y = pose.heading
  })

  return (
    <group ref={group} visible={false}>
      <mesh ref={ribbonMesh} material={ribbonMaterial} renderOrder={4} />
      <mesh ref={centerMesh} material={centerMaterial} renderOrder={5} />
    </group>
  )
}
