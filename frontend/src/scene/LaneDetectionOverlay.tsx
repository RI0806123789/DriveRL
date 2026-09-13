/**
 * CNN が認識した車線を路面へ重ねて描く 3D オーバーレイ。
 *
 * バウンディングボックス（DetectionOverlay）は車線には向かない。車線は細長い
 * 曲線なので矩形では道の形をまったく表せず、認識のずれが見えない。ここでは
 * backend が返す車線中心線の点列（frame.detections の cls===4 の lanePoints）を
 * 実際の車線標示の上へ直接重ね、ずれていればそのままずれて見えるようにする。
 *
 * ★ lanePoints は**自車座標系**（前方 +x / 左 +y）で届く。ジオメトリはこの
 *   ローカル座標のまま作り（RouteLines の buildRibbon をそのまま流用できる。
 *   「前方・左」は「東・北」と同じ右手系の関係にあるため、buildRibbon の
 *   ENU 前提の式がそのまま成り立つ）、**車両の補間済みの位置・向き**（60fps）を
 *   group の transform として毎フレーム乗せる。20Hz の frame をそのまま
 *   使うとカクつくため、scene/interpolation.ts の補間結果を Vehicles.tsx /
 *   RouteLines.tsx と同じやり方で使う。検出点列の作り直し（frame 受信のたび）
 *   と姿勢の更新（毎描画フレーム）を分けるのは RouteLines と同じ設計。
 *
 * ★ polygonOffset の重ね順は CLAUDE.md の表のとおり厳守。進路の矢羽根
 *   （高さ 0.13 / -5,-10）より手前の高さ 0.15 / -6,-12 に置く。
 *   depthTest は切らない（建物の裏まで透けてしまうため）。
 *
 * ★ 追跡中の車両（followTarget）1 台分だけを描く。表示トグルは
 *   DetectionOverlay と同じ view.detections に従うが、**カメラモードは
 *   問わない**（路面に描くので俯瞰・追従でも意味がある。運転席専用の
 *   DetectionOverlay とはここが異なる）。
 */

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

/**
 * 帯のジオメトリを点列に合わせる。**点数が同じなら作り直さず頂点だけ書き換える。**
 *
 * 車線の点列は毎フレーム変わるが、点数は `LANE_POLYLINE_POINTS`（既定 6）で
 * 固定なので、頂点数も添字も変わらない。それでも `BufferGeometry` ごと
 * 捨てて作り直すと、車線が見えているあいだ毎秒 40 個の GPU バッファを
 * 生成・破棄し続けることになる（code_review S-02）。
 * 認識器を差し替えて点数が変わった場合だけ作り直す。
 */
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
    // 頂点が動くと境界球が古くなり、フラスタムカリングで消えることがある
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
  const lastReceived = useRef(-1)
  const lastFollowTarget = useRef(-1)
  const hasLane = useRef(false)
  /** 自前で作ったジオメトリだけを持つ（EMPTY_GEOMETRY は共有物なので破棄しない） */
  const built = useRef<{ ribbon: RibbonSlot; center: RibbonSlot }>({
    ribbon: { geom: null, points: 0 },
    center: { geom: null, points: 0 },
  })

  // 色は palette.ts の 1 か所だけに書く約束（CLAUDE.md）。白線と昼夜どちらでも
  // 見分けが付く色を usePalette() 経由で読む。帯は半透明、中心線は不透明寄り。
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

  // アンマウント時に自前のジオメトリも手放す
  useEffect(() => {
    const b = built
    return () => {
      b.current.ribbon.geom?.dispose()
      b.current.center.geom?.dispose()
    }
  }, [])

  useFrame(() => {
    const g = group.current
    if (!g) return

    const store = useSimStore.getState()
    if (!store.view.detections) {
      g.visible = false
      return
    }

    const alpha = computeAlpha(performance.now(), store.status.renderPaused)
    const followTarget = store.followTarget
    const tracked = sampleVehicle(followTarget, alpha, pose)
    if (!tracked) {
      g.visible = false
      return
    }

    // 検出結果は受信フレームが変わったときだけ作り直す。追跡対象を切り替えた
    // 直後は次の frame を待たず必ず読み直す（DetectionOverlay と同じ理由）
    if (frameBuffer.received !== lastReceived.current || followTarget !== lastFollowTarget.current) {
      lastReceived.current = frameBuffer.received
      lastFollowTarget.current = followTarget

      const dets = frameBuffer.curr?.detections?.[String(followTarget)]
      const laneDet = findLaneDetection(dets)
      const points = laneDet?.lanePoints ?? null

      // 点数が同じなら頂点を書き換えるだけ（S-02）。作り直すのは点数が変わったときだけ
      const ok = syncRibbon(built.current.ribbon, points, LANE_WIDTH)
      syncRibbon(built.current.center, points, CENTERLINE_WIDTH)
      hasLane.current = ok

      // mesh に差すのは参照が変わったときだけでよいが、比較のほうが高くつかないので毎回入れる
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

    // ENU -> three（protocol.md 1.3）。lanePoints は自車座標系のままジオメトリ化
    // してあるので、車両の位置・向きへ持ち上げるのは group の transform だけでよい
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
