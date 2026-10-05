/** 追従中の車の見通しと死角（`frame.occlusion`）を、その車のローカル座標の板として路面に描く共通部品。**Canvas の中に置くこと。** */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import { useSimStore, type ViewToggles } from '../store/simStore'
import type { OcclusionView } from '../types/protocol'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { useHiddenFromMirrors } from './mirrorHidden'
import { OCCLUSION_MAX_VERTICES } from './occlusionGeometry'
import { composeVehicleMatrix, createTransformScratch } from './vehicleGeometry'

export type OcclusionWriter = (
  view: OcclusionView,
  positions: Float32Array,
  colors: Float32Array,
  capacity: number,
) => number

interface Props {
  /** このトグルが ON の間だけ描く */
  toggle: keyof Pick<ViewToggles, 'cameraFrustums' | 'occlusionShadows'>
  write: OcclusionWriter
  opacity: number
  renderOrder: number
}

/** 見通しと死角を描く車。俯瞰・実用モード・トグルが OFF なら -1 */
export function occlusionTarget(store: ReturnType<typeof useSimStore.getState>, toggle: Props['toggle']): number {
  if (store.mode !== 'dev' || store.cameraMode === 'orbit' || !store.view[toggle]) return -1
  return store.followTarget
}

export function OcclusionLayer({ toggle, write, opacity, renderOrder }: Props) {
  const resources = useMemo(() => {
    const positions = new Float32Array(OCCLUSION_MAX_VERTICES * 3)
    const colors = new Float32Array(OCCLUSION_MAX_VERTICES * 3)
    const position = new THREE.BufferAttribute(positions, 3)
    const color = new THREE.BufferAttribute(colors, 3)
    position.setUsage(THREE.DynamicDrawUsage)
    color.setUsage(THREE.DynamicDrawUsage)
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', position)
    geometry.setAttribute('color', color)
    geometry.setDrawRange(0, 0)
    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    return { positions, colors, position, color, geometry, material }
  }, [opacity])
  useEffect(
    () => () => {
      resources.geometry.dispose()
      resources.material.dispose()
    },
    [resources],
  )

  const mesh = useRef<THREE.Mesh>(null)
  // 路面に重ねた説明の板なので、鏡・周囲カメラの映像には写さない
  useHiddenFromMirrors(mesh)
  const scratch = useMemo(() => ({ transform: createTransformScratch(), pose: createPose() }), [])
  const drawn = useRef<{ view: OcclusionView | null; count: number }>({ view: null, count: 0 })

  useFrame((state) => {
    const m = mesh.current
    if (!m) return
    const store = useSimStore.getState()
    // 車体と同じ時刻・同じ区間で補間する。computeAlpha は表示の区間を進めるので、curr を読む前に呼ぶ
    const alpha = computeAlpha(state.clock.oldTime, store.status.renderPaused)
    const target = occlusionTarget(store, toggle)
    const view = target >= 0 ? frameBuffer.curr?.occlusion?.[String(target)] : undefined
    if (!view || !sampleVehicle(target, alpha, scratch.pose)) {
      m.visible = false
      return
    }
    if (view !== drawn.current.view) {
      const count = write(view, resources.positions, resources.colors, OCCLUSION_MAX_VERTICES)
      resources.position.needsUpdate = true
      resources.color.needsUpdate = true
      resources.geometry.setDrawRange(0, count)
      drawn.current = { view, count }
    }
    composeVehicleMatrix(scratch.transform, scratch.pose.x, scratch.pose.y, scratch.pose.heading, m.matrix)
    m.matrixWorldNeedsUpdate = true
    m.visible = drawn.current.count > 0
  })

  return (
    <mesh
      ref={mesh}
      geometry={resources.geometry}
      material={resources.material}
      matrixAutoUpdate={false}
      visible={false}
      renderOrder={renderOrder}
      frustumCulled={false}
    />
  )
}
