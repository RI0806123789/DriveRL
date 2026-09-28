/** 後退中の車の後ろへ、予測ガイド線（左右の線と 1/2/3m の目盛り）を路面に描く。**Canvas の中に置くこと。** */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import {
  GUIDE_LENGTH_M,
  GUIDE_MARKS_M,
  GUIDE_STEP_M,
  GUIDE_WIDTH_M,
  guideColor,
  reverseGuideLines,
  type GuidePoint,
} from './reverseGuideGeometry'
import { composeVehicleMatrix, createTransformScratch } from './vehicleGeometry'

/** 路面から浮かせる高さ [m] */
const LIFT = 0.03
const SEGMENTS = Math.round(GUIDE_LENGTH_M / GUIDE_STEP_M)
/** 左右の線の区間 + 目盛り */
const QUADS = SEGMENTS * 2 + GUIDE_MARKS_M.length

function makeGeometry(): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(QUADS * 4 * 3), 3))
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(QUADS * 4 * 3), 3))
  const index: number[] = []
  for (let q = 0; q < QUADS; q += 1) {
    const b = q * 4
    index.push(b, b + 1, b + 2, b, b + 2, b + 3)
  }
  g.setIndex(index)
  return g
}

/** 車両ローカル（前方 +x / 左 +y）の線分を、太さのある板として書き込む */
function writeQuad(
  pos: Float32Array,
  col: Float32Array,
  quad: number,
  a: GuidePoint,
  b: GuidePoint,
  color: THREE.Color,
): void {
  const dx = b.x - a.x
  const dy = b.y - a.y
  const len = Math.hypot(dx, dy) || 1
  const nx = (-dy / len) * GUIDE_WIDTH_M * 0.5
  const ny = (dx / len) * GUIDE_WIDTH_M * 0.5
  const corners: Array<[number, number]> = [
    [a.x + nx, a.y + ny],
    [a.x - nx, a.y - ny],
    [b.x - nx, b.y - ny],
    [b.x + nx, b.y + ny],
  ]
  corners.forEach(([x, y], k) => {
    const i = (quad * 4 + k) * 3
    // three の車両ローカルは X 前方・Y 上・Z 右
    pos[i] = x
    pos[i + 1] = LIFT
    pos[i + 2] = -y
    col[i] = color.r
    col[i + 1] = color.g
    col[i + 2] = color.b
  })
}

export function VehicleReverseGuide() {
  const mesh = useRef<THREE.Mesh>(null)
  const geometry = useMemo(makeGeometry, [])
  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        // 実車の後退ガイドと同じく映像に重ねる（自車のトランクの陰でも見えるように）
        depthTest: false,
        depthWrite: false,
        side: THREE.DoubleSide,
        fog: false,
      }),
    [],
  )
  useEffect(
    () => () => {
      geometry.dispose()
      material.dispose()
    },
    [geometry, material],
  )
  const scratch = useMemo(
    () => ({ transform: createTransformScratch(), pose: createPose(), color: new THREE.Color() }),
    [],
  )

  useFrame((state) => {
    const m = mesh.current
    if (!m) return
    const store = useSimStore.getState()
    const target = store.cameraMode === 'orbit' ? -1 : store.followTarget
    const reversing =
      target >= 0 && !!frameBuffer.curr?.vehicles.find((v) => v.id === target)?.reverse
    if (
      !reversing ||
      !sampleVehicle(target, computeAlpha(state.clock.oldTime, store.status.renderPaused), scratch.pose)
    ) {
      m.visible = false
      return
    }
    const guide = reverseGuideLines(scratch.pose.steer)
    const pos = geometry.getAttribute('position') as THREE.BufferAttribute
    const col = geometry.getAttribute('color') as THREE.BufferAttribute
    const p = pos.array as Float32Array
    const c = col.array as Float32Array
    let quad = 0
    for (const line of [guide.left, guide.right]) {
      for (let i = 0; i < SEGMENTS; i += 1) {
        scratch.color.set(guideColor(guide.distances[i]))
        writeQuad(p, c, quad, line[i], line[i + 1], scratch.color)
        quad += 1
      }
    }
    for (const mark of GUIDE_MARKS_M) {
      const i = Math.round(mark / GUIDE_STEP_M)
      scratch.color.set(guideColor(mark))
      writeQuad(p, c, quad, guide.left[i], guide.right[i], scratch.color)
      quad += 1
    }
    pos.needsUpdate = true
    col.needsUpdate = true
    geometry.computeBoundingSphere()
    composeVehicleMatrix(scratch.transform, scratch.pose.x, scratch.pose.y, scratch.pose.heading, m.matrix)
    m.matrixWorldNeedsUpdate = true
    m.visible = true
  })

  return (
    <mesh
      ref={mesh}
      geometry={geometry}
      material={material}
      matrixAutoUpdate={false}
      visible={false}
      renderOrder={20}
      frustumCulled={false}
    />
  )
}
