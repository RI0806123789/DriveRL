/** 車車間通信（V2X）でメッセージをやり取りしている車どうしを、半透明の水色の線で結ぶ。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { useHiddenFromMirrors } from './mirrorHidden'
import { V2X_MAX_PEERS, linkPairs, writeLinkSegments } from './v2xLinks'

const LINK_COLOR = '#4dd0e1'
const LINK_OPACITY = 0.85

export function V2XLinks({ maxVehicles }: { maxVehicles: number }) {
  const capacity = Math.max(1, maxVehicles * V2X_MAX_PEERS)
  const resources = useMemo(() => {
    const positions = new Float32Array(capacity * 2 * 3)
    const attribute = new THREE.BufferAttribute(positions, 3)
    attribute.setUsage(THREE.DynamicDrawUsage)
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', attribute)
    geometry.setDrawRange(0, 0)
    const material = new THREE.LineBasicMaterial({
      color: LINK_COLOR,
      transparent: true,
      opacity: LINK_OPACITY,
      depthWrite: false,
    })
    return { positions, attribute, geometry, material }
  }, [capacity])
  useEffect(
    () => () => {
      resources.geometry.dispose()
      resources.material.dispose()
    },
    [resources],
  )

  const lines = useRef<THREE.LineSegments>(null)
  // 画面の前に貼った板と同じく、鏡・周囲カメラの映像には写さない
  useHiddenFromMirrors(lines)
  const pose = useMemo(createPose, [])
  const drawn = useRef(0)

  useFrame((state) => {
    // 車体と同じ時刻・同じ区間で補間する（ずれると線の端が車から浮く）
    const alpha = computeAlpha(state.clock.oldTime, useSimStore.getState().status.renderPaused)
    const vehicles = frameBuffer.curr?.vehicles
    const count = vehicles
      ? writeLinkSegments(
          linkPairs(vehicles),
          (id) => (sampleVehicle(id, alpha, pose) ? { x: pose.x, y: pose.y } : null),
          resources.positions,
        )
      : 0
    if (count > 0) resources.attribute.needsUpdate = true
    if (count !== drawn.current) {
      resources.geometry.setDrawRange(0, count)
      drawn.current = count
    }
  })

  return (
    <lineSegments
      ref={lines}
      geometry={resources.geometry}
      material={resources.material}
      frustumCulled={false}
    />
  )
}
