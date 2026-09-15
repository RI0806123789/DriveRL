/** 3D 画面のクリック介入（memo F-05 / 5章「介入も現実の交通現象の一部」）。 */

import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { send } from '../store/connection'
import { useSimStore } from '../store/simStore'
import type { MapBounds } from '../types/protocol'
import { usePalette } from './usePalette'
import { vehicleColor } from './vehicleColors'

/** これ以上動いたらドラッグ（＝カメラ操作）とみなす [px] */
const DRAG_THRESHOLD_PX = 5
/** これ以上長押ししたらクリックとみなさない [ms] */
const CLICK_MAX_MS = 600

export interface InteractionPlaneProps {
  bounds: MapBounds | null
}

/** 介入用の当たり判定平面のサイズと中心（Three 座標） */
function planeFromBounds(bounds: MapBounds | null) {
  if (!bounds) return { size: 2000, cx: 0, cz: 0 }
  const w = bounds.maxX - bounds.minX
  const h = bounds.maxY - bounds.minY
  return {
    size: Math.max(w, h) * 1.6 + 200,
    cx: (bounds.minX + bounds.maxX) / 2,
    cz: -((bounds.minY + bounds.maxY) / 2),
  }
}

export function InteractionPlane({ bounds }: InteractionPlaneProps) {
  const interaction = useSimStore((s) => s.interaction)
  const obstacleRadius = useSimStore((s) => s.obstacleRadius)
  const mapLoaded = useSimStore((s) => s.map !== null)

  const { size, cx, cz } = useMemo(() => planeFromBounds(bounds), [bounds])

  const down = useRef<{ x: number; y: number; t: number } | null>(null)
  const cursor = useRef<{ x: number; z: number } | null>(null)
  const ghostRef = useRef<THREE.Group>(null)

  const active = interaction !== 'none' && mapLoaded

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    down.current = { x: e.nativeEvent.clientX, y: e.nativeEvent.clientY, t: performance.now() }
  }

  const handlePointerUp = (e: ThreeEvent<PointerEvent>) => {
    const start = down.current
    down.current = null
    if (!active || !start) return

    const dx = e.nativeEvent.clientX - start.x
    const dy = e.nativeEvent.clientY - start.y
    const moved = Math.hypot(dx, dy)
    if (moved > DRAG_THRESHOLD_PX) return
    if (performance.now() - start.t > CLICK_MAX_MS) return

    const enuX = e.point.x
    const enuY = -e.point.z

    if (interaction === 'obstacle') {
      send({ type: 'add_obstacle', x: enuX, y: enuY, radius: obstacleRadius })
    } else if (interaction === 'vehicle') {
      send({ type: 'spawn_vehicle', x: enuX, y: enuY })
    }
  }

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!active) {
      cursor.current = null
      return
    }
    cursor.current = { x: e.point.x, z: e.point.z }
  }

  const handlePointerOut = () => {
    cursor.current = null
  }

  useFrame(() => {
    const g = ghostRef.current
    if (!g) return
    const at = active ? cursor.current : null
    g.visible = at !== null
    if (at) g.position.set(at.x, 0, at.z)
  })

  return (
    <group>
      <mesh
        position={[cx, 0, cz]}
        rotation={[-Math.PI / 2, 0, 0]}
        visible={false}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
      >
        <planeGeometry args={[size, size]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>

      {active && (
        <group ref={ghostRef} visible={false}>
          <Ghost mode={interaction} radius={obstacleRadius} />
        </group>
      )}
    </group>
  )
}

function Ghost({ mode, radius }: { mode: 'obstacle' | 'vehicle'; radius: number }) {
  const palette = usePalette()
  const color = mode === 'obstacle' ? palette.obstacleCone : vehicleColor(0)

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
        <ringGeometry args={[mode === 'obstacle' ? radius * 1.6 : 3.0, mode === 'obstacle' ? radius * 2.1 : 3.5, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.7} side={THREE.DoubleSide} />
      </mesh>

      {mode === 'obstacle' ? (
        <mesh position={[0, 0.375, 0]}>
          <coneGeometry args={[radius, 0.75, 14]} />
          <meshBasicMaterial color={color} transparent opacity={0.45} />
        </mesh>
      ) : (
        <mesh position={[0, 0.6, 0]}>
          <boxGeometry args={[4.4, 1.2, 1.86]} />
          <meshBasicMaterial color={color} transparent opacity={0.35} />
        </mesh>
      )}
    </group>
  )
}
