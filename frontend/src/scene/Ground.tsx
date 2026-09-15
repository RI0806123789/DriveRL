/** 地面。マップの範囲より少し大きい平面を 1 枚敷き、影を受けさせる。 */

import { useMemo } from 'react'
import * as THREE from 'three'
import type { MapBounds } from '../types/protocol'
import { usePalette } from './usePalette'

export interface GroundProps {
  bounds: MapBounds | null
  showGrid: boolean
  receiveShadow: boolean
}

/** ENU の bounds から地面のサイズと中心（Three 座標）を求める */
function groundFromBounds(bounds: MapBounds | null) {
  if (!bounds) return { size: 1600, cx: 0, cz: 0 }
  const w = bounds.maxX - bounds.minX
  const h = bounds.maxY - bounds.minY
  const size = Math.max(w, h) * 2.2 + 200
  const cx = (bounds.minX + bounds.maxX) / 2
  const cz = -((bounds.minY + bounds.maxY) / 2)
  return { size, cx, cz }
}

export function Ground({ bounds, showGrid, receiveShadow }: GroundProps) {
  const palette = usePalette()
  const { size, cx, cz } = useMemo(() => groundFromBounds(bounds), [bounds])

  const gridArgs = useMemo(() => {
    const span = Math.round(size / 2 / 100) * 100 * 2
    const divisions = Math.max(4, Math.round(span / 50))
    return [
      span,
      divisions,
      new THREE.Color(palette.gridMajor),
      new THREE.Color(palette.gridMinor),
    ] as const
  }, [size, palette])

  return (
    <group position={[cx, 0, cz]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow={receiveShadow}>
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color={palette.ground} roughness={1} metalness={0} />
      </mesh>
      {showGrid && (
        <gridHelper args={gridArgs} position={[0, 0.012, 0]} renderOrder={1} />
      )}
    </group>
  )
}
