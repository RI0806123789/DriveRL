/** 建物メッシュ。 */

import { useEffect, useLayoutEffect, useMemo } from 'react'
import { usePalette } from './usePalette'
import type { MapBuilding } from '../types/protocol'
import { buildBuildingChunks, paintBuildingChunks } from './buildingGeometry'

export interface BuildingsProps {
  buildings: MapBuilding[]
  castShadow: boolean
  receiveShadow: boolean
}

export function Buildings({ buildings, castShadow, receiveShadow }: BuildingsProps) {
  const palette = usePalette()
  // 形はマップが変わったときだけ作る。昼夜の切り替えで作り直すと、金沢（35,607 棟）では
  //   押し出しとマージだけでメインスレッドが長く止まる
  const chunks = useMemo(() => buildBuildingChunks(buildings), [buildings])

  // 最初の画から色が付いているよう、描く前に塗る
  useLayoutEffect(() => {
    paintBuildingChunks(chunks, palette.buildingLow, palette.buildingHigh)
  }, [chunks, palette.buildingLow, palette.buildingHigh])

  useEffect(() => {
    return () => {
      for (const g of chunks) g.dispose()
    }
  }, [chunks])

  if (chunks.length === 0) return null

  return (
    <group>
      {chunks.map((geom, i) => (
        <mesh key={i} geometry={geom} castShadow={castShadow} receiveShadow={receiveShadow}>
          <meshStandardMaterial vertexColors roughness={0.86} metalness={0.06} />
        </mesh>
      ))}
    </group>
  )
}
