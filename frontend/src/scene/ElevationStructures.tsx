/** 立体の地図の地形・橋桁・トンネル。高さを持たない地図では何も描かない。 */

import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { useSimStore } from '../store/simStore'
import { usePalette } from './usePalette'
import { TUNNEL_LIGHT_FACTOR, elevationOf } from './elevationView'
import { buildDeckGeometry, buildTerrainGeometry, buildTunnelGeometry } from './elevationGeometry'

/** 構造物（コンクリート）の色。擬似カメラの LBL_STRUCTURE と同じ系統 */
const STRUCTURE_COLOR = '#a4a096'

export interface ElevationStructuresProps {
  castShadow: boolean
  receiveShadow: boolean
}

export function ElevationStructures({ castShadow, receiveShadow }: ElevationStructuresProps) {
  const map = useSimStore((s) => s.map)
  const palette = usePalette()
  const model = elevationOf(map)
  const geoms = useMemo(() => {
    if (!model) return null
    const profiles = [...model.profiles.values()]
    const decks = profiles.filter((p) => p.structure === 'bridge').map(buildDeckGeometry)
    const tunnels = profiles.filter((p) => p.structure === 'tunnel').map((p) => buildTunnelGeometry(p, model.terrain))
    const deck = decks.length ? mergeGeometries(decks, false) : null
    const tunnel = tunnels.length ? mergeGeometries(tunnels, false) : null
    for (const g of decks) g.dispose()
    for (const g of tunnels) g.dispose()
    return {
      terrain: model.terrain ? buildTerrainGeometry(model.terrain, model) : null,
      deck,
      tunnel,
    }
  }, [model])
  const tunnelColor = useMemo(() => new THREE.Color(STRUCTURE_COLOR).multiplyScalar(TUNNEL_LIGHT_FACTOR), [])

  useEffect(() => {
    return () => {
      geoms?.terrain?.dispose()
      geoms?.deck?.dispose()
      geoms?.tunnel?.dispose()
    }
  }, [geoms])

  if (!geoms) return null
  return (
    <group>
      {geoms.terrain && (
        <mesh geometry={geoms.terrain} receiveShadow={receiveShadow}>
          <meshStandardMaterial color={palette.ground} roughness={1} metalness={0} />
        </mesh>
      )}
      {geoms.deck && (
        <mesh geometry={geoms.deck} castShadow={castShadow} receiveShadow={receiveShadow}>
          <meshStandardMaterial color={STRUCTURE_COLOR} roughness={0.9} metalness={0} side={THREE.DoubleSide} />
        </mesh>
      )}
      {geoms.tunnel && (
        <mesh geometry={geoms.tunnel}>
          {/* 坑内は暗い（擬似カメラの TUNNEL_LIGHT_FACTOR と揃える） */}
          <meshStandardMaterial color={tunnelColor} roughness={1} metalness={0} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  )
}
