/** 乗車地点・降車地点の目印（実用モード）。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { useSimStore } from '../store/simStore'
import { isRidingPhase } from '../types/protocol'
import { usePalette } from './usePalette'

/** 柱の高さ [m]。 */
const PILLAR_HEIGHT = 10
const PILLAR_RADIUS = 0.3

export function TaxiMarkers() {
  const palette = usePalette()
  const phase = useSimStore((s) => s.taxi.phase)
  const pickup = useSimStore((s) => s.taxi.pickup)
  const dropoff = useSimStore((s) => s.taxi.dropoff)

  const pillarRef = useRef<THREE.Mesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)

  const resources = useMemo(() => {
    const pillar = new THREE.CylinderGeometry(PILLAR_RADIUS, PILLAR_RADIUS * 0.6, PILLAR_HEIGHT, 12, 1, true)
    pillar.translate(0, PILLAR_HEIGHT / 2, 0)
    const ring = new THREE.TorusGeometry(2.4, 0.16, 8, 36)
    ring.rotateX(-Math.PI / 2)
    const material = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide,
      depthWrite: false,
      toneMapped: false,
    })
    const ringMaterial = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.9,
      toneMapped: false,
    })
    return { pillar, ring, material, ringMaterial }
  }, [])

  useEffect(() => {
    resources.material.color.set(palette.taxiMarker)
    resources.ringMaterial.color.set(palette.taxiMarker)
  }, [resources, palette])

  useEffect(() => {
    const r = resources
    return () => {
      r.pillar.dispose()
      r.ring.dispose()
      r.material.dispose()
      r.ringMaterial.dispose()
    }
  }, [resources])

  const target = isRidingPhase(phase) ? dropoff : phase === 'idle' ? null : pickup

  useFrame(() => {
    const pillar = pillarRef.current
    const ring = ringRef.current
    if (!pillar || !ring) return
    const visible = target !== null
    pillar.visible = visible
    ring.visible = visible
    if (!visible || target === null) return
    const t = performance.now() * 0.0022
    pillar.position.set(target[0], 0, -target[1])
    ring.position.set(target[0], 0.08, -target[1])
    ring.scale.setScalar(1 + 0.08 * Math.sin(t))
  })

  return (
    <group>
      <mesh
        ref={pillarRef}
        geometry={resources.pillar}
        material={resources.material}
        visible={false}
        frustumCulled={false}
      />
      <mesh
        ref={ringRef}
        geometry={resources.ring}
        material={resources.ringMaterial}
        visible={false}
        frustumCulled={false}
      />
    </group>
  )
}
