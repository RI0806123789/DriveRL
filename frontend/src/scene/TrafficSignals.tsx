/** 交通信号機（車両用・横型3灯式）の描画。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { frameBuffer } from '../store/frameBuffer'
import type { MapNode, MapSignal } from '../types/protocol'
import { SIGNAL_LAMP_COLORS } from './palette'
import { buildSignalPlacement, createLampGeometry } from './signalGeometry'
import { usePalette } from './usePalette'

/** 添字は signalGeometry の ROLE_* と一致（0=青 / 1=黄 / 2=赤）。 */
const PHASE_COLORS = SIGNAL_LAMP_COLORS.map((c) => new THREE.Color(c))

export interface TrafficSignalsProps {
  signals: MapSignal[]
  nodes: MapNode[]
  castShadow: boolean
}

export function TrafficSignals({ signals, nodes, castShadow }: TrafficSignalsProps) {
  const palette = usePalette()
  const colorOff = useMemo(() => new THREE.Color(palette.signalLampOff), [palette.signalLampOff])
  const { structure, lamps } = useMemo(() => {
    const { parts, lamps: slots } = buildSignalPlacement(signals, nodes)
    const merged = parts.length > 0 ? mergeGeometries(parts, false) : null
    for (const g of parts) g.dispose()
    return { structure: merged, lamps: slots }
  }, [signals, nodes])

  const lampMeshRef = useRef<THREE.InstancedMesh>(null)
  const lastVersion = useRef(-1)

  useEffect(() => {
    lastVersion.current = -1
  }, [colorOff])

  const structureMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.signalHousing,
        roughness: 0.72,
        metalness: 0.35,
        side: THREE.DoubleSide,
      }),
    [palette.signalHousing],
  )

  const lampMaterial = useMemo(
    () => new THREE.MeshBasicMaterial({ toneMapped: false, vertexColors: true }),
    [],
  )

  const lampGeometry = useMemo(() => createLampGeometry(), [])

  useEffect(() => {
    return () => {
      structure?.dispose()
      structureMaterial.dispose()
      lampMaterial.dispose()
      lampGeometry.dispose()
    }
  }, [structure, structureMaterial, lampMaterial, lampGeometry])

  useEffect(() => {
    const mesh = lampMeshRef.current
    if (!mesh || lamps.length === 0) return
    const dummy = new THREE.Object3D()
    for (let i = 0; i < lamps.length; i++) {
      const lamp = lamps[i]
      dummy.position.copy(lamp.position)
      dummy.rotation.set(0, lamp.facing, 0)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
      mesh.setColorAt(i, colorOff)
    }
    mesh.count = lamps.length
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    mesh.computeBoundingSphere()
    lastVersion.current = -1
  }, [lamps])

  useFrame(() => {
    const mesh = lampMeshRef.current
    if (!mesh || lamps.length === 0) return
    if (frameBuffer.signalVersion === lastVersion.current) return
    lastVersion.current = frameBuffer.signalVersion

    const phases = frameBuffer.signals
    for (let i = 0; i < lamps.length; i++) {
      const lamp = lamps[i]
      const phase = phases[lamp.signalIndex]
      const lit = phase !== undefined && phase === lamp.role
      mesh.setColorAt(i, lit ? PHASE_COLORS[lamp.role] : colorOff)
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  })

  if (signals.length === 0) return null

  return (
    <group>
      {structure && (
        <mesh geometry={structure} material={structureMaterial} castShadow={castShadow} />
      )}
      <instancedMesh
        ref={lampMeshRef}
        args={[lampGeometry, lampMaterial, Math.max(1, lamps.length)]}
        count={0}
        frustumCulled={false}
      />
    </group>
  )
}
