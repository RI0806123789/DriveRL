/** 歩行者用信号機（縦 2 灯）の描画。**現示は車両信号の裏返し。** */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

import { frameBuffer } from '../store/frameBuffer'
import type { MapNode, MapSignal } from '../types/protocol'
import { SIGNAL_LAMP_COLORS } from './palette'
import {
  PED_ROLE_GREEN,
  PED_ROLE_RED,
  buildPedestrianSignalPlacement,
  createPedestrianLampGeometry,
  pedestrianWalkable,
} from './pedestrianSignalGeometry'
import { usePalette } from './usePalette'

/** 歩行者用は青と赤の 2 色だけ（黄は無い）。色は車両用と同じ出典 */
const GREEN = new THREE.Color(SIGNAL_LAMP_COLORS[0])
const RED = new THREE.Color(SIGNAL_LAMP_COLORS[2])

export interface PedestrianSignalsProps {
  signals: MapSignal[]
  nodes: MapNode[]
  castShadow: boolean
}

export function PedestrianSignals({ signals, nodes, castShadow }: PedestrianSignalsProps) {
  const palette = usePalette()
  const colorOff = useMemo(() => new THREE.Color(palette.signalLampOff), [palette.signalLampOff])

  const { structure, lamps } = useMemo(() => {
    const { parts, lamps: slots } = buildPedestrianSignalPlacement(signals, nodes)
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
        roughness: 0.75,
        metalness: 0.3,
        side: THREE.DoubleSide,
      }),
    [palette.signalHousing],
  )

  const lampMaterial = useMemo(
    () => new THREE.MeshBasicMaterial({ toneMapped: false, vertexColors: true }),
    [],
  )

  const lampGeometry = useMemo(() => createPedestrianLampGeometry(), [])

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
  }, [lamps, colorOff])

  useFrame(() => {
    const mesh = lampMeshRef.current
    if (!mesh || lamps.length === 0) return
    // 車両信号と同じく、現示が変わったときだけ塗り替える
    if (frameBuffer.signalVersion === lastVersion.current) return
    lastVersion.current = frameBuffer.signalVersion

    const phases = frameBuffer.signals
    for (let i = 0; i < lamps.length; i++) {
      const lamp = lamps[i]
      const phase = phases[lamp.signalIndex]
      if (phase === undefined) {
        mesh.setColorAt(i, colorOff)
        continue
      }
      const walk = pedestrianWalkable(phase)
      const lit = walk ? lamp.role === PED_ROLE_GREEN : lamp.role === PED_ROLE_RED
      mesh.setColorAt(i, lit ? (walk ? GREEN : RED) : colorOff)
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
