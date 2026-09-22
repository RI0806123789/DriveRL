/** 歩行者用信号機（縦 2 灯）の描画。**現示は車両信号の裏返し。** */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import type { MapSignal } from '../types/protocol'
import { SIGNAL_LAMP_COLORS } from './palette'
import {
  PED_MOUNT_HEIGHT,
  PED_ROLE_GREEN,
  PED_ROLE_RED,
  buildPedestrianSignalPlacement,
  createPedestrianHousingGeometry,
  createPedestrianLampGeometry,
  createPedestrianPoleGeometry,
  pedestrianWalkable,
} from './pedestrianSignalGeometry'
import { usePalette } from './usePalette'

/** 歩行者用は青と赤の 2 色だけ（黄は無い）。色は車両用と同じ出典 */
const GREEN = new THREE.Color(SIGNAL_LAMP_COLORS[0])
const RED = new THREE.Color(SIGNAL_LAMP_COLORS[2])

/** 灯火の状態。塗り替えを変わったものだけに絞るために覚える */
const LIT_OFF = 0
const LIT_ON = 1

export interface PedestrianSignalsProps {
  signals: MapSignal[]
  castShadow: boolean
}

export function PedestrianSignals({ signals, castShadow }: PedestrianSignalsProps) {
  const palette = usePalette()
  const colorOff = useMemo(() => new THREE.Color(palette.signalLampOff), [palette.signalLampOff])

  const { placements, lamps } = useMemo(
    () => buildPedestrianSignalPlacement(signals),
    [signals],
  )

  const poleRef = useRef<THREE.InstancedMesh>(null)
  const housingRef = useRef<THREE.InstancedMesh>(null)
  const lampMeshRef = useRef<THREE.InstancedMesh>(null)
  const lastLit = useRef<Int8Array>(new Int8Array(0))
  const lastVersion = useRef(-1)

  // 色だけ差し替える。マテリアルを作り直すと args が変わって instancedMesh が
  // count=0 に戻り、行列を書く useEffect は走らないので灯器が消える（S-01）
  const resources = useMemo(
    () => ({
      pole: createPedestrianPoleGeometry(),
      housing: createPedestrianHousingGeometry(),
      lamp: createPedestrianLampGeometry(),
      structureMaterial: new THREE.MeshStandardMaterial({
        roughness: 0.75,
        metalness: 0.3,
        side: THREE.DoubleSide,
      }),
      lampMaterial: new THREE.MeshBasicMaterial({ toneMapped: false, vertexColors: true }),
    }),
    [],
  )

  useEffect(() => {
    const r = resources
    return () => {
      r.pole.dispose()
      r.housing.dispose()
      r.lamp.dispose()
      r.structureMaterial.dispose()
      r.lampMaterial.dispose()
    }
  }, [resources])

  useEffect(() => {
    resources.structureMaterial.color.set(palette.signalHousing)
  }, [resources, palette.signalHousing])

  useEffect(() => {
    const pole = poleRef.current
    const housing = housingRef.current
    if (!pole || !housing) return
    const dummy = new THREE.Object3D()
    for (let i = 0; i < placements.length; i++) {
      const p = placements[i]
      dummy.rotation.set(0, 0, 0)
      dummy.position.set(p.x, 0, p.z)
      dummy.updateMatrix()
      pole.setMatrixAt(i, dummy.matrix)

      dummy.position.set(p.x, PED_MOUNT_HEIGHT, p.z)
      dummy.rotation.set(0, p.facing, 0)
      dummy.updateMatrix()
      housing.setMatrixAt(i, dummy.matrix)
    }
    for (const mesh of [pole, housing]) {
      mesh.count = placements.length
      mesh.instanceMatrix.needsUpdate = true
      mesh.computeBoundingSphere()
    }
  }, [placements])

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
    lastLit.current = new Int8Array(lamps.length).fill(-1)
    lastVersion.current = -1
  }, [lamps, colorOff])

  useFrame(() => {
    const mesh = lampMeshRef.current
    if (!mesh || lamps.length === 0) return
    // 車両信号と同じく、現示が変わったときだけ塗り替える
    if (frameBuffer.signalVersion === lastVersion.current) return
    lastVersion.current = frameBuffer.signalVersion

    const phases = frameBuffer.signals
    const seen = lastLit.current
    let dirty = false
    for (let i = 0; i < lamps.length; i++) {
      const lamp = lamps[i]
      const phase = phases[lamp.signalIndex]
      const walk = phase === undefined ? false : pedestrianWalkable(phase)
      const lit =
        phase === undefined
          ? false
          : walk
            ? lamp.role === PED_ROLE_GREEN
            : lamp.role === PED_ROLE_RED
      // 金沢は灯火が 10,944 個ある。変わっていないものまで塗り直さない
      const state = lit ? LIT_ON : LIT_OFF
      if (seen[i] === state) continue
      seen[i] = state
      dirty = true
      mesh.setColorAt(i, lit ? (walk ? GREEN : RED) : colorOff)
    }
    if (dirty && mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  })

  if (signals.length === 0) return null

  return (
    <group>
      <instancedMesh
        ref={poleRef}
        args={[resources.pole, resources.structureMaterial, Math.max(1, placements.length)]}
        count={0}
        castShadow={castShadow}
        frustumCulled={false}
      />
      <instancedMesh
        ref={housingRef}
        args={[resources.housing, resources.structureMaterial, Math.max(1, placements.length)]}
        count={0}
        castShadow={castShadow}
        frustumCulled={false}
      />
      <instancedMesh
        ref={lampMeshRef}
        args={[resources.lamp, resources.lampMaterial, Math.max(1, lamps.length)]}
        count={0}
        frustumCulled={false}
      />
    </group>
  )
}
