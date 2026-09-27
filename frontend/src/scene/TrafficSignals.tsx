/** 交通信号機（車両用・横型3灯式）の描画。 */

import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
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
  /** 灯火ごとに直前に塗った点灯状態（0=消灯 / 1=点灯 / 255=未塗り）。変わった灯火だけ塗り替える */
  const lastLit = useMemo(() => new Uint8Array(lamps.length).fill(255), [lamps])

  useEffect(() => {
    lastVersion.current = -1
    lastLit.fill(255)
  }, [colorOff, lastLit])

  // マテリアルは作り直さず色だけ差し替える（作り直すと使用中のものまで破棄することになる）
  const structureMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        roughness: 0.72,
        metalness: 0.35,
        side: THREE.DoubleSide,
      }),
    [],
  )
  useLayoutEffect(() => {
    structureMaterial.color.set(palette.signalHousing)
  }, [structureMaterial, palette.signalHousing])

  const lampMaterial = useMemo(
    () => new THREE.MeshBasicMaterial({ toneMapped: false, vertexColors: true }),
    [],
  )

  const lampGeometry = useMemo(() => createLampGeometry(), [])

  // 後始末は資源ごとに分ける。まとめると、どれか 1 つが変わっただけで使用中の残りまで破棄する
  useEffect(() => () => structure?.dispose(), [structure])
  useEffect(
    () => () => {
      structureMaterial.dispose()
      lampMaterial.dispose()
      lampGeometry.dispose()
    },
    [structureMaterial, lampMaterial, lampGeometry],
  )

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
    lastLit.fill(255)
  }, [lamps, lastLit])

  useFrame(() => {
    const mesh = lampMeshRef.current
    if (!mesh || lamps.length === 0) return
    if (frameBuffer.signalVersion === lastVersion.current) return
    lastVersion.current = frameBuffer.signalVersion

    // 変わった灯火だけ塗り替える（金沢は 2,528 基 × 3 灯あり、どこかの灯器はほぼ毎回変わる）
    const phases = frameBuffer.signals
    let changed = false
    for (let i = 0; i < lamps.length; i++) {
      const lamp = lamps[i]
      const phase = phases[lamp.signalIndex]
      const lit = phase !== undefined && phase === lamp.role ? 1 : 0
      if (lastLit[i] === lit) continue
      lastLit[i] = lit
      mesh.setColorAt(i, lit ? PHASE_COLORS[lamp.role] : colorOff)
      changed = true
    }
    if (changed && mesh.instanceColor) mesh.instanceColor.needsUpdate = true
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
