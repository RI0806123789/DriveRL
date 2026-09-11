/**
 * 交通信号機（車両用・横型3灯式）の描画。
 *
 * 配置の計算そのものは `signalGeometry.ts` にある（React に依存させず、
 * `scripts/verify-signal-geometry.ts` から Node で検証できるようにするため）。
 * ここは three のオブジェクトを組み立てて現示に応じて塗り替える役だけを持つ。
 *
 * 日本の設置基準に合わせている点:
 *   - 灯火は**運転者から見て左から 青・黄・赤**（赤が右端）
 *   - 灯器は**交差点の対面側**に置き、進入車両に正対させる
 *   - 支柱は進行方向の左側（左側通行のため）、灯器下端は路面から 5.0m
 *   - 灯火径 300mm、各灯にフード（庇）
 *
 * 描画の組み立て方:
 *   支柱・アーム・筐体・フードは変化しないので 1 メッシュへ統合する。
 *   灯火だけは色が変わるので InstancedMesh にし、現示が変わったときだけ
 *   インスタンスカラーを塗り替える（毎フレーム作り直さない）。
 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { frameBuffer } from '../store/frameBuffer'
import type { MapNode, MapSignal } from '../types/protocol'
import { SIGNAL_LAMP_COLORS } from './palette'
import { buildSignalPlacement, createLampGeometry } from './signalGeometry'
import { usePalette } from './usePalette'

/**
 * 添字は signalGeometry の ROLE_* と一致（0=青 / 1=黄 / 2=赤）。
 * 色そのものは `palette.SIGNAL_LAMP_COLORS` が持つ（認識結果の枠と同じ値を使うため）。
 */
const PHASE_COLORS = SIGNAL_LAMP_COLORS.map((c) => new THREE.Color(c))

export interface TrafficSignalsProps {
  signals: MapSignal[]
  nodes: MapNode[]
  castShadow: boolean
}

export function TrafficSignals({ signals, nodes, castShadow }: TrafficSignalsProps) {
  const palette = usePalette()
  // 消灯している灯火だけは背景に合わせる（夜は黒、昼は少し明るい灰）
  const colorOff = useMemo(() => new THREE.Color(palette.signalLampOff), [palette.signalLampOff])
  const { structure, lamps } = useMemo(() => {
    const { parts, lamps: slots } = buildSignalPlacement(signals, nodes)
    const merged = parts.length > 0 ? mergeGeometries(parts, false) : null
    for (const g of parts) g.dispose()
    return { structure: merged, lamps: slots }
  }, [signals, nodes])

  const lampMeshRef = useRef<THREE.InstancedMesh>(null)
  const lastVersion = useRef(-1)

  // 灯火は「現示が変わったフレームだけ」書き換えている。配色が変わっても
  // 現示は変わらないので、記録を捨てて次のフレームで必ず塗り直させる。
  // これが無いと、消灯している灯火だけ前の配色のまま残る。
  useEffect(() => {
    lastVersion.current = -1
  }, [colorOff])

  const structureMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.signalHousing,
        roughness: 0.72,
        metalness: 0.35,
        side: THREE.DoubleSide, // フードは開いた筒なので裏面も描く
      }),
    [palette.signalHousing],
  )

  // 灯火は自発光として扱いたいのでライティングを受けない Basic を使う。
  // vertexColors を立てることで InstancedMesh のインスタンスカラーが最終色に効く。
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

  // 灯火の位置・向きは一度決めたら変わらない
  useEffect(() => {
    const mesh = lampMeshRef.current
    if (!mesh || lamps.length === 0) return
    const dummy = new THREE.Object3D()
    for (let i = 0; i < lamps.length; i++) {
      const lamp = lamps[i]
      dummy.position.copy(lamp.position)
      // 円板の法線は +X に揃えてあるので、方位をそのまま入れれば正面が向く
      dummy.rotation.set(0, lamp.facing, 0)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
      mesh.setColorAt(i, colorOff)
    }
    mesh.count = lamps.length
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    mesh.computeBoundingSphere()
    lastVersion.current = -1 // 次の useFrame で必ず塗り直させる
  }, [lamps])

  // 現示が変わったときだけ塗り替える（信号は数秒に一度しか変わらない）
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
