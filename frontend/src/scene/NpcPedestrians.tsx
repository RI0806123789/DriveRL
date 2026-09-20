/** 街を歩く NPC 歩行者（protocol.md 2.3 の `frame.pedestrians`）。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import type { NpcPedestrianState } from '../types/protocol'
import { computeAlpha, createPedestrianPose, samplePedestrian } from './interpolation'
import {
  LIMB_JOINTS,
  composeLimbMatrix,
  composePedestrianMatrix,
  createPedestrianScratch,
  limbSwing,
  makeArmGeometry,
  makeHeadGeometry,
  makeHipGeometry,
  makeLegGeometry,
  makeTorsoGeometry,
  npcHueOffset,
  swingFor,
} from './pedestrianGeometry'
import { usePalette } from './usePalette'

/** バックエンドの `config.MAX_PEDESTRIANS` と揃える */
const MAX_PEDESTRIANS = 64

/** 手足の本数（腕 2 本・脚 2 本） */
const LIMBS_PER_SIDE = 2

/** 服の色を散らす幅。色相をこれだけの範囲で回す */
const HUE_SPREAD = 0.42

/** 明度の散らし幅。同じ色相でも濃淡で差が出る */
const LIGHT_SPREAD = 0.16

/** 1 人ぶんの色を `palette` の色から作る。色相と明度だけ散らす */
function shade(base: THREE.Color, id: number, out: THREE.Color): THREE.Color {
  const hsl = { h: 0, s: 0, l: 0 }
  base.getHSL(hsl)
  const t = npcHueOffset(id)
  return out.setHSL(
    (hsl.h + (t - 0.5) * HUE_SPREAD + 1) % 1,
    Math.min(1, Math.max(0, hsl.s * (0.7 + t * 0.6))),
    Math.min(0.92, Math.max(0.08, hsl.l + (t - 0.5) * LIGHT_SPREAD)),
  )
}

export interface NpcPedestriansProps {
  castShadow: boolean
}

export function NpcPedestrians({ castShadow }: NpcPedestriansProps) {
  const palette = usePalette()
  const renderPaused = useSimStore((s) => s.status.renderPaused)

  const headRef = useRef<THREE.InstancedMesh>(null)
  const torsoRef = useRef<THREE.InstancedMesh>(null)
  const hipRef = useRef<THREE.InstancedMesh>(null)
  const armRef = useRef<THREE.InstancedMesh>(null)
  const legRef = useRef<THREE.InstancedMesh>(null)

  const scratch = useMemo(
    () => ({
      pose: createPedestrianPose(),
      transform: createPedestrianScratch(),
      base: new THREE.Matrix4(),
      out: new THREE.Matrix4(),
      color: new THREE.Color(),
      prev: new Map<number, NpcPedestrianState>(),
      // 前フレームの索引を作り直した時点。60fps で 20Hz の frame を読むので、
      // 変わっていないフレームで毎回作り直さない
      indexedAt: -1,
    }),
    [],
  )

  // ★ palette は `args` に渡さない（マテリアルごと作り直されると物体が消える。code_review S-01）
  const resources = useMemo(() => {
    const white = () =>
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.74, metalness: 0.05 })
    return {
      head: makeHeadGeometry(),
      torso: makeTorsoGeometry(),
      hip: makeHipGeometry(),
      arm: makeArmGeometry(),
      leg: makeLegGeometry(),
      // 個体差は instanceColor で付けるので、マテリアル側の色は白のまま
      skin: white(),
      top: white(),
      bottom: white(),
    }
  }, [])

  useEffect(() => {
    const r = resources
    return () => {
      r.head.dispose()
      r.torso.dispose()
      r.hip.dispose()
      r.arm.dispose()
      r.leg.dispose()
      r.skin.dispose()
      r.top.dispose()
      r.bottom.dispose()
    }
  }, [resources])

  // 昼夜で配色が変わるので、色は palette が変わったときだけ書き直す
  useEffect(() => {
    const head = headRef.current
    const torso = torsoRef.current
    const hip = hipRef.current
    const arm = armRef.current
    const leg = legRef.current
    if (!head || !torso || !hip || !arm || !leg) return

    const skin = new THREE.Color(palette.pedestrianSkin)
    const top = new THREE.Color(palette.pedestrianTop)
    const bottom = new THREE.Color(palette.pedestrianBottom)
    const c = scratch.color
    for (let i = 0; i < MAX_PEDESTRIANS; i++) {
      head.setColorAt(i, shade(skin, i, c))
      torso.setColorAt(i, shade(top, i, c))
      hip.setColorAt(i, shade(bottom, i, c))
      for (let k = 0; k < LIMBS_PER_SIDE; k++) {
        arm.setColorAt(i * LIMBS_PER_SIDE + k, shade(top, i, c))
        leg.setColorAt(i * LIMBS_PER_SIDE + k, shade(bottom, i, c))
      }
    }
    for (const mesh of [head, torso, hip, arm, leg]) {
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    }
  }, [palette, scratch])

  useFrame(() => {
    const head = headRef.current
    const torso = torsoRef.current
    const hip = hipRef.current
    const arm = armRef.current
    const leg = legRef.current
    if (!head || !torso || !hip || !arm || !leg) return

    const curr = frameBuffer.curr
    const people = curr?.pedestrians
    if (!people || people.length === 0) {
      head.count = 0
      torso.count = 0
      hip.count = 0
      arm.count = 0
      leg.count = 0
      return
    }

    const prev = scratch.prev
    const prevFrame = frameBuffer.prev?.pedestrians
    if (prevFrame && scratch.indexedAt !== frameBuffer.received) {
      scratch.indexedAt = frameBuffer.received
      prev.clear()
      for (const p of prevFrame) prev.set(p.id, p)
    }

    const alpha = computeAlpha(performance.now(), renderPaused)
    const n = Math.min(people.length, MAX_PEDESTRIANS)
    for (let i = 0; i < n; i++) {
      const p = people[i]
      samplePedestrian(p, prevFrame ? prev.get(p.id) : undefined, alpha, scratch.pose)
      const swing = limbSwing(scratch.pose.stride, 1)
      composePedestrianMatrix(
        scratch.transform,
        scratch.pose.x,
        scratch.pose.y,
        scratch.pose.heading,
        swing.bob,
        scratch.base,
      )
      head.setMatrixAt(i, scratch.base)
      torso.setMatrixAt(i, scratch.base)
      hip.setMatrixAt(i, scratch.base)
      for (let k = 0; k < LIMB_JOINTS.length; k++) {
        composeLimbMatrix(scratch.transform, scratch.base, k, swingFor(k, swing), scratch.out)
        const target = LIMB_JOINTS[k].kind === 'arm' ? arm : leg
        const slot = i * LIMBS_PER_SIDE + (k % LIMBS_PER_SIDE)
        target.setMatrixAt(slot, scratch.out)
      }
    }

    head.count = n
    torso.count = n
    hip.count = n
    arm.count = n * LIMBS_PER_SIDE
    leg.count = n * LIMBS_PER_SIDE
    for (const mesh of [head, torso, hip, arm, leg]) {
      mesh.instanceMatrix.needsUpdate = true
    }
  })

  return (
    <group>
      <instancedMesh
        ref={headRef}
        args={[resources.head, resources.skin, MAX_PEDESTRIANS]}
        count={0}
        frustumCulled={false}
        castShadow={castShadow}
      />
      <instancedMesh
        ref={torsoRef}
        args={[resources.torso, resources.top, MAX_PEDESTRIANS]}
        count={0}
        frustumCulled={false}
        castShadow={castShadow}
      />
      <instancedMesh
        ref={hipRef}
        args={[resources.hip, resources.bottom, MAX_PEDESTRIANS]}
        count={0}
        frustumCulled={false}
        castShadow={castShadow}
      />
      <instancedMesh
        ref={armRef}
        args={[resources.arm, resources.top, MAX_PEDESTRIANS * LIMBS_PER_SIDE]}
        count={0}
        frustumCulled={false}
        castShadow={castShadow}
      />
      <instancedMesh
        ref={legRef}
        args={[resources.leg, resources.bottom, MAX_PEDESTRIANS * LIMBS_PER_SIDE]}
        count={0}
        frustumCulled={false}
        castShadow={castShadow}
      />
    </group>
  )
}
