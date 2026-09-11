/**
 * ユーザーが設置したパイロン（memo F-05 インタラクティブ介入）。
 *
 * 障害物は数が少なく、変化も稀なので InstancedMesh 1 組で描く。
 * frameBuffer.obstacleVersion が変わったときだけ行列を作り直すので、
 * React の再レンダリングは一切発生しない。
 *
 * **クリックで 1 個だけ取り消せる**（code_review F-25）。
 * `remove_obstacle` は protocol.md に定義されているのに送信箇所が無く、
 * 置き間違えても「すべて消す」しか手が無かった。
 * InstancedMesh のクリックは event.instanceId で何番目かが分かるので、
 * それを frameBuffer.obstacles の並びに対応させて id を引く。
 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { send } from '../store/connection'
import { frameBuffer } from '../store/frameBuffer'
import { usePalette } from './usePalette'

/** バックエンドの config.MAX_OBSTACLES と揃える */
const MAX_OBSTACLES = 64

/** バックエンドの config.OBSTACLE_HEIGHT [m] */
const CONE_HEIGHT = 0.75

// 底面が y=0 に来るよう平行移動しておく（インスタンスの位置＝接地点にできる）
const coneGeom = new THREE.ConeGeometry(1, 1, 14)
coneGeom.translate(0, 0.5, 0)

const baseGeom = new THREE.CylinderGeometry(1, 1, 1, 14)
baseGeom.translate(0, 0.5, 0)

export interface ObstaclesProps {
  castShadow: boolean
}

export function Obstacles({ castShadow }: ObstaclesProps) {
  const palette = usePalette()
  const coneRef = useRef<THREE.InstancedMesh>(null)
  const baseRef = useRef<THREE.InstancedMesh>(null)
  const lastVersion = useRef(-1)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  const coneMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.obstacleCone,
        roughness: 0.55,
        metalness: 0.05,
        emissive: new THREE.Color(palette.obstacleCone).multiplyScalar(0.12),
      }),
    [palette.obstacleCone],
  )
  const baseMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.obstacleBase,
        roughness: 0.9,
        metalness: 0,
      }),
    [palette.obstacleBase],
  )

  useEffect(() => {
    return () => {
      coneMaterial.dispose()
      baseMaterial.dispose()
    }
  }, [coneMaterial, baseMaterial])

  // マップ切替などでバッファが空になったら、次回必ず作り直させる
  useEffect(() => {
    lastVersion.current = -1
  }, [])

  /**
   * パイロンをクリックしたらその 1 個だけ消す。
   *
   * 地面（InteractionPlane）へのクリックは「置く」なので、こちらは必ず
   * stopPropagation して伝播を止める。止めないと、消したその場所に
   * すぐ新しいパイロンが置かれてしまう。
   */
  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    const index = e.instanceId
    if (index === undefined) return
    const target = frameBuffer.obstacles[index]
    if (!target) return
    e.stopPropagation()
    send({ type: 'remove_obstacle', id: target.id })
  }

  useFrame(() => {
    const cones = coneRef.current
    const bases = baseRef.current
    if (!cones || !bases) return
    if (frameBuffer.obstacleVersion === lastVersion.current) return
    lastVersion.current = frameBuffer.obstacleVersion

    const list = frameBuffer.obstacles
    const n = Math.min(list.length, MAX_OBSTACLES)

    for (let i = 0; i < n; i++) {
      const o = list[i]
      const r = Math.max(0.15, o.radius)
      // ENU -> Three（protocol.md 1.3）: three.z = -enu.y
      dummy.position.set(o.x, 0, -o.y)

      dummy.scale.set(r, CONE_HEIGHT, r)
      dummy.updateMatrix()
      cones.setMatrixAt(i, dummy.matrix)

      // 台座は少し広く、薄く
      dummy.scale.set(r * 1.55, 0.06, r * 1.55)
      dummy.updateMatrix()
      bases.setMatrixAt(i, dummy.matrix)
    }

    cones.count = n
    bases.count = n
    cones.instanceMatrix.needsUpdate = true
    bases.instanceMatrix.needsUpdate = true
    // 個数が変わると古い境界球のままカリングされてしまうことがある
    cones.computeBoundingSphere()
    bases.computeBoundingSphere()
  })

  return (
    <group>
      <instancedMesh
        ref={baseRef}
        args={[baseGeom, baseMaterial, MAX_OBSTACLES]}
        count={0}
        frustumCulled={false}
        onClick={handleClick}
      />
      <instancedMesh
        ref={coneRef}
        args={[coneGeom, coneMaterial, MAX_OBSTACLES]}
        count={0}
        onClick={handleClick}
        castShadow={castShadow}
        frustumCulled={false}
      />
    </group>
  )
}
