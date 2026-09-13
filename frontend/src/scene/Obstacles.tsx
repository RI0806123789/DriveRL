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

  // ★ **配色はマテリアルを作り直す理由にしない**（code_review S-01）。
  //   R3F は <instancedMesh args={[...]}> を要素ごとに比較し、違っていれば
  //   InstancedMesh ごと作り直す。作り直された直後は count={0} なのに、
  //   下の useFrame は `obstacleVersion` が変わらない限り早期 return するので
  //   行列も個数も書き直されない。結果、**日の出・日の入りで配色が切り替わると
  //   パイロンが画面から消え、マップを読み直すか障害物を足す／消すまで戻らない**。
  //   配色は autoTheme が自動で切り替えるので、利用者は何もしていないのに消える。
  const coneMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.obstacleCone,
        roughness: 0.55,
        metalness: 0.05,
        emissive: new THREE.Color(palette.obstacleCone).multiplyScalar(0.12),
      }),
    [], // ここで使う palette は初期色だけ。以降は下の useEffect が書き換える
  )
  const baseMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.obstacleBase,
        roughness: 0.9,
        metalness: 0,
      }),
    [], // 同上
  )

  // 配色が変わったら色だけ書き換える。自己発光はパイロンの色から作る
  // （夜に暗い地面へ沈まないようにするための下駄。倍率は生成時と揃える）
  useEffect(() => {
    coneMaterial.color.set(palette.obstacleCone)
    coneMaterial.emissive.set(palette.obstacleCone).multiplyScalar(0.12)
    baseMaterial.color.set(palette.obstacleBase)
  }, [coneMaterial, baseMaterial, palette.obstacleCone, palette.obstacleBase])

  useEffect(() => {
    return () => {
      coneMaterial.dispose()
      baseMaterial.dispose()
    }
  }, [coneMaterial, baseMaterial])

  // マップ切替などでバッファが空になったら、次回必ず作り直させる。
  // ★ deps には `args` に渡しているものを入れる（S-01）。マテリアルを作り直せば
  //   InstancedMesh も作り直されるので、そのときは行列を書き直す必要がある。
  useEffect(() => {
    lastVersion.current = -1
  }, [coneMaterial, baseMaterial])

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
