/** 最高速度標識（規制標識 323「最高速度」）の描画。 */

import { memo, useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useSimStore } from '../store/simStore'
import type { MapSign } from '../types/protocol'
import {
  createSignBoardGeometry,
  createSignFaceGeometry,
  createSignPoleGeometry,
  signBoardCenter,
  signBoardFacing,
  signPoleCenter,
  signSpeedKph,
} from './signGeometry'
import { usePalette } from './usePalette'

/** 数字を焼くキャンバスの一辺 [px]。標示板は画面上で小さいのでこれで足りる */
const TEXTURE_SIZE = 256

/** 赤縁の太さ。標示板の直径に対する比（60cm の標識で 6cm） */
const RING_WIDTH_RATIO = 0.1

/** 標識が 1 本も無いとき用の空配列。 */
const EMPTY_SIGNS: MapSign[] = []

/** 同じ規制速度の標識をまとめたもの。1 グループ = 1 テクスチャ = 1 InstancedMesh */
interface SignGroup {
  kph: number
  members: MapSign[]
}

/** 標示板の絵（白地・赤縁・黒数字）を描いてテクスチャにする。 */
function createSignTexture(
  kph: number,
  boardColor: string,
  ringColor: string,
  textColor: string,
): THREE.CanvasTexture {
  const size = TEXTURE_SIZE
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size

  const ctx = canvas.getContext('2d')
  if (ctx) {
    const half = size / 2
    const ringWidth = size * RING_WIDTH_RATIO

    ctx.fillStyle = boardColor
    ctx.fillRect(0, 0, size, size)

    ctx.strokeStyle = ringColor
    ctx.lineWidth = ringWidth
    ctx.beginPath()
    ctx.arc(half, half, half - ringWidth / 2, 0, Math.PI * 2)
    ctx.stroke()

    const label = String(kph)
    const inner = (half - ringWidth) * 2 * 0.86
    const font = (px: number) => `bold ${px.toFixed(1)}px "Helvetica Neue", Arial, sans-serif`
    let fontSize = size * 0.56
    ctx.font = font(fontSize)
    while (ctx.measureText(label).width > inner && fontSize > size * 0.2) {
      fontSize *= 0.9
      ctx.font = font(fontSize)
    }
    ctx.fillStyle = textColor
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, half, half + fontSize * 0.04)
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 4
  texture.needsUpdate = true
  return texture
}

export function SpeedSigns() {
  const palette = usePalette()
  const signs = useSimStore((s) => s.map?.signs ?? EMPTY_SIGNS)
  const showSigns = useSimStore((s) => s.view.showSigns)
  const castShadow = useSimStore((s) => s.view.shadows)

  const placed = showSigns ? signs : EMPTY_SIGNS

  const groups = useMemo<SignGroup[]>(() => {
    if (placed.length === 0) return []
    const byKph = new Map<number, MapSign[]>()
    for (const sign of placed) {
      const kph = signSpeedKph(sign)
      const members = byKph.get(kph)
      if (members) members.push(sign)
      else byKph.set(kph, [sign])
    }
    return Array.from(byKph, ([kph, members]) => ({ kph, members })).sort(
      (a, b) => a.kph - b.kph,
    )
  }, [placed])

  const poleGeometry = useMemo(() => createSignPoleGeometry(), [])
  const boardGeometry = useMemo(() => createSignBoardGeometry(), [])
  const faceGeometry = useMemo(() => createSignFaceGeometry(), [])

  const poleMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.signPole,
        roughness: 0.68,
        metalness: 0.4,
      }),
    [],
  )
  const boardMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.signBoard,
        roughness: 0.55,
        metalness: 0.05,
      }),
    [],
  )

  useEffect(() => {
    poleMaterial.color.set(palette.signPole)
    boardMaterial.color.set(palette.signBoard)
  }, [poleMaterial, boardMaterial, palette.signPole, palette.signBoard])

  const faceTextures = useMemo(
    () =>
      groups.map((g) =>
        createSignTexture(g.kph, palette.signBoard, palette.signRing, palette.signText),
      ),
    [groups, palette.signBoard, palette.signRing, palette.signText],
  )

  const faceMaterials = useMemo(
    () =>
      faceTextures.map(
        (map) =>
          new THREE.MeshBasicMaterial({
            map,
            toneMapped: false,
            polygonOffset: true,
            polygonOffsetFactor: -1,
            polygonOffsetUnits: -2,
          }),
      ),
    [faceTextures],
  )

  const poleRef = useRef<THREE.InstancedMesh>(null)
  const boardRef = useRef<THREE.InstancedMesh>(null)

  useEffect(() => {
    return () => {
      poleGeometry.dispose()
      boardGeometry.dispose()
      faceGeometry.dispose()
    }
  }, [poleGeometry, boardGeometry, faceGeometry])

  useEffect(() => {
    return () => {
      poleMaterial.dispose()
      boardMaterial.dispose()
    }
  }, [poleMaterial, boardMaterial])

  useEffect(() => {
    return () => {
      for (const m of faceMaterials) m.dispose()
    }
  }, [faceMaterials])

  useEffect(() => {
    return () => {
      for (const t of faceTextures) t.dispose()
    }
  }, [faceTextures])

  useEffect(() => {
    const poles = poleRef.current
    const boards = boardRef.current
    if (!poles || !boards || placed.length === 0) return

    const dummy = new THREE.Object3D()
    for (let i = 0; i < placed.length; i++) {
      const sign = placed[i]

      const [px, py, pz] = signPoleCenter(sign)
      dummy.position.set(px, py, pz)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      poles.setMatrixAt(i, dummy.matrix)

      const [bx, by, bz] = signBoardCenter(sign)
      dummy.position.set(bx, by, bz)
      dummy.rotation.set(0, signBoardFacing(sign), 0)
      dummy.updateMatrix()
      boards.setMatrixAt(i, dummy.matrix)
    }

    poles.count = placed.length
    boards.count = placed.length
    poles.instanceMatrix.needsUpdate = true
    boards.instanceMatrix.needsUpdate = true
    poles.computeBoundingSphere()
    boards.computeBoundingSphere()
  }, [placed, poleGeometry, poleMaterial, boardGeometry, boardMaterial])

  if (placed.length === 0) return null

  return (
    <group>
      <instancedMesh
        ref={poleRef}
        args={[poleGeometry, poleMaterial, placed.length]}
        count={0}
        castShadow={castShadow}
        frustumCulled={false}
      />
      <instancedMesh
        ref={boardRef}
        args={[boardGeometry, boardMaterial, placed.length]}
        count={0}
        castShadow={castShadow}
        frustumCulled={false}
      />
      {groups.map((group, i) => (
        <SignFaces
          key={`sign-face-${group.kph}`}
          members={group.members}
          geometry={faceGeometry}
          material={faceMaterials[i]}
        />
      ))}
    </group>
  )
}

interface SignFacesProps {
  members: MapSign[]
  geometry: THREE.BufferGeometry
  material: THREE.Material
}

const SignFaces = memo(function SignFaces({ members, geometry, material }: SignFacesProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null)

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh || members.length === 0) return
    const dummy = new THREE.Object3D()
    for (let i = 0; i < members.length; i++) {
      const sign = members[i]
      const [x, y, z] = signBoardCenter(sign)
      dummy.position.set(x, y, z)
      dummy.rotation.set(0, signBoardFacing(sign), 0)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    }
    mesh.count = members.length
    mesh.instanceMatrix.needsUpdate = true
    mesh.computeBoundingSphere()
  }, [members, geometry, material])

  if (members.length === 0) return null

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, members.length]}
      count={0}
      frustumCulled={false}
    />
  )
})
