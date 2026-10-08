import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { MapSign, SignKind } from '../types/protocol'
import { createSignPoleGeometry, signBoardCenter, signBoardFacing, signPoleCenter } from './signGeometry'
import { createTrafficSignGeometry, createTrafficSignTexture, trafficSignKey } from './trafficSignVisual'
import { usePalette } from './usePalette'

export function TrafficSigns({ signs, castShadow }: { signs: MapSign[]; castShadow: boolean }) {
  const groups = useMemo(() => {
    const result = new Map<string, MapSign[]>()
    for (const sign of signs) {
      if (!sign.kind || sign.kind === 'speed_limit') continue
      const key = trafficSignKey(sign)
      const members = result.get(key)
      if (members) members.push(sign)
      else result.set(key, [sign])
    }
    return [...result.entries()]
  }, [signs])
  return <group>{groups.map(([key, members]) => <SignInstances key={key} members={members} castShadow={castShadow} />)}</group>
}

function SignInstances({ members, castShadow }: { members: MapSign[]; castShadow: boolean }) {
  const palette = usePalette()
  const pole = useRef<THREE.InstancedMesh>(null)
  const board = useRef<THREE.InstancedMesh>(null)
  const face = useRef<THREE.InstancedMesh>(null)
  const kind = members[0].kind as SignKind
  const direction = members[0].direction ?? 'straight'
  const resources = useMemo(() => {
    const texture = createTrafficSignTexture(kind, direction)
    return {
      texture,
      poleGeometry: createSignPoleGeometry(),
      boardGeometry: createTrafficSignGeometry(kind, false),
      faceGeometry: createTrafficSignGeometry(kind, true),
      poleMaterial: new THREE.MeshStandardMaterial({ color: palette.signPole, roughness: 0.68, metalness: 0.4 }),
      boardMaterial: new THREE.MeshStandardMaterial({ color: palette.signBoard, roughness: 0.55 }),
      faceMaterial: new THREE.MeshBasicMaterial({ map: texture, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 }),
    }
  }, [kind, direction, palette.signPole, palette.signBoard])
  useEffect(() => () => {
    for (const resource of Object.values(resources)) resource.dispose()
  }, [resources])
  useEffect(() => {
    if (!pole.current || !board.current || !face.current) return
    const dummy = new THREE.Object3D()
    members.forEach((sign, i) => {
      dummy.position.set(...signPoleCenter(sign))
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      pole.current!.setMatrixAt(i, dummy.matrix)
      dummy.position.set(...signBoardCenter(sign))
      dummy.rotation.set(0, signBoardFacing(sign), 0)
      dummy.updateMatrix()
      board.current!.setMatrixAt(i, dummy.matrix)
      face.current!.setMatrixAt(i, dummy.matrix)
    })
    for (const ref of [pole, board, face]) {
      ref.current!.count = members.length
      ref.current!.instanceMatrix.needsUpdate = true
      ref.current!.computeBoundingSphere()
    }
  }, [members, resources])
  return <group>
    <instancedMesh ref={pole} args={[resources.poleGeometry, resources.poleMaterial, members.length]} count={0} castShadow={castShadow} />
    <instancedMesh ref={board} args={[resources.boardGeometry, resources.boardMaterial, members.length]} count={0} castShadow={castShadow} />
    <instancedMesh ref={face} args={[resources.faceGeometry, resources.faceMaterial, members.length]} count={0} />
  </group>
}
