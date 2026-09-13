/**
 * 最高速度標識（規制標識 323「最高速度」）の描画。
 *
 * 配置の計算そのものは `signGeometry.ts` にある（React に依存させず、
 * `scripts/verify-sign-geometry.ts` から Node で検証できるようにするため）。
 * ここは three のオブジェクトを組み立てる役だけを持つ。
 *
 * 日本の設置基準に合わせている点:
 *   - 標示板は直径 60cm の円板。**白地に赤縁、黒で規制速度（km/h）**
 *   - 標示板の下端は路面から 1.8m。支柱はそこから上端まで伸ばす
 *   - 標示板は進入車両に正対する（`heading + PI` を向く）
 *   - 支柱は進行方向の左側（左側通行）。左へ寄せるのはバックエンドの仕事で、
 *     `sign.x` / `sign.y` は既に寄せた後の支柱位置が入っている
 *
 * 描画の組み立て方:
 *   支柱と標示板は形が同じなので InstancedMesh 1 組ずつで描く（1 本 1 メッシュにしない）。
 *   数字だけは規制速度ごとにテクスチャが違い、three は instance ごとにテクスチャを
 *   持てないので、**規制速度の値ごとに InstancedMesh を 1 つ**作る。
 *   銀座では 4 種類しか出ないので、増えるメッシュは 4 つで済む。
 *
 * 標識は読み込んだマップに紐づく静的なものなので、`useFrame` は一切使わない。
 */

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

/**
 * 標識が 1 本も無いとき用の空配列。
 *
 * `map.signs` は任意プロパティ（標識に対応していない古いサーバーでは省略される）なので、
 * セレクタの中で必ず畳む。毎回 `[]` を書くと参照が変わって zustand が
 * 「変わった」と判断してしまうため、定数を 1 つだけ持つ。
 */
const EMPTY_SIGNS: MapSign[] = []

/** 同じ規制速度の標識をまとめたもの。1 グループ = 1 テクスチャ = 1 InstancedMesh */
interface SignGroup {
  kph: number
  members: MapSign[]
}

// ---------------------------------------------------------------------------
// 標示板のテクスチャ（Canvas 2D）
// ---------------------------------------------------------------------------

/**
 * 標示板の絵（白地・赤縁・黒数字）を描いてテクスチャにする。
 *
 * 呼び出し側が km/h ごとに 1 枚しか作らないので、**同じ規制速度は使い回される**。
 * 背景まで白で塗り潰しているのは、円の外側を透明にすると縁の 1px が
 * 黒く滲むため（円板ジオメトリの縁とテクスチャの円の縁がちょうど重なる）。
 */
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

    // 白地
    ctx.fillStyle = boardColor
    ctx.fillRect(0, 0, size, size)

    // 赤縁。外周がちょうど円板の縁に来るよう、線の中心を半径 - 太さ/2 に置く
    ctx.strokeStyle = ringColor
    ctx.lineWidth = ringWidth
    ctx.beginPath()
    ctx.arc(half, half, half - ringWidth / 2, 0, Math.PI * 2)
    ctx.stroke()

    // 黒数字。3 桁（100 km/h）でも赤縁に掛からないところまで縮める
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

// ---------------------------------------------------------------------------

export function SpeedSigns() {
  const palette = usePalette()
  const signs = useSimStore((s) => s.map?.signs ?? EMPTY_SIGNS)
  const showSigns = useSimStore((s) => s.view.showSigns)
  const castShadow = useSimStore((s) => s.view.shadows)

  // OFF の間は**そもそも作らない**（RouteLines が経路線 OFF でしているのと同じ）
  const placed = showSigns ? signs : EMPTY_SIGNS

  // 規制速度ごとにまとめる。銀座で 4 種類しか出ない
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

  // ★ **配色はマテリアルを作り直す理由にしない**（code_review S-01）。
  //   R3F は <instancedMesh args={[...]}> の要素を 1 つずつ比較し、違っていれば
  //   InstancedMesh ごと作り直す（Ground.tsx の F-27 の注記と同じ挙動）。
  //   作り直された直後は count={0} が再適用されるのに、行列を書く useEffect は
  //   deps が変わらないので走らない。結果、**日の出・日の入りで配色が切り替わった
  //   瞬間に支柱が消え、標示板だけが宙に浮いて残る**。配色は autoTheme が自動で
  //   切り替えるので、利用者は何もしていないのに消える。
  //   Vehicles.tsx と同じく「作り直さず色だけ差し替える」形にしてある。
  const poleMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.signPole,
        roughness: 0.68,
        metalness: 0.4,
      }),
    [], // ここで使う palette は初期色だけ。以降は下の useEffect が書き換える
  )
  // ★ 標示板の白地・赤縁・黒数字は昼夜で変えない。灯火と同じ理由で、
  //   現実の標識と違う色になっては意味を成さないため。
  //   それでも deps は空にしておく（配色の値が将来変わっても支柱と同じ事故を起こさない）。
  const boardMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: palette.signBoard,
        roughness: 0.55,
        metalness: 0.05,
      }),
    [], // 同上
  )

  useEffect(() => {
    poleMaterial.color.set(palette.signPole)
    boardMaterial.color.set(palette.signBoard)
  }, [poleMaterial, boardMaterial, palette.signPole, palette.signBoard])

  // テクスチャは規制速度ごとに 1 枚。配色（signBoard / signRing / signText）は
  // 昼夜同値なので実際には作り直されない。作り直された場合も SignFaces 側の
  // deps に material が入っているので行列は張り直される（S-01）
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
            // 標識は再帰反射材なので、照明で暗くしない（灯火と同じ扱い）
            toneMapped: false,
            // 板の前面から 2mm しか浮いていない。距離が離れると深度の分解能を
            // 下回るので、高さではなく polygonOffset で手前へ押す
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

  // 支柱と標示板の位置・向きは一度決めたら変わらない。
  // ★ deps には `args` に渡しているものを**すべて**入れる（code_review S-01）。
  //   `args` の要素が 1 つでも変われば R3F は InstancedMesh を作り直し、
  //   count は 0 に戻る。書き直す側がそれを見ていないと標識が消える。
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
      // 円板の軸は +X に揃えてあるので、方位をそのまま入れれば正面が向く
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

// ---------------------------------------------------------------------------
// 数字の面（規制速度 1 種類ぶん）
// ---------------------------------------------------------------------------

interface SignFacesProps {
  members: MapSign[]
  geometry: THREE.BufferGeometry
  material: THREE.Material
}

const SignFaces = memo(function SignFaces({ members, geometry, material }: SignFacesProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null)

  // deps は `args` に渡しているものと同じ（S-01）。テクスチャを作り直すと
  // material の参照が変わり、InstancedMesh も作り直されるため。
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
