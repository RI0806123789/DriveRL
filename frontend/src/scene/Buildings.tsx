/**
 * 建物メッシュ。
 *
 * 各建物の outline から THREE.Shape を作り、ExtrudeGeometry で height 分押し出す。
 * 建物は数千件になるため、必ず mergeGeometries でまとめること（統合しないと
 * ドローコールが爆発してフレームレートが落ちる）。
 *
 * ExtrudeGeometry は +Z 方向に押し出すので、rotateX(-PI/2) して
 *   (shape.x, shape.y, extrude.z) -> (x, height, -y)
 * すなわち protocol.md 1.3 の変換（three.x = enu.x, three.z = -enu.y）に一致させる。
 *
 * 高さに応じて頂点カラーを変え、街の起伏が分かるようにしている。
 */

import { useEffect, useMemo } from 'react'
import { usePalette } from './usePalette'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { MapBuilding } from '../types/protocol'

/** 1 メッシュにまとめる建物数。大きすぎると 1 回のマージが重くなる */
const CHUNK_SIZE = 400

/**
 * 低層 → 高層のグラデーション。
 *
 * 色は**頂点色として焼き込む**（建物ごとに高さが違うのでマテリアルでは表せない）。
 * そのため配色が変わったらジオメトリを作り直すことになる。
 * 切り替わるのは日の出と日の入りの 1 日 2 回だけなので、
 * マップ読み込みと同じ程度の一瞬の負荷は受け入れる。
 */
function buildingColor(
  height: number,
  low: THREE.Color,
  high: THREE.Color,
  target: THREE.Color,
): THREE.Color {
  const t = Math.min(1, Math.max(0, (height - 6) / 60))
  return target.copy(low).lerp(high, t)
}

function buildChunks(
  buildings: MapBuilding[],
  lowCss: string,
  highCss: string,
): THREE.BufferGeometry[] {
  const low = new THREE.Color(lowCss)
  const high = new THREE.Color(highCss)
  const chunks: THREE.BufferGeometry[] = []
  let pending: THREE.BufferGeometry[] = []
  const tmpColor = new THREE.Color()

  const flush = () => {
    if (pending.length === 0) return
    const merged = mergeGeometries(pending, false)
    for (const g of pending) g.dispose()
    pending = []
    if (merged) {
      merged.computeBoundingSphere()
      chunks.push(merged)
    }
  }

  for (const b of buildings) {
    const outline = b.outline
    if (!outline || outline.length < 3) continue

    // 最後の点が始点と重なっていたら落とす（Shape は自動で閉じる）
    let count = outline.length
    const first = outline[0]
    const last = outline[count - 1]
    if (Math.abs(first[0] - last[0]) < 1e-6 && Math.abs(first[1] - last[1]) < 1e-6) count -= 1
    if (count < 3) continue

    const shape = new THREE.Shape()
    shape.moveTo(outline[0][0], outline[0][1])
    for (let i = 1; i < count; i++) shape.lineTo(outline[i][0], outline[i][1])
    shape.closePath()

    const height = Number.isFinite(b.height) && b.height > 0 ? b.height : 9

    let geom: THREE.ExtrudeGeometry
    try {
      geom = new THREE.ExtrudeGeometry(shape, {
        depth: height,
        bevelEnabled: false,
        curveSegments: 1,
      })
    } catch {
      // 自己交差など三角形分割に失敗する形状はスキップする
      continue
    }

    // 押し出し方向（+Z）を上（+Y）へ向ける
    geom.rotateX(-Math.PI / 2)

    const posAttr = geom.getAttribute('position')
    if (!posAttr) {
      geom.dispose()
      continue
    }
    const vertexCount = posAttr.count
    const colors = new Float32Array(vertexCount * 3)
    const c = buildingColor(height, low, high, tmpColor)
    for (let i = 0; i < vertexCount; i++) {
      // 上に行くほどわずかに明るく（面の区別がつきやすい）
      const yv = posAttr.getY(i)
      const k = 0.82 + 0.28 * Math.min(1, yv / Math.max(1, height))
      colors[i * 3 + 0] = c.r * k
      colors[i * 3 + 1] = c.g * k
      colors[i * 3 + 2] = c.b * k
    }
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    pending.push(geom)
    if (pending.length >= CHUNK_SIZE) flush()
  }

  flush()
  return chunks
}

export interface BuildingsProps {
  buildings: MapBuilding[]
  castShadow: boolean
  receiveShadow: boolean
}

export function Buildings({ buildings, castShadow, receiveShadow }: BuildingsProps) {
  const palette = usePalette()
  const chunks = useMemo(
    () => buildChunks(buildings, palette.buildingLow, palette.buildingHigh),
    [buildings, palette.buildingLow, palette.buildingHigh],
  )

  useEffect(() => {
    return () => {
      for (const g of chunks) g.dispose()
    }
  }, [chunks])

  if (chunks.length === 0) return null

  return (
    <group>
      {chunks.map((geom, i) => (
        <mesh key={i} geometry={geom} castShadow={castShadow} receiveShadow={receiveShadow}>
          <meshStandardMaterial vertexColors roughness={0.86} metalness={0.06} />
        </mesh>
      ))}
    </group>
  )
}
