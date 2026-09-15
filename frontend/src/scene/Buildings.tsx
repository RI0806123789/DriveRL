/** 建物メッシュ。 */

import { useEffect, useMemo } from 'react'
import { usePalette } from './usePalette'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { MapBuilding } from '../types/protocol'

/** 1 メッシュにまとめる建物数。大きすぎると 1 回のマージが重くなる */
const CHUNK_SIZE = 400

/** 低層 → 高層のグラデーション。 */
function buildingColor(
  height: number,
  low: THREE.Color,
  high: THREE.Color,
  target: THREE.Color,
): THREE.Color {
  const t = Math.min(1, Math.max(0, (height - 6) / 60))
  return target.copy(low).lerp(high, t)
}

let warnedSkipped = false

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
  let skipped = 0

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
      skipped += 1
      continue
    }

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
  // 形を作れなかった建物は描かれないが、衝突判定はサーバー側の真値で行われる。
  // 「何も無いところで止まった」に見えるので 1 度だけ知らせる（code_review E-11）
  if (skipped > 0 && !warnedSkipped) {
    warnedSkipped = true
    console.warn(
      `[Buildings] ${skipped} 件の建物を描けませんでした（衝突判定はサーバー側で有効なままです）`,
    )
  }
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
