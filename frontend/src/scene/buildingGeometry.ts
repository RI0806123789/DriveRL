/** 建物の形と色。**React から切り離した純粋モジュール。** 形は一度だけ作り、色は配色が変わるたびに塗り直す。 */

import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { MapBuilding } from '../types/protocol.ts'

/** 1 メッシュにまとめる建物数。大きすぎると 1 回のマージが重くなる */
const CHUNK_SIZE = 400

/** 低層 → 高層のグラデーションの割合 0..1。 */
function heightShare(height: number): number {
  return Math.min(1, Math.max(0, (height - 6) / 60))
}

let warnedSkipped = false

/** まとめたジオメトリごとの、頂点の (低層 → 高層の割合, 高さによる陰り) */
const SHADES = new WeakMap<THREE.BufferGeometry, Float32Array>()

/** 形だけを作る。色は `paintBuildingChunks` が塗る（昼夜で配色が変わっても形は作り直さない） */
export function buildBuildingChunks(buildings: readonly MapBuilding[]): THREE.BufferGeometry[] {
  const chunks: THREE.BufferGeometry[] = []
  let pending: THREE.BufferGeometry[] = []
  let skipped = 0

  const flush = () => {
    if (pending.length === 0) return
    const merged = mergeGeometries(pending, false)
    for (const g of pending) g.dispose()
    pending = []
    if (merged) {
      merged.computeBoundingSphere()
      const vertices = merged.getAttribute('position').count
      merged.setAttribute('color', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3))
      // 塗るときにだけ使うので GPU へは送らない（属性に残すと転送の対象になる）
      SHADES.set(merged, merged.getAttribute('shade').array as Float32Array)
      merged.deleteAttribute('shade')
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
    // 頂点ごとに (低層 → 高層の割合, 高さによる陰り) を持つ。配色に依らない
    const vertexCount = posAttr.count
    const shade = new Float32Array(vertexCount * 2)
    const share = heightShare(height)
    for (let i = 0; i < vertexCount; i++) {
      const yv = posAttr.getY(i)
      shade[i * 2 + 0] = share
      shade[i * 2 + 1] = 0.82 + 0.28 * Math.min(1, yv / Math.max(1, height))
    }
    geom.setAttribute('shade', new THREE.BufferAttribute(shade, 2))

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

/** 色の属性だけを塗り直す。形（押し出し・マージ）はやり直さない */
export function paintBuildingChunks(
  chunks: readonly THREE.BufferGeometry[],
  lowCss: string,
  highCss: string,
): void {
  const low = new THREE.Color(lowCss)
  const high = new THREE.Color(highCss)
  const dr = high.r - low.r
  const dg = high.g - low.g
  const db = high.b - low.b
  for (const geom of chunks) {
    const s = SHADES.get(geom)
    if (!s) continue
    const color = geom.getAttribute('color') as THREE.BufferAttribute
    const c = color.array as Float32Array
    for (let i = 0, n = s.length / 2; i < n; i++) {
      const t = s[i * 2]
      const k = s[i * 2 + 1]
      c[i * 3 + 0] = (low.r + dr * t) * k
      c[i * 3 + 1] = (low.g + dg * t) * k
      c[i * 3 + 2] = (low.b + db * t) * k
    }
    color.needsUpdate = true
  }
}
