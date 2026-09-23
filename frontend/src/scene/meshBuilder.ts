/** 三角形を積み上げて 1 つの BufferGeometry にする。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'

export type Vec3 = readonly [number, number, number]

/** 頂点に付ける属性。 */
export interface PartStyle {
  /** 頂点色（マテリアルの色に掛かる。既定は白） */
  readonly color?: Vec3
  /** ドアと一緒に回る部品か（`doorPart` 属性） */
  readonly door?: boolean
}

const WHITE: Vec3 = [1, 1, 1]

function sub(a: Vec3, b: Vec3): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

function cross(a: Vec3, b: Vec3): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function normalize(v: Vec3, fallback: Vec3): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2])
  if (l < 1e-12) return [fallback[0], fallback[1], fallback[2]]
  return [v[0] / l, v[1] / l, v[2] / l]
}

/** 外向きの目安。点（格子なら行と列も）を受け取ってその点での外向きを返すか、固定の向きを渡す */
export type Outward = Vec3 | ((p: Vec3, i: number, j: number) => Vec3)

function outwardAt(o: Outward, p: Vec3, i = -1, j = -1): Vec3 {
  return typeof o === 'function' ? o(p, i, j) : o
}

export class MeshBuilder {
  private readonly pos: number[] = []
  private readonly nrm: number[] = []
  private readonly col: number[] = []
  private readonly doorAttr: number[] = []
  private readonly uvs: number[] = []
  private hasUv = false

  get triangleCount(): number {
    return this.pos.length / 9
  }

  private vertex(p: Vec3, n: Vec3, style: PartStyle, uv?: readonly [number, number]): void {
    this.pos.push(p[0], p[1], p[2])
    this.nrm.push(n[0], n[1], n[2])
    const c = style.color ?? WHITE
    this.col.push(c[0], c[1], c[2])
    this.doorAttr.push(style.door ? 1 : 0)
    if (uv) {
      this.hasUv = true
      this.uvs.push(uv[0], uv[1])
    } else {
      this.uvs.push(0, 0)
    }
  }

  /** 三角形 1 枚。法線を省くと面の法線を使う。巻き順は `outward` に合わせて直す */
  triangle(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    style: PartStyle = {},
    normals?: readonly [Vec3, Vec3, Vec3],
    outward?: Outward,
  ): void {
    const face = cross(sub(b, a), sub(c, a))
    if (Math.hypot(face[0], face[1], face[2]) < 1e-14) return
    let flip = false
    if (outward) {
      const centre: Vec3 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3]
      flip = dot(face, outwardAt(outward, centre)) < 0
    }
    const fn = normalize(flip ? [-face[0], -face[1], -face[2]] : face, [0, 1, 0])
    const [na, nb, nc] = normals ?? [fn, fn, fn]
    if (flip) {
      this.vertex(a, na, style)
      this.vertex(c, nc, style)
      this.vertex(b, nb, style)
    } else {
      this.vertex(a, na, style)
      this.vertex(b, nb, style)
      this.vertex(c, nc, style)
    }
  }

  /** 四角形 a-b-c-d（周回順）。 */
  quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, style: PartStyle = {}, outward?: Outward): void {
    this.triangle(a, b, c, style, undefined, outward)
    this.triangle(a, c, d, style, undefined, outward)
  }

  /** 格子状に並べた点を面にする。法線は隣の点との差分から滑らかに作る。 */
  grid(
    rows: ReadonlyArray<ReadonlyArray<Vec3>>,
    outward: Outward,
    style: PartStyle | ((i: number, j: number) => PartStyle) = {},
  ): void {
    const ni = rows.length
    if (ni < 2) return
    const nj = rows[0].length
    const normals: Vec3[][] = []
    for (let i = 0; i < ni; i++) {
      const row: Vec3[] = []
      for (let j = 0; j < nj; j++) {
        const p = rows[i][j]
        const du = sub(rows[Math.min(ni - 1, i + 1)][j], rows[Math.max(0, i - 1)][j])
        const dv = sub(rows[i][Math.min(nj - 1, j + 1)], rows[i][Math.max(0, j - 1)])
        const hint = outwardAt(outward, p, i, j)
        let n = normalize(cross(du, dv), hint)
        if (dot(n, hint) < 0) n = [-n[0], -n[1], -n[2]]
        row.push(n)
      }
      normals.push(row)
    }
    const styleAt = typeof style === 'function' ? style : () => style
    for (let i = 0; i + 1 < ni; i++) {
      for (let j = 0; j + 1 < nj; j++) {
        const s = styleAt(i, j)
        const a = rows[i][j]
        const b = rows[i + 1][j]
        const c = rows[i + 1][j + 1]
        const d = rows[i][j + 1]
        const hint = normals[i][j]
        this.triangle(a, b, c, s, [normals[i][j], normals[i + 1][j], normals[i + 1][j + 1]], hint)
        this.triangle(a, c, d, s, [normals[i][j], normals[i + 1][j + 1], normals[i][j + 1]], hint)
      }
    }
  }

  /** three のジオメトリを足す（インデックスは展開する）。行列を渡すとその場で掛ける */
  geometry(g: THREE.BufferGeometry, style: PartStyle = {}, matrix?: THREE.Matrix4): void {
    const src = g.index ? g.toNonIndexed() : g
    if (matrix) src.applyMatrix4(matrix)
    if (!src.attributes.normal) src.computeVertexNormals()
    const p = src.attributes.position
    const n = src.attributes.normal
    const uv = src.attributes.uv
    for (let i = 0; i < p.count; i++) {
      this.vertex(
        [p.getX(i), p.getY(i), p.getZ(i)],
        [n.getX(i), n.getY(i), n.getZ(i)],
        style,
        uv ? [uv.getX(i), uv.getY(i)] : undefined,
      )
    }
    if (src !== g) src.dispose()
  }

  /** 積み上げたものを 1 つのジオメトリにする。 */
  build(options: { color?: boolean; door?: boolean; uv?: boolean } = {}): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3))
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nrm, 3))
    if (options.color) g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3))
    if (options.door) g.setAttribute('doorPart', new THREE.Float32BufferAttribute(this.doorAttr, 1))
    if (options.uv || this.hasUv) g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uvs, 2))
    g.computeBoundingBox()
    g.computeBoundingSphere()
    return g
  }
}

/** 箱。`min`〜`max` で与える */
export function boxGeometry(min: Vec3, max: Vec3): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(max[0] - min[0], max[1] - min[1], max[2] - min[2])
  g.translate((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2)
  return g
}

/** 2 点を結ぶ板（XY 平面内で傾ける）。 */
export function slabGeometry(
  from: readonly [number, number],
  to: readonly [number, number],
  zMin: number,
  zMax: number,
  thickness: number,
): THREE.BufferGeometry {
  const dx = to[0] - from[0]
  const dy = to[1] - from[1]
  const g = new THREE.BoxGeometry(Math.hypot(dx, dy), thickness, zMax - zMin)
  g.rotateZ(Math.atan2(dy, dx))
  g.translate((from[0] + to[0]) / 2, (from[1] + to[1]) / 2, (zMin + zMax) / 2)
  return g
}

/** 3 点で決まる向きへ箱を置く（`along` の方向に長さ、`up` の方向に高さ）。 */
export function orientedBox(
  centre: Vec3,
  along: Vec3,
  up: Vec3,
  size: Vec3,
): THREE.BufferGeometry {
  const x = normalize(along, [1, 0, 0])
  const z = normalize(cross(x, up), [0, 0, 1])
  const y = cross(z, x)
  const g = new THREE.BoxGeometry(size[0], size[1], size[2])
  const m = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(x[0], x[1], x[2]),
    new THREE.Vector3(y[0], y[1], y[2]),
    new THREE.Vector3(z[0], z[1], z[2]),
  )
  m.setPosition(centre[0], centre[1], centre[2])
  g.applyMatrix4(m)
  return g
}

/** 2 点を結ぶ円柱（棒）。 */
export function rodGeometry(from: Vec3, to: Vec3, radius: number, segments = 6): THREE.BufferGeometry {
  const d = sub(to, from)
  const len = Math.hypot(d[0], d[1], d[2])
  const g = new THREE.CylinderGeometry(radius, radius, len, segments, 1, false)
  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(d[0] / len, d[1] / len, d[2] / len),
  )
  g.applyQuaternion(q)
  g.translate((from[0] + to[0]) / 2, (from[1] + to[1]) / 2, (from[2] + to[2]) / 2)
  return g
}

export const vec = { sub, cross, dot, normalize }
