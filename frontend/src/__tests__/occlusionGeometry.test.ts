/** 見通しと死角の描画の幾何（scene/occlusionGeometry.ts）とモック（store/mock/occlusion.ts）の単体テスト。 */

import assert from 'node:assert/strict'
import { describe, test } from 'node:test'

import {
  OCCLUSION_LIFT,
  polarPoint,
  sectorPolygon,
  writeFan,
  writeSeen,
  writeShadows,
  type Rgb,
} from '../scene/occlusionGeometry.ts'
import { mockOcclusion } from '../store/mock/occlusion.ts'
import type { OcclusionView, VehicleState } from '../types/protocol.ts'

const RED: Rgb = [1, 0, 0]
const HALF_FOV = (34 * Math.PI) / 180

function view(partial: Partial<OcclusionView>): OcclusionView {
  return { range: 30, los: [null, null], sectors: [0, 0, 0, 0], frontOccluded: 0, cameras: [], shadows: [], ...partial }
}

/** 書き込んだ三角形の面積の合計 [m²]（three の X-Z 平面） */
function area(positions: Float32Array, vertices: number): number {
  let sum = 0
  for (let v = 0; v + 2 < vertices; v += 3) {
    const ax = positions[v * 3]
    const az = positions[v * 3 + 2]
    const bx = positions[v * 3 + 3]
    const bz = positions[v * 3 + 5]
    const cx = positions[v * 3 + 6]
    const cz = positions[v * 3 + 8]
    sum += Math.abs((bx - ax) * (cz - az) - (cx - ax) * (bz - az)) / 2
  }
  return sum
}

describe('自車座標から three の車両ローカルへ', () => {
  test('左（+y）は -Z、前（+x）は +X、高さは路面の少し上', () => {
    const positions = new Float32Array(9)
    const colors = new Float32Array(9)
    const n = writeFan(
      [
        { x: 0, y: 0 },
        { x: 10, y: 2 },
        { x: 10, y: -2 },
      ],
      RED,
      positions,
      colors,
      0,
    )
    assert.equal(n, 3)
    assert.equal(positions[3], 10)
    assert.equal(positions[5], -2, '左 2m の点は three の Z = -2')
    assert.equal(positions[8], 2, '右 2m の点は three の Z = +2')
    assert.ok(positions.every((_, i) => i % 3 !== 1 || positions[i] === Math.fround(OCCLUSION_LIFT)))
  })

  test('左カメラ（yaw = +90 度）の扇は車の左（three の -Z）へ広がる', () => {
    const positions = new Float32Array(3000)
    const colors = new Float32Array(3000)
    const n = writeSeen(
      view({ cameras: [{ key: 'left', at: [0, 0.74], yaw: Math.PI / 2, seen: [[-HALF_FOV, HALF_FOV, 20]] }] }),
      RED,
      positions,
      colors,
    )
    assert.ok(n > 0)
    for (let v = 0; v < n; v += 1) assert.ok(positions[v * 3 + 2] <= -0.74 + 1e-6, `頂点 ${v} が右へ出た`)
  })
})

describe('扇', () => {
  test('奥行き 0 からの扇は、カメラの位置から始まり弧は奥行きちょうど', () => {
    const at: [number, number] = [0.35, -0.36]
    const poly = sectorPolygon(at, 0, -0.3, 0.3, 0, 25)
    assert.deepEqual(poly[0], { x: 0.35, y: -0.36 })
    for (const p of poly.slice(1)) assert.ok(Math.abs(Math.hypot(p.x - at[0], p.y - at[1]) - 25) < 1e-9)
  })

  test('見えている扇の面積は 0.5 d² Δθ にほぼ等しい（弦で近似する分だけ小さい）', () => {
    const positions = new Float32Array(6000)
    const colors = new Float32Array(6000)
    const n = writeSeen(
      view({
        cameras: [
          { key: 'front', at: [0.35, -0.36], yaw: 0, seen: [[-HALF_FOV, 0, 30], [0, HALF_FOV, 10]] },
        ],
      }),
      RED,
      positions,
      colors,
    )
    const expected = 0.5 * 900 * HALF_FOV + 0.5 * 100 * HALF_FOV
    const got = area(positions, n)
    assert.ok(got <= expected && got > expected * 0.995, `面積 ${got} / 期待 ${expected}`)
  })

  test('死角は方位 from..to・奥行き near..range に収まる', () => {
    const positions = new Float32Array(3000)
    const colors = new Float32Array(3000)
    const at: [number, number] = [0.35, -0.36]
    const n = writeShadows(
      view({
        cameras: [{ key: 'front', at, yaw: 0, seen: [] }],
        shadows: [{ camera: 'front', kind: 'dynamic', from: -0.1, to: 0.1, near: 12 }],
      }),
      () => RED,
      positions,
      colors,
    )
    assert.ok(n > 0)
    for (let v = 0; v < n; v += 1) {
      const x = positions[v * 3] - at[0]
      const y = -positions[v * 3 + 2] - at[1]
      const d = Math.hypot(x, y)
      const a = Math.atan2(y, x)
      assert.ok(d >= 12 * Math.cos(0.1) - 1e-4 && d <= 30 + 1e-4, `奥行き ${d}`)
      assert.ok(a >= -0.1 - 1e-6 && a <= 0.1 + 1e-6, `方位 ${a}`)
    }
  })

  test('描かない種類と、結果の無いカメラの死角は書かない', () => {
    const positions = new Float32Array(3000)
    const colors = new Float32Array(3000)
    const n = writeShadows(
      view({
        cameras: [{ key: 'front', at: [0, 0], yaw: 0, seen: [] }],
        shadows: [
          { camera: 'front', kind: 'static', from: -0.1, to: 0.1, near: 12 },
          { camera: 'rear', kind: 'dynamic', from: -0.1, to: 0.1, near: 12 },
        ],
      }),
      (kind) => (kind === 'dynamic' ? RED : null),
      positions,
      colors,
    )
    assert.equal(n, 0)
  })

  test('頂点の上限を超えて書かない', () => {
    const positions = new Float32Array(30)
    const colors = new Float32Array(30)
    const n = writeSeen(
      view({ cameras: [{ key: 'front', at: [0, 0], yaw: 0, seen: [[-HALF_FOV, HALF_FOV, 30]] }] }),
      RED,
      positions,
      colors,
      10,
    )
    assert.ok(n <= 10 && n % 3 === 0, `${n} 頂点`)
  })
})

describe('モックの見通しと死角', () => {
  const car = (id: number, x: number, y: number, heading = 0): VehicleState =>
    ({ id, active: true, x, y, heading, speed: 0, steer: 0, collided: false, reachedGoal: false, goal: [0, 0] }) as VehicleState

  test('前に車がいれば前方カメラに陰ができ、何も無い左右は見通し 30m', () => {
    const me = car(0, 0, 0)
    const ahead = car(1, 12, 0)
    const result = mockOcclusion(me, [me, ahead])
    assert.deepEqual(result.los, [30, 30])
    assert.ok(result.frontOccluded > 0 && result.frontOccluded < 0.3)
    const shadow = result.shadows.find((s) => s.camera === 'front')
    assert.ok(shadow && shadow.kind === 'dynamic' && shadow.from < 0.05 && shadow.to > -0.05)
    assert.equal(result.cameras.length, 4)
    // 何も遮らなければ扇の面積の割合は 68 / 90
    assert.ok(Math.abs(result.sectors[1] - 68 / 90) < 0.01, `後方 ${result.sectors[1]}`)
    assert.ok(result.sectors[0] < result.sectors[1])
  })

  test('左にいる車は左カメラにだけ陰を作る（左右を取り違えない）', () => {
    const me = car(0, 0, 0)
    const left = car(1, 0, 8)
    const result = mockOcclusion(me, [me, left])
    assert.deepEqual(
      result.shadows.map((s) => s.camera),
      ['left'],
    )
    assert.ok(result.sectors[2] < result.sectors[3], `左 ${result.sectors[2]} / 右 ${result.sectors[3]}`)
    // 見通し距離は画角の中で見えている奥行きの最大なので、細い陰 1 つでは縮まない
    assert.deepEqual(result.los, [30, 30])
    const p = polarPoint([0, 0.74], Math.PI / 2, 0, 7.26)
    assert.ok(Math.abs(p.x) < 1e-9 && Math.abs(p.y - 8) < 1e-9)
  })
})
