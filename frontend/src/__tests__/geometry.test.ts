/** カメラ位置（scene/cameraMath.ts）と角度の補間（scene/interpolationMath.ts）の単体テスト。 */

import assert from 'node:assert/strict'
import { describe, test } from 'node:test'

import {
  DRIVER_FORWARD,
  DRIVER_RIGHT,
  SURROUND_CAMERAS,
  driverEye,
  driverLookAt,
  firstActiveSlot,
  followEye,
  followLerpFactor,
  surroundEye,
  surroundLookAt,
  vehicleLocalToThree,
  type Vec3,
} from '../scene/cameraMath.ts'
import { angleDelta, lerpAngle } from '../scene/interpolationMath.ts'

const EPS = 1e-9
const HEADINGS = [0, Math.PI / 6, Math.PI / 2, 2.5, Math.PI, -Math.PI / 2, -2.2]
const CAR = { x: 12.5, y: -40.25 }

function near(actual: number, expected: number, label: string): void {
  assert.ok(Math.abs(actual - expected) <= EPS, `${label}: ${actual} ≠ ${expected}`)
}

/** three 空間の点を ENU 平面へ戻す（protocol.md 1.3 の逆） */
function enu(p: Vec3): [number, number] {
  return [p.x, -p.z]
}

/** 車から見た点の位置（前方・右方向の成分）[m] */
function inCarFrame(p: Vec3, heading: number): { forward: number; right: number } {
  const [ex, ey] = enu(p)
  const dx = ex - CAR.x
  const dy = ey - CAR.y
  return {
    forward: dx * Math.cos(heading) + dy * Math.sin(heading),
    right: dx * Math.sin(heading) - dy * Math.cos(heading),
  }
}

describe('ENU → three の写し方（protocol.md 1.3）', () => {
  test('車両の中心の真上は (x, 高さ, -y)', () => {
    for (const heading of HEADINGS) {
      const p = vehicleLocalToThree([0, 1.5, 0], CAR.x, CAR.y, heading)
      near(p.x, CAR.x, 'x')
      near(p.y, 1.5, '高さ')
      near(p.z, -CAR.y, 'z')
    }
  })

  test('車両ローカルの前・右は、どの向きでも進行方向の前・右へ写る', () => {
    for (const heading of HEADINGS) {
      const front = inCarFrame(vehicleLocalToThree([2, 0, 0], CAR.x, CAR.y, heading), heading)
      near(front.forward, 2, `前 ${heading}`)
      near(front.right, 0, `前の横ずれ ${heading}`)
      const right = inCarFrame(vehicleLocalToThree([0, 0, 1], CAR.x, CAR.y, heading), heading)
      near(right.forward, 0, `右の縦ずれ ${heading}`)
      near(right.right, 1, `右 ${heading}`)
    }
  })

  test('向きを変えても 2 点間の距離は変わらない', () => {
    const a: [number, number, number] = [1.2, 0.3, -0.7]
    const b: [number, number, number] = [-2.1, 1.1, 0.4]
    const local = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
    for (const heading of HEADINGS) {
      const pa = vehicleLocalToThree(a, CAR.x, CAR.y, heading)
      const pb = vehicleLocalToThree(b, CAR.x, CAR.y, heading)
      near(Math.hypot(pa.x - pb.x, pa.y - pb.y, pa.z - pb.z), local, `距離 ${heading}`)
    }
  })
})

describe('カメラの置き場所', () => {
  test('運転席は右ハンドル（進行方向の右・前寄り）', () => {
    for (const heading of HEADINGS) {
      const eye = inCarFrame(driverEye(CAR.x, CAR.y, heading), heading)
      near(eye.forward, DRIVER_FORWARD, `前 ${heading}`)
      near(eye.right, DRIVER_RIGHT, `右 ${heading}`)
      assert.ok(eye.right > 0)
    }
  })

  test('運転席は前を・追従カメラは後ろから見る', () => {
    for (const heading of HEADINGS) {
      const look = inCarFrame(driverLookAt(CAR.x, CAR.y, heading), heading)
      assert.ok(look.forward > 0, `運転席の注視点 ${heading}`)
      near(look.right, 0, `運転席の注視点の横ずれ ${heading}`)
      assert.ok(inCarFrame(followEye(CAR.x, CAR.y, heading), heading).forward < 0, `追従 ${heading}`)
    }
  })

  test('周囲カメラは車体の向き + 取り付けの向きを、少し見下ろす', () => {
    for (const cam of SURROUND_CAMERAS) {
      for (const heading of HEADINGS) {
        const eye = surroundEye(cam, CAR.x, CAR.y, heading)
        const look = surroundLookAt(cam, CAR.x, CAR.y, heading)
        const [ex, ey] = enu(eye)
        const [lx, ly] = enu(look)
        const yaw = Math.atan2(ly - ey, lx - ex)
        near(angleDelta(heading + (cam.yawDeg * Math.PI) / 180, yaw), 0, `${cam.key} ${heading}`)
        assert.ok(look.y < eye.y, `${cam.key} は見下ろす`)
        const side = inCarFrame(eye, heading).right
        if (cam.key === 'left') assert.ok(side < 0, '左カメラは左側')
        if (cam.key === 'right') assert.ok(side > 0, '右カメラは右側')
      }
    }
  })
})

describe('追従と補間', () => {
  test('追従の係数は 0〜1 で、時間とともに増え、rate 0 なら常に 1', () => {
    assert.equal(followLerpFactor(0, 0.016), 1)
    assert.equal(followLerpFactor(4.5, -1), 0)
    let prev = 0
    for (const dt of [0.001, 0.016, 0.05, 0.2, 1, 10]) {
      const k = followLerpFactor(4.5, dt)
      assert.ok(k > prev && k < 1 + EPS, `${dt}: ${k}`)
      prev = k
    }
  })

  test('角度差は近いほうへ回る（359° → 1° は +2°）', () => {
    const deg = Math.PI / 180
    near(angleDelta(359 * deg, 1 * deg), 2 * deg, '359→1')
    near(angleDelta(1 * deg, 359 * deg), -2 * deg, '1→359')
    near(angleDelta(0, 3 * Math.PI), Math.PI, 'π は +π 側')
    for (let a = -10; a <= 10; a += 0.7) {
      for (let b = -10; b <= 10; b += 1.3) {
        const d = angleDelta(a, b)
        assert.ok(d > -Math.PI - EPS && d <= Math.PI + EPS, `${a}, ${b}: ${d}`)
      }
    }
  })

  test('角度の補間は端で両端に一致し、中間は近いほうを通る', () => {
    const deg = Math.PI / 180
    near(lerpAngle(350 * deg, 10 * deg, 0), 350 * deg, 't=0')
    near(angleDelta(lerpAngle(350 * deg, 10 * deg, 1), 10 * deg), 0, 't=1')
    near(angleDelta(lerpAngle(350 * deg, 10 * deg, 0.5), 0), 0, 't=0.5 は 0° を通る')
  })
})

describe('追従する車の選び方', () => {
  test('走っている車のうちスロット番号が最小のもの。いなければ -1', () => {
    assert.equal(firstActiveSlot(null), -1)
    assert.equal(firstActiveSlot([]), -1)
    assert.equal(firstActiveSlot([{ id: 0, active: false }]), -1)
    assert.equal(
      firstActiveSlot([
        { id: 5, active: true },
        { id: 1, active: false },
        { id: 3, active: true },
      ]),
      3,
    )
  })
})
