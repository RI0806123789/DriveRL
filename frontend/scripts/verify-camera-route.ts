/** 運転席カメラと進路矢印の幾何を検証する（ブラウザ不要）。 */

import {
  DRIVER_EYE_HEIGHT,
  DRIVER_FORWARD,
  DRIVER_RIGHT,
  FOLLOW_BACK,
  FOLLOW_UP,
  driverEye,
  driverLookAt,
  followEye,
  followLookAt,
} from '../src/scene/cameraMath.ts'
import {
  CHEVRON_SPACING,
  RIBBON_WIDTH,
  buildChevrons,
  buildRibbon,
  chevronsAlong,
  writeRibbonPositions,
} from '../src/scene/routeArrowGeometry.ts'
import type { Point2 } from '../src/scene/routeArrowGeometry.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

/** three 空間の点を ENU に戻す（enu.y = -three.z） */
function toEnu(p: { x: number; y: number; z: number }): { x: number; y: number; h: number } {
  return { x: p.x, y: -p.z, h: p.y }
}

const HEADINGS: Array<[string, number]> = [
  ['東向き', 0],
  ['北向き', Math.PI / 2],
  ['西向き', Math.PI],
  ['南向き', -Math.PI / 2],
  ['北東向き', Math.PI / 4],
]

console.log('='.repeat(70))
console.log('1. 運転席の視点位置（日本車 = 右ハンドル）')
console.log('='.repeat(70))

for (const [name, h] of HEADINGS) {
  const eye = toEnu(driverEye(0, 0, h))

  const fwdX = Math.cos(h)
  const fwdY = Math.sin(h)
  const rightX = Math.sin(h)
  const rightY = -Math.cos(h)

  const lateral = eye.x * rightX + eye.y * rightY
  const longitudinal = eye.x * fwdX + eye.y * fwdY

  check(
    `${name}: 視点が車体中心より右（右ハンドル）`,
    Math.abs(lateral - DRIVER_RIGHT) < 1e-9,
    `右方向 ${lateral.toFixed(3)}m`,
  )
  check(
    `${name}: 視点が車体中心より前`,
    Math.abs(longitudinal - DRIVER_FORWARD) < 1e-9,
    `前方向 ${longitudinal.toFixed(3)}m`,
  )
  check(
    `${name}: アイポイントの高さ`,
    Math.abs(eye.h - DRIVER_EYE_HEIGHT) < 1e-9,
    `${eye.h.toFixed(2)}m`,
  )

  check(
    `${name}: 視点が車体の内側にある`,
    Math.abs(longitudinal) < 4.4 / 2 && Math.abs(lateral) < 1.8 / 2,
    `前後 ${longitudinal.toFixed(2)}m / 左右 ${lateral.toFixed(2)}m`,
  )

  const look = toEnu(driverLookAt(0, 0, h))
  const dirX = look.x - eye.x
  const dirY = look.y - eye.y
  const len = Math.hypot(dirX, dirY)
  const dot = (dirX / len) * fwdX + (dirY / len) * fwdY
  check(`${name}: 視線が進行方向を向く`, dot > 0.999, `内積 ${dot.toFixed(5)}`)
  check(
    `${name}: 視線がやや下向き（路面が見える）`,
    look.h < eye.h,
    `視点 ${eye.h.toFixed(2)}m → 注視点 ${look.h.toFixed(2)}m`,
  )
}

console.log('')
console.log('='.repeat(70))
console.log('2. 追従カメラ（後方上空）')
console.log('='.repeat(70))

for (const [name, h] of HEADINGS) {
  const eye = toEnu(followEye(0, 0, h))
  const look = toEnu(followLookAt(0, 0, h))
  const fwdX = Math.cos(h)
  const fwdY = Math.sin(h)

  const behind = -(eye.x * fwdX + eye.y * fwdY)
  check(`${name}: カメラが車両の後方`, Math.abs(behind - FOLLOW_BACK) < 1e-9, `${behind.toFixed(2)}m 後ろ`)
  check(`${name}: カメラが上空`, Math.abs(eye.h - FOLLOW_UP) < 1e-9, `高さ ${eye.h.toFixed(2)}m`)

  const dirX = look.x - eye.x
  const dirY = look.y - eye.y
  const len = Math.hypot(dirX, dirY)
  const dot = (dirX / len) * fwdX + (dirY / len) * fwdY
  check(`${name}: 視線が進行方向を向く`, dot > 0.999, `内積 ${dot.toFixed(5)}`)
}

console.log('')
console.log('='.repeat(70))
console.log('3. 進路矢印のリボン')
console.log('='.repeat(70))

{
  const straight: Point2[] = Array.from({ length: 26 }, (_, i) => [i * 2, 0] as Point2)
  const ribbon = buildRibbon(straight)
  check('リボンが生成される', ribbon !== null)

  if (ribbon) {
    const pos = ribbon.getAttribute('position')
    check('頂点数が点数 x 2', pos.count === straight.length * 2, `${pos.count} 頂点`)

    let minW = Infinity
    let maxW = -Infinity
    for (let i = 0; i < straight.length; i++) {
      const w = Math.hypot(
        pos.getX(i * 2) - pos.getX(i * 2 + 1),
        pos.getZ(i * 2) - pos.getZ(i * 2 + 1),
      )
      minW = Math.min(minW, w)
      maxW = Math.max(maxW, w)
    }
    check(
      `リボン幅が ${RIBBON_WIDTH}m`,
      Math.abs(minW - RIBBON_WIDTH) < 1e-6 && Math.abs(maxW - RIBBON_WIDTH) < 1e-6,
      `${minW.toFixed(3)}〜${maxW.toFixed(3)}m`,
    )

    let nan = false
    for (let i = 0; i < pos.count * 3; i++) {
      if (!Number.isFinite((pos.array as ArrayLike<number>)[i])) nan = true
    }
    check('NaN が無い', !nan)
    check('インデックスがある', ribbon.getIndex() !== null)
  }

  const curve: Point2[] = Array.from({ length: 40 }, (_, i) => {
    const t = (i / 39) * Math.PI * 0.5
    return [30 * Math.cos(t), 30 * Math.sin(t)] as Point2
  })
  const curved = buildRibbon(curve)
  if (curved) {
    const pos = curved.getAttribute('position')
    let minW = Infinity
    let maxW = -Infinity
    for (let i = 0; i < curve.length; i++) {
      const w = Math.hypot(
        pos.getX(i * 2) - pos.getX(i * 2 + 1),
        pos.getZ(i * 2) - pos.getZ(i * 2 + 1),
      )
      minW = Math.min(minW, w)
      maxW = Math.max(maxW, w)
    }
    check(
      'カーブでもリボン幅が保たれる',
      Math.abs(maxW - RIBBON_WIDTH) < 0.05 && Math.abs(minW - RIBBON_WIDTH) < 0.05,
      `${minW.toFixed(3)}〜${maxW.toFixed(3)}m`,
    )
  }

  check('点が 1 個以下ならリボンを作らない', buildRibbon([[0, 0]]) === null)

  const before: Point2[] = [
    [0, 0],
    [4, 0.5],
    [8, 1.6],
    [12, 3.4],
    [16, 6.0],
    [20, 9.3],
  ]
  const after: Point2[] = [
    [0, 0],
    [4, -0.4],
    [8, -1.1],
    [12, -2.2],
    [16, -3.8],
    [20, -5.9],
  ]
  const reused = buildRibbon(before, RIBBON_WIDTH)
  const fresh = buildRibbon(after, RIBBON_WIDTH)
  if (reused && fresh) {
    const target = reused.getAttribute('position').array as Float32Array
    writeRibbonPositions(target, after, RIBBON_WIDTH)
    const expected = fresh.getAttribute('position').array as Float32Array
    let maxDiff = 0
    for (let i = 0; i < expected.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(target[i] - expected[i]))
    }
    check(
      '書き換えた頂点が作り直したものと一致する',
      target.length === expected.length && maxDiff === 0,
      `最大差 ${maxDiff}`,
    )
    const nrm = fresh.getAttribute('normal').array as Float32Array
    let flat = true
    for (let i = 0; i < nrm.length; i += 3) {
      if (Math.abs(nrm[i]) > 1e-6 || Math.abs(nrm[i + 2]) > 1e-6) flat = false
    }
    check('リボンの法線は常に ±Y（点列に依らない）', flat)
  }
}

console.log('')
console.log('='.repeat(70))
console.log('4. 矢羽根が進行方向を指しているか')
console.log('='.repeat(70))

for (const [name, h] of HEADINGS) {
  const dirX = Math.cos(h)
  const dirY = Math.sin(h)
  const path: Point2[] = Array.from(
    { length: 31 },
    (_, i) => [dirX * i * 2, dirY * i * 2] as Point2,
  )
  const chevrons = chevronsAlong(path)
  check(`${name}: 矢羽根ができる`, chevrons.length > 0, `${chevrons.length} 個`)

  if (chevrons.length > 0) {
    let allForward = true
    for (const c of chevrons) {
      const midX = (c.backLeft[0] + c.backRight[0]) / 2
      const midY = (c.backLeft[1] + c.backRight[1]) / 2
      const vx = c.tip[0] - midX
      const vy = c.tip[1] - midY
      const len = Math.hypot(vx, vy)
      if ((vx / len) * dirX + (vy / len) * dirY < 0.999) allForward = false
    }
    check(`${name}: すべての矢羽根が進行方向を指す`, allForward)

    if (chevrons.length >= 2) {
      const d = Math.hypot(
        chevrons[1].tip[0] - chevrons[0].tip[0],
        chevrons[1].tip[1] - chevrons[0].tip[1],
      )
      check(
        `${name}: 矢羽根の間隔が ${CHEVRON_SPACING}m`,
        Math.abs(d - CHEVRON_SPACING) < 0.2,
        `${d.toFixed(2)}m`,
      )
    }
  }
}

{
  const short: Point2[] = [
    [0, 0],
    [3, 0],
  ]
  check('短すぎる経路には矢羽根を置かない', chevronsAlong(short).length === 0)
  check('短すぎる経路のメッシュは null', buildChevrons(short) === null)

  const path: Point2[] = Array.from({ length: 31 }, (_, i) => [i * 2, 0] as Point2)
  const mesh = buildChevrons(path)
  check('矢羽根メッシュが生成される', mesh !== null)
  if (mesh) {
    const pos = mesh.getAttribute('position')
    check('三角形 1 枚につき 3 頂点', pos.count % 3 === 0, `${pos.count} 頂点`)
    let sameY = true
    const y0 = pos.getY(0)
    for (let i = 1; i < pos.count; i++) if (Math.abs(pos.getY(i) - y0) > 1e-9) sameY = false
    check('矢羽根が水平に置かれている', sameY, `高さ ${y0.toFixed(3)}m`)
  }
}

console.log('')
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
