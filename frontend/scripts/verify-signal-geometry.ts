/** 交通信号機（車両用・歩行者用）の向きを検証する（ブラウザ不要）。 */

import * as THREE from 'three'
import {
  HOUSING_H,
  MOUNT_HEIGHT,
  ROLE_GREEN,
  ROLE_RED,
  ROLE_YELLOW,
  buildSignalPlacement,
  createLampGeometry,
} from '../src/scene/signalGeometry.ts'
import {
  PED_MOUNT_HEIGHT,
  PED_ROLE_GREEN,
  PED_ROLE_RED,
  PED_SIDE_MARGIN,
  buildPedestrianSignalPlacement,
  pedestrianWalkable,
} from '../src/scene/pedestrianSignalGeometry.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

/** three 空間のベクトルを ENU に戻す（enu.y = -three.z） */
function toEnu(x: number, z: number): { x: number; y: number } {
  return { x, y: -z }
}

/** 2 つの方位の差を (-pi, pi] に畳む */
function angleDiff(a: number, b: number): number {
  return Math.atan2(Math.sin(a - b), Math.cos(a - b))
}

console.log('='.repeat(70))
console.log('1. 灯火の円板が「前方 = +X」規約になっているか')
console.log('='.repeat(70))

{
  const g = createLampGeometry()
  g.computeVertexNormals()
  const n = g.attributes.normal
  const nx = n.getX(0)
  const ny = n.getY(0)
  const nz = n.getZ(0)
  check(
    '法線が +X を向いている',
    Math.abs(nx - 1) < 1e-6 && Math.abs(ny) < 1e-6 && Math.abs(nz) < 1e-6,
    `(${nx.toFixed(3)}, ${ny.toFixed(3)}, ${nz.toFixed(3)})`,
  )
  check('color 属性がある（instanceColor を効かせるため）', g.getAttribute('color') !== undefined)
}

console.log('')
console.log('='.repeat(70))
console.log('2. 灯器が進入車両に正対しているか（4 方位で確認）')
console.log('='.repeat(70))

const HEADINGS: Array<[string, number]> = [
  ['東向き', 0],
  ['北向き', Math.PI / 2],
  ['西向き', Math.PI],
  ['南向き', -Math.PI / 2],
]

for (const [name, heading] of HEADINGS) {
  const roadWidth = 12
  const { lamps, placements } = buildSignalPlacement(
    [{ nodeId: 0, x: 0, y: 0, heading, roadWidth }],
    [{ id: 0, x: 0, y: 0 }],
  )
  const pl = placements[0]

  const headEnu = toEnu(pl.headX, pl.headZ)
  const forwardDot = headEnu.x * Math.cos(heading) + headEnu.y * Math.sin(heading)
  check(`${name}: 灯器が交差点の対面側にある`, forwardDot > 0, `前方成分 ${forwardDot.toFixed(2)}m`)

  const facingDiff = angleDiff(pl.facing, heading + Math.PI)
  check(
    `${name}: 灯器の正面が進行方向の真逆`,
    Math.abs(facingDiff) < 1e-9,
    `ずれ ${((facingDiff * 180) / Math.PI).toFixed(3)}度`,
  )

  const poleEnu = toEnu(pl.poleX, pl.poleZ)
  const leftX = -Math.sin(heading)
  const leftY = Math.cos(heading)
  const poleLateral = poleEnu.x * leftX + poleEnu.y * leftY
  check(
    `${name}: 支柱が進行方向の左側（左側通行）`,
    poleLateral > 0,
    `左方向成分 ${poleLateral.toFixed(2)}m`,
  )

  const green = lamps.find((l) => l.role === ROLE_GREEN)!
  const yellow = lamps.find((l) => l.role === ROLE_YELLOW)!
  const red = lamps.find((l) => l.role === ROLE_RED)!
  const lateral = (l: typeof green) => {
    const e = toEnu(l.position.x, l.position.z)
    const c = toEnu(pl.headX, pl.headZ)
    return (e.x - c.x) * leftX + (e.y - c.y) * leftY
  }
  const lg = lateral(green)
  const ly = lateral(yellow)
  const lr = lateral(red)
  check(
    `${name}: 運転者から見て 左=青 / 中=黄 / 右=赤`,
    lg > ly && ly > lr && Math.abs(ly) < 1e-9,
    `青 ${lg.toFixed(2)} / 黄 ${ly.toFixed(2)} / 赤 ${lr.toFixed(2)}（左が正）`,
  )

  check(
    `${name}: 灯器下端が路面から 5.0m`,
    Math.abs(green.position.y - HOUSING_H / 2 - MOUNT_HEIGHT) < 1e-9,
    `${(green.position.y - HOUSING_H / 2).toFixed(2)}m`,
  )
}

console.log('')
console.log('='.repeat(70))
console.log('3. 灯火の法線が実際に運転者の方を向くか（行列を組んで確認）')
console.log('='.repeat(70))

for (const [name, heading] of HEADINGS) {
  const { lamps } = buildSignalPlacement(
    [{ nodeId: 0, x: 0, y: 0, heading, roadWidth: 12 }],
    [{ id: 0, x: 0, y: 0 }],
  )
  const lamp = lamps[0]

  const dummy = new THREE.Object3D()
  dummy.position.copy(lamp.position)
  dummy.rotation.set(0, lamp.facing, 0)
  dummy.updateMatrix()

  const normal = new THREE.Vector3(1, 0, 0).applyQuaternion(dummy.quaternion)
  const enuNormal = toEnu(normal.x, normal.z)
  const want = { x: Math.cos(heading + Math.PI), y: Math.sin(heading + Math.PI) }
  const dot = enuNormal.x * want.x + enuNormal.y * want.y
  check(
    `${name}: 灯火の法線が運転者を向く`,
    dot > 0.999,
    `内積 ${dot.toFixed(6)}（1.0 が完全一致）`,
  )
}

console.log('')
console.log('='.repeat(70))
console.log('4. 交差点の 4 方向を作っても破綻しないか')
console.log('='.repeat(70))

{
  const roadWidth = 14
  const four = HEADINGS.map(([, h], i) => ({
    nodeId: 0,
    x: -Math.cos(h) * 10,
    y: -Math.sin(h) * 10,
    heading: h,
    roadWidth,
  }))
  const { parts, lamps, placements } = buildSignalPlacement(four, [{ id: 0, x: 0, y: 0 }])
  check('4 方向ぶんの灯火ができる', lamps.length === 12, `${lamps.length} 灯`)
  check('部品が生成される', parts.length > 0, `${parts.length} 個`)

  let minDist = Infinity
  for (let i = 0; i < placements.length; i++) {
    for (let j = i + 1; j < placements.length; j++) {
      const d = Math.hypot(
        placements[i].headX - placements[j].headX,
        placements[i].headZ - placements[j].headZ,
      )
      minDist = Math.min(minDist, d)
    }
  }
  check('灯器どうしが重なっていない', minDist > 2.0, `最小間隔 ${minDist.toFixed(2)}m`)

  const nan = parts.some((g) => {
    const p = g.getAttribute('position')
    for (let i = 0; i < p.count * 3; i++) {
      if (!Number.isFinite((p.array as ArrayLike<number>)[i])) return true
    }
    return false
  })
  check('ジオメトリに NaN が無い', !nan)
}

console.log('')
console.log('='.repeat(70))
console.log('歩行者用信号（縦 2 灯・現示は車両信号の裏返し）')
console.log('='.repeat(70))

{
  // 東を向いて進む車の信号。停止線は原点、道路幅 10m
  const signal = { nodeId: 1, x: 0, y: 0, heading: 0, roadWidth: 10 }
  const { lamps, placements } = buildPedestrianSignalPlacement([signal], [{ id: 1, x: 0, y: 0 }])

  check('1 基の車両信号に対して灯器が 2 基（横断歩道の両端）', placements.length === 2)
  check('灯器 1 基につき 2 灯（赤・青）', lamps.length === 4)

  const sides = placements.map((p) => toEnu(p.x, p.z).y)
  check(
    '灯器は道路の左右へ同じだけ離れて立つ',
    Math.abs(sides[0] + sides[1]) < 1e-6 &&
      Math.abs(Math.abs(sides[0]) - (signal.roadWidth / 2 + PED_SIDE_MARGIN)) < 1e-6,
    `y=${sides.map((v) => v.toFixed(2)).join(' / ')}`,
  )
  check(
    '灯器は停止線より交差点側（車の進行方向）にある',
    placements.every((p) => toEnu(p.x, p.z).x > 0),
  )

  // ★ 渡ってくる人に正対すること。道路と同じ向きだと横からしか見えない
  for (const p of placements) {
    const enu = toEnu(p.x, p.z)
    // 左側（ENU の +y）に立つ灯器は右（-y 方向）を向く
    const want = enu.y > 0 ? -Math.PI / 2 : Math.PI / 2
    check(
      `${enu.y > 0 ? '左' : '右'}側の灯器は横断してくる人に正対する`,
      Math.abs(angleDiff(p.facing, want)) < 1e-6,
      `${((p.facing * 180) / Math.PI).toFixed(0)}°`,
    )
  }

  const red = lamps.filter((l) => l.role === PED_ROLE_RED)
  const green = lamps.filter((l) => l.role === PED_ROLE_GREEN)
  check(
    '赤が上・青が下（日本の歩行者信号の並び）',
    red.every((r) => green.every((g) => r.position.y > g.position.y)),
  )
  check(
    `灯器の高さは ${PED_MOUNT_HEIGHT}m 前後（見上げる位置）`,
    lamps.every((l) => l.position.y > 2 && l.position.y < 3.2),
  )
  check(
    'どの灯も対応する車両信号を指している',
    lamps.every((l) => l.signalIndex === 0),
  )
}

{
  // ★ 歩行者信号は車両信号の裏返し。ここがずれると、車が走っている横を渡らせてしまう
  check('車両が赤なら渡れる', pedestrianWalkable(2))
  check('車両が青なら渡れない', !pedestrianWalkable(0))
  check('車両が黄でも渡れない', !pedestrianWalkable(1))
}

console.log('')
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
