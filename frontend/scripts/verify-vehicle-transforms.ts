/** 車両の姿勢・部品の取り付け位置・ライトの点灯条件を検証する（ブラウザ不要）。 */

import * as THREE from 'three'
import {
  BODY_OFFSET,
  CABIN_OFFSET,
  NOSE_OFFSET,
  WHEEL_OFFSETS,
  WHEEL_RADIUS,
  composeVehicleMatrix,
  composeWheelMatrix,
  createTransformScratch,
  makeBodyGeometry,
  makeCabinGeometry,
  makeNoseGeometry,
  composeLightMatrix,
  makeWheelGeometry,
} from '../src/scene/vehicleGeometry.ts'
import {
  BLINK_HZ,
  HEADLIGHT_FOG,
  HEADLIGHT_RAIN,
  LIGHTS_PER_VEHICLE,
  LIGHT_HEAD,
  LIGHT_SLOTS,
  LIGHT_TAIL,
  LIGHT_TURN,
  blinkOn,
  headlightsOn,
  lightIntensity,
  lightStateFor,
} from '../src/scene/vehicleLights.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

/** 2 つの行列の最大要素差 */
function matrixDiff(a: THREE.Matrix4, b: THREE.Matrix4): number {
  let worst = 0
  for (let i = 0; i < 16; i++) worst = Math.max(worst, Math.abs(a.elements[i] - b.elements[i]))
  return worst
}

/** 旧実装（シーングラフ）で同じ姿勢を組み立て、ワールド行列を返す */
function legacyWorld(
  enuX: number,
  enuY: number,
  heading: number,
  build: (parent: THREE.Object3D) => THREE.Object3D,
): THREE.Matrix4 {
  const root = new THREE.Group()
  root.position.set(enuX, 0, -enuY)
  root.rotation.y = heading
  const leaf = build(root)
  root.updateMatrixWorld(true)
  return leaf.matrixWorld.clone()
}

const scratch = createTransformScratch()
const out = new THREE.Matrix4()
const base = new THREE.Matrix4()

const POSES: Array<{ x: number; y: number; heading: number; steer: number; roll: number }> = [
  { x: 0, y: 0, heading: 0, steer: 0, roll: 0 },
  { x: 12.5, y: -30.25, heading: Math.PI / 2, steer: 0.35, roll: 1.2 },
  { x: -104.75, y: 88.5, heading: -2.4, steer: -0.42, roll: -9.7 },
  { x: 300, y: 300, heading: Math.PI, steer: 0.61, roll: 41.3 },
  { x: -7.125, y: 0.5, heading: 5.9, steer: -0.05, roll: -0.001 },
]

console.log('='.repeat(70))
console.log('車両の姿勢（InstancedMesh 化の前後で一致するか）')
console.log('='.repeat(70))

let worstBody = 0
for (const p of POSES) {
  composeVehicleMatrix(scratch, p.x, p.y, p.heading, base)
  const legacy = legacyWorld(p.x, p.y, p.heading, (root) => root)
  worstBody = Math.max(worstBody, matrixDiff(base, legacy))
}
check('車両のワールド行列が旧実装と一致する', worstBody < 1e-9, `最大差 ${worstBody.toExponential(1)}`)

{
  composeVehicleMatrix(scratch, 10, 20, 0, base)
  const pos = new THREE.Vector3().setFromMatrixPosition(base)
  check(
    'three.z = -enu.y になっている',
    Math.abs(pos.x - 10) < 1e-9 && Math.abs(pos.z + 20) < 1e-9,
    `(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`,
  )
  composeVehicleMatrix(scratch, 0, 0, 0, base)
  const fwd = new THREE.Vector3(1, 0, 0).applyMatrix4(base)
  check('heading=0 で前方が ENU +x', Math.abs(fwd.x - 1) < 1e-9 && Math.abs(fwd.z) < 1e-9)
  composeVehicleMatrix(scratch, 0, 0, Math.PI / 2, base)
  const left = new THREE.Vector3(1, 0, 0).applyMatrix4(base)
  check(
    'heading=+90度 で前方が ENU +y',
    Math.abs(left.x) < 1e-9 && Math.abs(left.z + 1) < 1e-9,
    `three=(${left.x.toFixed(3)}, ${left.z.toFixed(3)})`,
  )
}

let worstWheel = 0
for (const p of POSES) {
  composeVehicleMatrix(scratch, p.x, p.y, p.heading, base)
  for (let k = 0; k < WHEEL_OFFSETS.length; k++) {
    const spec = WHEEL_OFFSETS[k]
    composeWheelMatrix(scratch, base, k, p.steer, p.roll, out)
    const legacy = legacyWorld(p.x, p.y, p.heading, (root) => {
      const holder = new THREE.Group()
      holder.position.set(spec.position[0], spec.position[1], spec.position[2])
      if (spec.steered) holder.rotation.y = p.steer
      const mesh = new THREE.Object3D()
      mesh.rotation.z = p.roll
      holder.add(mesh)
      root.add(holder)
      return mesh
    })
    worstWheel = Math.max(worstWheel, matrixDiff(out, legacy))
  }
}
check('車輪のワールド行列が旧実装と一致する', worstWheel < 1e-9, `最大差 ${worstWheel.toExponential(1)}`)

{
  composeVehicleMatrix(scratch, 0, 0, 0, base)
  const noSteer: THREE.Matrix4[] = []
  const steered: THREE.Matrix4[] = []
  for (let k = 0; k < WHEEL_OFFSETS.length; k++) {
    noSteer.push(composeWheelMatrix(scratch, base, k, 0, 0, new THREE.Matrix4()))
    steered.push(composeWheelMatrix(scratch, base, k, 0.4, 0, new THREE.Matrix4()))
  }
  check(
    '前輪 2 本は舵角で姿勢が変わる',
    matrixDiff(noSteer[0], steered[0]) > 1e-3 && matrixDiff(noSteer[1], steered[1]) > 1e-3,
  )
  check(
    '後輪 2 本は舵角で姿勢が変わらない',
    matrixDiff(noSteer[2], steered[2]) < 1e-12 && matrixDiff(noSteer[3], steered[3]) < 1e-12,
  )
}

{
  composeVehicleMatrix(scratch, 0, 0, 0, base)
  const ys: number[] = []
  for (let k = 0; k < WHEEL_OFFSETS.length; k++) {
    composeWheelMatrix(scratch, base, k, 0, 0, out)
    ys.push(new THREE.Vector3().setFromMatrixPosition(out).y)
  }
  check(
    '4 輪とも車輪半径の高さにある（＝接地している）',
    ys.every((y) => Math.abs(y - WHEEL_RADIUS) < 1e-9),
    `y = ${ys.map((v) => v.toFixed(3)).join(' / ')}`,
  )
  const zs = WHEEL_OFFSETS.map((w) => w.position[2])
  check(
    '左右の車輪が対称に付いている',
    Math.abs(zs[0] + zs[1]) < 1e-9 && Math.abs(zs[2] + zs[3]) < 1e-9,
    `前 ${zs[0]} / ${zs[1]}、後 ${zs[2]} / ${zs[3]}`,
  )
  const xs = WHEEL_OFFSETS.map((w) => w.position[0])
  check('前輪が後輪より前にある', xs[0] > 0 && xs[1] > 0 && xs[2] < 0 && xs[3] < 0)
}

console.log()
console.log('='.repeat(70))
console.log('ジオメトリに焼き込んだ取り付け位置')
console.log('='.repeat(70))

function centerOf(geom: THREE.BufferGeometry): THREE.Vector3 {
  geom.computeBoundingBox()
  const box = geom.boundingBox!
  return box.getCenter(new THREE.Vector3())
}

const cases: Array<[string, THREE.BufferGeometry, readonly [number, number, number]]> = [
  ['車体', makeBodyGeometry(), BODY_OFFSET],
  ['ノーズ', makeNoseGeometry(), NOSE_OFFSET],
  ['キャビン', makeCabinGeometry(), CABIN_OFFSET],
]
for (const [name, geom, offset] of cases) {
  const c = centerOf(geom)
  const ok =
    Math.abs(c.x - offset[0]) < 1e-6 &&
    Math.abs(c.y - offset[1]) < 1e-6 &&
    Math.abs(c.z - offset[2]) < 1e-6
  check(
    `${name}の中心が (${offset.join(', ')}) にある`,
    ok,
    `(${c.x.toFixed(3)}, ${c.y.toFixed(3)}, ${c.z.toFixed(3)})`,
  )
}

{
  const geom = makeBodyGeometry()
  geom.computeBoundingBox()
  check(
    '車体が地面より上にある',
    geom.boundingBox!.min.y > 0,
    `最下端 ${geom.boundingBox!.min.y.toFixed(3)}m`,
  )
}

{
  const geom = makeWheelGeometry()
  geom.computeBoundingBox()
  const size = geom.boundingBox!.getSize(new THREE.Vector3())
  const RADIAL_SEGMENTS = 14
  const minSpan = 2 * WHEEL_RADIUS * Math.cos(Math.PI / RADIAL_SEGMENTS)
  const maxSpan = 2 * WHEEL_RADIUS
  const spans = (v: number) => v >= minSpan - 1e-6 && v <= maxSpan + 1e-6
  check(
    '車輪の軸が車体左右（Z）を向いている',
    spans(size.x) && spans(size.y) && Math.abs(size.z - 0.24) < 1e-6,
    `外形 ${size.x.toFixed(3)} x ${size.y.toFixed(3)} x ${size.z.toFixed(3)}m` +
      `（転がり面の許容 ${minSpan.toFixed(3)}〜${maxSpan.toFixed(3)}m）`,
  )
}

{
  const hidden = new THREE.Matrix4().makeScale(0, 0, 0)
  const p = new THREE.Vector3(1, 1, 1).applyMatrix4(hidden)
  check('非表示スロットの行列が 1 点に潰れる', p.lengthSq() === 0)
}

console.log()
console.log('='.repeat(70))
console.log('ライト（前照灯・制動灯・方向指示器）')
console.log('='.repeat(70))

function state(over: Partial<Parameters<typeof lightStateFor>[0]>) {
  return lightStateFor({
    braking: false,
    turnSignal: 0,
    hazard: false,
    headlights: false,
    blink: true,
    ...over,
  })
}

{
  const kinds = LIGHT_SLOTS.map((s) => s.kind)
  check(
    '前 2 / 後ろ 2 / ウインカー 4 の 8 灯',
    LIGHTS_PER_VEHICLE === 8 &&
      kinds.filter((k) => k === LIGHT_HEAD).length === 2 &&
      kinds.filter((k) => k === LIGHT_TAIL).length === 2 &&
      kinds.filter((k) => k === LIGHT_TURN).length === 4,
  )
  check(
    '前照灯は車体の前、尾灯は後ろ',
    LIGHT_SLOTS.filter((s) => s.kind === LIGHT_HEAD).every((s) => s.position[0] > 0) &&
      LIGHT_SLOTS.filter((s) => s.kind === LIGHT_TAIL).every((s) => s.position[0] < 0),
  )
  check(
    'ウインカーは車体の左右いちばん外側',
    LIGHT_SLOTS.filter((s) => s.kind === LIGHT_TURN).every(
      (s) => Math.abs(s.position[2]) > 0.8,
    ),
  )
  check(
    'side の符号と取り付け位置の左右が一致する（右が +Z）',
    LIGHT_SLOTS.every((s) => Math.sign(s.position[2]) === s.side),
  )
  check('すべての灯体が地面より上', LIGHT_SLOTS.every((s) => s.position[1] > 0))
}

{
  // ★ 左のウインカーが右側に出ると、外から見て曲がる向きが逆になる
  const scratch = createTransformScratch()
  const base = new THREE.Matrix4()
  const out = new THREE.Matrix4()
  // ENU の東（heading 0）を向いた車。three では -Z が北＝進行方向の左
  composeVehicleMatrix(scratch, 0, 0, 0, base)
  const left = LIGHT_SLOTS.findIndex((s) => s.kind === LIGHT_TURN && s.side === -1)
  composeLightMatrix(scratch, base, LIGHT_SLOTS[left].position, out)
  const p = new THREE.Vector3().setFromMatrixPosition(out)
  check(
    '東を向いた車の左ウインカーは北側（three の -Z）に出る',
    p.z < 0,
    `z=${p.z.toFixed(2)}`,
  )
}

{
  check('晴れの昼は前照灯を点けない', !headlightsOn(0, 0, false))
  check('夜は天候によらず点ける', headlightsOn(0, 0, true))
  check(`小雨（${HEADLIGHT_RAIN} 以上）で点ける`, headlightsOn(0.35, 0, false))
  check(`霧（${HEADLIGHT_FOG} 以上）で点ける`, headlightsOn(0, 0.75, false))
  check('ごく弱い雨では点けない', !headlightsOn(0.1, 0, false))
}

{
  const brake = state({ braking: true })
  check('制動灯はブレーキ指令だけで点く（前照灯とは独立）', brake.brake && !brake.head)

  const night = state({ headlights: true })
  check('前照灯を点けると尾灯も点く', night.head && night.tail && !night.brake)

  const tailIndex = LIGHT_SLOTS.findIndex((s) => s.kind === LIGHT_TAIL)
  check(
    '尾灯は制動灯より暗い（ブレーキが見分けられる）',
    lightIntensity(tailIndex, night) < lightIntensity(tailIndex, state({ braking: true })),
    `${lightIntensity(tailIndex, night)} < ${lightIntensity(tailIndex, state({ braking: true }))}`,
  )

  const left = state({ turnSignal: -1 })
  check('左ウインカーは左だけ点く', left.left && !left.right)
  const right = state({ turnSignal: 1 })
  check('右ウインカーは右だけ点く', right.right && !right.left)
  const off = state({ turnSignal: -1, blink: false })
  check('点滅の消灯位相では消える', !off.left && !off.right)

  // ★ 乗降を待っている間はハザード。方向指示器より優先する
  const hazard = state({ hazard: true, turnSignal: 1 })
  check('ハザードは左右同時に点く（方向指示器より優先）', hazard.left && hazard.right)
}

{
  // 保安基準は毎分 60〜120 回（1〜2Hz）
  check(`点滅は毎分 ${BLINK_HZ * 60} 回で 60〜120 回に収まる`, BLINK_HZ >= 1 && BLINK_HZ <= 2)
  const period = 1000 / BLINK_HZ
  check(
    '点滅のデューティは半分',
    blinkOn(0) && !blinkOn(period * 0.75) && blinkOn(period),
  )
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
