/** 車両の姿勢・部品の位置・ライト・ナンバープレートを検証する（ブラウザ不要）。 */

import * as THREE from 'three'
import {
  BODY_OFFSET,
  CABIN_OFFSET,
  COLUMN_TILT,
  DASH_REAR_X,
  DRIVER_SEAT_Z,
  GAUGE_SLOTS,
  NEEDLE_START,
  NEEDLE_SWEEP,
  NOSE_OFFSET,
  PASSENGER_SEAT_Z,
  PEDAL_SLOTS,
  PEDAL_TRAVEL,
  STEERING_CENTER,
  STEERING_RADIUS,
  STEERING_RATIO,
  VEHICLE_HEIGHT,
  VEHICLE_LENGTH,
  VEHICLE_WIDTH,
  WHEEL_OFFSETS,
  WHEEL_RADIUS,
  composeFixedMatrix,
  composeNeedleMatrix,
  composePedalMatrix,
  composeSteeringMatrix,
  composeVehicleMatrix,
  composeWheelMatrix,
  createTransformScratch,
  makeBodyGeometry,
  makeCabinGeometry,
  makeGaugeFaceGeometry,
  makeGlassGeometry,
  makeInteriorGeometry,
  makeNeedleGeometry,
  makeNoseGeometry,
  makePedalGeometry,
  makeSteeringGeometry,
  needleAngle,
  pedalPress,
  steeringAngle,
  composeLightMatrix,
  composePlateMatrix,
  makeWheelGeometry,
} from '../src/scene/vehicleGeometry.ts'
import { DRIVER_EYE_HEIGHT, DRIVER_FORWARD, DRIVER_RIGHT } from '../src/scene/cameraMath.ts'
import {
  DEFAULT_REGION,
  PLATES_PER_VEHICLE,
  PLATE_H,
  PLATE_SLOTS,
  PLATE_W,
  plateRegion,
  plateSerial,
  plateLabel,
  plateTextFor,
  plateUvRow,
  withPlateNames,
} from '../src/scene/licensePlate.ts'
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
console.log('ナンバープレート（日本の中型・自家用）')
console.log('='.repeat(70))

{
  check('中型プレートの実寸は 330 × 165mm', PLATE_W === 0.33 && PLATE_H === 0.165)
  check('前後に 1 枚ずつ', PLATES_PER_VEHICLE === 2)
  check(
    '前のプレートは車体の前、後ろのプレートは後ろ',
    PLATE_SLOTS[0].position[0] > 0 && PLATE_SLOTS[1].position[0] < 0,
  )
  check(
    'どちらも車体の中心線上（左右にずれていない）',
    PLATE_SLOTS.every((p) => p.position[2] === 0),
  )
  check(
    'バンパーの高さに収まる（地面より上・屋根より下）',
    PLATE_SLOTS.every((p) => p.position[1] > 0.2 && p.position[1] < 1),
  )
}

{
  // ★ 後ろのプレートが前を向いていると、真後ろから見て裏面（無地）しか見えない
  const scratch = createTransformScratch()
  const base = new THREE.Matrix4()
  const out = new THREE.Matrix4()
  composeVehicleMatrix(scratch, 0, 0, 0, base)

  const normals = PLATE_SLOTS.map((slot) => {
    composePlateMatrix(scratch, base, slot.position, slot.yaw, out)
    const n = new THREE.Vector3(1, 0, 0).transformDirection(out)
    return n
  })
  check('前のプレートは車の前方を向く', normals[0].x > 0.99, `x=${normals[0].x.toFixed(2)}`)
  check('後ろのプレートは真後ろを向く', normals[1].x < -0.99, `x=${normals[1].x.toFixed(2)}`)
}

{
  // 一連指定番号は上位を中黒で埋め、4 桁のときだけハイフンを挟む
  check('1 桁は中黒 3 つで埋める', plateSerial(0) === '・・・0', plateSerial(0))
  check('車両 #7 は ・・・7', plateSerial(7) === '・・・7', plateSerial(7))
  check('2 桁', plateSerial(12) === '・・12', plateSerial(12))
  check('3 桁', plateSerial(123) === '・123', plateSerial(123))
  check('4 桁だけハイフンが入る', plateSerial(1234) === '12-34', plateSerial(1234))
  check('どの桁でも 4 文字幅に収まる', [0, 9, 42, 700, 8888].every((n) => plateSerial(n).length <= 5))
}

{
  check('銀座は品川ナンバー', plateRegion('ginza') === '品川')
  check('金沢は石川ナンバー', plateRegion('kanazawa') === '石川')
  check('未知のプリセットは既定の地名', plateRegion('unknown') === DEFAULT_REGION)
  check('プリセット未選択でも落ちない', plateRegion(null) === DEFAULT_REGION)

  const text = plateTextFor(3, 'umeda')
  check(
    'プレートの文字が 4 つとも揃う',
    text.region === 'なにわ' && text.classNumber === '300' && text.kana.length === 1 &&
      text.serial === '・・・3',
    `${text.region} ${text.classNumber} ${text.kana} ${text.serial}`,
  )
}

{
  // ★ Canvas は上から 0,1,2… と描くが、テクスチャの v は下から数える
  const count = 8
  const rows = Array.from({ length: count }, (_, i) => plateUvRow(i, count))
  check('段は 0〜count-1 に収まる', rows.every((r) => r >= 0 && r < count))
  check('車両ごとに違う段を使う', new Set(rows).size === count)
  check('車両 #0 はいちばん上の段（v では最後）', rows[0] === count - 1, `row=${rows[0]}`)
}

{
  // ★ 実用モードの画面は、サーバーの文言中の「車両 #N」をプレート表記へ差し替える
  //   （地名の対応表はクライアントにしか無い）。書式は docs/protocol.md 2.10
  check(
    'プレートの 1 行表記',
    plateLabel(0, 'ginza') === '品川 300 さ ・・・0',
    plateLabel(0, 'ginza'),
  )
  const before = '車両 #1 が来られなくなったため、車両 #5 が向かっています'
  const after = withPlateNames(before, 'kanazawa')
  check(
    '文中の「車両 #N」をすべて置き換える',
    after === '石川 300 さ ・・・1 が来られなくなったため、石川 300 さ ・・・5 が向かっています',
    after,
  )
  check(
    '該当が無ければそのまま返す',
    withPlateNames('目的地へ向かっています', 'ginza') === '目的地へ向かっています',
  )
  check(
    'モックの接頭辞が付いていても置き換わる',
    withPlateNames('（モック）車両 #2 が迎えに向かっています', 'ginza').includes('さ ・・・2'),
  )
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
console.log('車体の作り込み（外接寸法・窓・内装）')
console.log('='.repeat(70))

/** ジオメトリの外接箱 */
function boundsOf(g: THREE.BufferGeometry): THREE.Box3 {
  g.computeBoundingBox()
  return g.boundingBox!.clone()
}

{
  // ★ ここが一番大事。外接寸法は backend の `config.VEHICLE_*` と揃っていて、
  //   衝突判定と擬似カメラの検出枠がこれを前提にしている。はみ出すと
  //   「当たっていないのに当たる」「枠から車がはみ出す」が起きる
  const limitX = VEHICLE_LENGTH / 2
  const limitZ = VEHICLE_WIDTH / 2
  const pieces: Array<[string, THREE.BufferGeometry]> = [
    ['車体', makeBodyGeometry()],
    ['ボンネットの飾り', makeNoseGeometry()],
    ['窓ガラス', makeGlassGeometry()],
    ['内装', makeInteriorGeometry()],
  ]
  for (const [name, g] of pieces) {
    const b = boundsOf(g)
    const ok =
      b.min.x >= -limitX - 1e-6 &&
      b.max.x <= limitX + 1e-6 &&
      b.min.y >= -1e-6 &&
      b.max.y <= VEHICLE_HEIGHT + 1e-6 &&
      b.min.z >= -limitZ - 1e-6 &&
      b.max.z <= limitZ + 1e-6
    check(
      name + 'が外接寸法 ' + VEHICLE_LENGTH + '×' + VEHICLE_WIDTH + '×' + VEHICLE_HEIGHT + 'm に収まる',
      ok,
      'X[' + b.min.x.toFixed(2) + ', ' + b.max.x.toFixed(2) + '] Y[' +
        b.min.y.toFixed(2) + ', ' + b.max.y.toFixed(2) + '] Z[' +
        b.min.z.toFixed(2) + ', ' + b.max.z.toFixed(2) + ']',
    )
    g.dispose()
  }
}

{
  const body = makeBodyGeometry()
  const glass = makeGlassGeometry()
  const bb = boundsOf(body)
  const gb = boundsOf(glass)
  check('窓はベルトラインより上にある', gb.min.y > 0.9, '窓の下端 ' + gb.min.y.toFixed(2) + 'm')
  check(
    '窓は車体の幅を越えない',
    gb.max.z <= bb.max.z + 1e-6,
    '窓 ' + gb.max.z.toFixed(2) + 'm / 車体 ' + bb.max.z.toFixed(2) + 'm',
  )
  body.dispose()
  glass.dispose()
}

{
  // 運転席は右（+Z）。`cameraMath` のアイポイントと同じ側でないと、
  // 運転席視点なのに助手席に座ることになる
  check(
    '運転席は右ハンドル側（cameraMath と同じ +Z）',
    Math.sign(DRIVER_SEAT_Z) === Math.sign(DRIVER_RIGHT) && DRIVER_SEAT_Z > 0,
    '座席 ' + DRIVER_SEAT_Z.toFixed(2) + ' / カメラ ' + DRIVER_RIGHT.toFixed(2),
  )
  check('助手席は反対側にある', Math.sign(PASSENGER_SEAT_Z) === -Math.sign(DRIVER_SEAT_Z))
  check(
    'ハンドルは運転席の正面（アイポイントより前）にある',
    Math.abs(STEERING_CENTER[2] - DRIVER_SEAT_Z) < 1e-6 && STEERING_CENTER[0] > DRIVER_FORWARD,
    'ハンドル X=' + STEERING_CENTER[0].toFixed(2) + ' / アイポイント X=' + DRIVER_FORWARD.toFixed(2),
  )
  check(
    'アイポイントはハンドルより上（メーターを見下ろせる）',
    DRIVER_EYE_HEIGHT > STEERING_CENTER[1],
    '目 ' + DRIVER_EYE_HEIGHT.toFixed(2) + 'm / ハンドル ' + STEERING_CENTER[1].toFixed(2) + 'm',
  )
  // ★ ダッシュボードを目へ近づけすぎると、運転席視点の下半分が壁で埋まる。
  //   最初 0.52m に置いて、目の 0.17m 先が壁になり前が見えなくなった
  const legroom = DASH_REAR_X - DRIVER_FORWARD
  check(
    'アイポイントからダッシュボードまで 0.3m 以上ある',
    legroom >= 0.3,
    legroom.toFixed(2) + 'm',
  )
}

console.log()
console.log('='.repeat(70))
console.log('可動部（ハンドル・ペダル・メーター）')
console.log('='.repeat(70))

{
  // 実車のギア比。最大舵角 0.55rad でおよそ 1.5 回転（540 度）になる
  const full = steeringAngle(0.55)
  const turns = full / (Math.PI * 2)
  check(
    '最大舵角でハンドルが ' + turns.toFixed(2) + ' 回転する',
    turns > 1.2 && turns < 1.8,
    ((full * 180) / Math.PI).toFixed(0) + ' 度 / 比 ' + STEERING_RATIO,
  )
  check('舵角 0 でハンドルも 0', Math.abs(steeringAngle(0)) < 1e-12)
  check('左右で符号が反転する', Math.abs(steeringAngle(0.3) + steeringAngle(-0.3)) < 1e-12)
}

{
  composeVehicleMatrix(scratch, 0, 0, 0, base)
  const centre = new THREE.Vector3(STEERING_CENTER[0], STEERING_CENTER[1], STEERING_CENTER[2])

  // ★ 回しても中心が動かないこと。ジオメトリ側を傾けていると回転軸まで傾き、
  //   「斜めに首を振る」動きになる
  let worst = 0
  for (const steer of [-0.5, -0.2, 0, 0.2, 0.5]) {
    composeSteeringMatrix(scratch, base, steer, out)
    const pos = new THREE.Vector3().setFromMatrixPosition(out)
    worst = Math.max(worst, pos.distanceTo(centre))
  }
  check('ハンドルは回しても中心が動かない', worst < 1e-9, '最大 ' + worst.toExponential(1) + 'm')

  const top = new THREE.Vector3(0, STEERING_RADIUS, 0)
  composeSteeringMatrix(scratch, base, 0, out)
  const neutral = top.clone().applyMatrix4(out)
  composeSteeringMatrix(scratch, base, 0.1, out)
  const left = top.clone().applyMatrix4(out)
  check(
    '左へ切るとリムの上端が左（-Z）へ回る',
    left.z < neutral.z - 1e-3,
    'Z ' + neutral.z.toFixed(3) + ' → ' + left.z.toFixed(3),
  )

  composeSteeringMatrix(scratch, base, 0, out)
  const normal = new THREE.Vector3(1, 0, 0).transformDirection(out)
  const tilt = Math.asin(Math.max(-1, Math.min(1, normal.y)))
  check(
    'ハンドル面がコラムの傾き ' + ((COLUMN_TILT * 180) / Math.PI).toFixed(0) + ' 度で寝ている',
    Math.abs(Math.abs(tilt) - COLUMN_TILT) < 1e-6,
    ((tilt * 180) / Math.PI).toFixed(1) + ' 度',
  )
}

{
  check(
    'アクセルは加速指令が正のときだけ踏まれる',
    pedalPress('throttle', 0.8) === 0.8 && pedalPress('throttle', -0.8) === 0,
  )
  check(
    'ブレーキは負のときだけ踏まれる',
    pedalPress('brake', -0.8) === 0.8 && pedalPress('brake', 0.8) === 0,
  )
  check('踏み込みは 0..1 に収まる', pedalPress('throttle', 3) === 1 && pedalPress('brake', -3) === 1)

  composeVehicleMatrix(scratch, 0, 0, 0, base)
  const tip = new THREE.Vector3(0, -0.22, 0)
  for (let k = 0; k < PEDAL_SLOTS.length; k++) {
    const kind = PEDAL_SLOTS[k].kind
    const name = kind === 'throttle' ? 'アクセル' : 'ブレーキ'
    const press = kind === 'throttle' ? 1 : -1
    composePedalMatrix(scratch, base, k, 0, out)
    const rest = tip.clone().applyMatrix4(out)
    composePedalMatrix(scratch, base, k, press, out)
    const down = tip.clone().applyMatrix4(out)
    check(
      name + 'を踏むと踏面が前方（+X）へ回る',
      down.x > rest.x + 1e-3,
      'X ' + rest.x.toFixed(3) + ' → ' + down.x.toFixed(3),
    )
    const pivot = new THREE.Vector3(
      PEDAL_SLOTS[k].pivot[0],
      PEDAL_SLOTS[k].pivot[1],
      PEDAL_SLOTS[k].pivot[2],
    )
    const at = new THREE.Vector3().setFromMatrixPosition(out)
    check(name + 'の支点は動かない', at.distanceTo(pivot) < 1e-9)
    check(
      name + 'は運転席側にある',
      Math.sign(PEDAL_SLOTS[k].pivot[2]) === Math.sign(DRIVER_SEAT_Z),
      'Z=' + PEDAL_SLOTS[k].pivot[2].toFixed(2),
    )
  }
  check(
    '踏み切っても ' + ((PEDAL_TRAVEL * 180) / Math.PI).toFixed(0) + ' 度までしか回らない',
    PEDAL_TRAVEL > 0.1 && PEDAL_TRAVEL < 0.8,
  )
}

{
  check(
    '針は 0 で始点、1 で終点を指す',
    needleAngle(0) === NEEDLE_START &&
      Math.abs(needleAngle(1) - (NEEDLE_START + NEEDLE_SWEEP)) < 1e-12,
  )
  check(
    '針は範囲の外を指さない',
    needleAngle(-1) === NEEDLE_START && needleAngle(2) === NEEDLE_START + NEEDLE_SWEEP,
  )
  check(
    '針の振れ角は 1 回転に満たない',
    NEEDLE_SWEEP > 0 && NEEDLE_SWEEP < Math.PI * 2,
    ((NEEDLE_SWEEP * 180) / Math.PI).toFixed(0) + ' 度',
  )

  composeVehicleMatrix(scratch, 0, 0, 0, base)
  const tipLocal = new THREE.Vector3(0, GAUGE_SLOTS[0].radius * 0.8, 0)
  composeNeedleMatrix(scratch, base, 0, 0, out)
  const slow = tipLocal.clone().applyMatrix4(out)
  composeNeedleMatrix(scratch, base, 0, 1, out)
  const fast = tipLocal.clone().applyMatrix4(out)
  check('針は速度で動く', slow.distanceTo(fast) > 0.05, slow.distanceTo(fast).toFixed(3) + 'm')

  for (let k = 0; k < GAUGE_SLOTS.length; k++) {
    composeFixedMatrix(scratch, base, GAUGE_SLOTS[k].center, out)
    const face = new THREE.Vector3().setFromMatrixPosition(out)
    composeNeedleMatrix(scratch, base, k, 0.5, out)
    const pivot = new THREE.Vector3().setFromMatrixPosition(out)
    check(
      'メーター ' + k + ' の針が文字盤の中心にある',
      face.distanceTo(pivot) < 0.02,
      face.distanceTo(pivot).toFixed(4) + 'm',
    )
  }

  const faceGeom = makeGaugeFaceGeometry(GAUGE_SLOTS[0].radius)
  const fb = boundsOf(faceGeom)
  check(
    '文字盤は薄い板になっている',
    fb.max.x - fb.min.x < 1e-6,
    '厚み ' + (fb.max.x - fb.min.x).toExponential(1) + 'm',
  )
  faceGeom.dispose()
}

{
  const steeringGeom = makeSteeringGeometry()
  const sb = boundsOf(steeringGeom)
  const reach = Math.max(Math.abs(sb.min.y), sb.max.y, Math.abs(sb.min.z), sb.max.z)
  check(
    'ハンドルは回してもキャビンの中に収まる',
    STEERING_CENTER[1] + reach < VEHICLE_HEIGHT &&
      Math.abs(STEERING_CENTER[2]) + reach < VEHICLE_WIDTH / 2,
    '半径 ' + reach.toFixed(3) + 'm',
  )
  steeringGeom.dispose()

  const pedalGeom = makePedalGeometry()
  const pb = boundsOf(pedalGeom)
  check('ペダルは支点が原点にある', Math.abs(pb.max.y) < 1e-6, '上端 ' + pb.max.y.toFixed(4) + 'm')
  pedalGeom.dispose()

  const needleGeom = makeNeedleGeometry(GAUGE_SLOTS[0].radius)
  const nb = boundsOf(needleGeom)
  check('針は根元が原点にある', Math.abs(nb.min.y) < 1e-6, '下端 ' + nb.min.y.toFixed(4) + 'm')
  check('針は文字盤からはみ出さない', nb.max.y <= GAUGE_SLOTS[0].radius + 1e-6)
  needleGeom.dispose()
}

console.log()
console.log('='.repeat(70))
console.log('描画コスト（台数によらず一定であること）')
console.log('='.repeat(70))

{
  /**
   * 1 台あたりの部品と個数。**instancedMesh の数 = ドローコールの数**なので、
   * 部品を増やすほどここが伸びる。
   */
  const parts: Array<[string, () => THREE.BufferGeometry, number]> = [
    ['車体', makeBodyGeometry, 1],
    ['ボンネットの飾り', makeNoseGeometry, 1],
    ['窓ガラス', makeGlassGeometry, 1],
    ['内装', makeInteriorGeometry, 1],
    ['ハンドル', makeSteeringGeometry, 1],
    ['ペダル', makePedalGeometry, PEDAL_SLOTS.length],
    ['文字盤', () => makeGaugeFaceGeometry(GAUGE_SLOTS[0].radius), GAUGE_SLOTS.length],
    ['針', () => makeNeedleGeometry(GAUGE_SLOTS[0].radius), GAUGE_SLOTS.length],
    ['車輪', makeWheelGeometry, WHEEL_OFFSETS.length],
  ]

  let tris = 0
  for (const [, make, n] of parts) {
    const g = make()
    const count = g.index ? g.index.count / 3 : g.attributes.position.count / 3
    tris += count * n
    g.dispose()
  }

  // 金沢は建物 35,607 棟・標識 15,719 本がある。車の作り込みがそれに並ぶと
  // 重いフレームの原因が読めなくなるので、1 台 3,000 三角形を上限にしておく
  check(
    '1 台あたりの三角形が 3,000 以下',
    tris <= 3000,
    Math.round(tris) + ' 三角形（8 台で ' + Math.round(tris * 8) + '）',
  )
  check(
    '部品ごとの instancedMesh が 15 個以下',
    parts.length + 2 <= 15,
    (parts.length + 2) + ' 個（灯体・プレートを含む。台数によらず一定）',
  )
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
