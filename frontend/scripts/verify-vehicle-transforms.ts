/** 車両の姿勢・部品の位置・ライト・ナンバープレートを検証する（ブラウザ不要）。 */

import * as THREE from 'three'
import {
  BELT_Y,
  BODY_HALF_W,
  GLASS_TOP_Y,
  COLUMN_TILT,
  DASH_REAR_X,
  DASH_TOP_Y,
  GAUGE_KINDS,
  GAUGE_SPEED_MAX_KMH,
  NAV_SCREEN,
  NAV_SPAN_MAX_M,
  NAV_SPAN_MIN_M,
  REAR_DOOR_X,
  ROOF_Y,
  TAXI_DOOR_HINGE,
  TAXI_DOOR_OPEN,
  TURN_INDICATOR_SIZE,
  TURN_INDICATOR_SLOTS,
  DRIVER_SEAT_Z,
  GAUGE_SLOTS,
  NEEDLE_START,
  NEEDLE_SWEEP,
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
  composeBodyMatrix,
  composeCaliperMatrix,
  composeFixedMatrix,
  composeNeedleMatrix,
  composePedalMatrix,
  composeSteeringMatrix,
  composeTurnIndicatorMatrix,
  composeVehicleMatrix,
  composeWheelMatrix,
  createTransformScratch,
  gaugeUvRow,
  makeGaugeFaceGeometry,
  makeNeedleGeometry,
  makePedalGeometry,
  makeNavScreenGeometry,
  makePlateGeometry,
  makeSteeringGeometry,
  makeTurnIndicatorGeometry,
  navSpanFor,
  needleAngle,
  POWER_NEEDLE_TAU_S,
  dampNeedle,
  pedalPress,
  powerRatio,
  speedRatio,
  steeringAngle,
  composePlateMatrix,
} from '../src/scene/vehicleGeometry.ts'
import {
  DRIVER_EYE_HEIGHT,
  DRIVER_EYE_LOCAL,
  DRIVER_FORWARD,
  DRIVER_FOV_DEG,
  DRIVER_LOOK_AHEAD,
  DRIVER_LOOK_DROP,
  DRIVER_RIGHT,
  driverEye,
  driverLookAt,
  vehicleLocalToThree,
} from '../src/scene/cameraMath.ts'
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
  HEADLIGHT_POOL,
  HEADLIGHT_RAIN,
  LIGHTS_PER_VEHICLE,
  LIGHT_HEAD,
  LIGHT_SLOTS,
  LIGHT_STOP,
  LIGHT_TAIL,
  LIGHT_TURN,
  blinkOn,
  composeLampMatrix,
  headlightsOn,
  lightIntensity,
  lightStateFor,
  makeHeadlightPoolGeometry,
  makeLampLensGeometry,
} from '../src/scene/vehicleLights.ts'
import {
  makeBodyChromeGeometry,
  makeBodyPaintGeometry,
  makeBodyTrimGeometry,
  makeFarBodyGeometry,
  makeVehicleGlass,
} from '../src/scene/vehicleBody.ts'
import { AC_VENTS, ROOM_MIRROR_BODY_DEPTH, ROOM_MIRROR_RIM, makeInteriorGeometry } from '../src/scene/vehicleInterior.ts'
import {
  DOOR_MIRROR_DEPTH,
  DOOR_MIRROR_HALF,
  MIRRORS_PER_FRAME,
  MIRROR_FACES,
  MIRROR_INTERVAL_SEC,
  REAR_GLASS_CENTER,
  doorMirrorLocal,
  makeMirrorQuadGeometry,
  mirrorCorners,
  mirrorView,
  mirrorsDue,
  type MirrorFace,
} from '../src/scene/mirrorView.ts'
import { orientedBox } from '../src/scene/meshBuilder.ts'
import {
  RIM_RADIUS,
  TYRE_WIDTH,
  makeCaliperGeometry,
  makeFarWheelGeometry,
  makeRimGeometry,
  makeTyreGeometry,
} from '../src/scene/vehicleWheels.ts'
import { ARCH_HALF_SPAN, OUTLINE_LENGTH, archY, surfaceAt } from '../src/scene/vehicleShape.ts'
import {
  MAX_PITCH,
  MAX_ROLL,
  TILT_OVERSHOOT,
  composeTiltMatrix,
  createTiltState,
  lateralAccel,
  resetTilt,
  stepTilt,
  tiltLocalPoint,
  tiltTarget,
} from '../src/scene/vehicleMotion.ts'
import { LOD_ENTER_M, LOD_EXIT_M, LOD_FAR, LOD_NEAR, nearestViewDistance, nextLod } from '../src/scene/vehicleLod.ts'
import {
  DROP_OPACITY,
  WIPER_HI,
  WIPER_INT,
  WIPER_LO,
  WIPER_OFF,
  WIPER_PERIOD,
  WIPER_PIVOTS,
  WIPER_REACH,
  WIPER_SWEEP,
  WINDSHIELD_LENGTH,
  advanceWiper,
  composeWiperLocal,
  createWiperState,
  dropCoverage,
  lastWipeAge,
  makeWiperGeometry,
  strokeAngle,
  windshieldHalfWidth,
  wiperAngle,
  wiperModeFor,
} from '../src/scene/wiper.ts'
import {
  SIGN_ROWS,
  SIGN_ROW_STATUS,
  STATUS_SIGN,
  TAXI_LAMP,
  TAXI_LAMP_OVERHANG,
  TAXI_LAMP_TOP_Y,
  lampGlow,
  makeTaxiSignGeometry,
  signUvRow,
  statusRow,
  taxiSignStatus,
} from '../src/scene/taxiSign.ts'
import { FAR_PARTS, NEAR_PARTS, VEHICLE_PARTS, type PartKey } from '../src/scene/vehicleParts.ts'
import type { TaxiPhase } from '../src/types/protocol.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

/** ジオメトリの外接箱（行列を渡すと掛けた後の箱） */
function boundsOf(g: THREE.BufferGeometry, matrix?: THREE.Matrix4): THREE.Box3 {
  if (!matrix) {
    g.computeBoundingBox()
    return g.boundingBox!.clone()
  }
  const box = new THREE.Box3()
  const p = g.attributes.position
  const v = new THREE.Vector3()
  for (let i = 0; i < p.count; i++) box.expandByPoint(v.set(p.getX(i), p.getY(i), p.getZ(i)).applyMatrix4(matrix))
  return box
}

/** 外接寸法（車両ローカル）の内側か */
function insideBox(b: THREE.Box3, tol = 1e-6): boolean {
  return (
    b.min.x >= -VEHICLE_LENGTH / 2 - tol &&
    b.max.x <= VEHICLE_LENGTH / 2 + tol &&
    b.min.y >= -tol &&
    b.max.y <= VEHICLE_HEIGHT + tol &&
    b.min.z >= -VEHICLE_WIDTH / 2 - tol &&
    b.max.z <= VEHICLE_WIDTH / 2 + tol
  )
}

/** ドアミラーの張り出しを許す範囲（車両ローカル）。筐体・腕・台座がここに入る */
const DOOR_MIRROR_ZONE = { x: [0.8, 1.05], y: [0.94, 1.13], minAbsZ: 0.79 } as const
/** ドアミラーが外接寸法の横からはみ出してよい量 [m]（CLAUDE.md の例外） */
const DOOR_MIRROR_OVERHANG_LIMIT = 0.18

function inDoorMirrorZone(v: THREE.Vector3): boolean {
  const z = DOOR_MIRROR_ZONE
  return v.x >= z.x[0] && v.x <= z.x[1] && v.y >= z.y[0] && v.y <= z.y[1] && Math.abs(v.z) >= z.minAbsZ
}

/** ドアミラーの範囲を除いた外接箱と、ドアミラーが横へはみ出した量の最大 */
function boundsBesideMirrors(g: THREE.BufferGeometry, matrix?: THREE.Matrix4): { box: THREE.Box3; overhang: number } {
  const box = new THREE.Box3()
  let overhang = 0
  const p = g.attributes.position
  const v = new THREE.Vector3()
  for (let i = 0; i < p.count; i++) {
    v.set(p.getX(i), p.getY(i), p.getZ(i))
    if (inDoorMirrorZone(v)) {
      overhang = Math.max(overhang, Math.abs(v.z) - VEHICLE_WIDTH / 2)
      if (matrix) v.applyMatrix4(matrix)
      continue
    }
    if (matrix) v.applyMatrix4(matrix)
    box.expandByPoint(v)
  }
  return { box, overhang }
}

function describeBox(b: THREE.Box3): string {
  return (
    'X[' + b.min.x.toFixed(3) + ', ' + b.max.x.toFixed(3) + '] Y[' + b.min.y.toFixed(3) + ', ' + b.max.y.toFixed(3) +
    '] Z[' + b.min.z.toFixed(3) + ', ' + b.max.z.toFixed(3) + ']'
  )
}

/** 点が車体の外板の面から外へどれだけ出ているか [m]（負なら面の内側）。同じ高さで外形をなぞって最寄りを探す */
function surfaceGap(p: readonly [number, number, number]): number {
  let best = Infinity
  let gap = 0
  const z = Math.abs(p[2])
  for (let k = 0; k <= 4000; k++) {
    const h = surfaceAt((OUTLINE_LENGTH * k) / 4000, p[1])
    const d = Math.hypot(h.point[0] - p[0], h.point[2] - z)
    if (d < best) {
      best = d
      gap = (p[0] - h.point[0]) * h.normal[0] + (z - h.point[2]) * h.normal[2]
    }
  }
  return gap
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

/** 車両ローカル座標の点を、**実際の運転席カメラ**で投影したときの画面 x（-1..1）。 */
function screenX(local: readonly [number, number, number]): number {
  const cam = new THREE.PerspectiveCamera(DRIVER_FOV_DEG, 16 / 9, 0.1, 100)
  const eye = driverEye(0, 0, 0)
  const at = driverLookAt(0, 0, 0)
  cam.position.set(eye.x, eye.y, eye.z)
  cam.up.set(0, 1, 0)
  cam.lookAt(at.x, at.y, at.z)
  cam.updateMatrixWorld(true)
  cam.updateProjectionMatrix()
  const m = createTransformScratch()
  const b = new THREE.Matrix4()
  const o = new THREE.Matrix4()
  composeVehicleMatrix(m, 0, 0, 0, b)
  composeFixedMatrix(m, b, local, o)
  return new THREE.Vector3().setFromMatrixPosition(o).project(cam).x
}

/** 針先（割合 `ratio`）の画面 x。時計回りかどうかはこれで決める */
function needleScreenX(slot: number, ratio: number): number {
  const cam = new THREE.PerspectiveCamera(DRIVER_FOV_DEG, 16 / 9, 0.1, 100)
  const eye = driverEye(0, 0, 0)
  const at = driverLookAt(0, 0, 0)
  cam.position.set(eye.x, eye.y, eye.z)
  cam.up.set(0, 1, 0)
  cam.lookAt(at.x, at.y, at.z)
  cam.updateMatrixWorld(true)
  cam.updateProjectionMatrix()
  const m = createTransformScratch()
  const b = new THREE.Matrix4()
  const o = new THREE.Matrix4()
  composeVehicleMatrix(m, 0, 0, 0, b)
  composeNeedleMatrix(m, b, slot, ratio, o)
  const tip = new THREE.Vector3(0, GAUGE_SLOTS[slot].radius * 0.8, 0).applyMatrix4(o)
  return tip.project(cam).x
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
      // 左の車輪はホイールの面を外へ向けるため半回転させ、転がる向きを戻す
      if (spec.position[2] < 0) mesh.rotation.set(0, Math.PI, -p.roll)
      else mesh.rotation.z = p.roll
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

  // ホイールの面（ジオメトリの +Z）は左右とも車の外側を向く
  for (let k = 0; k < WHEEL_OFFSETS.length; k++) {
    composeWheelMatrix(scratch, base, k, 0, 0, out)
    const face = new THREE.Vector3(0, 0, 1).transformDirection(out)
    const side = Math.sign(WHEEL_OFFSETS[k].position[2])
    check(`車輪 ${k} のホイールの面が外（${side > 0 ? '+Z' : '-Z'}）を向く`, face.z * side > 0.99, `z=${face.z.toFixed(2)}`)
  }
  // 左右で同じ向きに転がる（前へ進むと、上端が前へ動く）
  const tops = WHEEL_OFFSETS.map((_, k) => {
    const a = new THREE.Vector3(0, WHEEL_RADIUS, 0).applyMatrix4(composeWheelMatrix(scratch, base, k, 0, 0, new THREE.Matrix4()))
    const b = new THREE.Vector3(0, WHEEL_RADIUS, 0).applyMatrix4(composeWheelMatrix(scratch, base, k, 0, -0.1, new THREE.Matrix4()))
    return b.x - a.x
  })
  check('4 輪とも前へ転がる（左右で回る向きが揃う）', tops.every((d) => d > 1e-3), tops.map((d) => d.toFixed(3)).join(' / '))
  // キャリパーは転がらず、左右とも同じ「後ろ寄りの上」に付く
  const calipers = WHEEL_OFFSETS.map((_, k) => {
    composeCaliperMatrix(scratch, base, k, 0, Math.PI * 0.78, out)
    return new THREE.Vector3(Math.cos(Math.PI * 0.78) * 0.15, Math.sin(Math.PI * 0.78) * 0.15, -0.03).applyMatrix4(out)
  })
  check(
    'キャリパーは左右とも車輪の後ろ寄りの上にある',
    calipers.every((c, k) => c.x < WHEEL_OFFSETS[k].position[0] && c.y > WHEEL_RADIUS),
    calipers.map((c, k) => `${(c.x - WHEEL_OFFSETS[k].position[0]).toFixed(2)},${(c.y - WHEEL_RADIUS).toFixed(2)}`).join(' / '),
  )
  check(
    'キャリパーはホイールの内側（車体側）にある',
    calipers.every((c, k) => Math.abs(c.z) < Math.abs(WHEEL_OFFSETS[k].position[2])),
  )
}

console.log()
console.log('='.repeat(70))
console.log('車体と車輪のジオメトリ')
console.log('='.repeat(70))

{
  for (const [name, g] of [
    ['近景の車体', makeBodyPaintGeometry()],
    ['遠景の車体', makeFarBodyGeometry()],
  ] as const) {
    g.computeBoundingBox()
    check(`${name}が地面より上にある`, g.boundingBox!.min.y > 0.1, `最下端 ${g.boundingBox!.min.y.toFixed(3)}m`)
    g.dispose()
  }
}

{
  // 転がり面の外形は分割数によらず、半径の cos(π/分割数) 倍〜半径に収まる
  const spans = (v: number, segments: number) =>
    v >= 2 * WHEEL_RADIUS * Math.cos(Math.PI / segments) - 1e-6 && v <= 2 * WHEEL_RADIUS + 1e-6
  const tyre = makeTyreGeometry()
  const tb = boundsOf(tyre)
  const ts = tb.getSize(new THREE.Vector3())
  check(
    'タイヤの軸が車体左右（Z）を向き、幅が 0.24m',
    spans(ts.x, 12) && spans(ts.y, 12) && Math.abs(ts.z - TYRE_WIDTH) < 1e-6,
    `外形 ${ts.x.toFixed(3)} x ${ts.y.toFixed(3)} x ${ts.z.toFixed(3)}m`,
  )
  // リムとスポークはタイヤの幅の中へ収める。はみ出すと車輪が太くなり、外接寸法の検証が落ちる
  const rim = makeRimGeometry()
  const rb = boundsOf(rim)
  check(
    'ホイール（リム・スポーク・ディスク）がタイヤの幅と半径の中に収まる',
    rb.min.z >= -TYRE_WIDTH / 2 && rb.max.z <= TYRE_WIDTH / 2 && rb.max.x <= RIM_RADIUS + 0.01 && rb.max.y <= RIM_RADIUS + 0.01,
    `Z ${rb.min.z.toFixed(3)}〜${rb.max.z.toFixed(3)} / 半径 ${Math.max(rb.max.x, rb.max.y).toFixed(3)}m`,
  )
  const cal = makeCaliperGeometry()
  const cb = boundsOf(cal)
  const cp = cal.attributes.position
  let calR = 0
  for (let i = 0; i < cp.count; i++) calR = Math.max(calR, Math.hypot(cp.getX(i), cp.getY(i)))
  check(
    'キャリパーはホイールの中（リムの内側・スポークより奥）に収まる',
    cb.max.z < 0.07 && calR < RIM_RADIUS - 0.01,
    `Z 〜${cb.max.z.toFixed(3)}m / 半径 ${calR.toFixed(3)}m（リム ${RIM_RADIUS.toFixed(3)}m）`,
  )
  const far = makeFarWheelGeometry()
  const fs = boundsOf(far).getSize(new THREE.Vector3())
  check('遠景の車輪も同じ外形', spans(fs.x, 12) && Math.abs(fs.z - TYRE_WIDTH) < 1e-6, `${fs.x.toFixed(3)} x ${fs.z.toFixed(3)}m`)
  for (const g of [tyre, rim, cal, far]) g.dispose()
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
  // 板の四隅が外接寸法の内側にあり、車体の面から浮きすぎず、面の中にも潜らない
  const plate = makePlateGeometry(PLATE_W, PLATE_H)
  const scratchP = createTransformScratch()
  const id = new THREE.Matrix4()
  const m = new THREE.Matrix4()
  for (const [k, slot] of PLATE_SLOTS.entries()) {
    composePlateMatrix(scratchP, id, slot.position, slot.yaw, m)
    const b = boundsOf(plate, m)
    check(
      `${k === 0 ? '前' : '後ろ'}のプレートが外接寸法の内側にある`,
      Math.abs(b.min.x) <= VEHICLE_LENGTH / 2 && Math.abs(b.max.x) <= VEHICLE_LENGTH / 2,
      `X ${b.min.x.toFixed(3)}〜${b.max.x.toFixed(3)}`,
    )
    const gap = surfaceGap(slot.position)
    check(
      `${k === 0 ? '前' : '後ろ'}のプレートが車体の面のすぐ外（面から 2cm 以内で、面より外）にある`,
      gap > 0 && gap < 0.02,
      `${(gap * 100).toFixed(1)}cm`,
    )
  }
  plate.dispose()
}

{
  // 後ろのプレートが前を向いていると、真後ろから見て裏面（無地）しか見えない
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
  // Canvas は上から 0,1,2… と描くが、テクスチャの v は下から数える
  const count = 8
  const rows = Array.from({ length: count }, (_, i) => plateUvRow(i, count))
  check('段は 0〜count-1 に収まる', rows.every((r) => r >= 0 && r < count))
  check('車両ごとに違う段を使う', new Set(rows).size === count)
  check('車両 #0 はいちばん上の段（v では最後）', rows[0] === count - 1, `row=${rows[0]}`)
}

{
  // 実用モードの画面は、サーバーの文言中の「車両 #N」をプレート表記へ差し替える
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
  const count = (k: number) => kinds.filter((x) => x === k).length
  check(
    '前照灯 2 / 尾灯 2 / 方向指示器 6（前・ミラー・後ろ）/ ハイマウント 1 の 11 灯',
    LIGHTS_PER_VEHICLE === 11 && count(LIGHT_HEAD) === 2 && count(LIGHT_TAIL) === 2 && count(LIGHT_TURN) === 6 && count(LIGHT_STOP) === 1,
    `${count(LIGHT_HEAD)} / ${count(LIGHT_TAIL)} / ${count(LIGHT_TURN)} / ${count(LIGHT_STOP)}`,
  )
  check(
    '前照灯は車体の前、尾灯とハイマウントは後ろ',
    LIGHT_SLOTS.filter((s) => s.kind === LIGHT_HEAD).every((s) => s.position[0] > 0) &&
      LIGHT_SLOTS.filter((s) => s.kind === LIGHT_TAIL || s.kind === LIGHT_STOP).every((s) => s.position[0] < 0),
  )
  // 方向指示器は、同じ端の前照灯・尾灯より外側（ミラーのものは車体のいちばん外側）
  const outboard = LIGHT_SLOTS.filter((s) => s.kind === LIGHT_TURN).every((t) => {
    const partner = LIGHT_SLOTS.find(
      (s) => (s.kind === LIGHT_HEAD || s.kind === LIGHT_TAIL) && s.side === t.side && Math.sign(s.position[0]) === Math.sign(t.position[0]),
    )
    const onMirror = t.position[1] > 0.98
    return onMirror ? Math.abs(t.position[2]) > 0.85 : partner !== undefined && Math.abs(t.position[2]) > Math.abs(partner.position[2])
  })
  check('方向指示器は前照灯・尾灯より外側（ドアミラーのものは車体のいちばん外側）', outboard)
  check(
    'side の符号と取り付け位置の左右が一致する（右が +Z、中央は 0）',
    LIGHT_SLOTS.every((s) => (s.side === 0 ? Math.abs(s.position[2]) < 1e-9 : Math.sign(s.position[2]) === s.side)),
  )
  check('すべての灯体が地面より上', LIGHT_SLOTS.every((s) => s.position[1] > 0))
  // 車体の面の灯火は、レンズの中を面が通る（浮いてもいないし、潜り切ってもいない）。ミラーとハイマウントは除く
  let worstGap = 0
  let sunk = 0
  for (const s of LIGHT_SLOTS) {
    if (s.position[1] > 0.95) continue
    const gap = surfaceGap(s.position)
    worstGap = Math.max(worstGap, Math.abs(gap))
    if (gap + s.size[0] / 2 < 0) sunk++
  }
  check('車体の面の灯火がレンズの厚みの中で面に付いている', worstGap < 0.02 && sunk === 0, `中心と面の差 最大 ${(worstGap * 100).toFixed(1)}cm / 潜り切り ${sunk}`)
  // レンズを伸ばした箱が外接寸法の内側にある
  const lens = makeLampLensGeometry()
  const lb = boundsOf(lens)
  check(
    'レンズの型は寸法 1 の箱（インスタンスの行列で各灯の大きさへ伸ばす）',
    Math.abs(lb.max.x - 0.5) < 1e-6 && Math.abs(lb.min.y + 0.5) < 1e-6 && Math.abs(lb.max.z - 0.5) < 1e-6,
  )
  const idm = new THREE.Matrix4()
  const lm = new THREE.Matrix4()
  let outside = 0
  let mirrorLampOverhang = 0
  for (const s of LIGHT_SLOTS) {
    composeLampMatrix(idm, s, lm)
    const b = boundsOf(lens, lm)
    if (inDoorMirrorZone(new THREE.Vector3(...s.position))) {
      mirrorLampOverhang = Math.max(mirrorLampOverhang, Math.max(Math.abs(b.min.z), Math.abs(b.max.z)) - VEHICLE_WIDTH / 2)
      continue
    }
    if (!insideBox(b)) outside++
  }
  check('灯火のレンズがすべて外接寸法の内側にある（ドアミラーのものを除く）', outside === 0, `${outside} 個がはみ出す`)
  check(
    `ドアミラーのサイドターンランプのはみ出しは ${DOOR_MIRROR_OVERHANG_LIMIT * 100}cm まで`,
    mirrorLampOverhang <= DOOR_MIRROR_OVERHANG_LIMIT,
    `${(mirrorLampOverhang * 100).toFixed(1)}cm`,
  )
  const stop = LIGHT_SLOTS.find((s) => s.kind === LIGHT_STOP)!
  check(
    'ハイマウントストップランプはリアガラスの上端（屋根の縁の下）で、後ろ上を向く',
    stop.position[1] > 1.3 && stop.position[1] < ROOF_Y && stop.facing[0] < 0 && stop.facing[1] > 0,
    `(${stop.position.map((v) => v.toFixed(2)).join(', ')})`,
  )
  const mirrorTurn = LIGHT_SLOTS.filter((s) => s.kind === LIGHT_TURN && s.position[1] > 0.98)
  // 筐体の座標で見て、外側（運転席から遠い側）の側面にあり、筐体の高さと奥行きの中に収まる
  check(
    'サイドターンランプはドアミラーの筐体の外側の側面に付く',
    mirrorTurn.length === 2 &&
      mirrorTurn.every((s) => {
        // 筐体の x は右の筐体では外向き、左の筐体では内向き（左右の筐体は鏡像）
        const l = doorMirrorLocal(s.side, s.position)
        const outward = s.side * l[0]
        return (
          outward >= DOOR_MIRROR_HALF[0] * 0.85 &&
          outward <= DOOR_MIRROR_HALF[0] + 0.01 &&
          Math.abs(l[1]) <= DOOR_MIRROR_HALF[1] &&
          l[2] <= 0 &&
          l[2] >= -DOOR_MIRROR_DEPTH
        )
      }),
  )
  lens.dispose()
}

{
  // 左のウインカーが右側に出ると、外から見て曲がる向きが逆になる
  const scratch = createTransformScratch()
  const base = new THREE.Matrix4()
  const out = new THREE.Matrix4()
  // ENU の東（heading 0）を向いた車。three では -Z が北＝進行方向の左
  composeVehicleMatrix(scratch, 0, 0, 0, base)
  const left = LIGHT_SLOTS.findIndex((s) => s.kind === LIGHT_TURN && s.side === -1)
  composeLampMatrix(base, LIGHT_SLOTS[left], out)
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

  // 乗降を待っている間はハザード。方向指示器より優先する
  const hazard = state({ hazard: true, turnSignal: 1 })
  check('ハザードは左右同時に点く（方向指示器より優先）', hazard.left && hazard.right)

  // ハイマウントは制動灯と同じ指令（braking）だけで点く。尾灯のように常時は点かない
  const stopIndex = LIGHT_SLOTS.findIndex((s) => s.kind === LIGHT_STOP)
  check(
    'ハイマウントはブレーキで点き、前照灯（夜）だけでは点かない',
    lightIntensity(stopIndex, state({ braking: true })) === 1 && lightIntensity(stopIndex, night) === 0,
  )
  // ドアミラーのサイドターンも、同じ側の方向指示器と同時に点く
  const mirrorLeft = LIGHT_SLOTS.findIndex((s) => s.kind === LIGHT_TURN && s.side === -1 && s.position[1] > 0.98)
  check('左のサイドターンは左ウインカーで点く', lightIntensity(mirrorLeft, left) === 1 && lightIntensity(mirrorLeft, right) === 0)
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

{
  // 外接寸法（4.4 × 1.8 × 1.45m）は衝突判定・擬似カメラ・正解ラベルが前提にしている。
  //   はみ出してよいのは、実用モードのタクシーの表示灯（高さだけ）・ドアミラー（横だけ）・車輪（取り付け位置が元から外）だけ
  const idm = new THREE.Matrix4()
  const pieces: Array<[string, THREE.BufferGeometry]> = [
    ['車体（塗装）', makeBodyPaintGeometry()],
    ['車体（黒い樹脂）', makeBodyTrimGeometry()],
    ['車体（メッキ）', makeBodyChromeGeometry()],
    ['窓ガラス', makeVehicleGlass()],
    ['遠景の車体', makeFarBodyGeometry()],
    ['内装', makeInteriorGeometry()],
  ]
  let mirrorOverhang = 0
  for (const [name, g] of pieces) {
    const { box, overhang } = boundsBesideMirrors(g)
    mirrorOverhang = Math.max(mirrorOverhang, overhang)
    check(
      name + 'が外接寸法 ' + VEHICLE_LENGTH + '×' + VEHICLE_WIDTH + '×' + VEHICLE_HEIGHT + 'm に収まる（ドアミラーを除く）',
      insideBox(box),
      describeBox(box),
    )
    g.dispose()
  }
  check(
    `ドアミラーの横へのはみ出しは片側 ${DOOR_MIRROR_OVERHANG_LIMIT * 100}cm まで（外接寸法の例外）`,
    mirrorOverhang > 0.1 && mirrorOverhang <= DOOR_MIRROR_OVERHANG_LIMIT,
    `${(mirrorOverhang * 100).toFixed(1)}cm`,
  )
  const wiper = makeWiperGeometry()
  let wiperWorst = new THREE.Box3()
  for (let k = 0; k < WIPER_PIVOTS.length; k++) {
    for (const a of [0, WIPER_SWEEP / 2, WIPER_SWEEP]) {
      const m = composeWiperLocal(k, a, new THREE.Matrix4())
      wiperWorst = wiperWorst.union(boundsOf(wiper, m))
    }
  }
  check('ワイパー（どの角度でも）が外接寸法に収まる', insideBox(wiperWorst), describeBox(wiperWorst))
  wiper.dispose()
  const sign = makeTaxiSignGeometry()
  const sb = boundsOf(sign, idm)
  check(
    'タクシーの表示（表示灯と表示板）は、高さ以外は外接寸法に収まる',
    insideBox(new THREE.Box3(sb.min, new THREE.Vector3(sb.max.x, Math.min(sb.max.y, VEHICLE_HEIGHT), sb.max.z))),
    describeBox(sb),
  )
  sign.dispose()
}

{
  const body = makeBodyPaintGeometry()
  const glass = makeVehicleGlass()
  const bb = boundsBesideMirrors(body).box
  const gb = boundsOf(glass)
  check('窓はベルトラインより上にある', gb.min.y > 0.9, '窓の下端 ' + gb.min.y.toFixed(2) + 'm')
  check(
    '窓は車体の幅を越えない',
    gb.max.z <= bb.max.z + 1e-6,
    '窓 ' + gb.max.z.toFixed(2) + 'm / 車体 ' + bb.max.z.toFixed(2) + 'm',
  )
  check('車体の半幅はドアの面で ' + BODY_HALF_W + 'm（フェンダーだけ外接寸法の手前まで張り出す）', bb.max.z <= VEHICLE_WIDTH / 2 && bb.max.z > BODY_HALF_W)
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
  // ダッシュボードを目へ近づけすぎると、運転席視点の下半分が壁で埋まる。
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

  // 回しても中心が動かないこと。ジオメトリ側を傾けていると回転軸まで傾き、
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
    Math.abs(NEEDLE_SWEEP) > 0 && Math.abs(NEEDLE_SWEEP) < Math.PI * 2,
    Math.abs((NEEDLE_SWEEP * 180) / Math.PI).toFixed(0) + ' 度',
  )
  // 運転者から見て時計回りに振れること。**符号を手で決めない**（`screenX`）
  const nx0 = needleScreenX(0, 0)
  const nxHalf = needleScreenX(0, 0.5)
  const nx1 = needleScreenX(0, 1)
  const dialX = screenX(GAUGE_SLOTS[0].center)
  check(
    '針は 0 のとき文字盤の左寄りを指す',
    nx0 < dialX,
    '画面 x=' + nx0.toFixed(3) + '（文字盤 ' + dialX.toFixed(3) + '）',
  )
  check(
    '針は 0.5 のとき真上（文字盤の中心の真上）を指す',
    Math.abs(nxHalf - dialX) < 0.01,
    '画面 x=' + nxHalf.toFixed(3) + ' / 文字盤 ' + dialX.toFixed(3),
  )
  check(
    '速度が上がると針が時計回り（左 → 上 → 右）へ回る',
    nx0 < nxHalf && nxHalf < nx1,
    nx0.toFixed(3) + ' → ' + nxHalf.toFixed(3) + ' → ' + nx1.toFixed(3),
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
console.log('運転席から室内が見えるか（画角と遮蔽）')
console.log('='.repeat(70))

{
  /** 運転席カメラの画角に入っているか（入らなければ室内を作り込んでも映らない）。 */
  const eyeX = DRIVER_FORWARD
  const eyeY = DRIVER_EYE_HEIGHT
  /** 視線の俯角 [度]（下向きが正） */
  const lookDown = (Math.atan2(DRIVER_LOOK_DROP, DRIVER_LOOK_AHEAD) * 180) / Math.PI
  /** 画角の下端 [度] */
  const bottom = lookDown + DRIVER_FOV_DEG / 2
  const downTo = (x: number, y: number) => (Math.atan2(eyeY - y, x - eyeX) * 180) / Math.PI

  const rimDrop = STEERING_RADIUS * Math.cos(COLUMN_TILT)
  const rimShift = STEERING_RADIUS * Math.sin(COLUMN_TILT)

  const seen: Array<[string, number, number]> = [
    ['ハンドルの上端', STEERING_CENTER[0] - rimShift, STEERING_CENTER[1] + rimDrop],
    ['ハンドルの下端', STEERING_CENTER[0] + rimShift, STEERING_CENTER[1] - rimDrop],
    ['速度計', GAUGE_SLOTS[0].center[0], GAUGE_SLOTS[0].center[1]],
    ['回転計', GAUGE_SLOTS[1].center[0], GAUGE_SLOTS[1].center[1]],
    ['アクセルの支点', PEDAL_SLOTS[0].pivot[0], PEDAL_SLOTS[0].pivot[1]],
    ['アクセルの踏面', PEDAL_SLOTS[0].pivot[0], PEDAL_SLOTS[0].pivot[1] - 0.22],
    ['ブレーキの支点', PEDAL_SLOTS[1].pivot[0], PEDAL_SLOTS[1].pivot[1]],
    ['ブレーキの踏面', PEDAL_SLOTS[1].pivot[0], PEDAL_SLOTS[1].pivot[1] - 0.22],
  ]

  for (const [name, x, y] of seen) {
    const d = downTo(x, y)
    check(
      name + 'が運転席の画角に入る',
      d <= bottom,
      '俯角 ' + d.toFixed(1) + ' 度 / 下端 ' + bottom.toFixed(1) + ' 度',
    )
  }

  // メーターはハンドルの「リングの中」から覗く。ハブに重なると隠れる
  //    （実際にハブとちょうど同じ高さにあって見えなかった）
  const hubRadius = 0.052
  for (let k = 0; k < GAUGE_SLOTS.length; k++) {
    const g = GAUGE_SLOTS[k]
    // 目からメーターへの視線が、ハンドル面（STEERING_CENTER[0]）を横切る高さ
    const t = (STEERING_CENTER[0] - eyeX) / (g.center[0] - eyeX)
    const crossY = eyeY + (g.center[1] - eyeY) * t
    const offset = Math.abs(crossY - STEERING_CENTER[1])
    check(
      'メーター ' + k + ' への視線がハンドルのハブを外れる',
      offset > hubRadius,
      'ハンドル中心から ' + offset.toFixed(3) + 'm（ハブ半径 ' + hubRadius + 'm）',
    )
    check(
      'メーター ' + k + ' への視線がリングの内側を通る',
      offset < STEERING_RADIUS,
      offset.toFixed(3) + 'm < リム半径 ' + STEERING_RADIUS + 'm',
    )
  }

  // 前が見えなくなるほど下を向いていないこと
  check(
    '視線より上も同じだけ見える（空が見える）',
    DRIVER_FOV_DEG / 2 - lookDown > 30,
    '上端 ' + (DRIVER_FOV_DEG / 2 - lookDown).toFixed(1) + ' 度',
  )
}

console.log()
console.log('='.repeat(70))
console.log('メーター（速度計とパワーメーター）')
console.log('='.repeat(70))

{
  // EV なので回転計は持たない
  check(
    'メーターは速度計とパワーメーターの 2 つ',
    GAUGE_KINDS.length === 2 && GAUGE_KINDS[0] === 'speed' && GAUGE_KINDS[1] === 'power',
    GAUGE_KINDS.join(' / '),
  )
  check(
    '文字盤の並びが GAUGE_SLOTS と一致する',
    GAUGE_SLOTS.every((g, i) => g.kind === GAUGE_KINDS[i]),
  )

  {
    const n = GAUGE_KINDS.length
    const rows = GAUGE_KINDS.map((_, i) => gaugeUvRow(i, n))
    check('文字盤の段は 0〜種類数-1 に収まる', rows.every((r) => r >= 0 && r < n))
    check('種類ごとに違う段を使う', new Set(rows).size === n)
    check(
      'いちばん上に描いた speed が v では最後の段',
      gaugeUvRow(GAUGE_KINDS.indexOf('speed'), n) === n - 1,
      'row=' + gaugeUvRow(GAUGE_KINDS.indexOf('speed'), n),
    )
  }

  // 速度計。目盛りは固定なので、設定できる最高速度（40 m/s = 144 km/h）を覆うこと
  check(
    '速度計の目盛りが maxSpeed の上限（144 km/h）を覆う',
    GAUGE_SPEED_MAX_KMH >= 144,
    '0〜' + GAUGE_SPEED_MAX_KMH + ' km/h',
  )
  check('停車で針が 0 を指す', speedRatio(0) === 0)
  // 目盛りは固定なので、既定の最高速度でどれだけ振れるかは選んだ上限で決まる。
  // 読み取れる範囲（目盛りの 1/4 以上）に入っていること
  check(
    '既定の最高速度（13.9 m/s ≒ 50 km/h）で針が読める範囲まで振れる',
    speedRatio(13.9) >= 0.25,
    (speedRatio(13.9) * 100).toFixed(1) + '%（上限 ' + GAUGE_SPEED_MAX_KMH + ' km/h）',
  )
  check('目盛りを超えても振り切れない', speedRatio(100) === 1)
  check(
    '速度が上がるほど針が進む',
    speedRatio(5) < speedRatio(10) && speedRatio(10) < speedRatio(20),
  )

  // パワーメーター。中央が 0 で、回生（負）と出力（正）に振れる
  check(
    'アクセルもブレーキも踏んでいないとき、針が中央を指す',
    Math.abs(powerRatio(0) - 0.5) < 1e-12,
    powerRatio(0).toFixed(3),
  )
  check('目いっぱいの出力で右いっぱい', powerRatio(1) === 1)
  check('目いっぱいの回生で左いっぱい', powerRatio(-1) === 0)
  check(
    '出力と回生が中央に対して対称',
    Math.abs(powerRatio(0.4) - 0.5 - (0.5 - powerRatio(-0.4))) < 1e-12,
  )
  check('範囲の外へは振れない', powerRatio(3) === 1 && powerRatio(-3) === 0)

  // POWER の針の慣性。経路追従や方策の指令は 20Hz の段差で届き、0 と 1 の間を行き来する
  check(
    'POWER の針は時定数ぶんで 63% 寄る',
    Math.abs(dampNeedle(0, 1, POWER_NEEDLE_TAU_S, POWER_NEEDLE_TAU_S) - (1 - Math.exp(-1))) < 1e-12,
  )
  check(
    'POWER の針は間隔が長くても目標を行き過ぎない',
    dampNeedle(0.5, 1, 10, POWER_NEEDLE_TAU_S) <= 1 &&
      dampNeedle(0.5, 0, 10, POWER_NEEDLE_TAU_S) >= 0 &&
      dampNeedle(0.5, 1, 0, POWER_NEEDLE_TAU_S) === 0.5,
  )
  {
    // 0 と 1 を 50ms ごとに行き来する指令（振り切れる指令の最悪の形）を 60fps で追わせる
    let needle = powerRatio(0)
    let maxStep = 0
    for (let frame = 1; frame <= 180; frame++) {
      const target = powerRatio(Math.floor(frame / 3) % 2 === 0 ? 0 : 1)
      const next = dampNeedle(needle, target, 1 / 60, POWER_NEEDLE_TAU_S)
      maxStep = Math.max(maxStep, Math.abs(next - needle))
      needle = next
    }
    check(
      'POWER の針は振り切れる指令でも 1 フレームで目盛りの 1 割以上跳ばない',
      maxStep < 0.1,
      '最大 ' + maxStep.toFixed(3),
    )
    let settled = powerRatio(0)
    for (let frame = 0; frame < 60; frame++) settled = dampNeedle(settled, 1, 1 / 60, POWER_NEEDLE_TAU_S)
    check('POWER の針は 1 秒で指令に追いつく', Math.abs(settled - 1) < 0.01, settled.toFixed(4))
  }

  // 針の角度。中央（0.5）が真上を向くこと
  check(
    'パワーメーターの中央が真上を向く',
    Math.abs(needleAngle(powerRatio(0))) < 1e-12,
    (needleAngle(powerRatio(0)) * 180 / Math.PI).toFixed(1) + ' 度',
  )
  // 出力側（正）は時計回り。**画面へ投影して決める**
  const powerSlot = GAUGE_SLOTS.findIndex((g) => g.kind === 'power')
  const mid = needleScreenX(powerSlot, powerRatio(0))
  check(
    '出力側で針が右（時計回り）へ振れる',
    needleScreenX(powerSlot, powerRatio(0.5)) > mid,
    '画面 x=' + needleScreenX(powerSlot, powerRatio(0.5)).toFixed(3) + '（中央 ' + mid.toFixed(3) + '）',
  )
  check(
    '回生側で針が左へ振れる',
    needleScreenX(powerSlot, powerRatio(-0.5)) < mid,
    '画面 x=' + needleScreenX(powerSlot, powerRatio(-0.5)).toFixed(3) + '（中央 ' + mid.toFixed(3) + '）',
  )
}

console.log()
console.log('='.repeat(70))
console.log('メーターのウインカー表示')
console.log('='.repeat(70))

{
  check('左右 2 つある', TURN_INDICATOR_SLOTS.length === 2)
  check(
    '左が -1・右が +1（frame の turnSignal と同じ符号）',
    TURN_INDICATOR_SLOTS[0].side === -1 && TURN_INDICATOR_SLOTS[1].side === 1,
  )
  // どちらが画面の左かは**投影で決める**（+Z が左か右かを手で書かない）
  const leftX = screenX(TURN_INDICATOR_SLOTS[0].center)
  const rightX = screenX(TURN_INDICATOR_SLOTS[1].center)
  check(
    '左の表示が運転席から見て画面の左にある',
    leftX < rightX,
    '左 x=' + leftX.toFixed(3) + ' / 右 x=' + rightX.toFixed(3),
  )
  check(
    '左右がメーターの中心に対して対称',
    Math.abs(
      TURN_INDICATOR_SLOTS[0].center[2] +
        TURN_INDICATOR_SLOTS[1].center[2] -
        2 * DRIVER_SEAT_Z,
    ) < 1e-9,
  )

  // 2 つのメーターの「間」に収まっていること
  const gapMin = Math.min(GAUGE_SLOTS[0].center[2], GAUGE_SLOTS[1].center[2])
  const gapMax = Math.max(GAUGE_SLOTS[0].center[2], GAUGE_SLOTS[1].center[2])
  for (let k = 0; k < TURN_INDICATOR_SLOTS.length; k++) {
    const z = TURN_INDICATOR_SLOTS[k].center[2]
    check(
      'ウインカー表示 ' + k + ' が 2 つのメーターの間にある',
      z > gapMin && z < gapMax,
      'Z=' + z.toFixed(3) + '（メーター ' + gapMin.toFixed(2) + '〜' + gapMax.toFixed(2) + '）',
    )
  }

  // メーターの円と重ならないこと（間は狭いので、上へ逃がしてある）
  for (let k = 0; k < TURN_INDICATOR_SLOTS.length; k++) {
    const c = TURN_INDICATOR_SLOTS[k].center
    let worst = Infinity
    for (const g of GAUGE_SLOTS) {
      worst = Math.min(worst, Math.hypot(c[1] - g.center[1], c[2] - g.center[2]) - g.radius)
    }
    check(
      'ウインカー表示 ' + k + ' が文字盤の円と重ならない',
      worst > 0,
      '余白 ' + (worst * 1000).toFixed(0) + 'mm',
    )
  }

  // ハンドルのリングの内側から覗く位置なので、**ハブとスポークを外すこと**。
  //   スポークは下・左・右の 3 本なので、視線がハンドル中心より「上」を通れば当たらない
  const eyeX = DRIVER_FORWARD
  const eyeY = DRIVER_EYE_HEIGHT
  const eyeZ = DRIVER_SEAT_Z
  const hubRadius = 0.052
  for (let k = 0; k < TURN_INDICATOR_SLOTS.length; k++) {
    const c = TURN_INDICATOR_SLOTS[k].center
    const t = (STEERING_CENTER[0] - eyeX) / (c[0] - eyeX)
    const crossY = eyeY + (c[1] - eyeY) * t
    const crossZ = eyeZ + (c[2] - eyeZ) * t
    const d = Math.hypot(crossY - STEERING_CENTER[1], crossZ - STEERING_CENTER[2])
    check(
      'ウインカー表示 ' + k + ' への視線がハンドルのハブを外れる',
      d > hubRadius,
      'ハンドル中心から ' + d.toFixed(3) + 'm（ハブ半径 ' + hubRadius + 'm）',
    )
    check(
      'ウインカー表示 ' + k + ' への視線がリングの内側を通る',
      d < STEERING_RADIUS,
      d.toFixed(3) + 'm < リム半径 ' + STEERING_RADIUS + 'm',
    )
    check(
      'ウインカー表示 ' + k + ' への視線がスポークの無い上側を通る',
      crossY > STEERING_CENTER[1],
      '高さ ' + crossY.toFixed(3) + 'm（ハンドル中心 ' + STEERING_CENTER[1].toFixed(3) + 'm）',
    )
  }

  // 矢印の向きも投影で確かめる。左の矢印は画面の左を指すこと
  {
    const cam = new THREE.PerspectiveCamera(DRIVER_FOV_DEG, 16 / 9, 0.1, 100)
    const eye = driverEye(0, 0, 0)
    const at = driverLookAt(0, 0, 0)
    cam.position.set(eye.x, eye.y, eye.z)
    cam.up.set(0, 1, 0)
    cam.lookAt(at.x, at.y, at.z)
    cam.updateMatrixWorld(true)
    cam.updateProjectionMatrix()
    composeVehicleMatrix(scratch, 0, 0, 0, base)
    const tipLocal = new THREE.Vector3(0, 0, TURN_INDICATOR_SIZE)
    for (let k = 0; k < TURN_INDICATOR_SLOTS.length; k++) {
      composeTurnIndicatorMatrix(scratch, base, k, out)
      const tipX = tipLocal.clone().applyMatrix4(out).project(cam).x
      const baseX = screenX(TURN_INDICATOR_SLOTS[k].center)
      const wantLeft = TURN_INDICATOR_SLOTS[k].side < 0
      check(
        (wantLeft ? '左' : '右') + 'の矢印が画面の' + (wantLeft ? '左' : '右') + 'を指す',
        wantLeft ? tipX < baseX : tipX > baseX,
        '先端 x=' + tipX.toFixed(4) + ' / 根元 x=' + baseX.toFixed(4),
      )
    }
  }

  // 点灯は車外の方向指示器と同じ `lightStateFor` から取ること
  const left = lightStateFor({ braking: false, turnSignal: -1, hazard: false, headlights: false, blink: true })
  check('左を出すと左だけ点く', left.left && !left.right)
  const right = lightStateFor({ braking: false, turnSignal: 1, hazard: false, headlights: false, blink: true })
  check('右を出すと右だけ点く', right.right && !right.left)
  const off = lightStateFor({ braking: false, turnSignal: -1, hazard: false, headlights: false, blink: false })
  check('点滅の消えている位相では消灯する', !off.left && !off.right)
  const hazard = lightStateFor({ braking: false, turnSignal: 0, hazard: true, headlights: false, blink: true })
  check('ハザードで左右とも点く', hazard.left && hazard.right)

  const geom = makeTurnIndicatorGeometry(TURN_INDICATOR_SIZE)
  geom.computeBoundingBox()
  const b = geom.boundingBox!
  check(
    '矢印が板（厚みを持たない）',
    Math.abs(b.max.x - b.min.x) < 1e-9,
    '厚み ' + (b.max.x - b.min.x).toExponential(1) + 'm',
  )
  geom.dispose()
}

console.log()
console.log('='.repeat(70))
console.log('カーナビ（インパネ中央）')
console.log('='.repeat(70))

{
  const eyeX = DRIVER_FORWARD
  const eyeY = DRIVER_EYE_HEIGHT
  const eyeZ = DRIVER_SEAT_Z
  const c = NAV_SCREEN.center

  check('画面は車体の中央（インパネ中央部）にある', Math.abs(c[2]) < 1e-9, 'Z=' + c[2].toFixed(2))
  check(
    '画面はメーターより下にある',
    c[1] < GAUGE_SLOTS[0].center[1],
    'ナビ ' + c[1].toFixed(2) + 'm / メーター ' + GAUGE_SLOTS[0].center[1].toFixed(2) + 'm',
  )

  // 運転席の画角に入るか（垂直だけ見る。横は画面のほうが広い）
  const down = (Math.atan2(eyeY - c[1], c[0] - eyeX) * 180) / Math.PI
  const lookDown = (Math.atan2(DRIVER_LOOK_DROP, DRIVER_LOOK_AHEAD) * 180) / Math.PI
  check(
    'ナビが運転席の画角に入る',
    down <= lookDown + DRIVER_FOV_DEG / 2,
    '俯角 ' + down.toFixed(1) + ' 度 / 下端 ' + (lookDown + DRIVER_FOV_DEG / 2).toFixed(1) + ' 度',
  )

  // ハンドルのリムに隠れないこと（中央にあるので、リムの外を通るはず）
  const t = (STEERING_CENTER[0] - eyeX) / (c[0] - eyeX)
  const crossY = eyeY + (c[1] - eyeY) * t
  const crossZ = eyeZ + (c[2] - eyeZ) * t
  const d = Math.hypot(crossY - STEERING_CENTER[1], crossZ - STEERING_CENTER[2])
  check(
    'ナビへの視線がハンドルのリムの外を通る',
    d > STEERING_RADIUS,
    'ハンドル中心から ' + d.toFixed(3) + 'm（リム半径 ' + STEERING_RADIUS + 'm）',
  )

  // 画面がダッシュボードに収まること
  check(
    'ナビの画面がダッシュボードの高さに収まる',
    c[1] - NAV_SCREEN.height / 2 > 0.7 && c[1] + NAV_SCREEN.height / 2 < DASH_TOP_Y,
    'Y ' + (c[1] - NAV_SCREEN.height / 2).toFixed(2) + '〜' + (c[1] + NAV_SCREEN.height / 2).toFixed(2),
  )

  // 縮尺。**停車で寄り、速度が上がるほど引く**
  const maxSpeed = 13.9
  check('停車でいちばん寄る', navSpanFor(0, maxSpeed) === NAV_SPAN_MIN_M, navSpanFor(0, maxSpeed) + 'm')
  check(
    '最高速でいちばん引く',
    navSpanFor(maxSpeed, maxSpeed) === NAV_SPAN_MAX_M,
    navSpanFor(maxSpeed, maxSpeed) + 'm',
  )
  check(
    '速度が上がるほど広く見える',
    navSpanFor(3, maxSpeed) < navSpanFor(8, maxSpeed) && navSpanFor(8, maxSpeed) < navSpanFor(13, maxSpeed),
    [3, 8, 13].map((v) => navSpanFor(v, maxSpeed).toFixed(0) + 'm').join(' → '),
  )
  check(
    '最高速を超えても引きすぎない',
    navSpanFor(100, maxSpeed) === NAV_SPAN_MAX_M,
  )
  check(
    '引く幅が寄る幅より広い（実車のナビと同じ向き）',
    NAV_SPAN_MAX_M > NAV_SPAN_MIN_M,
    NAV_SPAN_MIN_M + 'm 〜 ' + NAV_SPAN_MAX_M + 'm',
  )

  const geom = makeNavScreenGeometry(NAV_SCREEN.width, NAV_SCREEN.height)
  geom.computeBoundingBox()
  const b = geom.boundingBox!
  check(
    'ナビの画面が板（厚みを持たない）',
    Math.abs(b.max.x - b.min.x) < 1e-6,
    '厚み ' + (b.max.x - b.min.x).toExponential(1) + 'm',
  )
  check(
    'ナビの画面が横長',
    b.max.z - b.min.z > b.max.y - b.min.y,
    (b.max.z - b.min.z).toFixed(2) + 'm x ' + (b.max.y - b.min.y).toFixed(2) + 'm',
  )
  geom.dispose()
}

console.log()
console.log('='.repeat(70))
console.log('面の重なり（Z ファイティングの種）')
console.log('='.repeat(70))

interface PlaneTri {
  readonly p: readonly [THREE.Vector3, THREE.Vector3, THREE.Vector3]
  /** 向きを揃えた法線（最も大きい成分が正）と、元の向きがそれと同じか */
  readonly n: THREE.Vector3
  readonly same: boolean
  readonly d: number
}

function planeTrisOf(g: THREE.BufferGeometry, matrix?: THREE.Matrix4): PlaneTri[] {
  const src = g.index ? g.toNonIndexed() : g
  const pos = src.attributes.position
  const out: PlaneTri[] = []
  for (let i = 0; i + 2 < pos.count; i += 3) {
    const p = [0, 1, 2].map((k) => {
      const v = new THREE.Vector3(pos.getX(i + k), pos.getY(i + k), pos.getZ(i + k))
      return matrix ? v.applyMatrix4(matrix) : v
    }) as [THREE.Vector3, THREE.Vector3, THREE.Vector3]
    const n = new THREE.Vector3().subVectors(p[1], p[0]).cross(new THREE.Vector3().subVectors(p[2], p[0]))
    const area = n.length() / 2
    if (area < 1e-7) continue
    n.normalize()
    const ax = Math.abs(n.x) >= Math.abs(n.y) && Math.abs(n.x) >= Math.abs(n.z) ? n.x : Math.abs(n.y) >= Math.abs(n.z) ? n.y : n.z
    const same = ax > 0
    if (!same) n.negate()
    out.push({ p, n, same, d: n.dot(p[0]) })
  }
  if (src !== g) src.dispose()
  return out
}

/** 同じ平面（向き 0.8 度以内・互いの頂点が相手の平面から 0.5mm 以内）の上で、面積が重なる三角形の組を数える */
function coplanarOverlaps(a: PlaneTri[], b: PlaneTri[] | null, doubleSided: boolean): { count: number; sample: string[] } {
  const key = (t: PlaneTri, dShift: number) =>
    `${Math.round(t.n.x * 40)},${Math.round(t.n.y * 40)},${Math.round(t.n.z * 40)},${Math.round(t.d / 0.02) + dShift}`
  const target = b ?? a
  const buckets = new Map<string, number[]>()
  target.forEach((t, i) => {
    const k = key(t, 0)
    const list = buckets.get(k)
    if (list) list.push(i)
    else buckets.set(k, [i])
  })
  let count = 0
  const sample: string[] = []
  const shrink = (tri: THREE.Vector2[]) => {
    const c = tri[0].clone().add(tri[1]).add(tri[2]).multiplyScalar(1 / 3)
    return tri.map((v) => {
      const d = v.clone().sub(c)
      const l = d.length()
      return l < 0.0015 ? c.clone() : c.clone().add(d.multiplyScalar((l - 0.001) / l))
    })
  }
  const separated = (t1: THREE.Vector2[], t2: THREE.Vector2[]) => {
    for (const tri of [t1, t2]) {
      for (let e = 0; e < 3; e++) {
        const p0 = tri[e]
        const p1 = tri[(e + 1) % 3]
        const axis = new THREE.Vector2(-(p1.y - p0.y), p1.x - p0.x)
        const a1 = t1.map((v) => v.dot(axis))
        const a2 = t2.map((v) => v.dot(axis))
        if (Math.max(...a1) <= Math.min(...a2) + 1e-12 || Math.max(...a2) <= Math.min(...a1) + 1e-12) return true
      }
    }
    return false
  }
  a.forEach((t, i) => {
    for (const dShift of [-1, 0, 1]) {
      const list = buckets.get(key(t, dShift))
      if (!list) continue
      for (const j of list) {
        if (b === null && j <= i) continue
        const u = target[j]
        if (t.n.dot(u.n) < 0.9999 || Math.abs(t.d - u.d) > 0.02) continue
        // 平面の距離は原点から測ると傾きの差で大きくぶれるので、相手の頂点までの距離で見る
        const off = (x: PlaneTri, y: PlaneTri) => Math.max(...y.p.map((v) => Math.abs(x.n.dot(v) - x.d)))
        if (off(t, u) > 0.0005 || off(u, t) > 0.0005) continue
        if (t.same !== u.same && !doubleSided) continue
        const ex = Math.abs(t.n.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)
        const bu = ex.cross(t.n).normalize()
        const bv = t.n.clone().cross(bu)
        const flat = (tri: readonly THREE.Vector3[]) => tri.map((v) => new THREE.Vector2(v.dot(bu), v.dot(bv)))
        if (separated(shrink(flat(t.p)), shrink(flat(u.p)))) continue
        count++
        if (sample.length < 4) sample.push(`(${t.p[0].x.toFixed(3)}, ${t.p[0].y.toFixed(3)}, ${t.p[0].z.toFixed(3)})`)
      }
    }
  })
  return { count, sample }
}

{
  // 窓ガラスは transparent + depthWrite: false で描くので深度を書かない。比べるのは不透明どうし。
  //   内装は両面（DoubleSide）なので、背中合わせの面でもちらつく
  const paint = planeTrisOf(makeBodyPaintGeometry())
  const trim = planeTrisOf(makeBodyTrimGeometry())
  const chrome = planeTrisOf(makeBodyChromeGeometry())
  const interior = planeTrisOf(makeInteriorGeometry())
  const far = planeTrisOf(makeFarBodyGeometry())
  const sign = planeTrisOf(makeTaxiSignGeometry())
  const pairs: Array<[string, PlaneTri[], PlaneTri[] | null, boolean]> = [
    ['車体（塗装）どうし', paint, null, false],
    ['黒い樹脂どうし', trim, null, false],
    ['メッキどうし', chrome, null, false],
    ['内装どうし', interior, null, true],
    ['遠景の車体どうし', far, null, false],
    ['塗装と黒い樹脂', paint, trim, false],
    ['塗装とメッキ', paint, chrome, false],
    ['黒い樹脂とメッキ', trim, chrome, false],
    ['車体（塗装）と内装', paint, interior, true],
    ['黒い樹脂と内装', trim, interior, true],
    ['メッキと内装', chrome, interior, true],
    ['タクシーの表示と塗装', sign, paint, false],
    ['タクシーの表示と内装', sign, interior, true],
  ]
  for (const [name, a, b, both] of pairs) {
    const r = coplanarOverlaps(a, b, both)
    check(`${name}が同じ平面で重ならない`, r.count === 0, r.count === 0 ? '' : `${r.count} 組: ${r.sample.join(' / ')}`)
  }
}

console.log()
console.log('='.repeat(70))
console.log('描画の予算（近景・遠景・部品の数）')
console.log('='.repeat(70))

function trianglesOf(g: THREE.BufferGeometry): number {
  return (g.index ? g.index.count : g.attributes.position.count) / 3
}

const PART_GEOMETRY: Record<PartKey, () => THREE.BufferGeometry> = {
  bodyNear: makeBodyPaintGeometry,
  trim: makeBodyTrimGeometry,
  chrome: makeBodyChromeGeometry,
  interior: makeInteriorGeometry,
  steering: makeSteeringGeometry,
  pedal: makePedalGeometry,
  gauge: () => makeGaugeFaceGeometry(GAUGE_SLOTS[0].radius),
  needle: () => makeNeedleGeometry(GAUGE_SLOTS[0].radius),
  indicator: () => makeTurnIndicatorGeometry(TURN_INDICATOR_SIZE),
  tyre: makeTyreGeometry,
  rim: makeRimGeometry,
  caliper: makeCaliperGeometry,
  wiper: makeWiperGeometry,
  bodyFar: makeFarBodyGeometry,
  wheelFar: makeFarWheelGeometry,
  glass: makeVehicleGlass,
  lamp: makeLampLensGeometry,
  plate: () => makePlateGeometry(PLATE_W, PLATE_H),
  sign: makeTaxiSignGeometry,
  pool: makeHeadlightPoolGeometry,
}

/** 1 台あたりの三角形の上限（近景・遠景）と、車両の instancedMesh の上限 */
const NEAR_TRIS_LIMIT = 16000
const FAR_TRIS_LIMIT = 2500
const PARTS_LIMIT = 20

{
  const partTris = new Map<PartKey, number>()
  for (const p of VEHICLE_PARTS) {
    const g = PART_GEOMETRY[p.key]()
    partTris.set(p.key, trianglesOf(g) * p.per)
    g.dispose()
  }
  const sum = (parts: ReadonlyArray<{ key: PartKey }>) => parts.reduce((a, p) => a + (partTris.get(p.key) ?? 0), 0)
  const shared = VEHICLE_PARTS.filter((p) => p.tier === 'both')
  const near = sum(NEAR_PARTS) + sum(shared)
  const far = sum(FAR_PARTS) + sum(shared)
  console.log('        部品ごと（1 台ぶん）: ' + VEHICLE_PARTS.map((p) => `${p.key} ${partTris.get(p.key)}`).join(' / '))
  check(
    `近景の 1 台が ${NEAR_TRIS_LIMIT.toLocaleString()} 三角形以下`,
    near <= NEAR_TRIS_LIMIT,
    `${near.toLocaleString()} 三角形（うち共通 ${sum(shared).toLocaleString()}）`,
  )
  check(
    `遠景の 1 台が ${FAR_TRIS_LIMIT.toLocaleString()} 三角形以下（作り込む前の 1 台 1,960 より少ない）`,
    far <= FAR_TRIS_LIMIT && far < 1960,
    `${far.toLocaleString()} 三角形（8 台で ${(far * 8).toLocaleString()}）`,
  )
  check(
    `車両の instancedMesh が ${PARTS_LIMIT} 個以下（近景 ${NEAR_PARTS.length} / 遠景 ${FAR_PARTS.length} / 共通 ${shared.length}）`,
    VEHICLE_PARTS.length <= PARTS_LIMIT,
    `${VEHICLE_PARTS.length} 個。近景の車が 1 台も無ければ近景の ${NEAR_PARTS.length} 個は描かない`,
  )
  check(
    '近景・遠景の片方にしか出ない部品は、もう片方の段に同じ役の部品がある（車体・車輪）',
    NEAR_PARTS.some((p) => p.key === 'bodyNear') &&
      FAR_PARTS.some((p) => p.key === 'bodyFar') &&
      NEAR_PARTS.some((p) => p.key === 'tyre') &&
      FAR_PARTS.some((p) => p.key === 'wheelFar'),
  )
}

console.log()
console.log('='.repeat(70))
console.log('近景と遠景の切り替え（LOD）')
console.log('='.repeat(70))

{
  check('近景に入る距離より出る距離が遠い（境目で行き来しない）', LOD_EXIT_M > LOD_ENTER_M, `${LOD_ENTER_M}m / ${LOD_EXIT_M}m`)
  check('遠景の車は入る距離の手前で近景になる', nextLod(LOD_FAR, LOD_ENTER_M - 1, false) === LOD_NEAR)
  check('近景の車は入る距離を少し越えても近景のまま', nextLod(LOD_NEAR, (LOD_ENTER_M + LOD_EXIT_M) / 2, false) === LOD_NEAR)
  check('出る距離を越えたら遠景へ戻る', nextLod(LOD_NEAR, LOD_EXIT_M + 1, false) === LOD_FAR)
  check('運転席から見ている車は遠くても近景（内装が要る）', nextLod(LOD_FAR, 1000, true) === LOD_NEAR)
  // 視点が 2 つ（画面のカメラと車載カメラ）あれば近いほうで測る
  const d = nearestViewDistance([[0, 300, 0], [10, 1.2, -5]], 10, 5)
  check('距離はいちばん近い視点で測る', Math.abs(d - 0.5) < 1e-9, `${d.toFixed(3)}m`)
  check('俯瞰（上空 300m）の車は遠景', nearestViewDistance([[0, 300, 0]], 0, 0) > LOD_EXIT_M)
}

console.log()
console.log('='.repeat(70))
console.log('ホイールアーチとタイヤ（傾いても当たらない）')
console.log('='.repeat(70))

{
  // 車体は傾くが車輪は路面に残る。アーチの縁とタイヤのすき間は、傾きで縁が下がる分を足しても残ること
  const pitchDrop = (x: number) => Math.abs(x) * Math.sin(MAX_PITCH * TILT_OVERSHOOT)
  const rollDrop = (VEHICLE_WIDTH / 2) * Math.sin(MAX_ROLL * TILT_OVERSHOOT)
  let worst = Infinity
  let at = 0
  for (const w of WHEEL_OFFSETS.filter((v) => v.position[2] > 0)) {
    const cx = w.position[0]
    for (let k = 0; k <= 200; k++) {
      const dx = -WHEEL_RADIUS + (2 * WHEEL_RADIUS * k) / 200
      const edge = archY(cx + dx)
      if (edge === null) continue
      const tyreTop = WHEEL_RADIUS + Math.sqrt(Math.max(0, WHEEL_RADIUS ** 2 - dx * dx))
      const gap = edge - tyreTop - pitchDrop(cx + dx) - rollDrop
      if (gap < worst) {
        worst = gap
        at = cx + dx
      }
    }
  }
  check('傾き切ってもアーチの縁とタイヤの間に 1cm 以上ある', worst >= 0.01, `最小 ${(worst * 100).toFixed(1)}cm（X=${at.toFixed(2)}）`)
  // 前輪を切り切っても、タイヤの内側の角がホイールハウスの奥の壁（z=0.52）に届かない
  const steer = 0.55
  const front = WHEEL_OFFSETS[0].position
  const innerEdge = front[2] - (TYRE_WIDTH / 2) * Math.cos(steer) - WHEEL_RADIUS * Math.sin(steer)
  check('前輪を切り切ってもホイールハウスの奥の壁に当たらない', innerEdge > 0.52 + 0.02, `タイヤの内側 ${innerEdge.toFixed(3)}m / 壁 0.52m`)
  const reach = WHEEL_RADIUS * Math.cos(steer) + (TYRE_WIDTH / 2) * Math.sin(steer)
  check('前輪を切り切ってもタイヤの前後がアーチの幅に収まる', reach < ARCH_HALF_SPAN, `${reach.toFixed(3)}m < ${ARCH_HALF_SPAN}m`)
}

console.log()
console.log('='.repeat(70))
console.log('車体の傾き（加減速のピッチ・旋回のロール）')
console.log('='.repeat(70))

{
  const front = new THREE.Vector3(VEHICLE_LENGTH / 2, 0.6, 0)
  const roof = new THREE.Vector3(0, VEHICLE_HEIGHT, 0)
  const brake = tiltTarget(-6, 0)
  const m = composeTiltMatrix(brake.pitch, 0, new THREE.Matrix4())
  check('減速すると前が沈む', front.clone().applyMatrix4(m).y < front.y - 0.01, `${((brake.pitch * 180) / Math.PI).toFixed(2)} 度`)
  const accel = tiltTarget(3, 0)
  check('加速すると前が上がる（後ろが沈む）', front.clone().applyMatrix4(composeTiltMatrix(accel.pitch, 0, new THREE.Matrix4())).y > front.y)
  const left = tiltTarget(0, lateralAccel(10, 0.1))
  const lm = composeTiltMatrix(0, left.roll, new THREE.Matrix4())
  check('左へ曲がると屋根が外側（右 = +Z）へ傾く', roof.clone().applyMatrix4(lm).z > 0.01, `${((left.roll * 180) / Math.PI).toFixed(2)} 度`)
  check(
    '傾きは数度まで（上限で止まる）',
    Math.abs(tiltTarget(-100, 100).pitch) <= MAX_PITCH + 1e-12 && Math.abs(tiltTarget(0, 100).roll) <= MAX_ROLL + 1e-12,
    `ピッチ ${((MAX_PITCH * 180) / Math.PI).toFixed(1)} 度 / ロール ${((MAX_ROLL * 180) / Math.PI).toFixed(1)} 度`,
  )

  // 20Hz の段差で届く速度を 60fps で線形に補間したもの（実際の入力の形）を入れても、傾きがガタつかない
  const s = createTiltState()
  resetTilt(s, 13.9)
  let maxStep = 0
  let minPitch = 0
  let prev = 0
  for (let frame = 1; frame <= 180; frame++) {
    const t = frame / 60
    const seg = Math.floor(t * 20)
    const alpha = t * 20 - seg
    const v0 = Math.max(0, 13.9 - 6 * (seg / 20))
    const v1 = Math.max(0, 13.9 - 6 * ((seg + 1) / 20))
    stepTilt(s, v0 + (v1 - v0) * alpha, 0, 1 / 60)
    maxStep = Math.max(maxStep, Math.abs(s.pitch - prev))
    minPitch = Math.min(minPitch, s.pitch)
    prev = s.pitch
  }
  check(
    `急ブレーキで前へ沈み、上限の ${TILT_OVERSHOOT} 倍を越えない`,
    minPitch < -MAX_PITCH * 0.5 && minPitch >= -MAX_PITCH * TILT_OVERSHOOT - 1e-9,
    `${((minPitch * 180) / Math.PI).toFixed(2)} 度`,
  )
  check('1 フレームの変化が 0.1 度未満（段差でガタつかない）', maxStep < (0.1 * Math.PI) / 180, `${((maxStep * 180) / Math.PI).toFixed(3)} 度`)

  // 傾き切った車体が外接寸法からどれだけはみ出すか（認識結果の枠とのずれ）
  const body = makeBodyPaintGeometry()
  let excess = 0
  for (const [p, r] of [
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ]) {
    const tb = boundsBesideMirrors(body, composeTiltMatrix(p * MAX_PITCH * TILT_OVERSHOOT, r * MAX_ROLL * TILT_OVERSHOOT, new THREE.Matrix4())).box
    excess = Math.max(
      excess,
      tb.max.y - VEHICLE_HEIGHT,
      tb.max.z - VEHICLE_WIDTH / 2,
      -VEHICLE_WIDTH / 2 - tb.min.z,
      tb.max.x - VEHICLE_LENGTH / 2,
      -VEHICLE_LENGTH / 2 - tb.min.x,
    )
  }
  check('傾き切っても外接寸法からのはみ出しが 5cm 以内（ドアミラーを除く）', excess <= 0.05, `${(excess * 100).toFixed(1)}cm`)
  body.dispose()

  // 運転席の目は車体と一緒に動く（車体の行列を掛けた目の位置と一致する）
  const scratchT = createTransformScratch()
  const baseT = new THREE.Matrix4()
  const tiltT = new THREE.Matrix4()
  const bodyT = new THREE.Matrix4()
  let worstEye = 0
  for (const [x, y, h, p, r] of [
    [3, -2, 0.7, 0.02, -0.03],
    [-40, 12, -2.1, -0.025, 0.03],
  ]) {
    composeVehicleMatrix(scratchT, x, y, h, baseT)
    composeTiltMatrix(p, r, tiltT)
    composeBodyMatrix(baseT, tiltT, bodyT)
    const expect = new THREE.Vector3(...DRIVER_EYE_LOCAL).applyMatrix4(bodyT)
    const got = vehicleLocalToThree(tiltLocalPoint(DRIVER_EYE_LOCAL, p, r), x, y, h)
    worstEye = Math.max(worstEye, expect.distanceTo(new THREE.Vector3(got.x, got.y, got.z)))
  }
  check('運転席の目は傾いた車体と一緒に動く（車内が前後に揺れない）', worstEye < 1e-9, `${worstEye.toExponential(1)}m`)
}

console.log()
console.log('='.repeat(70))
console.log('ワイパーとフロントガラスの水滴')
console.log('='.repeat(70))

/** 往復の始まりが何回変わったか（`rain` を一定にして `sec` 秒回す） */
function countStrokes(rain: number, sec: number): number {
  const w = createWiperState(0)
  let strokes = 0
  let lastStart = -Infinity
  for (let t = 0; t < sec; t += 1 / 60) {
    advanceWiper(w, t, rain)
    if (w.start !== lastStart) {
      strokes++
      lastStart = w.start
    }
  }
  return strokes
}

{
  check('雨が無ければ止まっている', wiperModeFor(0, WIPER_OFF) === WIPER_OFF)
  check(
    '濃霧の雨（0.15）と小雨（0.35）は間欠、雨（0.85）は低速、それより強い雨は高速（backend の percep/weather.py の PRESETS）',
    wiperModeFor(0.15, WIPER_OFF) === WIPER_INT &&
      wiperModeFor(0.35, WIPER_OFF) === WIPER_INT &&
      wiperModeFor(0.85, WIPER_OFF) === WIPER_LO &&
      wiperModeFor(0.95, WIPER_OFF) === WIPER_HI,
  )
  check('しきい値の近くで段を行き来しない', wiperModeFor(0.87, WIPER_HI) === WIPER_HI && wiperModeFor(0.87, WIPER_LO) === WIPER_LO)
  check('往復は高速のほうが短い', WIPER_PERIOD[WIPER_HI] < WIPER_PERIOD[WIPER_LO])
  check(
    '往復の始まりと終わりは止まる位置、中ほどで振り切る',
    strokeAngle(0, 1.4) === 0 && Math.abs(strokeAngle(0.7, 1.4) - WIPER_SWEEP) < 1e-9 && strokeAngle(1.4, 1.4) === 0,
  )

  // 連続なら往復が続き、間欠なら休みが入る。途中で雨がやんでも、振り切ってから止まる
  const hi = countStrokes(0.95, 10)
  check('強い雨の 10 秒で往復が途切れず続く', hi >= Math.floor(10 / WIPER_PERIOD[WIPER_HI]) - 1, `${hi} 往復`)
  const intermittent = countStrokes(0.35, 10)
  check('小雨の 10 秒は休みを挟んで 2〜3 往復', intermittent >= 2 && intermittent <= 3, `${intermittent} 往復`)
  const ws = createWiperState(0)
  advanceWiper(ws, 0, 0.9)
  advanceWiper(ws, 0.3, 0)
  check('雨がやんでも往復の途中なら振り切ってから止まる', wiperAngle(ws, 0.3) > 0 && wiperAngle(ws, 5) === 0)

  // 拭いた跡：ブレードが通った直後は 0 秒、届かない所は拭かれない
  const wa = createWiperState(0)
  advanceWiper(wa, 0, 0.95)
  const up = (WIPER_PERIOD[WIPER_HI] / (2 * Math.PI)) * Math.acos(1 - 2 * 0.5)
  check('ブレードが通った直後は拭いてから 0 秒', Math.abs(lastWipeAge(wa, up + 1e-6, WIPER_SWEEP * 0.5)) < 1e-3)
  check('ブレードの届かない角度は拭かれない', lastWipeAge(wa, 1, WIPER_SWEEP * 1.1) === Infinity)

  // 振り上げたブレードの先がガラスの外へ出ない
  let worstOut = -Infinity
  for (const [s0, z0] of WIPER_PIVOTS) {
    for (let k = 0; k <= 20; k++) {
      const a = (WIPER_SWEEP * k) / 20
      const s = s0 + Math.sin(a) * WIPER_REACH[1]
      const z = z0 + Math.cos(a) * WIPER_REACH[1]
      worstOut = Math.max(worstOut, s - WINDSHIELD_LENGTH, Math.abs(z) - windshieldHalfWidth(Math.min(s, WINDSHIELD_LENGTH)))
    }
  }
  check('ブレードの先はどの角度でもガラスの内側', worstOut <= 0.01, `はみ出し ${(worstOut * 100).toFixed(1)}cm`)
}

/** 運転席カメラで見た画面を粗い格子に塗り、何割を覆うかを数える（深度つき） */
class DriverRaster {
  readonly w = 320
  readonly h = 180
  readonly depth = new Float32Array(320 * 180).fill(Infinity)
  readonly cam: THREE.PerspectiveCamera
  constructor() {
    this.cam = new THREE.PerspectiveCamera(DRIVER_FOV_DEG, 16 / 9, 0.1, 100)
    const eye = driverEye(0, 0, 0)
    const at = driverLookAt(0, 0, 0)
    this.cam.position.set(eye.x, eye.y, eye.z)
    this.cam.lookAt(at.x, at.y, at.z)
    this.cam.updateMatrixWorld(true)
    this.cam.updateProjectionMatrix()
  }
  /** 三角形を塗る。`write` なら深度を書き、そうでなければ見えている格子の数を返す */
  draw(g: THREE.BufferGeometry, matrix: THREE.Matrix4 | null, cull: boolean, write: boolean): number {
    const src = g.index ? g.toNonIndexed() : g
    const pos = src.attributes.position
    const eye = this.cam.position
    const seen = new Uint8Array(this.w * this.h)
    let count = 0
    const v = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()]
    const ndc = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()]
    for (let i = 0; i + 2 < pos.count; i += 3) {
      for (let k = 0; k < 3; k++) {
        v[k].set(pos.getX(i + k), pos.getY(i + k), pos.getZ(i + k))
        if (matrix) v[k].applyMatrix4(matrix)
        ndc[k].copy(v[k]).project(this.cam)
      }
      if (ndc.some((p) => p.z < -1 || p.z > 1)) continue
      if (cull) {
        const n = new THREE.Vector3().subVectors(v[1], v[0]).cross(new THREE.Vector3().subVectors(v[2], v[0]))
        if (n.dot(new THREE.Vector3().subVectors(eye, v[0])) <= 0) continue
      }
      const sx = ndc.map((p) => (p.x * 0.5 + 0.5) * this.w)
      const sy = ndc.map((p) => (0.5 - p.y * 0.5) * this.h)
      const area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0])
      if (Math.abs(area) < 1e-12) continue
      const x0 = Math.max(0, Math.floor(Math.min(...sx)))
      const x1 = Math.min(this.w - 1, Math.ceil(Math.max(...sx)))
      const y0 = Math.max(0, Math.floor(Math.min(...sy)))
      const y1 = Math.min(this.h - 1, Math.ceil(Math.max(...sy)))
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const px = x + 0.5
          const py = y + 0.5
          const w0 = ((sx[1] - px) * (sy[2] - py) - (sx[2] - px) * (sy[1] - py)) / area
          const w1 = ((sx[2] - px) * (sy[0] - py) - (sx[0] - px) * (sy[2] - py)) / area
          const w2 = 1 - w0 - w1
          if (w0 < 0 || w1 < 0 || w2 < 0) continue
          const z = w0 * ndc[0].z + w1 * ndc[1].z + w2 * ndc[2].z
          const idx = y * this.w + x
          if (write) {
            if (z < this.depth[idx]) this.depth[idx] = z
          } else if (z <= this.depth[idx] + 1e-6 && !seen[idx]) {
            seen[idx] = 1
            count++
          }
        }
      }
    }
    if (src !== g) src.dispose()
    return count
  }
  get cells(): number {
    return this.w * this.h
  }
}

{
  // 視界を遮るもの：運転席から見て、車体・内装の手前に何割見えているか
  const raster = new DriverRaster()
  const interior = makeInteriorGeometry()
  const paint = makeBodyPaintGeometry()
  const trim = makeBodyTrimGeometry()
  raster.draw(interior, null, false, true)
  raster.draw(paint, null, true, true)
  raster.draw(trim, null, true, true)

  const wiper = makeWiperGeometry()
  const parked = WIPER_PIVOTS.reduce((a, _, k) => a + raster.draw(wiper, composeWiperLocal(k, 0, new THREE.Matrix4()), false, false), 0)
  check('止めたワイパーはダッシュボードの陰に隠れる（画面の 0.3% 以下）', parked / raster.cells <= 0.003, `${((parked / raster.cells) * 100).toFixed(2)}%`)
  let worstSweep = 0
  for (let k = 1; k <= 12; k++) {
    const a = (WIPER_SWEEP * k) / 12
    const c = WIPER_PIVOTS.reduce((acc, _, i) => acc + raster.draw(wiper, composeWiperLocal(i, a, new THREE.Matrix4()), false, false), 0)
    worstSweep = Math.max(worstSweep, c)
  }
  check('動いているワイパーが視界を塞ぐのは画面の 4% まで', worstSweep / raster.cells <= 0.04, `最大 ${((worstSweep / raster.cells) * 100).toFixed(2)}%`)
  wiper.dispose()

  // 水滴：降り続けて拭かない所（上限）と、強い雨で拭き続ける所（平均）
  const saturated = dropCoverage(Infinity, 1)
  check('拭かない所の水滴が視界を覆うのは 12% まで（不透明度を掛けた割合）', saturated <= 0.12, `${(saturated * 100).toFixed(1)}%`)
  let sum = 0
  let n = 0
  const period = WIPER_PERIOD[WIPER_HI]
  const wHi = createWiperState(0)
  for (let t = 0; t < period * 6; t += 1 / 60) advanceWiper(wHi, t, 1)
  const t0 = wHi.start
  for (let t = t0; t < t0 + period; t += period / 40) {
    for (let k = 1; k < 20; k++) {
      sum += dropCoverage(lastWipeAge(wHi, t, (WIPER_SWEEP * k) / 20), 1)
      n++
    }
  }
  check('強い雨でも、拭いている所の水滴は平均 4% まで', sum / n <= 0.04, `${((sum / n) * 100).toFixed(2)}%（粒の不透明度 ${DROP_OPACITY}）`)

  // ルームミラーは小さく保つ（`vehicleInterior` と同じ寸法・向きで本体の箱を作って測る）
  const rm = MIRROR_FACES[0]
  const mirror = orientedBox(
    [
      rm.center[0] - rm.normal[0] * (0.002 + ROOM_MIRROR_BODY_DEPTH / 2),
      rm.center[1] - rm.normal[1] * (0.002 + ROOM_MIRROR_BODY_DEPTH / 2),
      rm.center[2] - rm.normal[2] * (0.002 + ROOM_MIRROR_BODY_DEPTH / 2),
    ],
    rm.right,
    rm.up,
    [rm.width + 2 * ROOM_MIRROR_RIM, rm.height + 2 * ROOM_MIRROR_RIM, ROOM_MIRROR_BODY_DEPTH],
  )
  const raster2 = new DriverRaster()
  const mirrorCells = raster2.draw(mirror, null, false, false)
  check('ルームミラーが視界を塞ぐのは画面の 3% まで', mirrorCells / raster2.cells <= 0.03, `${((mirrorCells / raster2.cells) * 100).toFixed(2)}%`)
  mirror.dispose()

  // 吹き出し口とルームミラーは運転席の画角に入る（作り込んでも映らなければ意味がない）
  const inView = (p: THREE.Vector3) => {
    const q = p.clone().project(raster.cam)
    return q.z > -1 && q.z < 1 && Math.abs(q.x) <= 1 && Math.abs(q.y) <= 1
  }
  // 助手席の端の吹き出し口は運転席から 60 度以上左にあり、画角の外になる（実車と同じ）
  const visibleVents = AC_VENTS.filter((v) => v.z > -0.5)
  check(
    `運転席側と中央の吹き出し口（${visibleVents.length} つ）が運転席の画角に入る`,
    visibleVents.length === 3 && visibleVents.every((v) => inView(new THREE.Vector3(DASH_REAR_X - 0.01, v.y, v.z))),
  )
  check('ルームミラーが運転席の画角に入る', inView(new THREE.Vector3(rm.center[0], rm.center[1], rm.center[2])))

  // タクシーの表示板（助手席側のダッシュボードの上）は運転席の視界を塞ぎすぎない
  const sign = makeTaxiSignGeometry()
  const signCells = raster.draw(sign, null, false, false)
  check('タクシーの表示が運転席の視界を塞ぐのは画面の 2% まで', signCells / raster.cells <= 0.02, `${((signCells / raster.cells) * 100).toFixed(2)}%`)
  sign.dispose()
  interior.dispose()
  paint.dispose()
  trim.dispose()
}

console.log()
console.log('='.repeat(70))
console.log('実用モードのタクシー（表示灯・状態の表示・ドア）')
console.log('='.repeat(70))

{
  const phases: TaxiPhase[] = ['idle', 'approaching', 'waiting', 'riding', 'arrived']
  const expected: Record<TaxiPhase, string> = {
    idle: 'vacant',
    approaching: 'dispatched',
    waiting: 'dispatched',
    riding: 'occupied',
    arrived: 'paying',
  }
  check(
    '呼ばれた車は段階どおり（迎車 → 賃走 → 支払）、ほかの車は空車',
    phases.every(
      (ph) => taxiSignStatus(3, { phase: ph, vehicleId: 3 }) === expected[ph] && taxiSignStatus(2, { phase: ph, vehicleId: 3 }) === 'vacant',
    ),
    phases.map((ph) => `${ph}=${taxiSignStatus(3, { phase: ph, vehicleId: 3 })}`).join(' '),
  )
  check('表示灯は空車のときだけ点く', lampGlow('vacant') === 1 && lampGlow('occupied') < 0.5 && lampGlow('dispatched') < 0.5)
  const rows = (['vacant', 'dispatched', 'occupied', 'paying'] as const).map(statusRow)
  check('状態ごとにアトラスの別の段を使う', new Set(rows).size === 4 && rows.every((r) => r >= SIGN_ROW_STATUS && r < SIGN_ROWS))
  check('Canvas の段は v では下から数える', signUvRow(0) === SIGN_ROWS - 1 && signUvRow(SIGN_ROWS - 1) === 0)

  // 表示灯は外接寸法の高さ（1.45m）を越える唯一の部品（CLAUDE.md の例外）。越える量と置き場所を縛る
  const g = makeTaxiSignGeometry()
  const b = boundsOf(g)
  check(
    `表示灯のはみ出しは 13cm まで（上端 ${TAXI_LAMP_TOP_Y.toFixed(3)}m）`,
    TAXI_LAMP_OVERHANG > 0 && TAXI_LAMP_OVERHANG <= 0.13 && b.max.y <= TAXI_LAMP_TOP_Y + 1e-9,
    `${(TAXI_LAMP_OVERHANG * 100).toFixed(1)}cm`,
  )
  check('表示灯は屋根の平らな所に載る（屋根の縁より内側）', TAXI_LAMP.width / 2 + 0.02 < 0.66 && Math.abs(TAXI_LAMP.centerX) < 0.3)
  // 表示板はフロントガラスの内側（ガラスを突き抜けない）で、助手席側
  const nws = new THREE.Vector3(0.583, 0.813, 0).normalize()
  const pos = g.attributes.position
  let worstInside = -Infinity
  for (let i = 0; i < pos.count; i++) {
    const p = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i))
    if (p.y > 1.2) continue
    worstInside = Math.max(worstInside, nws.dot(new THREE.Vector3(p.x - 1.15, p.y - BELT_Y, 0)))
  }
  check('表示板がフロントガラスの内側にある（ガラスの面から 5mm 以上）', worstInside < -0.005, `${(worstInside * 1000).toFixed(1)}mm`)
  check('表示板は助手席側（左）のダッシュボードの上', STATUS_SIGN.z[1] < 0 && Math.abs(STATUS_SIGN.y[0] - DASH_TOP_Y) < 0.01)
  g.dispose()

  // ドア：印の付いた頂点はすべて左後席ドアの範囲にあり、開けると後ろの縁が外（-Z）へ出る
  const doorGeoms: Array<[string, THREE.BufferGeometry]> = [
    ['塗装', makeBodyPaintGeometry()],
    ['黒い樹脂', makeBodyTrimGeometry()],
    ['メッキ', makeBodyChromeGeometry()],
    ['窓ガラス', makeVehicleGlass()],
    ['内装', makeInteriorGeometry()],
  ]
  for (const [name, dg] of doorGeoms) {
    const flag = dg.attributes.doorPart
    const p2 = dg.attributes.position
    let n = 0
    let stray = 0
    for (let i = 0; i < flag.count; i++) {
      if (flag.getX(i) < 0.5) continue
      n++
      const x = p2.getX(i)
      const z = p2.getZ(i)
      if (z > 0 || x < REAR_DOOR_X[0] - 0.03 || x > REAR_DOOR_X[1] + 0.03) stray++
    }
    check(`${name}：ドアと一緒に回る頂点がある`, n > 0, `${n} 頂点`)
    check(`${name}：ドアの印は左後席ドアの範囲だけ`, stray === 0, `${stray} 頂点がはみ出す`)
    dg.dispose()
  }
  check('蝶番はドアの前の縁（左側）', Math.abs(TAXI_DOOR_HINGE[0] - REAR_DOOR_X[1]) < 0.05 && TAXI_DOOR_HINGE[1] < 0)
  // シェーダー（vehicleMaterials の DOOR）と同じ式で回す
  const a = -TAXI_DOOR_OPEN
  const rx = REAR_DOOR_X[0] - TAXI_DOOR_HINGE[0]
  const rz = -BODY_HALF_W - TAXI_DOOR_HINGE[1]
  const zOpen = TAXI_DOOR_HINGE[1] - rx * Math.sin(a) + rz * Math.cos(a)
  check('ドアを開けると後ろの縁が外（-Z）へ出る', zOpen < -BODY_HALF_W - 0.5, `z=${zOpen.toFixed(2)}`)
}

console.log()
console.log('='.repeat(70))
console.log('ルームミラーとドアミラー（後ろの景色を映す）')
console.log('='.repeat(70))

{
  const eye = new THREE.Vector3(...DRIVER_EYE_LOCAL)
  const v3 = (p: readonly [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])
  const reflected = (f: MirrorFace) => {
    const n = v3(f.normal)
    const d = v3(f.center).sub(eye).normalize()
    return d.sub(n.multiplyScalar(2 * d.dot(v3(f.normal))))
  }
  check(
    '鏡は 3 枚（ルームミラーと左右のドアミラー）',
    MIRROR_FACES.map((f) => f.key).join(',') === 'room,right,left',
  )
  check(
    '鏡面の向き（法線・右・上）は直交する単位ベクトルで、右 = 上 × 法線',
    MIRROR_FACES.every((f) => {
      const n = v3(f.normal)
      const r = v3(f.right)
      const u = v3(f.up)
      return (
        Math.abs(n.length() - 1) < 1e-9 &&
        Math.abs(r.length() - 1) < 1e-9 &&
        Math.abs(u.length() - 1) < 1e-9 &&
        Math.abs(n.dot(r)) < 1e-9 &&
        Math.abs(n.dot(u)) < 1e-9 &&
        u.clone().cross(n).distanceTo(r) < 1e-9
      )
    }),
  )
  check('鏡面は運転席の目の側を向く', MIRROR_FACES.every((f) => eye.clone().sub(v3(f.center)).dot(v3(f.normal)) > 0))

  // ルームミラーの真ん中には、リアガラス越しの真後ろがほぼ水平に映る（下を向きすぎると地平線が入らない）
  const room = MIRROR_FACES[0]
  const rr = reflected(room)
  const tRear = (REAR_GLASS_CENTER[0] - room.center[0]) / rr.x
  const hit = v3(room.center).add(rr.clone().multiplyScalar(tRear))
  const roomPitch = (Math.asin(rr.y) * 180) / Math.PI
  check(
    'ルームミラーの真ん中はリアガラスを通って真後ろを映し、下向きは 3 度まで',
    rr.x < 0 && hit.y > BELT_Y + 0.1 && hit.y < GLASS_TOP_Y - 0.05 && Math.abs(hit.z) < 0.1 && roomPitch <= 0 && roomPitch >= -3,
    `リアガラスで y=${hit.y.toFixed(3)} z=${hit.z.toFixed(3)} / ${roomPitch.toFixed(1)} 度`,
  )
  // ドアミラーの真ん中は、車の真後ろから少し外・少し下を映す
  for (const f of MIRROR_FACES.slice(1)) {
    const r = reflected(f)
    const side = f.key === 'right' ? 1 : -1
    const outward = (Math.atan2(side * r.z, -r.x) * 180) / Math.PI
    const pitch = (Math.asin(r.y) * 180) / Math.PI
    check(
      `${f.key === 'right' ? '右' : '左'}のドアミラーの真ん中は真後ろから外へ 3〜10 度・下へ 0〜5 度`,
      r.x < 0 && outward >= 3 && outward <= 10 && pitch <= 0 && pitch >= -5,
      `外へ ${outward.toFixed(1)} 度 / ${pitch.toFixed(1)} 度`,
    )
  }

  // 映像の左右：車の後ろの右にある物は鏡の右に、左にある物は鏡の左に映る（運転席から見て）
  const driver = new DriverRaster().cam
  const screenXOf = (f: MirrorFace, world: THREE.Vector3): number | null => {
    const view = mirrorView(f)
    const cam = new THREE.PerspectiveCamera()
    cam.projectionMatrix.makePerspective(view.left, view.right, view.top, view.bottom, view.near, view.far)
    cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert()
    view.matrix.decompose(cam.position, cam.quaternion, cam.scale)
    cam.updateMatrixWorld(true)
    const ndc = world.clone().project(cam)
    if (Math.abs(ndc.x) > 1 || Math.abs(ndc.y) > 1 || ndc.z < -1 || ndc.z > 1) return null
    const [pa, pb, pc] = mirrorCorners(f).map(v3)
    const u = (ndc.x + 1) / 2
    const w = (ndc.y + 1) / 2
    const q = pa.clone().add(pb.clone().sub(pa).multiplyScalar(u)).add(pc.clone().sub(pa).multiplyScalar(w))
    return q.project(driver).x
  }
  const pairs: Array<[MirrorFace, THREE.Vector3, THREE.Vector3]> = [
    [MIRROR_FACES[0], new THREE.Vector3(-14, 0.8, 1.2), new THREE.Vector3(-14, 0.8, -1.2)],
    [MIRROR_FACES[1], new THREE.Vector3(-14, 0.9, 3.6), new THREE.Vector3(-14, 0.9, 1.8)],
    [MIRROR_FACES[2], new THREE.Vector3(-14, 0.9, -1.8), new THREE.Vector3(-14, 0.9, -3.4)],
  ]
  for (const [f, rightPoint, leftPoint] of pairs) {
    const xr = screenXOf(f, rightPoint)
    const xl = screenXOf(f, leftPoint)
    check(
      `${f.key === 'room' ? 'ルームミラー' : f.key === 'right' ? '右のドアミラー' : '左のドアミラー'}：後ろの右にある物ほど、運転席から見て鏡の右に映る`,
      xr !== null && xl !== null && xr > xl,
      xr !== null && xl !== null ? `右 ${xr.toFixed(4)} / 左 ${xl.toFixed(4)}` : '鏡に映らない',
    )
  }

  // 鏡より前（運転席の反対側）の物は映さない。ルームミラーからはフロントガラスが見えない
  {
    const view = mirrorView(room)
    const cam = new THREE.PerspectiveCamera()
    cam.projectionMatrix.makePerspective(view.left, view.right, view.top, view.bottom, view.near, view.far)
    view.matrix.decompose(cam.position, cam.quaternion, cam.scale)
    cam.updateMatrixWorld(true)
    const front = new THREE.Vector3(0.95, 1.15, 0.02).project(cam)
    const rear = new THREE.Vector3(-15, 0.6, 0).project(cam)
    check(
      'ルームミラーの映像には、鏡より前のフロントガラスが入らず、15m 後ろの道路が入る',
      !(Math.abs(front.x) <= 1 && Math.abs(front.y) <= 1 && front.z >= -1 && front.z <= 1) &&
        Math.abs(rear.x) <= 1 && Math.abs(rear.y) <= 1 && rear.z >= -1 && rear.z <= 1,
    )
  }

  // ドアミラーに自分の車が映る割合（実車のように内側の端に少しだけ車体が見える）
  const carBox = new THREE.Box3(
    new THREE.Vector3(-VEHICLE_LENGTH / 2, 0, -VEHICLE_WIDTH / 2),
    new THREE.Vector3(VEHICLE_LENGTH / 2, VEHICLE_HEIGHT, VEHICLE_WIDTH / 2),
  )
  for (const f of MIRROR_FACES.slice(1)) {
    const view = mirrorView(f)
    const rot = new THREE.Matrix4().extractRotation(view.matrix)
    const origin = v3(view.eye)
    let own = 0
    let total = 0
    for (let i = 0; i < 24; i++) {
      for (let j = 0; j < 14; j++) {
        const x = view.left + ((i + 0.5) / 24) * (view.right - view.left)
        const y = view.bottom + ((j + 0.5) / 14) * (view.top - view.bottom)
        const dir = new THREE.Vector3(x, y, -view.near).applyMatrix4(rot).normalize()
        const start = origin.clone().add(dir.clone().multiplyScalar(view.near / Math.max(1e-9, -new THREE.Vector3(x, y, -view.near).normalize().z)))
        const ray = new THREE.Ray(start, dir)
        const p = ray.intersectBox(carBox, new THREE.Vector3())
        if (p && p.distanceTo(start) < 8) own++
        total++
      }
    }
    check(
      `${f.key === 'right' ? '右' : '左'}のドアミラーに映る自分の車は 3〜35%（内側の端に少しだけ）`,
      own / total >= 0.03 && own / total <= 0.35,
      `${((own / total) * 100).toFixed(1)}%`,
    )
  }

  // 映像を貼る板は運転席の側を向き、鏡の面のすぐ手前にある
  check(
    '映像を貼る板は運転席の側を向く（表が見える）',
    MIRROR_FACES.every((f) => {
      const g = makeMirrorQuadGeometry(f, 0.001)
      const pos = g.attributes.position
      const a = new THREE.Vector3().fromBufferAttribute(pos, 0)
      const b = new THREE.Vector3().fromBufferAttribute(pos, 1)
      const c = new THREE.Vector3().fromBufferAttribute(pos, 2)
      const n = b.clone().sub(a).cross(c.clone().sub(a))
      g.dispose()
      return n.dot(eye.clone().sub(a)) > 0
    }),
  )

  // 運転席から見える鏡：ルームミラーと右（運転席側）のドアミラーは画角に入る
  const inDriverView = (p: readonly [number, number, number]) => {
    const q = v3(p).project(driver)
    return q.z > -1 && q.z < 1 && Math.abs(q.x) <= 1 && Math.abs(q.y) <= 1
  }
  check('ルームミラーと運転席側のドアミラーが運転席の画角に入る', inDriverView(MIRROR_FACES[0].center) && inDriverView(MIRROR_FACES[1].center))

  // 描き直しの順番：見えている鏡だけを、間隔を空けて、1 フレームに上限まで
  const simulate = (visible: boolean[], fps: number, seconds: number) => {
    const ages = visible.map(() => Infinity)
    const counts = visible.map(() => 0)
    let worst = 0
    for (let k = 0; k < fps * seconds; k++) {
      for (let i = 0; i < ages.length; i++) ages[i] += 1 / fps
      const due = mirrorsDue(ages, visible)
      worst = Math.max(worst, due.length)
      for (const i of due) {
        counts[i]++
        ages[i] = 0
      }
    }
    return { counts, worst }
  }
  const two = simulate([true, true, false], 60, 1)
  check(
    `見えている 2 枚は 60fps の画面で毎秒 ${Math.round(1 / MIRROR_INTERVAL_SEC)} 回前後描き直し、見えていない鏡は描かない`,
    two.counts[0] >= 25 && two.counts[1] >= 25 && two.counts[2] === 0,
    two.counts.join(' / '),
  )
  const three = simulate([true, true, true], 60, 1)
  check(
    `3 枚とも見えていても 1 フレームに ${MIRRORS_PER_FRAME} 枚まで、それぞれ毎秒 18 回以上`,
    three.worst <= MIRRORS_PER_FRAME && three.counts.every((c) => c >= 18),
    `${three.counts.join(' / ')}（1 フレーム最大 ${three.worst} 枚）`,
  )
  const slow = simulate([true, true, false], 30, 1)
  check('画面が 30fps に落ちても、見えている鏡は毎秒 14 回以上描き直す', slow.counts[0] >= 14 && slow.counts[1] >= 14, slow.counts.join(' / '))
}

console.log()
console.log('='.repeat(70))
console.log('前照灯が路面に落とす光')
console.log('='.repeat(70))

{
  const pool = makeHeadlightPoolGeometry()
  const b = boundsOf(pool)
  check('路面の少し上に平らに貼る', Math.abs(b.max.y - b.min.y) < 1e-6 && b.min.y > 0.02 && b.min.y < 0.1, `y=${b.min.y.toFixed(3)}`)
  check('車の前（前端より先）から前方へ伸びる', b.min.x > VEHICLE_LENGTH / 2 && b.max.x >= HEADLIGHT_POOL.far - 1e-6, `X ${b.min.x.toFixed(1)}〜${b.max.x.toFixed(1)}m`)
  pool.dispose()
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
