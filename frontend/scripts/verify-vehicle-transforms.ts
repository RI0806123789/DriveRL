/** 車両の姿勢と部品の取り付け位置を検証する（ブラウザ不要）。 */

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
  makeWheelGeometry,
} from '../src/scene/vehicleGeometry.ts'

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
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
