/** 最高速度標識の向きと高さを検証する（ブラウザ不要）。 */

import * as THREE from 'three'
import {
  SIGN_BOARD_THICKNESS,
  SIGN_BOTTOM_HEIGHT,
  SIGN_DIAMETER,
  SIGN_FACE_OFFSET,
  SIGN_RADIUS,
  SIGN_TOP_HEIGHT,
  createSignBoardGeometry,
  createSignFaceGeometry,
  createSignPoleGeometry,
  signBoardCenter,
  signBoardFacing,
  signBoardNormal,
  signPoleCenter,
  signPoleHeight,
  signSpeedKph,
  type SignPlacementInput,
} from '../src/scene/signGeometry.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

/** 進行方向 heading を three 空間の単位ベクトルにする（three.z = -enu.y） */
function headingVector(heading: number): THREE.Vector3 {
  return new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading))
}

/** 標識 1 本ぶんの入力を作る。x, y は既に路端へ寄せた支柱位置 */
function makeSign(heading: number, speedLimit = 13.9): SignPlacementInput {
  return { x: 12.5, y: -7.5, heading, speedLimit }
}

const HEADINGS: Array<[string, number]> = [
  ['東向き', 0],
  ['北向き', Math.PI / 2],
  ['西向き', Math.PI],
  ['南向き', -Math.PI / 2],
]

console.log('='.repeat(70))
console.log('1. 標示板が進入車両に正対しているか（4 方位で確認）')
console.log('='.repeat(70))

for (const [name, heading] of HEADINGS) {
  const sign = makeSign(heading)

  const [nx, ny, nz] = signBoardNormal(sign)
  const forward = headingVector(heading)
  const dot = nx * forward.x + ny * forward.y + nz * forward.z
  check(
    `${name}: 標示板の法線が進行方向と向かい合う`,
    dot < -0.999,
    `内積 ${dot.toFixed(6)}（-1.0 が完全な正対）`,
  )

  const dummy = new THREE.Object3D()
  const [bx, by, bz] = signBoardCenter(sign)
  dummy.position.set(bx, by, bz)
  dummy.rotation.set(0, signBoardFacing(sign), 0)
  dummy.updateMatrix()
  const rotated = new THREE.Vector3(1, 0, 0).applyQuaternion(dummy.quaternion)
  check(
    `${name}: 行列を組んだ +X が signBoardNormal と一致`,
    Math.abs(rotated.x - nx) < 1e-9 &&
      Math.abs(rotated.y - ny) < 1e-9 &&
      Math.abs(rotated.z - nz) < 1e-9,
    `(${rotated.x.toFixed(3)}, ${rotated.y.toFixed(3)}, ${rotated.z.toFixed(3)})`,
  )
  check(
    `${name}: 行列の +X も進行方向と向かい合う`,
    rotated.dot(forward) < -0.999,
    `内積 ${rotated.dot(forward).toFixed(6)}`,
  )
}

console.log('')
console.log('='.repeat(70))
console.log('2. 標示板の下端が路面から 1.8m にあるか')
console.log('='.repeat(70))

for (const [name, heading] of HEADINGS) {
  const sign = makeSign(heading)
  const [, cy] = signBoardCenter(sign)
  const bottom = cy - SIGN_RADIUS
  const top = cy + SIGN_RADIUS
  check(
    `${name}: 下端が路面から ${SIGN_BOTTOM_HEIGHT.toFixed(1)}m`,
    Math.abs(bottom - SIGN_BOTTOM_HEIGHT) < 1e-9,
    `${bottom.toFixed(3)}m`,
  )
  check(
    `${name}: 上端が下端 + 直径`,
    Math.abs(top - (SIGN_BOTTOM_HEIGHT + SIGN_DIAMETER)) < 1e-9,
    `${top.toFixed(3)}m（直径 ${SIGN_DIAMETER.toFixed(2)}m）`,
  )
}

console.log('')
console.log('='.repeat(70))
console.log('3. 支柱が地面から立ち、標示板の上端まで届いているか')
console.log('='.repeat(70))

for (const [name, heading] of HEADINGS) {
  const sign = makeSign(heading)
  const height = signPoleHeight(sign)
  const [, py] = signPoleCenter(sign)
  const base = py - height / 2
  const top = py + height / 2
  const boardTop = signBoardCenter(sign)[1] + SIGN_RADIUS

  check(`${name}: 支柱の下端が地面 y=0`, Math.abs(base) < 1e-9, `${base.toFixed(3)}m`)
  check(
    `${name}: 支柱の上端が標示板の上端以上`,
    top >= boardTop - 1e-9,
    `支柱 ${top.toFixed(3)}m / 標示板 ${boardTop.toFixed(3)}m`,
  )
  check(
    `${name}: 支柱の高さが SIGN_TOP_HEIGHT と一致`,
    Math.abs(height - SIGN_TOP_HEIGHT) < 1e-9,
    `${height.toFixed(3)}m`,
  )
}

console.log('')
console.log('='.repeat(70))
console.log('4. 標示板が支柱の真上にあるか（左寄せはバックエンド側で済んでいる）')
console.log('='.repeat(70))

for (const [name, heading] of HEADINGS) {
  const sign = makeSign(heading)
  const [bx, , bz] = signBoardCenter(sign)
  const [px, , pz] = signPoleCenter(sign)
  check(
    `${name}: 標示板の中心が支柱の真上`,
    Math.abs(bx - px) < 1e-9 && Math.abs(bz - pz) < 1e-9,
    `ずれ ${Math.hypot(bx - px, bz - pz).toFixed(6)}m`,
  )
  check(
    `${name}: ENU -> three の変換が three.z = -enu.y`,
    Math.abs(px - sign.x) < 1e-9 && Math.abs(pz + sign.y) < 1e-9,
    `(${px.toFixed(2)}, ${pz.toFixed(2)}) ← ENU(${sign.x.toFixed(2)}, ${sign.y.toFixed(2)})`,
  )
}

console.log('')
console.log('='.repeat(70))
console.log('5. ジオメトリと行列に NaN が無いか')
console.log('='.repeat(70))

{
  const pole = createSignPoleGeometry()
  const board = createSignBoardGeometry()
  const face = createSignFaceGeometry()
  const geometries: Array<[string, THREE.BufferGeometry]> = [
    ['支柱', pole],
    ['標示板', board],
    ['数字の面', face],
  ]

  for (const [name, g] of geometries) {
    const p = g.getAttribute('position')
    let bad = false
    for (let i = 0; i < p.count * 3; i++) {
      if (!Number.isFinite((p.array as ArrayLike<number>)[i])) bad = true
    }
    check(`${name}のジオメトリに NaN が無い`, !bad, `${p.count} 頂点`)
  }

  board.computeBoundingBox()
  const bb = board.boundingBox!
  check(
    '標示板の軸が +X（厚みが X、直径が Y/Z）',
    Math.abs(bb.max.x - SIGN_BOARD_THICKNESS / 2) < 1e-6 &&
      Math.abs(bb.max.y - SIGN_RADIUS) < 1e-6 &&
      Math.abs(bb.max.z - SIGN_RADIUS) < 1e-6,
    `厚み ${(bb.max.x * 2).toFixed(3)}m / 直径 ${(bb.max.y * 2).toFixed(3)}m`,
  )

  face.computeVertexNormals()
  const n = face.getAttribute('normal')
  check(
    '数字の面の法線が +X',
    Math.abs(n.getX(0) - 1) < 1e-6 && Math.abs(n.getY(0)) < 1e-6 && Math.abs(n.getZ(0)) < 1e-6,
    `(${n.getX(0).toFixed(3)}, ${n.getY(0).toFixed(3)}, ${n.getZ(0).toFixed(3)})`,
  )
  face.computeBoundingBox()
  const fb = face.boundingBox!
  const lift = fb.min.x
  check(
    '数字の面が標示板の前面より手前にある',
    Math.abs(lift - (SIGN_BOARD_THICKNESS / 2 + SIGN_FACE_OFFSET)) < 1e-6,
    `${(lift * 1000).toFixed(1)}mm（板の前面は ${((SIGN_BOARD_THICKNESS / 2) * 1000).toFixed(1)}mm）`,
  )

  let badMatrix = false
  const dummy = new THREE.Object3D()
  for (const [, heading] of HEADINGS) {
    const sign = makeSign(heading)
    for (const [x, y, z] of [signBoardCenter(sign), signPoleCenter(sign)]) {
      dummy.position.set(x, y, z)
      dummy.rotation.set(0, signBoardFacing(sign), 0)
      dummy.updateMatrix()
      if (dummy.matrix.elements.some((e) => !Number.isFinite(e))) badMatrix = true
    }
  }
  check('インスタンス行列に NaN が無い', !badMatrix, `${HEADINGS.length * 2} 件`)

  pole.dispose()
  board.dispose()
  face.dispose()
}

console.log('')
console.log('='.repeat(70))
console.log('6. 規制速度 [m/s] から表示する数字 [km/h] への丸め')
console.log('='.repeat(70))

{
  const CASES: Array<[number, number]> = [
    [5.56, 20],
    [8.33, 30],
    [11.111, 40],
    [13.9, 50],
    [16.667, 60],
    [22.222, 80],
    [27.78, 100],
  ]
  for (const [mps, kph] of CASES) {
    const got = signSpeedKph(makeSign(0, mps))
    check(
      `${mps} m/s -> ${kph} km/h`,
      got === kph,
      `${got} km/h（素の値 ${(mps * 3.6).toFixed(3)}）`,
    )
  }
  check(
    '整数以外が出ない',
    CASES.every(([mps]) => Number.isInteger(signSpeedKph(makeSign(0, mps)))),
  )
}

console.log('')
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
