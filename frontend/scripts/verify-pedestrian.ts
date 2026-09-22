/** 徒歩キャラクターと NPC 歩行者の寸法・姿勢・当たり判定を検証する（ブラウザ不要）。 */

import * as THREE from 'three'
import {
  ARM_LENGTH,
  EYE_HEIGHT,
  LEG_LENGTH,
  LIMB_JOINTS,
  PEDESTRIAN_HEIGHT,
  composeLimbMatrix,
  composePedestrianMatrix,
  createPedestrianScratch,
  eyeLookAt,
  eyePosition,
  limbSwing,
  makeArmGeometry,
  makeHeadGeometry,
  makeHipGeometry,
  makeLegGeometry,
  makeTorsoGeometry,
  npcHueOffset,
} from '../src/scene/pedestrianGeometry.ts'
import {
  TELEPORT_DISTANCE_M,
  createPedestrianPose,
  samplePedestrian,
} from '../src/scene/interpolationMath.ts'
import type { NpcPedestrianState } from '../src/types/protocol.ts'
import {
  AIM_DISTANCE_M,
  PEDESTRIAN_RADIUS,
  VEHICLE_HALF_LENGTH,
  VEHICLE_HALF_WIDTH,
  aimedVehicle,
  alightPosition,
  alightSpot,
  buildBuildingIndex,
  isInsideBuilding,
  resolveMove,
  touchesBuilding,
  touchesVehicle,
} from '../src/scene/pedestrianCollision.ts'
import type { MapBuilding } from '../src/types/protocol.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

function matrixDiff(a: THREE.Matrix4, b: THREE.Matrix4): number {
  let worst = 0
  for (let i = 0; i < 16; i++) worst = Math.max(worst, Math.abs(a.elements[i] - b.elements[i]))
  return worst
}

console.log('='.repeat(70))
console.log('体の寸法（人として成立しているか）')
console.log('='.repeat(70))

{
  const geoms: Array<[string, THREE.BufferGeometry]> = [
    ['頭', makeHeadGeometry()],
    ['胴', makeTorsoGeometry()],
    ['腰', makeHipGeometry()],
  ]
  let top = -Infinity
  let bottom = Infinity
  for (const [, g] of geoms) {
    g.computeBoundingBox()
    top = Math.max(top, g.boundingBox!.max.y)
    bottom = Math.min(bottom, g.boundingBox!.min.y)
  }
  check(
    '頭頂が身長の ±10cm に収まる',
    Math.abs(top - PEDESTRIAN_HEIGHT) < 0.1,
    `頭頂 ${top.toFixed(3)}m / 宣言 ${PEDESTRIAN_HEIGHT}m`,
  )
  check('目線が頭頂より低く、腰より高い', EYE_HEIGHT < top && EYE_HEIGHT > 0.9, `${EYE_HEIGHT}m`)

  const hipJoint = LIMB_JOINTS.find((j) => j.kind === 'leg')!
  const foot = hipJoint.position[1] - LEG_LENGTH
  check('脚の下端が地面に接する（±5cm）', Math.abs(foot) < 0.05, `足元 ${foot.toFixed(3)}m`)

  const shoulder = LIMB_JOINTS.find((j) => j.kind === 'arm')!
  const hand = shoulder.position[1] - ARM_LENGTH
  check('手が腰より下、膝より上に来る', hand < hipJoint.position[1] && hand > 0.3, `手先 ${hand.toFixed(3)}m`)
}

{
  // 頂点は float32 なので、1e-8 程度の誤差は許す
  const arm = makeArmGeometry()
  arm.computeBoundingBox()
  check(
    '腕の支点（原点）が上端にある',
    Math.abs(arm.boundingBox!.max.y) < 1e-6,
    `上端 ${arm.boundingBox!.max.y.toExponential(1)}m`,
  )
  const leg = makeLegGeometry()
  leg.computeBoundingBox()
  check(
    '脚の支点（原点）が上端にある',
    Math.abs(leg.boundingBox!.max.y) < 1e-6,
    `上端 ${leg.boundingBox!.max.y.toExponential(1)}m`,
  )
}

console.log()
console.log('='.repeat(70))
console.log('歩行の姿勢')
console.log('='.repeat(70))

{
  const s = limbSwing(0.7, 1)
  check('左右の脚が逆位相で振れる', Math.abs(s.legLeft + s.legRight) < 1e-9, `${s.legLeft.toFixed(3)} / ${s.legRight.toFixed(3)}`)
  check(
    '腕は同じ側の脚と逆に振れる（対角の歩き方）',
    Math.sign(s.armRight) !== Math.sign(s.legRight),
    `右腕 ${s.armRight.toFixed(3)} / 右脚 ${s.legRight.toFixed(3)}`,
  )
  const still = limbSwing(0, 0)
  check(
    '止まっているときは振れない',
    still.legLeft === 0 && still.armRight === 0 && still.bob === 0,
  )
  const wide = limbSwing(Math.PI / 2, 1)
  check('振れ幅が 45 度を超えない', Math.abs(wide.legRight) < Math.PI / 4, `${wide.legRight.toFixed(3)} rad`)
}

{
  const scratch = createPedestrianScratch()
  const base = new THREE.Matrix4()
  const out = new THREE.Matrix4()
  let worst = 0
  const poses = [
    { x: 0, y: 0, h: 0, swing: 0 },
    { x: 12.5, y: -30.25, h: Math.PI / 2, swing: 0.4 },
    { x: -104.75, y: 88.5, h: -2.4, swing: -0.62 },
  ]
  for (const p of poses) {
    composePedestrianMatrix(scratch, p.x, p.y, p.h, 0, base)
    for (let i = 0; i < LIMB_JOINTS.length; i++) {
      composeLimbMatrix(scratch, base, i, p.swing, out)
      const root = new THREE.Group()
      root.position.set(p.x, 0, -p.y)
      root.rotation.y = p.h
      const joint = new THREE.Object3D()
      joint.position.set(...(LIMB_JOINTS[i].position as unknown as [number, number, number]))
      joint.rotation.z = p.swing
      root.add(joint)
      root.updateMatrixWorld(true)
      worst = Math.max(worst, matrixDiff(out, joint.matrixWorld))
    }
  }
  check('手足のワールド行列がシーングラフと一致する', worst < 1e-9, `最大差 ${worst.toExponential(1)}`)
}

{
  const scratch = createPedestrianScratch()
  const base = new THREE.Matrix4()
  composePedestrianMatrix(scratch, 10, 20, 0, 0, base)
  const pos = new THREE.Vector3().setFromMatrixPosition(base)
  check(
    'three.z = -enu.y になっている（車両と同じ規約）',
    Math.abs(pos.x - 10) < 1e-9 && Math.abs(pos.z + 20) < 1e-9,
    `(${pos.x.toFixed(2)}, ${pos.z.toFixed(2)})`,
  )

  const out = new THREE.Matrix4()
  composePedestrianMatrix(scratch, 0, 0, 0, 0, base)
  composeLimbMatrix(scratch, base, 0, 0.5, out)
  const forward = new THREE.Vector3(0, -0.5, 0).applyMatrix4(out)
  const rest = new THREE.Vector3(0, -0.5, 0).applyMatrix4(
    composeLimbMatrix(scratch, base, 0, 0, new THREE.Matrix4()),
  )
  check('振り角が正なら手足は前（+X）へ出る', forward.x > rest.x + 0.05, `x ${rest.x.toFixed(3)} → ${forward.x.toFixed(3)}`)
}

{
  const eye = eyePosition(5, -7, 0, 0)
  check(
    '目線は前へ出ていて、高さが EYE_HEIGHT',
    eye.x > 5 && Math.abs(eye.y - EYE_HEIGHT) < 1e-9 && Math.abs(eye.z - 7) < 1e-9,
    `(${eye.x.toFixed(2)}, ${eye.y.toFixed(2)}, ${eye.z.toFixed(2)})`,
  )
  const up = eyeLookAt(0, 0, 0, 0.5)
  const down = eyeLookAt(0, 0, 0, -0.5)
  check('pitch が正なら上を向く', up.y > EYE_HEIGHT && down.y < EYE_HEIGHT, `${down.y.toFixed(2)} / ${up.y.toFixed(2)}`)
  const east = eyeLookAt(0, 0, 0, 0)
  const north = eyeLookAt(0, 0, Math.PI / 2, 0)
  check(
    'heading=0 は東、+90 度は北を向く',
    east.x > 5 && Math.abs(east.z) < 1e-6 && Math.abs(north.x) < 1e-6 && north.z < -5,
    `東 (${east.x.toFixed(1)}, ${east.z.toFixed(1)}) / 北 (${north.x.toFixed(1)}, ${north.z.toFixed(1)})`,
  )
}

console.log()
console.log('='.repeat(70))
console.log('当たり判定')
console.log('='.repeat(70))

const buildings: MapBuilding[] = [
  { id: 0, height: 9, outline: [[10, 10], [30, 10], [30, 30], [10, 30]] },
  { id: 1, height: 12, outline: [[-40, -20], [-20, -20], [-20, -5], [-40, -5]] },
]
const index = buildBuildingIndex(buildings, 8)

check('建物の内側を内側と判定する', isInsideBuilding(index, 20, 20))
check('建物の外側を外側と判定する', !isInsideBuilding(index, 5, 20))
check('索引の外（建物が無い区画）は素通し', !isInsideBuilding(index, 5000, 5000))
check('2 棟目も引ける', isInsideBuilding(index, -30, -12) && !isInsideBuilding(index, -30, 0))

{
  // 建物の西側の壁ぎわで、東（壁向き）＋北（壁沿い）に動く
  const start = { x: 10 - PEDESTRIAN_RADIUS - 0.05, y: 20 }
  const res = resolveMove(index, start.x, start.y, 0.5, 0.5, [])
  check(
    '壁に向かう成分だけが削られ、壁沿いには動ける',
    Math.abs(res.x - start.x) < 1e-9 && res.y > start.y + 0.4 && res.blocked,
    `dx ${(res.x - start.x).toFixed(3)} / dy ${(res.y - start.y).toFixed(3)}`,
  )
  const free = resolveMove(index, -100, -100, 0.5, 0.5, [])
  check(
    '何も無ければそのまま動く',
    Math.abs(free.x + 99.5) < 1e-9 && Math.abs(free.y + 99.5) < 1e-9 && !free.blocked,
  )
}

{
  const car = { id: 3, x: 0, y: 0, heading: 0 }
  check(
    '車体の正面（長手方向）に沿って判定が伸びる',
    touchesVehicle(car, VEHICLE_HALF_LENGTH, 0) && !touchesVehicle(car, VEHICLE_HALF_LENGTH + 1.2, 0),
  )
  check(
    '車体の側面は短い',
    touchesVehicle(car, 0, VEHICLE_HALF_WIDTH) && !touchesVehicle(car, 0, VEHICLE_HALF_WIDTH + 1.2),
  )
  const turned = { id: 3, x: 0, y: 0, heading: Math.PI / 2 }
  check(
    '車体を 90 度回すと当たり方も回る',
    touchesVehicle(turned, 0, VEHICLE_HALF_LENGTH) && !touchesVehicle(turned, VEHICLE_HALF_LENGTH, 0),
  )
  const blocked = resolveMove(index, -100, -100, 0, 0.5, [{ id: 0, x: -100, y: -99, heading: Math.PI / 2 }])
  check('車体は歩行者を通さない', Math.abs(blocked.y + 100) < 1e-9 && blocked.blocked)
}

{
  const cars = [
    { id: 1, x: 5, y: 0, heading: 0 },
    { id: 2, x: -5, y: 0, heading: 0 },
    { id: 3, x: 0, y: 40, heading: 0 },
  ]
  check('正面の車両を拾う', aimedVehicle(0, 0, 0, cars) === 1)
  check('背後の車両は拾わない', aimedVehicle(0, 0, Math.PI / 2, cars) === -1)
  check(
    `${AIM_DISTANCE_M}m より遠い車両は拾わない`,
    aimedVehicle(0, 0, Math.PI / 2, [cars[2]]) === -1,
  )
  check(
    '真横に接していれば角度を問わず拾える（乗り込む直前）',
    aimedVehicle(0, VEHICLE_HALF_WIDTH + 0.5, Math.PI / 2, [{ id: 7, x: 0, y: 0, heading: 0 }]) === 7,
  )
}

{
  const car = { id: 0, x: 0, y: 0, heading: 0 }
  const spot = alightPosition(car)
  check(
    '降車位置は車体の左（左側通行なので歩道側）',
    spot.y > VEHICLE_HALF_WIDTH && Math.abs(spot.x) < 1e-9,
    `(${spot.x.toFixed(2)}, ${spot.y.toFixed(2)})`,
  )
  check('降車位置は車体と重ならない', !touchesVehicle(car, spot.x, spot.y))

  // 歩道側が建物で塞がっている区画（道路際まで建物が迫っているとき）
  const walls: MapBuilding[] = [
    { id: 0, height: 9, outline: [[-8, 1.5], [8, 1.5], [8, 12], [-8, 12]] },
  ]
  const blocked = buildBuildingIndex(walls)
  const away = alightSpot(car, blocked)
  check(
    '歩道側が建物なら反対側へ降ろす',
    !touchesBuilding(blocked, away.x, away.y) && away.y < 0,
    `(${away.x.toFixed(2)}, ${away.y.toFixed(2)})`,
  )

  const both: MapBuilding[] = [
    { id: 0, height: 9, outline: [[-8, 1.5], [8, 1.5], [8, 12], [-8, 12]] },
    { id: 1, height: 9, outline: [[-8, -12], [8, -12], [8, -1.5], [-8, -1.5]] },
  ]
  const narrow = buildBuildingIndex(both)
  const behind = alightSpot(car, narrow)
  check(
    '左右とも建物なら車の後ろへ降ろす',
    !touchesBuilding(narrow, behind.x, behind.y) && behind.x < -VEHICLE_HALF_LENGTH,
    `(${behind.x.toFixed(2)}, ${behind.y.toFixed(2)})`,
  )

  const sealed = buildBuildingIndex([
    { id: 0, height: 9, outline: [[-30, -30], [30, -30], [30, 30], [-30, 30]] },
  ])
  const last = alightSpot(car, sealed)
  check(
    '逃げ場が無ければ車の位置（建物の中には置かない）',
    last.x === car.x && last.y === car.y,
  )

  // 建物が無ければ、歩道側に降ろす従来の位置と一致する
  const plain = alightSpot(car, null)
  check(
    '建物が無ければ従来どおり歩道側',
    Math.abs(plain.x - spot.x) < 1e-9 && Math.abs(plain.y - spot.y) < 1e-9,
  )
}

console.log()
console.log('='.repeat(70))
console.log('NPC 歩行者（frame.pedestrians の補間と色の散らし）')
console.log('='.repeat(70))

function npc(over: Partial<NpcPedestrianState>): NpcPedestrianState {
  return { id: 0, x: 0, y: 0, heading: 0, stride: 0, crossing: false, ...over }
}

{
  const pose = createPedestrianPose()
  samplePedestrian(npc({ x: 10, y: 4 }), npc({ x: 0, y: 0 }), 0.25, pose)
  check(
    '2 フレームの間を補間する',
    Math.abs(pose.x - 2.5) < 1e-9 && Math.abs(pose.y - 1) < 1e-9,
    `(${pose.x.toFixed(2)}, ${pose.y.toFixed(2)})`,
  )

  // 見えない場所の歩行者は車の近くへ回される（protocol.md 2.3）。
  //   跳んだ先まで線を引くと、街を横切る人が見える
  const jump = TELEPORT_DISTANCE_M + 5
  samplePedestrian(npc({ x: jump }), npc({ x: 0 }), 0.5, pose)
  check(
    `${TELEPORT_DISTANCE_M}m を超えて跳んだら補間しない`,
    Math.abs(pose.x - jump) < 1e-9,
    `x=${pose.x.toFixed(2)}`,
  )

  samplePedestrian(
    npc({ heading: (Math.PI * 179) / 180 * -1 }),
    npc({ heading: (Math.PI * 179) / 180 }),
    0.5,
    pose,
  )
  check(
    '向きは近いほうへ回す（179° → -179° は 2° だけ動く）',
    Math.abs(Math.abs(pose.heading) - Math.PI) < 0.02,
    `${((pose.heading * 180) / Math.PI).toFixed(1)}°`,
  )

  // 位相は 2π で折り返すので、角度と同じ扱いにしないと手足が逆回りする
  samplePedestrian(npc({ stride: 0.1 }), npc({ stride: Math.PI * 2 - 0.1 }), 0.5, pose)
  check(
    '歩行位相も近いほうへ回す（2π の折り返しで手足が逆回りしない）',
    Math.abs(Math.atan2(Math.sin(pose.stride), Math.cos(pose.stride))) < 0.02,
    `${pose.stride.toFixed(3)} rad`,
  )

  samplePedestrian(npc({ x: 3, crossing: true }), undefined, 1, pose)
  check('前フレームが無ければそのまま置く', pose.x === 3 && pose.crossing)
}

{
  const hues = Array.from({ length: 64 }, (_, i) => npcHueOffset(i))
  check(
    '服の色は 0.0〜1.0 に収まる',
    hues.every((h) => h >= 0 && h < 1),
  )
  let closest = 1
  for (let i = 1; i < hues.length; i++) {
    closest = Math.min(closest, Math.abs(hues[i] - hues[i - 1]))
  }
  check(
    '隣り合うスロット番号で色が似ない',
    closest > 0.3,
    `隣どうしの最小差 ${closest.toFixed(3)}`,
  )
  const buckets = new Set(hues.map((h) => Math.floor(h * 8)))
  check('64 人ぶんが色相全体へ散る', buckets.size === 8, `${buckets.size} / 8 区画`)
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
