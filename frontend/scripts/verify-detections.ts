/** 認識結果オーバーレイの座標変換と、運転席の 4 分割（前後左右のカメラ）を検証する（ブラウザ不要）。 */

import { readFileSync } from 'node:fs'
import * as THREE from 'three'
import {
  BACKEND_CAMERA,
  BACKEND_FOCAL_PX,
  projectBox,
  projectToViewport,
} from '../src/scene/detectionProjection.ts'
import {
  DRIVER_FOV_DEG,
  SURROUND_CAMERAS,
  SURROUND_FOV_DEG,
  surroundEye,
  surroundLookAt,
} from '../src/scene/cameraMath.ts'
import { QUAD_PANES, frontPanePointer, quadViewport } from '../src/scene/quadLayout.ts'
import { GUIDE_LENGTH_M, guideColor, reverseGuideLines } from '../src/scene/reverseGuideGeometry.ts'
import { assistSummary, paneAssist } from '../src/scene/assistLabels.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures += 1
}

function near(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= tol
}

const DEG = Math.PI / 180
const WIDE = 16 / 9
const SQUARE = 1.0

console.log('擬似カメラ諸元')
console.log(
  `  ${BACKEND_CAMERA.width}x${BACKEND_CAMERA.height} / 水平 ${BACKEND_CAMERA.fovDeg} 度` +
    ` / 焦点距離 ${BACKEND_FOCAL_PX.toFixed(2)} px`,
)
const backendFovV = 2 * Math.atan(BACKEND_CAMERA.height * 0.5 / BACKEND_FOCAL_PX) / DEG
console.log(`  擬似カメラの垂直画角 = ${backendFovV.toFixed(2)} 度`)
console.log(`  three の垂直画角     = ${DRIVER_FOV_DEG} 度`)

console.log('\n中心と対称性')
for (const aspect of [WIDE, SQUARE, 4 / 3]) {
  const c = projectToViewport(0.5, 0.5, aspect)
  check(
    `画像中心は画面中心に写る (aspect=${aspect.toFixed(2)})`,
    near(c.x, 0.5) && near(c.y, 0.5),
    `(${c.x.toFixed(4)}, ${c.y.toFixed(4)})`,
  )
}
{
  const l = projectToViewport(0.2, 0.5, WIDE)
  const r = projectToViewport(0.8, 0.5, WIDE)
  check('左右対称', near(l.x - 0.5, -(r.x - 0.5)), `${l.x.toFixed(4)} / ${r.x.toFixed(4)}`)
  const t = projectToViewport(0.5, 0.2, WIDE)
  const b = projectToViewport(0.5, 0.8, WIDE)
  check('上下対称', near(t.y - 0.5, -(b.y - 0.5)), `${t.y.toFixed(4)} / ${b.y.toFixed(4)}`)
}

console.log('\n単調性（並び順が入れ替わらない）')
{
  let monotonic = true
  let prev = -Infinity
  for (let i = 0; i <= 10; i += 1) {
    const p = projectToViewport(i / 10, 0.5, WIDE)
    if (p.x <= prev) monotonic = false
    prev = p.x
  }
  check('x は nx について単調増加', monotonic)

  monotonic = true
  prev = -Infinity
  for (let i = 0; i <= 10; i += 1) {
    const p = projectToViewport(0.5, i / 10, WIDE)
    if (p.y <= prev) monotonic = false
    prev = p.y
  }
  check('y は ny について単調増加', monotonic)
}

console.log('\n画角の違いが実際に効いていること（素通しとの差）')
{
  const bottom = projectToViewport(0.5, 1.0, WIDE)
  check(
    '画像下端は画面下端より内側',
    bottom.y < 0.999 && bottom.y > 0.5,
    `y=${bottom.y.toFixed(4)}（素通しなら 1.0、ずれ ${((1 - bottom.y) * 100).toFixed(1)}%）`,
  )
  const right = projectToViewport(1.0, 0.5, WIDE)
  check(
    '画像右端は画面右端より内側（16:9）',
    right.x < 0.999 && right.x > 0.5,
    `x=${right.x.toFixed(4)}（素通しなら 1.0、ずれ ${((1 - right.x) * 100).toFixed(1)}%）`,
  )
  check(
    'ずれは丸め誤差では説明できない大きさ',
    1 - bottom.y > 0.05 || 1 - right.x > 0.05,
  )
}
console.log('')
console.log('検出枠が出る範囲の広さ（擬似カメラの画いっぱい。運転席の画角を広げると縮む）')
{
  // 下限。**運転席の画角（DRIVER_FOV_DEG）を広げるほど検出枠の出る範囲は狭くなる**
  // ので、下回ったら画角を見直すか、擬似カメラ（percep/types.py の CameraSpec）と
  // 揃えること。素通し（画角が一致）なら 100% になる
  const MIN_W = 0.3
  const MIN_H = 0.4
  const MIN_AREA = 0.12

  for (const aspect of [WIDE, 4 / 3]) {
    const scope = projectBox([0, 0, 1, 1], aspect)
    const area = scope.width * scope.height
    check(
      `検出枠が出る範囲の横幅が画面の ${(MIN_W * 100).toFixed(0)}% 以上 (aspect=${aspect.toFixed(2)})`,
      scope.width >= MIN_W,
      `${(scope.width * 100).toFixed(1)}%`,
    )
    check(
      `検出枠が出る範囲の高さが画面の ${(MIN_H * 100).toFixed(0)}% 以上`,
      scope.height >= MIN_H,
      `${(scope.height * 100).toFixed(1)}%`,
    )
    check(
      `検出枠が出る範囲の面積が画面の ${(MIN_AREA * 100).toFixed(0)}% 以上`,
      area >= MIN_AREA,
      `${(area * 100).toFixed(1)}%`,
    )
  }

  const scope = projectBox([0, 0, 1, 1], WIDE)
  check(
    '検出枠が出る範囲は画面の中央にある',
    near(scope.left + scope.width / 2, 0.5, 1e-9) &&
      near(scope.top + scope.height / 2, 0.5, 1e-9),
  )
}


console.log('\nアスペクト比の効き方')
{
  const a = projectToViewport(0.8, 0.8, WIDE)
  const b = projectToViewport(0.8, 0.8, SQUARE)
  check('横位置はアスペクト比で変わる', !near(a.x, b.x), `${a.x.toFixed(4)} / ${b.x.toFixed(4)}`)
  check('縦位置はアスペクト比で変わらない', near(a.y, b.y), `${a.y.toFixed(4)}`)
  check('横に広いほど中心へ寄る', Math.abs(a.x - 0.5) < Math.abs(b.x - 0.5))
}

console.log('\nボックスの矩形化')
{
  const rect = projectBox([0.3, 0.3, 0.7, 0.7], WIDE)
  check(
    '中心対称なボックスは画面中心に来る',
    near(rect.left + rect.width / 2, 0.5, 1e-9) && near(rect.top + rect.height / 2, 0.5, 1e-9),
    `left=${rect.left.toFixed(4)} top=${rect.top.toFixed(4)}`,
  )
  check('幅・高さが正', rect.width > 0 && rect.height > 0)

  const clipped = projectBox([-0.5, -0.5, 1.5, 1.5], WIDE)
  const inside =
    clipped.left >= 0 &&
    clipped.top >= 0 &&
    clipped.left + clipped.width <= 1 + 1e-9 &&
    clipped.top + clipped.height <= 1 + 1e-9
  check('視野外のボックスも 0〜1 に丸められる', inside)

  const flipped = projectBox([0.7, 0.7, 0.3, 0.3], WIDE)
  check('逆順の座標でも矩形になる', flipped.width > 0 && flipped.height > 0)
}

console.log('\n周囲カメラの取り付け（バックエンドの percep/types.py と同じ値か）')
{
  const source = readFileSync(new URL('../../backend/app/percep/types.py', import.meta.url), 'utf8')
  const pitch = Number(/^SURROUND_PITCH_DEG = (-?[\d.]+)/m.exec(source)?.[1])
  const read = (name: string): Record<string, number> => {
    const block = new RegExp(`^${name} = CameraSpec\\(([\\s\\S]*?)^\\)`, 'm').exec(source)?.[1] ?? ''
    const out: Record<string, number> = {}
    for (const m of block.matchAll(/(\w+)=(-?[\d.]+|SURROUND_PITCH_DEG)/g)) {
      out[m[1]] = m[2] === 'SURROUND_PITCH_DEG' ? pitch : Number(m[2])
    }
    return out
  }
  for (const cam of SURROUND_CAMERAS) {
    const py = read(`${cam.key.toUpperCase()}_CAMERA`)
    const same =
      py.forward === cam.forward &&
      py.right === cam.right &&
      py.eye_height === cam.height &&
      py.yaw_deg === cam.yawDeg &&
      py.pitch_deg === cam.pitchDeg
    check(
      `${cam.key}: 前 / 右 / 高さ / 向き / 俯角がバックエンドと一致`,
      same,
      `フロント ${[cam.forward, cam.right, cam.height, cam.yawDeg, cam.pitchDeg].join(' / ')}` +
        ` ・バックエンド ${[py.forward, py.right, py.eye_height, py.yaw_deg, py.pitch_deg].join(' / ')}`,
    )
  }
  const fovV = (2 * Math.atan(BACKEND_CAMERA.height * 0.5 / BACKEND_FOCAL_PX)) / DEG
  check('周囲のペインの垂直画角は擬似カメラの垂直画角と同じ', near(SURROUND_FOV_DEG, fovV, 1e-9), `${SURROUND_FOV_DEG.toFixed(3)} 度`)
}

/** バックエンドの `project_components` と同じ式（ENU の点 → 擬似カメラの正規化座標） */
function backendProject(
  cam: (typeof SURROUND_CAMERAS)[number],
  car: { x: number; y: number; heading: number },
  p: { x: number; y: number; z: number },
): { u: number; v: number; depth: number } {
  const c = Math.cos(car.heading)
  const s = Math.sin(car.heading)
  const ex = car.x + c * cam.forward + s * cam.right
  const ey = car.y + s * cam.forward - c * cam.right
  const yaw = car.heading + cam.yawDeg * DEG
  const pitch = cam.pitchDeg * DEG
  const rx = p.x - ex
  const ry = p.y - ey
  const rz = p.z - cam.height
  const fwd = rx * Math.cos(yaw) + ry * Math.sin(yaw)
  const right = rx * Math.sin(yaw) - ry * Math.cos(yaw)
  const depth = Math.cos(pitch) * fwd + Math.sin(pitch) * rz
  const up = -Math.sin(pitch) * fwd + Math.cos(pitch) * rz
  return {
    u: 0.5 + (BACKEND_FOCAL_PX * right) / depth / BACKEND_CAMERA.width,
    v: 0.5 - (BACKEND_FOCAL_PX * up) / depth / BACKEND_CAMERA.height,
    depth,
  }
}

/** three の周囲カメラ（QuadViewRenderer と同じ置き方）で、ENU の点がペインのどこに写るか */
function threeProject(
  cam: (typeof SURROUND_CAMERAS)[number],
  car: { x: number; y: number; heading: number },
  p: { x: number; y: number; z: number },
  aspect: number,
): { x: number; y: number } {
  const camera = new THREE.PerspectiveCamera(SURROUND_FOV_DEG, aspect, 0.15, 8000)
  const eye = surroundEye(cam, car.x, car.y, car.heading)
  const look = surroundLookAt(cam, car.x, car.y, car.heading)
  camera.position.set(eye.x, eye.y, eye.z)
  camera.lookAt(look.x, look.y, look.z)
  camera.updateMatrixWorld()
  const ndc = new THREE.Vector3(p.x, p.z, -p.y).project(camera)
  return { x: (ndc.x + 1) / 2, y: (1 - ndc.y) / 2 }
}

console.log('\n周囲カメラの枠は three のカメラの画と重なるか（擬似カメラの式 → ペインの座標 と three の投影を突き合わせる）')
{
  const cars = [
    { x: 0, y: 0, heading: 0 },
    { x: 12.5, y: -7.25, heading: 1.1 },
    { x: -40, y: 22, heading: -2.4 },
  ]
  for (const cam of SURROUND_CAMERAS) {
    let worst = 0
    let count = 0
    for (const car of cars) {
      for (const bearing of [-25, -10, 0, 10, 25]) {
        for (const dist of [4, 9, 25]) {
          for (const z of [0, 0.75, 1.6]) {
            const yaw = car.heading + (cam.yawDeg + bearing) * DEG
            const c = Math.cos(car.heading)
            const s = Math.sin(car.heading)
            const ex = car.x + c * cam.forward + s * cam.right
            const ey = car.y + s * cam.forward - c * cam.right
            const p = { x: ex + Math.cos(yaw) * dist, y: ey + Math.sin(yaw) * dist, z }
            const b = backendProject(cam, car, p)
            if (b.depth <= 0.5 || b.u < 0 || b.u > 1 || b.v < 0 || b.v > 1) continue
            for (const aspect of [WIDE, 4 / 3]) {
              const viaBox = projectToViewport(b.u, b.v, aspect, SURROUND_FOV_DEG)
              const viaThree = threeProject(cam, car, p, aspect)
              worst = Math.max(worst, Math.abs(viaBox.x - viaThree.x), Math.abs(viaBox.y - viaThree.y))
              count += 1
            }
          }
        }
      }
    }
    check(`${cam.key}: ${count} 点で枠の位置と three の画が一致`, count > 0 && worst < 1e-6, `最大差 ${worst.toExponential(2)}`)
  }
}

console.log('\n左右の向き（手で決めずに投影で確かめる）')
{
  const car = { x: 3, y: -2, heading: 0.7 }
  const at = (fx: number, fy: number, z = 0.8) => {
    const c = Math.cos(car.heading)
    const s = Math.sin(car.heading)
    return { x: car.x + fx * c - fy * s, y: car.y + fx * s + fy * c, z }
  }
  const pane = (key: string) => SURROUND_CAMERAS.find((c) => c.key === key)!
  const rearLeft = threeProject(pane('rear'), car, at(-12, 2), WIDE)
  const rearRight = threeProject(pane('rear'), car, at(-12, -2), WIDE)
  check('後方: 車の左後ろは画の右、右後ろは画の左に写る（後ろを向いているため）', rearLeft.x > 0.5 && rearRight.x < 0.5,
    `左後ろ x=${rearLeft.x.toFixed(3)} / 右後ろ x=${rearRight.x.toFixed(3)}`)
  const leftAhead = threeProject(pane('left'), car, at(3, 8), WIDE)
  const leftBehind = threeProject(pane('left'), car, at(-3, 8), WIDE)
  check('左側方: 車の前寄りは画の右、後ろ寄りは画の左に写る', leftAhead.x > 0.5 && leftBehind.x < 0.5,
    `前寄り x=${leftAhead.x.toFixed(3)} / 後ろ寄り x=${leftBehind.x.toFixed(3)}`)
  const rightAhead = threeProject(pane('right'), car, at(3, -8), WIDE)
  const rightBehind = threeProject(pane('right'), car, at(-3, -8), WIDE)
  check('右側方: 車の前寄りは画の左、後ろ寄りは画の右に写る', rightAhead.x < 0.5 && rightBehind.x > 0.5,
    `前寄り x=${rightAhead.x.toFixed(3)} / 後ろ寄り x=${rightBehind.x.toFixed(3)}`)
  const ground = threeProject(pane('rear'), car, at(-8, 0, 0), WIDE)
  const high = threeProject(pane('rear'), car, at(-8, 0, 3), WIDE)
  check('後方: 路面は高い所より画の下に写る（上下が逆になっていない）', ground.y > high.y,
    `路面 y=${ground.y.toFixed(3)} / 高さ 3m y=${high.y.toFixed(3)}`)
}

console.log('\n4 分割の配置')
{
  const w = 1600
  const h = 900
  const rects = QUAD_PANES.map((_, i) => quadViewport(i, w, h))
  const area = rects.reduce((a, r) => a + r.w * r.h, 0)
  check('4 枚で画面をすき間なく覆う', near(area, w * h, 1e-6), `${area} / ${w * h}`)
  const front = rects[QUAD_PANES.indexOf('front')]
  const rear = rects[QUAD_PANES.indexOf('rear')]
  check('前方は左上（WebGL の viewport は左下原点なので y が大きい）', front.x === 0 && front.y === h / 2)
  check('後方は右上', rear.x === w / 2 && rear.y === h / 2)
  check('左側方は左下・右側方は右下',
    quadViewport(QUAD_PANES.indexOf('left'), w, h).y === 0 && quadViewport(QUAD_PANES.indexOf('right'), w, h).x === w / 2)
  const center = frontPanePointer(w / 4, h / 4, w, h)
  const corner = frontPanePointer(0, 0, w, h)
  check('前方ペインの中心はカメラの中心、左上は (-1, 1)',
    !!center && near(center.x, 0) && near(center.y, 0) && !!corner && near(corner.x, -1) && near(corner.y, 1))
  check('前方ペインの外のクリックは当てない', frontPanePointer(w * 0.75, h * 0.25, w, h) === null && frontPanePointer(w * 0.25, h * 0.75, w, h) === null)
  check('ペインの縦横比は画面と同じ（前方は画面のカメラをそのまま使うため）', near(front.w / front.h, w / h))
}

console.log('\n後退の予測ガイド線')
{
  const straight = reverseGuideLines(0)
  const halfW = 0.9
  check('まっすぐなら左右の線は車幅の位置で平行', straight.left.every((p) => near(p.y, halfW, 1e-9)) && straight.right.every((p) => near(p.y, -halfW, 1e-9)))
  check('線は後ろのバンパーから始まり、後ろへ 5m 伸びる',
    near(straight.left[0].x, -2.2, 1e-9) && near(straight.left[straight.left.length - 1].x, -2.2 - GUIDE_LENGTH_M, 1e-9))
  const left = reverseGuideLines(0.3)
  const end = left.left[left.left.length - 1]
  check('ハンドルを左へ切って下がると、車の後ろは左へ回り込む', end.y > halfW + 0.5, `左の線の終わり y=${end.y.toFixed(2)}`)
  // backend/app/sim/vehicle.py と同じ更新式（車体の中心を、向きの変化率 v/L·tanδ で積分する）で下がってみる
  for (const steer of [-0.4, -0.1, 0.25, 0.5]) {
    const guide = reverseGuideLines(steer)
    let x = 0
    let y = 0
    let h = 0
    let moved = 0
    const v = -0.5
    const dt = 0.001
    while (moved < GUIDE_LENGTH_M) {
      h += (v / 2.6) * Math.tan(steer) * dt
      x += v * Math.cos(h) * dt
      y += v * Math.sin(h) * dt
      moved += Math.abs(v) * dt
    }
    const cornerX = x + Math.cos(h) * -2.2 - Math.sin(h) * halfW
    const cornerY = y + Math.sin(h) * -2.2 + Math.cos(h) * halfW
    const g = guide.left[guide.left.length - 1]
    const miss = Math.hypot(g.x - cornerX, g.y - cornerY)
    check(`舵角 ${steer}rad: ガイド線の終わりが車両モデルの積分と一致`, miss < 0.02, `ずれ ${miss.toFixed(4)}m`)
  }
  check('目盛りの色: 1m まで赤・2m まで黄・その先は緑',
    guideColor(0.5) !== guideColor(1.5) && guideColor(1.5) !== guideColor(2.5) && guideColor(1) === guideColor(0.2))
}

console.log('\n介入の表示（ペインごと）')
{
  check('巻き込み確認は曲がる側のペインだけに出す',
    paneAssist('blind_spot', 'left', -1) !== null && paneAssist('blind_spot', 'right', -1) === null)
  check('後退 AEB は後方ペイン', paneAssist('reverse_stop', 'rear', 0)?.level === 'stop' && paneAssist('reverse_stop', 'front', 0) === null)
  check('前方の障害物で止めているのは前方ペイン', paneAssist('front_hold', 'front', 0)?.level === 'stop')
  check('介入していなければ何も出さない', assistSummary('') === null && QUAD_PANES.every((p) => paneAssist('', p, 0) === null))
}

console.log(failures === 0 ? '\nすべて OK' : `\n${failures} 件が NG`)
process.exit(failures === 0 ? 0 : 1)
