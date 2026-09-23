/** 認識結果オーバーレイの座標変換を検証する（ブラウザ不要）。 */

import {
  BACKEND_CAMERA,
  BACKEND_FOCAL_PX,
  projectBox,
  projectToViewport,
} from '../src/scene/detectionProjection.ts'
import { DRIVER_FOV_DEG } from '../src/scene/cameraMath.ts'

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

console.log(failures === 0 ? '\nすべて OK' : `\n${failures} 件が NG`)
process.exit(failures === 0 ? 0 : 1)
