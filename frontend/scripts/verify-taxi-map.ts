/** スマホ画面の 2D 地図の座標変換を検証する（ブラウザ不要）。 */

import {
  FIT_MIN_SPAN_M,
  ZOOM_MAX,
  ZOOM_MIN,
  applyLayerTransform,
  baseScale,
  boundsCenter,
  clampCenter,
  clampZoom,
  createProjection,
  fitView,
  layerSizeFor,
  layerTransform,
  lerpAngle,
  lerpView,
  normalizeAngle,
  rotationForHeading,
  toCanvas,
  toEnu,
  viewsClose,
  visibleBounds,
  zoomAround,
} from '../src/panel/taxiMapMath.ts'
import type { MapProjection, MapView } from '../src/panel/taxiMapMath.ts'
import type { MapBounds, Vec2 } from '../src/types/protocol.ts'

/** 北向き前提の軸ごとの投影。**回転があるときは `toCanvas()` を直に使うこと。** */
const cvx = (p: MapProjection, x: number): number => toCanvas(p, x, p.centerY)[0]
const cvy = (p: MapProjection, y: number): number => toCanvas(p, p.centerX, y)[1]

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

/** 銀座（0.84km 四方）と金沢（12.3km 四方）を模した範囲 */
const GINZA: MapBounds = { minX: -420, maxX: 420, minY: -420, maxY: 420 }
const KANAZAWA: MapBounds = { minX: -6150, maxX: 6150, minY: -6150, maxY: 6150 }
/** 縦横が違う範囲（収まり方の確認用） */
const OBLONG: MapBounds = { minX: -100, maxX: 500, minY: -50, maxY: 150 }

const W = 320
const H = 320

/** 実装と同じ「画面の対角を一辺とする正方形」のレイヤ投影を作る。 */
function squareLayer(view: MapView, bounds: MapBounds = GINZA): MapProjection {
  const side = layerSizeFor(W, H, true)
  return { ...createProjection(bounds, W, H, view), width: side, height: side }
}

/** レイヤを貼り直したとき、地物が本来の画素からどれだけずれるか [px]。 */
function pasteError(
  layer: MapProjection,
  current: MapProjection,
  points: readonly Vec2[] = [
    [0, 0],
    [60, 40],
    [-120, 75],
  ],
): number {
  const t = layerTransform(layer, current)
  let worst = 0
  for (const [x, y] of points) {
    const [lx, ly] = toCanvas(layer, x, y)
    const [px, py] = applyLayerTransform(t, layer.width, layer.height, lx, ly)
    const [ex, ey] = toCanvas(current, x, y)
    worst = Math.max(worst, Math.hypot(px - ex, py - ey))
  }
  return worst
}

console.log('='.repeat(70))
console.log('投影（ENU ↔ キャンバス）')
console.log('='.repeat(70))

for (const [name, bounds] of [
  ['銀座', GINZA],
  ['金沢', KANAZAWA],
  ['横長', OBLONG],
] as const) {
  const [cx, cy] = boundsCenter(bounds)
  const p = createProjection(bounds, W, H, { zoom: 1, centerX: cx, centerY: cy, rotation: 0 })

  const cornerX = cvx(p, bounds.minX)
  const cornerY = cvy(p, bounds.maxY)
  const farX = cvx(p, bounds.maxX)
  const farY = cvy(p, bounds.minY)
  check(
    `${name}: 倍率 1 でマップ全体が画面に収まる`,
    cornerX >= -0.01 && cornerY >= -0.01 && farX <= W + 0.01 && farY <= H + 0.01,
    `左上 (${cornerX.toFixed(1)}, ${cornerY.toFixed(1)}) 右下 (${farX.toFixed(1)}, ${farY.toFixed(1)})`,
  )

  let worst = 0
  for (const [x, y] of [
    [bounds.minX, bounds.minY],
    [bounds.maxX, bounds.maxY],
    [cx, cy],
    [cx + 37.5, cy - 12.25],
  ]) {
    const back = toEnu(p, cvx(p, x), cvy(p, y))
    worst = Math.max(worst, Math.abs(back[0] - x), Math.abs(back[1] - y))
  }
  check(`${name}: 投影 → 逆投影で元に戻る`, worst < 1e-6, `最大差 ${worst.toExponential(1)}m`)
}

{
  const [cx, cy] = boundsCenter(GINZA)
  const p = createProjection(GINZA, W, H, { zoom: 1, centerX: cx, centerY: cy, rotation: 0 })
  check(
    '中心が画面の中央に来る',
    Math.abs(cvx(p, cx) - W / 2) < 1e-9 && Math.abs(cvy(p, cy) - H / 2) < 1e-9,
  )
  check(
    '北（+y）が上、東（+x）が右',
    cvy(p, cy + 100) < cvy(p, cy) && cvx(p, cx + 100) > cvx(p, cx),
    `y: ${cvy(p, cy + 100).toFixed(1)} < ${cvy(p, cy).toFixed(1)}`,
  )
}

console.log()
console.log('='.repeat(70))
console.log('拡大と移動')
console.log('='.repeat(70))

{
  check('倍率の下限で頭打ちになる', clampZoom(0.01) === ZOOM_MIN && clampZoom(-5) === ZOOM_MIN)
  check('倍率の上限で頭打ちになる', clampZoom(1e6) === ZOOM_MAX)
  check('倍率が数値でなければ下限に落ちる', clampZoom(NaN) === ZOOM_MIN)

  const [cx, cy] = boundsCenter(GINZA)
  const base = createProjection(GINZA, W, H, { zoom: 1, centerX: cx, centerY: cy, rotation: 0 })
  const zoomed = createProjection(GINZA, W, H, { zoom: 4, centerX: cx, centerY: cy, rotation: 0 })
  check(
    '倍率 4 で 1m あたりのピクセル数がちょうど 4 倍',
    Math.abs(zoomed.scale / base.scale - 4) < 1e-9,
    `${base.scale.toFixed(4)} → ${zoomed.scale.toFixed(4)} px/m`,
  )

  const box = visibleBounds(zoomed)
  check(
    '拡大すると見えている範囲が狭くなる',
    box.maxX - box.minX < GINZA.maxX - GINZA.minX,
    `${(box.maxX - box.minX).toFixed(0)}m 幅`,
  )
}

{
  // 画面の左上（マップの北西寄り）を掴んだままホイールを回す
  const [cx, cy] = boundsCenter(GINZA)
  const view = { zoom: 1, centerX: cx, centerY: cy, rotation: 0 }
  const anchor: [number, number] = [-300, 260]
  const before = createProjection(GINZA, W, H, view)
  const canvasBefore = [cvx(before, anchor[0]), cvy(before, anchor[1])]

  const next = zoomAround(view, GINZA, anchor, 2)
  const after = createProjection(GINZA, W, H, next)
  const canvasAfter = [cvx(after, anchor[0]), cvy(after, anchor[1])]

  check(
    'ホイールで拡大しても、掴んだ点は画面上で動かない',
    Math.abs(canvasBefore[0] - canvasAfter[0]) < 1e-6 &&
      Math.abs(canvasBefore[1] - canvasAfter[1]) < 1e-6,
    `(${canvasBefore[0].toFixed(2)}, ${canvasBefore[1].toFixed(2)}) → ` +
      `(${canvasAfter[0].toFixed(2)}, ${canvasAfter[1].toFixed(2)})`,
  )
  check('倍率が上がっている', next.zoom === 2, `${next.zoom}`)

  const capped = zoomAround({ zoom: ZOOM_MAX, centerX: cx, centerY: cy, rotation: 0 }, GINZA, anchor, 2)
  check('上限に達していれば中心も動かさない', capped.centerX === cx && capped.centerY === cy)
}

{
  const far = clampCenter({ zoom: 3, centerX: 99999, centerY: -99999, rotation: 0 }, GINZA)
  check(
    '中心はマップの範囲から出ない',
    far.centerX === GINZA.maxX && far.centerY === GINZA.minY,
    `(${far.centerX}, ${far.centerY})`,
  )
}

console.log()
console.log('='.repeat(70))
console.log('配車に合わせた自動ズーム')
console.log('='.repeat(70))

{
  const empty = fitView(GINZA, [], W, H)
  const [cx, cy] = boundsCenter(GINZA)
  check(
    '収めるものが無ければマップ全体へ戻る',
    empty.zoom === ZOOM_MIN && empty.centerX === cx && empty.centerY === cy,
    `zoom ${empty.zoom}`,
  )

  const pair: Vec2[] = [
    [-200, -150],
    [120, 90],
  ]
  const fitted = fitView(GINZA, pair, W, H)
  const p = createProjection(GINZA, W, H, fitted)
  const inside = pair.every(([x, y]) => {
    const cxp = cvx(p, x)
    const cyp = cvy(p, y)
    return cxp >= 0 && cxp <= W && cyp >= 0 && cyp <= H
  })
  check('2 点が画面の中に収まる', inside)
  check(
    '中心が 2 点の真ん中に来る',
    Math.abs(fitted.centerX - (pair[0][0] + pair[1][0]) / 2) < 1e-9 &&
      Math.abs(fitted.centerY - (pair[0][1] + pair[1][1]) / 2) < 1e-9,
  )
  check('全体表示より寄っている', fitted.zoom > ZOOM_MIN, `zoom ${fitted.zoom.toFixed(2)}`)

  // 余白: 2 点がちょうど端に貼り付かないこと
  const margin = Math.min(
    cvx(p, pair[0][0]),
    W - cvx(p, pair[1][0]),
    cvy(p, pair[1][1]),
    H - cvy(p, pair[0][1]),
  )
  check('端に貼り付かず余白が残る', margin > 10, `最小の余白 ${margin.toFixed(1)}px`)
}

{
  // 1 点だけのときに寄りすぎないこと（乗車地点で待っている場面）
  const single = fitView(GINZA, [[50, -50]], W, H)
  const p = createProjection(GINZA, W, H, single)
  const spanM = W / p.scale
  check(
    `1 点でも ${FIT_MIN_SPAN_M}m 程度は見渡せる`,
    spanM >= FIT_MIN_SPAN_M * 0.9,
    `画面の横幅 ${spanM.toFixed(0)}m`,
  )
}

{
  // 金沢（12.3km 四方）で 300m の区間へ寄る。**エリアが広いほど大きな倍率が要る**
  const near = fitView(KANAZAWA, [[0, 0], [200, 220]], W, H)
  check(
    '広いマップでも寄れる（上限で頭打ちにならない）',
    near.zoom < ZOOM_MAX,
    `zoom ${near.zoom.toFixed(1)} / 上限 ${ZOOM_MAX}`,
  )
  const p = createProjection(KANAZAWA, W, H, near)
  check('寄った先の 1px が 2m 未満', 1 / p.scale < 2, `${(1 / p.scale).toFixed(2)} m/px`)

  const tightest = createProjection(KANAZAWA, W, H, { zoom: ZOOM_MAX, centerX: 0, centerY: 0, rotation: 0 })
  check(
    '上限まで寄れば交差点 1 つぶん（200m 以下）は見える',
    W / tightest.scale <= 200,
    `画面の横幅 ${(W / tightest.scale).toFixed(0)}m`,
  )
}

console.log()
console.log('='.repeat(70))
console.log('見え方の補間とレイヤの貼り替え')
console.log('='.repeat(70))

{
  const from: MapView = { zoom: 1, centerX: 0, centerY: 0, rotation: 0 }
  const to = { zoom: 16, centerX: 300, centerY: -200, rotation: 0 }
  check('t=0 は始点そのもの', lerpView(from, to, 0).zoom === from.zoom)
  check('t=1 は終点そのもの', Math.abs(lerpView(from, to, 1).zoom - to.zoom) < 1e-9)

  const mid = lerpView(from, to, 0.5)
  check(
    '倍率は対数の中点（線形だと寄りが跳ねる）',
    Math.abs(mid.zoom - 4) < 1e-9,
    `zoom ${mid.zoom.toFixed(3)}（線形なら 8.5）`,
  )
  check('中心は線形の中点', Math.abs(mid.centerX - 150) < 1e-9 && Math.abs(mid.centerY + 100) < 1e-9)
  check('範囲外の t は 0〜1 に丸める', lerpView(from, to, -5).zoom === from.zoom)

  // 指数収束で寄せると必ず近づく（行き過ぎない）
  let cur = from
  for (let i = 0; i < 200; i++) cur = lerpView(cur, to, 0.1)
  const p = createProjection(GINZA, W, H, to)
  check('繰り返し寄せると目標に収束する', viewsClose(cur, to, p.scale), `zoom ${cur.zoom.toFixed(3)}`)
}

{
  const a = { zoom: 4, centerX: 10, centerY: 10, rotation: 0 }
  const p = createProjection(GINZA, W, H, a)
  check('同じ見え方は「近い」と判定する', viewsClose(a, a, p.scale))
  check(
    '1px 動いただけでも「まだ動いている」とみなす',
    !viewsClose(a, { ...a, centerX: 10 + 2 / p.scale }, p.scale),
  )
}

{
  // 別の倍率で描いたレイヤを貼り直しても、地物の位置がずれないこと
  const layer = squareLayer({ zoom: 2, centerX: 0, centerY: 0, rotation: 0 })
  const current = createProjection(GINZA, W, H, { zoom: 5, centerX: 40, centerY: -25, rotation: 0 })
  const t = layerTransform(layer, current)
  check('貼り替えても地物が同じ画素に乗る', pasteError(layer, current) < 1e-6,
    `最大差 ${pasteError(layer, current).toExponential(1)}px`)
  check(
    '倍率が上がったぶんだけ引き伸ばされる',
    Math.abs(t.scale - current.scale / layer.scale) < 1e-12,
    `${t.scale.toFixed(3)} 倍`,
  )
}

{
  const same = createProjection(GINZA, W, H, { zoom: 3, centerX: 5, centerY: -5, rotation: 0 })
  const t = layerTransform(same, same)
  check(
    '同じ見え方なら等倍・無回転で中央に貼る',
    Math.abs(t.x - W / 2) < 1e-9 &&
      Math.abs(t.y - H / 2) < 1e-9 &&
      Math.abs(t.angle) < 1e-12 &&
      Math.abs(t.scale - 1) < 1e-12,
  )
}

{
  check(
    '倍率 1 の 1px は、マップ全体を画面に収めた大きさ',
    Math.abs(baseScale(GINZA, W, H) - createProjection(GINZA, W, H, { zoom: 1, centerX: 0, centerY: 0, rotation: 0 }).scale) < 1e-12,
  )
}

console.log()
console.log('='.repeat(70))
console.log('向きの追従（乗車中は進行方向が上）')
console.log('='.repeat(70))

{
  // 進行方向が画面の上を向くこと。ENU の heading をひととおり試す
  let worst = 0
  for (const heading of [0, 0.7, Math.PI / 2, 2.4, Math.PI, -1.2, -Math.PI / 2]) {
    const p = createProjection(GINZA, W, H, {
      zoom: 8,
      centerX: 0,
      centerY: 0,
      rotation: rotationForHeading(heading),
    })
    // 自車の 50m 先が、画面では真上に来る
    const [ax, ay] = toCanvas(p, 0, 0)
    const [bx, by] = toCanvas(p, Math.cos(heading) * 50, Math.sin(heading) * 50)
    worst = Math.max(worst, Math.abs(bx - ax), Math.abs(by - ay + 50 * p.scale))
  }
  check('進行方向がどの向きでも画面の真上に来る', worst < 1e-9, `最大差 ${worst.toExponential(1)}px`)
}

{
  const north = createProjection(GINZA, W, H, { zoom: 4, centerX: 0, centerY: 0, rotation: 0 })
  const turned = createProjection(GINZA, W, H, {
    zoom: 4,
    centerX: 0,
    centerY: 0,
    rotation: 0.9,
  })
  const [nx, ny] = toCanvas(north, 0, 0)
  const [tx, ty] = toCanvas(turned, 0, 0)
  check(
    '回しても画面の中心は動かない',
    Math.abs(nx - tx) < 1e-9 && Math.abs(ny - ty) < 1e-9,
  )

  // 往復して戻ること（回転を入れた逆変換の取り違えを捕まえる）
  let worst = 0
  for (const [x, y] of [[0, 0], [120, -80], [-300, 210]] as Vec2[]) {
    const [cx2, cy2] = toCanvas(turned, x, y)
    const [bx, by] = toEnu(turned, cx2, cy2)
    worst = Math.max(worst, Math.hypot(bx - x, by - y))
  }
  check('回した状態でも ENU ↔ キャンバスが往復する', worst < 1e-9, `最大差 ${worst.toExponential(1)}m`)
}

{
  const p = createProjection(GINZA, W, H, { zoom: 6, centerX: 0, centerY: 0, rotation: 0 })
  const turned = createProjection(GINZA, W, H, {
    zoom: 6,
    centerX: 0,
    centerY: 0,
    rotation: Math.PI / 4,
  })
  const flat = visibleBounds(p)
  const tilted = visibleBounds(turned)
  check(
    '斜めに回すと見える範囲（AABB）は広がる',
    tilted.maxX - tilted.minX > flat.maxX - flat.minX,
    `${(flat.maxX - flat.minX).toFixed(1)}m → ${(tilted.maxX - tilted.minX).toFixed(1)}m`,
  )
}

{
  // 回した状態で 2 点を収める
  const pickup: Vec2 = [-160, 90]
  const dropoff: Vec2 = [140, -70]
  const rotation = rotationForHeading(1.1)
  const fitted = fitView(GINZA, [pickup, dropoff], W, H, rotation)
  const p = createProjection(GINZA, W, H, fitted)
  let inside = true
  for (const [x, y] of [pickup, dropoff]) {
    const [cx2, cy2] = toCanvas(p, x, y)
    if (cx2 < 0 || cx2 > W || cy2 < 0 || cy2 > H) inside = false
  }
  check('回した向きのままでも 2 点が画面に収まる', inside)
  check('自動ズームは指定した向きをそのまま返す', Math.abs(fitted.rotation - rotation) < 1e-12)
}

{
  // レイヤを回して貼っても地物がずれないこと（実装と同じ正方形レイヤで確かめる）
  const layer = squareLayer({ zoom: 3, centerX: 10, centerY: -5, rotation: 0.4 })
  const current = createProjection(GINZA, W, H, {
    zoom: 3.2,
    centerX: 18,
    centerY: -12,
    rotation: 1.1,
  })
  const worst = pasteError(layer, current)
  check('回しながら貼り替えても地物が同じ画素に乗る', worst < 1e-6, `最大差 ${worst.toExponential(1)}px`)

  const t = layerTransform(layer, current)
  check(
    '貼るときの回転は「いまの向き − 描いたときの向き」',
    Math.abs(normalizeAngle(t.angle - (current.rotation - layer.rotation))) < 1e-12,
    `${t.angle.toFixed(3)} rad`,
  )
}

{
  const side = layerSizeFor(W, H, true)
  check(
    'レイヤはどの向きに回しても画面の隅まで覆える',
    side / 2 >= Math.hypot(W, H) / 2 - 1e-9,
    `一辺 ${side}px / 画面の対角 ${Math.hypot(W, H).toFixed(1)}px`,
  )
  check(
    '回さないときの一辺は画面の長辺どまり',
    layerSizeFor(W, H, false) === Math.max(W, H),
    `${layerSizeFor(W, H, false)}px`,
  )
}

{
  // 北へ戻すとき、遠回りしないこと
  check(
    '350° から 10° へは +20° 回る（-340° ではない）',
    Math.abs(normalizeAngle(lerpAngle((350 * Math.PI) / 180, (10 * Math.PI) / 180, 1) - (10 * Math.PI) / 180)) < 1e-12 &&
      lerpAngle((350 * Math.PI) / 180, (10 * Math.PI) / 180, 0.5) > (350 * Math.PI) / 180,
  )
  check('t=0 は元の向きのまま', Math.abs(lerpAngle(1.2, -2.5, 0) - 1.2) < 1e-12)
  check(
    '繰り返し寄せると目標の向きに収束する',
    (() => {
      let a = Math.PI * 0.95
      for (let i = 0; i < 400; i++) a = lerpAngle(a, 0, 0.1)
      return Math.abs(normalizeAngle(a)) < 1e-3
    })(),
  )
}

{
  const a: MapView = { zoom: 4, centerX: 0, centerY: 0, rotation: 0 }
  const b: MapView = { zoom: 4, centerX: 0, centerY: 0, rotation: 0.05 }
  const p = createProjection(GINZA, W, H, a)
  check('向きが違えば「まだ動いている」とみなす', !viewsClose(a, b, p.scale))
  check(
    '向きの差がごく僅かなら落ち着いたとみなす',
    viewsClose(a, { ...a, rotation: 0.0005 }, p.scale),
  )
}

{
  // 補間の途中で倍率と向きを別々の速さにできること
  const from: MapView = { zoom: 2, centerX: 0, centerY: 0, rotation: 0 }
  const to: MapView = { zoom: 8, centerX: 100, centerY: 0, rotation: 1.2 }
  const mid = lerpView(from, to, 0.5, 0.1)
  check(
    '向きだけゆっくり追わせられる',
    Math.abs(mid.rotation - 0.12) < 1e-12 && Math.abs(mid.centerX - 50) < 1e-12,
    `向き ${mid.rotation.toFixed(3)} rad / 中心 ${mid.centerX.toFixed(1)}m`,
  )
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
