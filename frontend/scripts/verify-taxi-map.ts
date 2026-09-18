/** スマホ画面の 2D 地図の座標変換を検証する（ブラウザ不要）。 */

import {
  FIT_MIN_SPAN_M,
  ZOOM_MAX,
  ZOOM_MIN,
  baseScale,
  boundsCenter,
  clampCenter,
  clampZoom,
  createProjection,
  fitView,
  layerPlacement,
  lerpView,
  toCanvasX,
  toCanvasY,
  toEnu,
  viewsClose,
  visibleBounds,
  zoomAround,
} from '../src/panel/taxiMapMath.ts'
import type { MapBounds, Vec2 } from '../src/types/protocol.ts'

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

console.log('='.repeat(70))
console.log('投影（ENU ↔ キャンバス）')
console.log('='.repeat(70))

for (const [name, bounds] of [
  ['銀座', GINZA],
  ['金沢', KANAZAWA],
  ['横長', OBLONG],
] as const) {
  const [cx, cy] = boundsCenter(bounds)
  const p = createProjection(bounds, W, H, { zoom: 1, centerX: cx, centerY: cy })

  const cornerX = toCanvasX(p, bounds.minX)
  const cornerY = toCanvasY(p, bounds.maxY)
  const farX = toCanvasX(p, bounds.maxX)
  const farY = toCanvasY(p, bounds.minY)
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
    const back = toEnu(p, toCanvasX(p, x), toCanvasY(p, y))
    worst = Math.max(worst, Math.abs(back[0] - x), Math.abs(back[1] - y))
  }
  check(`${name}: 投影 → 逆投影で元に戻る`, worst < 1e-6, `最大差 ${worst.toExponential(1)}m`)
}

{
  const [cx, cy] = boundsCenter(GINZA)
  const p = createProjection(GINZA, W, H, { zoom: 1, centerX: cx, centerY: cy })
  check(
    '中心が画面の中央に来る',
    Math.abs(toCanvasX(p, cx) - W / 2) < 1e-9 && Math.abs(toCanvasY(p, cy) - H / 2) < 1e-9,
  )
  check(
    '北（+y）が上、東（+x）が右',
    toCanvasY(p, cy + 100) < toCanvasY(p, cy) && toCanvasX(p, cx + 100) > toCanvasX(p, cx),
    `y: ${toCanvasY(p, cy + 100).toFixed(1)} < ${toCanvasY(p, cy).toFixed(1)}`,
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
  const base = createProjection(GINZA, W, H, { zoom: 1, centerX: cx, centerY: cy })
  const zoomed = createProjection(GINZA, W, H, { zoom: 4, centerX: cx, centerY: cy })
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
  const view = { zoom: 1, centerX: cx, centerY: cy }
  const anchor: [number, number] = [-300, 260]
  const before = createProjection(GINZA, W, H, view)
  const canvasBefore = [toCanvasX(before, anchor[0]), toCanvasY(before, anchor[1])]

  const next = zoomAround(view, GINZA, anchor, 2)
  const after = createProjection(GINZA, W, H, next)
  const canvasAfter = [toCanvasX(after, anchor[0]), toCanvasY(after, anchor[1])]

  check(
    'ホイールで拡大しても、掴んだ点は画面上で動かない',
    Math.abs(canvasBefore[0] - canvasAfter[0]) < 1e-6 &&
      Math.abs(canvasBefore[1] - canvasAfter[1]) < 1e-6,
    `(${canvasBefore[0].toFixed(2)}, ${canvasBefore[1].toFixed(2)}) → ` +
      `(${canvasAfter[0].toFixed(2)}, ${canvasAfter[1].toFixed(2)})`,
  )
  check('倍率が上がっている', next.zoom === 2, `${next.zoom}`)

  const capped = zoomAround({ zoom: ZOOM_MAX, centerX: cx, centerY: cy }, GINZA, anchor, 2)
  check('上限に達していれば中心も動かさない', capped.centerX === cx && capped.centerY === cy)
}

{
  const far = clampCenter({ zoom: 3, centerX: 99999, centerY: -99999 }, GINZA)
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
    const cxp = toCanvasX(p, x)
    const cyp = toCanvasY(p, y)
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
    toCanvasX(p, pair[0][0]),
    W - toCanvasX(p, pair[1][0]),
    toCanvasY(p, pair[1][1]),
    H - toCanvasY(p, pair[0][1]),
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

  const tightest = createProjection(KANAZAWA, W, H, { zoom: ZOOM_MAX, centerX: 0, centerY: 0 })
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
  const from: { zoom: number; centerX: number; centerY: number } = { zoom: 1, centerX: 0, centerY: 0 }
  const to = { zoom: 16, centerX: 300, centerY: -200 }
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
  const a = { zoom: 4, centerX: 10, centerY: 10 }
  const p = createProjection(GINZA, W, H, a)
  check('同じ見え方は「近い」と判定する', viewsClose(a, a, p.scale))
  check(
    '1px 動いただけでも「まだ動いている」とみなす',
    !viewsClose(a, { ...a, centerX: 10 + 2 / p.scale }, p.scale),
  )
}

{
  // 別の倍率で描いたレイヤを貼り直しても、地物の位置がずれないこと
  const layerView = { zoom: 2, centerX: 0, centerY: 0 }
  const currentView = { zoom: 5, centerX: 40, centerY: -25 }
  const layer = createProjection(GINZA, W, H, layerView)
  const current = createProjection(GINZA, W, H, currentView)
  const place = layerPlacement(layer, current)

  let worst = 0
  for (const [x, y] of [[0, 0], [60, 40], [-120, 75]] as Vec2[]) {
    // レイヤ内での位置 → 貼り付け後の画面位置
    const inLayerX = toCanvasX(layer, x)
    const inLayerY = toCanvasY(layer, y)
    const pastedX = place.dx + (inLayerX / layer.width) * place.dw
    const pastedY = place.dy + (inLayerY / layer.height) * place.dh
    worst = Math.max(
      worst,
      Math.abs(pastedX - toCanvasX(current, x)),
      Math.abs(pastedY - toCanvasY(current, y)),
    )
  }
  check('貼り替えても地物が同じ画素に乗る', worst < 1e-6, `最大差 ${worst.toExponential(1)}px`)
  check(
    '倍率が上がったぶんだけ引き伸ばされる',
    Math.abs(place.dw / layer.width - current.scale / layer.scale) < 1e-9,
    `${(place.dw / layer.width).toFixed(3)} 倍`,
  )
}

{
  const same = createProjection(GINZA, W, H, { zoom: 3, centerX: 5, centerY: -5 })
  const place = layerPlacement(same, same)
  check(
    '同じ見え方なら等倍で原点に貼る',
    Math.abs(place.dx) < 1e-9 &&
      Math.abs(place.dy) < 1e-9 &&
      Math.abs(place.dw - W) < 1e-9 &&
      Math.abs(place.dh - H) < 1e-9,
  )
}

{
  check(
    '倍率 1 の 1px は、マップ全体を画面に収めた大きさ',
    Math.abs(baseScale(GINZA, W, H) - createProjection(GINZA, W, H, { zoom: 1, centerX: 0, centerY: 0 }).scale) < 1e-12,
  )
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
