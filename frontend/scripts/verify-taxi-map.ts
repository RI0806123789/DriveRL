/** スマホ画面の 2D 地図の座標変換を検証する（ブラウザ不要）。 */

import {
  ZOOM_MAX,
  ZOOM_MIN,
  boundsCenter,
  clampCenter,
  clampZoom,
  createProjection,
  toCanvasX,
  toCanvasY,
  toEnu,
  visibleBounds,
  zoomAround,
} from '../src/panel/taxiMapMath.ts'
import type { MapBounds } from '../src/types/protocol.ts'

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
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
