/** スマホ画面の 2D 地図の投影と描画。**React から切り離した純粋モジュール。** */

import type { MapBuilding, MapBounds, MapEdge, Vec2 } from '../types/protocol'

/** 地図の見え方（倍率 1 でマップ全体がちょうど収まる） */
export interface MapView {
  zoom: number
  /** 画面中心の ENU 座標 */
  centerX: number
  centerY: number
}

/** ENU ↔ キャンバス座標の変換。**y は上下が反転する**（北が上） */
export interface MapProjection {
  width: number
  height: number
  /** ENU 1m あたりのピクセル数 */
  scale: number
  centerX: number
  centerY: number
}

export const ZOOM_MIN = 1
/**
 * 倍率の上限。**マップの一辺に対する比なので、広いエリアほど大きな値が要る**
 * （金沢は 12.3km 四方あり、40 倍では 300m 幅までしか寄れなかった）。
 * 120 なら銀座で 7m 幅、金沢で 102m 幅まで寄れる。
 */
export const ZOOM_MAX = 120

/** 倍率 1（マップ全体が収まる状態）の 1m あたりピクセル数。 */
export function baseScale(
  bounds: MapBounds,
  width: number,
  height: number,
  padding = 6,
): number {
  const spanX = Math.max(1e-6, bounds.maxX - bounds.minX)
  const spanY = Math.max(1e-6, bounds.maxY - bounds.minY)
  return Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY)
}

/** マップ全体が収まる倍率を 1 として投影を作る。 */
export function createProjection(
  bounds: MapBounds,
  width: number,
  height: number,
  view: MapView,
  padding = 6,
): MapProjection {
  return {
    width,
    height,
    scale: baseScale(bounds, width, height, padding) * clampZoom(view.zoom),
    centerX: view.centerX,
    centerY: view.centerY,
  }
}

export function clampZoom(zoom: number): number {
  if (!Number.isFinite(zoom)) return ZOOM_MIN
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom))
}

/** マップの中心（倍率 1 のときの見え方） */
export function boundsCenter(bounds: MapBounds): Vec2 {
  return [(bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2]
}

export function toCanvasX(p: MapProjection, x: number): number {
  return p.width / 2 + (x - p.centerX) * p.scale
}

export function toCanvasY(p: MapProjection, y: number): number {
  return p.height / 2 - (y - p.centerY) * p.scale
}

/** キャンバス座標を ENU へ戻す（地図をクリックして地点を選ぶのに使う）。 */
export function toEnu(p: MapProjection, cx: number, cy: number): Vec2 {
  return [
    p.centerX + (cx - p.width / 2) / p.scale,
    p.centerY - (cy - p.height / 2) / p.scale,
  ]
}

/** 見えている範囲の ENU 矩形（描画を間引くのに使う） */
export function visibleBounds(p: MapProjection, margin = 0): MapBounds {
  const halfW = p.width / 2 / p.scale + margin
  const halfH = p.height / 2 / p.scale + margin
  return {
    minX: p.centerX - halfW,
    maxX: p.centerX + halfW,
    minY: p.centerY - halfH,
    maxY: p.centerY + halfH,
  }
}

/** 中心を動かしても地図が視界から外れないように留める。 */
export function clampCenter(view: MapView, bounds: MapBounds): MapView {
  return {
    zoom: clampZoom(view.zoom),
    centerX: Math.min(bounds.maxX, Math.max(bounds.minX, view.centerX)),
    centerY: Math.min(bounds.maxY, Math.max(bounds.minY, view.centerY)),
  }
}

/** 自動で寄るときに、最低これだけの範囲は見せる [m]（1 点だけのとき近づきすぎない） */
export const FIT_MIN_SPAN_M = 150

/**
 * 指定した点が全部入る見え方を作る（配車の段階に合わせた自動ズーム）。
 * 点が無ければマップ全体へ戻す。
 */
export function fitView(
  bounds: MapBounds,
  points: readonly Vec2[],
  width: number,
  height: number,
  marginRatio = 0.26,
): MapView {
  const [cx, cy] = boundsCenter(bounds)
  if (points.length === 0) return { zoom: ZOOM_MIN, centerX: cx, centerY: cy }

  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const [px, py] of points) {
    if (px < x0) x0 = px
    if (py < y0) y0 = py
    if (px > x1) x1 = px
    if (py > y1) y1 = py
  }

  const spanX = Math.max(FIT_MIN_SPAN_M, x1 - x0)
  const spanY = Math.max(FIT_MIN_SPAN_M, y1 - y0)
  const room = Math.max(0.05, 1 - marginRatio)
  const want = Math.min((width * room) / spanX, (height * room) / spanY)

  return clampCenter(
    {
      zoom: clampZoom(want / baseScale(bounds, width, height)),
      centerX: (x0 + x1) / 2,
      centerY: (y0 + y1) / 2,
    },
    bounds,
  )
}

/**
 * 見え方を補間する。**倍率は対数で混ぜる**（線形だと桁の違いで寄り方が跳ねる。
 * 天候のフォグ距離と同じ理由）。
 */
export function lerpView(from: MapView, to: MapView, t: number): MapView {
  const k = Math.max(0, Math.min(1, t))
  const logFrom = Math.log(clampZoom(from.zoom))
  const logTo = Math.log(clampZoom(to.zoom))
  return {
    zoom: Math.exp(logFrom + (logTo - logFrom) * k),
    centerX: from.centerX + (to.centerX - from.centerX) * k,
    centerY: from.centerY + (to.centerY - from.centerY) * k,
  }
}

/** 2 つの見え方が実質同じか（画面上のずれと倍率比で見る）。 */
export function viewsClose(
  a: MapView,
  b: MapView,
  scale: number,
  pixelEpsilon = 0.4,
  zoomEpsilon = 0.004,
): boolean {
  const moved = Math.hypot(a.centerX - b.centerX, a.centerY - b.centerY) * scale
  const ratio = Math.abs(Math.log(clampZoom(a.zoom) / clampZoom(b.zoom)))
  return moved <= pixelEpsilon && ratio <= zoomEpsilon
}

/**
 * 別の見え方で描いたレイヤを、いまの見え方へ貼るときの矩形。
 * **道路と建物を毎フレーム描き直さないため**にある（金沢は 58,120 本ある）。
 */
export function layerPlacement(
  layer: MapProjection,
  current: MapProjection,
): { dx: number; dy: number; dw: number; dh: number } {
  const topLeft = toEnu(layer, 0, 0)
  const bottomRight = toEnu(layer, layer.width, layer.height)
  const dx = toCanvasX(current, topLeft[0])
  const dy = toCanvasY(current, topLeft[1])
  return {
    dx,
    dy,
    dw: toCanvasX(current, bottomRight[0]) - dx,
    dh: toCanvasY(current, bottomRight[1]) - dy,
  }
}

/** ある点を固定したままズームする（ホイール操作）。 */
export function zoomAround(
  view: MapView,
  bounds: MapBounds,
  anchor: Vec2,
  factor: number,
): MapView {
  const next = clampZoom(view.zoom * factor)
  const ratio = next === view.zoom ? 1 : view.zoom / next
  return clampCenter(
    {
      zoom: next,
      centerX: anchor[0] + (view.centerX - anchor[0]) * ratio,
      centerY: anchor[1] + (view.centerY - anchor[1]) * ratio,
    },
    bounds,
  )
}

/**
 * 道路。**見えない区間と、縮尺に対して短すぎる区間は引かない**。
 * 金沢は 58,120 本あり、全体表示で全部なぞると 1 フレームを使い切る。
 * 引いた本数を返す（測るときの手がかり）。
 */
export function drawRoads(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  edges: readonly MapEdge[],
  color: string,
  lineWidth: number,
  minPixels = 2.5,
): number {
  const box = visibleBounds(p, 50)
  const minSpanM = minPixels / p.scale
  let drawn = 0
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.lineCap = 'round'
  ctx.beginPath()
  for (const e of edges) {
    const line = e.polyline
    if (!line || line.length < 2) continue
    const a = line[0]
    const b = line[line.length - 1]
    if (
      (a[0] < box.minX && b[0] < box.minX) ||
      (a[0] > box.maxX && b[0] > box.maxX) ||
      (a[1] < box.minY && b[1] < box.minY) ||
      (a[1] > box.maxY && b[1] > box.maxY)
    ) {
      continue
    }
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) < minSpanM) continue
    ctx.moveTo(toCanvasX(p, a[0]), toCanvasY(p, a[1]))
    for (let i = 1; i < line.length; i++) {
      ctx.lineTo(toCanvasX(p, line[i][0]), toCanvasY(p, line[i][1]))
    }
    drawn++
  }
  ctx.stroke()
  ctx.restore()
  return drawn
}

/** 建物。**投影して 2px 未満になるものは描かない**（見えないのに件数だけかさむ） */
export function drawBuildings(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  buildings: readonly MapBuilding[],
  color: string,
  minPixels = 2,
): number {
  const box = visibleBounds(p, 30)
  let drawn = 0
  ctx.save()
  ctx.fillStyle = color
  ctx.beginPath()
  for (const b of buildings) {
    const pts = b.outline
    if (!pts || pts.length < 3) continue
    let x0 = Infinity
    let y0 = Infinity
    let x1 = -Infinity
    let y1 = -Infinity
    for (const [px, py] of pts) {
      if (px < x0) x0 = px
      if (py < y0) y0 = py
      if (px > x1) x1 = px
      if (py > y1) y1 = py
    }
    if (x1 < box.minX || x0 > box.maxX || y1 < box.minY || y0 > box.maxY) continue
    if ((x1 - x0) * p.scale < minPixels && (y1 - y0) * p.scale < minPixels) continue
    ctx.moveTo(toCanvasX(p, pts[0][0]), toCanvasY(p, pts[0][1]))
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(toCanvasX(p, pts[i][0]), toCanvasY(p, pts[i][1]))
    }
    ctx.closePath()
    drawn++
  }
  ctx.fill()
  ctx.restore()
  return drawn
}

/** 経路 */
export function drawRoute(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  route: readonly Vec2[],
  color: string,
  lineWidth: number,
): void {
  if (route.length < 2) return
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(toCanvasX(p, route[0][0]), toCanvasY(p, route[0][1]))
  for (let i = 1; i < route.length; i++) {
    ctx.lineTo(toCanvasX(p, route[i][0]), toCanvasY(p, route[i][1]))
  }
  ctx.stroke()
  ctx.restore()
}

/** 車両（進行方向を向いた三角）。 */
export function drawHeadingMark(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  x: number,
  y: number,
  heading: number,
  color: string,
  size: number,
  outline?: string,
): void {
  const cx = toCanvasX(p, x)
  const cy = toCanvasY(p, y)
  // ENU の反時計回りは、y を反転したキャンバスでは時計回りになる
  const a = -heading
  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(a)
  ctx.beginPath()
  ctx.moveTo(size, 0)
  ctx.lineTo(-size * 0.72, size * 0.66)
  ctx.lineTo(-size * 0.72, -size * 0.66)
  ctx.closePath()
  ctx.fillStyle = color
  ctx.fill()
  if (outline) {
    ctx.strokeStyle = outline
    ctx.lineWidth = 1
    ctx.stroke()
  }
  ctx.restore()
}

/** 乗降地点の目印。 */
export function drawPin(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  x: number,
  y: number,
  color: string,
  radius: number,
  ring: boolean,
): void {
  const cx = toCanvasX(p, x)
  const cy = toCanvasY(p, y)
  ctx.save()
  ctx.beginPath()
  ctx.arc(cx, cy, radius, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
  if (ring) {
    ctx.beginPath()
    ctx.arc(cx, cy, radius * 2.1, 0, Math.PI * 2)
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 0.55
    ctx.stroke()
  }
  ctx.restore()
}
