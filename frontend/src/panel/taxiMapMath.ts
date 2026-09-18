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
export const ZOOM_MAX = 40

/** マップ全体が収まる倍率を 1 として投影を作る。 */
export function createProjection(
  bounds: MapBounds,
  width: number,
  height: number,
  view: MapView,
  padding = 6,
): MapProjection {
  const spanX = Math.max(1e-6, bounds.maxX - bounds.minX)
  const spanY = Math.max(1e-6, bounds.maxY - bounds.minY)
  const fit = Math.min(
    (width - padding * 2) / spanX,
    (height - padding * 2) / spanY,
  )
  const zoom = clampZoom(view.zoom)
  return {
    width,
    height,
    scale: fit * zoom,
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

/** 道路。**始点と終点だけの直線に落とす**（金沢の 58,120 本を毎回なぞらないため） */
export function drawRoads(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  edges: readonly MapEdge[],
  color: string,
  lineWidth: number,
): void {
  const box = visibleBounds(p, 50)
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
    ctx.moveTo(toCanvasX(p, a[0]), toCanvasY(p, a[1]))
    for (let i = 1; i < line.length; i++) {
      ctx.lineTo(toCanvasX(p, line[i][0]), toCanvasY(p, line[i][1]))
    }
  }
  ctx.stroke()
  ctx.restore()
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
