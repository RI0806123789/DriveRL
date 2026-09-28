/** 運転席の 4 分割表示の配置と、ペインごとのカメラ（純粋関数）。 */

import type { SurroundKey } from '../types/protocol.ts'
import { DRIVER_FOV_DEG, SURROUND_CAMERAS, SURROUND_FOV_DEG, type SurroundCamera } from './cameraMath.ts'

export type QuadPane = 'front' | SurroundKey

/** ペインの並び（左上・右上・左下・右下）。issue #57 の指定どおり */
export const QUAD_PANES: readonly QuadPane[] = ['front', 'rear', 'left', 'right']

export const PANE_LABELS: Record<QuadPane, string> = {
  front: '前方',
  rear: '後方',
  left: '左側方',
  right: '右側方',
}

/** CSS の配置（左上原点・0〜1） */
export interface PaneRect {
  left: number
  top: number
  width: number
  height: number
}

export function quadPaneRect(index: number): PaneRect {
  return { left: (index % 2) * 0.5, top: Math.floor(index / 2) * 0.5, width: 0.5, height: 0.5 }
}

/** WebGL の viewport（**左下原点**・CSS px）。`setViewport` / `setScissor` にそのまま渡す */
export function quadViewport(
  index: number,
  width: number,
  height: number,
): { x: number; y: number; w: number; h: number } {
  const rect = quadPaneRect(index)
  const w = width * rect.width
  const h = height * rect.height
  return { x: width * rect.left, y: height - height * rect.top - h, w, h }
}

/** 画面上のポインタ（CSS px・左上原点）を、前方ペインのカメラの正規化座標（-1〜1・上が +）へ。前方ペインの外なら null */
export function frontPanePointer(
  offsetX: number,
  offsetY: number,
  width: number,
  height: number,
): { x: number; y: number } | null {
  const rect = quadPaneRect(QUAD_PANES.indexOf('front'))
  const u = (offsetX / Math.max(width, 1) - rect.left) / rect.width
  const v = (offsetY / Math.max(height, 1) - rect.top) / rect.height
  if (u < 0 || u > 1 || v < 0 || v > 1) return null
  return { x: u * 2 - 1, y: -(v * 2 - 1) }
}

/** ペインの**垂直**画角。前方は運転席の画角のまま、周囲は擬似カメラの画角に揃える */
export function paneFov(pane: QuadPane): number {
  return pane === 'front' ? DRIVER_FOV_DEG : SURROUND_FOV_DEG
}

export function surroundCameraFor(pane: QuadPane): SurroundCamera | null {
  return SURROUND_CAMERAS.find((c) => c.key === pane) ?? null
}
