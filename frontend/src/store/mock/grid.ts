/** モックの碁盤の目と乱数。モックの各部（地図・交通・配車）が共有する土台 */

import type { Vec2 } from '../../types/protocol.ts'

/** 0〜1 の一様乱数を返す関数 */
export type Rng = () => number

export function makeRng(seed: number): Rng {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

export const GRID_N = 9
export const GRID_SPACING = 100
export const GRID_HALF = ((GRID_N - 1) * GRID_SPACING) / 2
export const ROAD_WIDTH = 7.0
export const SIM_HZ = 20
export const FRAME_MS = 1000 / SIM_HZ

export function nodeId(gx: number, gy: number): number {
  return gy * GRID_N + gx
}
export function nodeX(gx: number): number {
  return gx * GRID_SPACING - GRID_HALF
}
export function nodeY(gy: number): number {
  return gy * GRID_SPACING - GRID_HALF
}

export function clampGrid(v: number): number {
  return Math.max(0, Math.min(GRID_N - 1, v))
}

/** 格子の 1 区間（`from` から `to` へ進む向き）のキー。信号・標識・車列の引き当てに使う */
export function hopKey(fx: number, fy: number, tx: number, ty: number): string {
  return `${fx},${fy}>${tx},${ty}`
}

/** ENU 座標を、いちばん近いグリッドノードへ寄せる（サーバーの道路スナップの代わり） */
export function snapToGrid(x: number, y: number): { gx: number; gy: number; point: Vec2 } {
  const gx = clampGrid(Math.round((x + GRID_HALF) / GRID_SPACING))
  const gy = clampGrid(Math.round((y + GRID_HALF) / GRID_SPACING))
  return { gx, gy, point: [nodeX(gx), nodeY(gy)] }
}
