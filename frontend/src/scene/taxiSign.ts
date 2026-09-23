/** 実用モードのタクシーの表示（屋根の表示灯と、フロントガラス越しの空車／迎車／賃走／支払）。**純粋モジュール。** */

import * as THREE from 'three'
import type { Vec3 } from './meshBuilder.ts'
import { isBoardablePhase, isRidingPhase, isWaitingPhase, type TaxiPhase } from '../types/protocol.ts'
import { DASH_TOP_Y, ROOF_Y, VEHICLE_HEIGHT } from './vehicleGeometry.ts'

/** アトラスの段（Canvas の上から）。0〜2 は固定、3 以降が状態の表示 */
export const SIGN_ROW_LAMP_TEXT = 0
export const SIGN_ROW_LAMP_BODY = 1
export const SIGN_ROW_DARK = 2
export const SIGN_ROW_STATUS = 3

export type TaxiSignStatus = 'vacant' | 'dispatched' | 'occupied' | 'paying'

/** 状態ごとの文字と段 */
export const TAXI_SIGN_STATUSES: ReadonlyArray<{ readonly status: TaxiSignStatus; readonly text: string }> = [
  { status: 'vacant', text: '空車' },
  { status: 'dispatched', text: '迎車' },
  { status: 'occupied', text: '賃走' },
  { status: 'paying', text: '支払' },
]

/** 屋根の表示灯の文字。**一般的な語だけを使い、社名・マークは描かない** */
export const TAXI_LAMP_TEXT = 'TAXI'

export const SIGN_ROWS = SIGN_ROW_STATUS + TAXI_SIGN_STATUSES.length

/** 車両 id の表示。**段階の判定は `types/protocol.ts` の関数だけを使う**（各所で書かない） */
export function taxiSignStatus(id: number, taxi: { phase: TaxiPhase; vehicleId: number }): TaxiSignStatus {
  if (id !== taxi.vehicleId) return 'vacant'
  if (isRidingPhase(taxi.phase)) return isWaitingPhase(taxi.phase) ? 'paying' : 'occupied'
  if (isBoardablePhase(taxi.phase)) return 'dispatched'
  return 'vacant'
}

/** 状態の表示に使うアトラスの段 */
export function statusRow(status: TaxiSignStatus): number {
  return SIGN_ROW_STATUS + TAXI_SIGN_STATUSES.findIndex((s) => s.status === status)
}

/** 表示灯の明るさ。空車のときだけ点ける（乗せている間・迎えに行く間は消す） */
export function lampGlow(status: TaxiSignStatus): number {
  return status === 'vacant' ? 1 : 0.1
}

/** Canvas の段を、テクスチャ座標 v の段へ（Canvas は上から、v は下から数える） */
export function signUvRow(row: number): number {
  return SIGN_ROWS - 1 - row
}

/** 屋根の表示灯。**外接寸法の高さ（1.45m）を越える唯一の部品**（CLAUDE.md の例外） */
export const TAXI_LAMP = {
  centerX: -0.05,
  /** 幅（左右） */
  width: 0.4,
  /** 奥行き（下端・上端） */
  depthBottom: 0.16,
  depthTop: 0.1,
  bottomY: ROOF_Y - 0.004,
  height: 0.12,
} as const

/** 表示灯の上端 [m]。verify:vehicles がこの高さを上限にする */
export const TAXI_LAMP_TOP_Y = TAXI_LAMP.bottomY + TAXI_LAMP.height
/** 外接寸法からのはみ出し [m] */
export const TAXI_LAMP_OVERHANG = TAXI_LAMP_TOP_Y - VEHICLE_HEIGHT

/** 状態の表示板（助手席側のダッシュボードの上。フロントガラス越しに前から読める） */
export const STATUS_SIGN = {
  x: [0.955, 0.982] as const,
  y: [DASH_TOP_Y + 0.002, DASH_TOP_Y + 0.066] as const,
  z: [-0.62, -0.36] as const,
} as const

/** 1 面ぶん（4 点）を UV つきで足す。`row` が負なら状態の段（インスタンスごと） */
function face(
  pos: number[],
  uv: number[],
  rows: number[],
  nrm: number[],
  corners: readonly [Vec3, Vec3, Vec3, Vec3],
  uvs: readonly [readonly [number, number], readonly [number, number], readonly [number, number], readonly [number, number]],
  row: number,
): void {
  const a = corners
  const e1 = new THREE.Vector3(a[1][0] - a[0][0], a[1][1] - a[0][1], a[1][2] - a[0][2])
  const e2 = new THREE.Vector3(a[2][0] - a[0][0], a[2][1] - a[0][1], a[2][2] - a[0][2])
  const n = new THREE.Vector3().crossVectors(e1, e2).normalize()
  for (const k of [0, 1, 2, 0, 2, 3]) {
    pos.push(a[k][0], a[k][1], a[k][2])
    uv.push(uvs[k][0], uvs[k][1])
    rows.push(row)
    nrm.push(n.x, n.y, n.z)
  }
}

/** 表示灯と表示板を 1 つのジオメトリに（属性 `signRow` でアトラスの段を指す）。 */
export function makeTaxiSignGeometry(): THREE.BufferGeometry {
  const pos: number[] = []
  const uv: number[] = []
  const rows: number[] = []
  const nrm: number[] = []
  const L = TAXI_LAMP
  const hw = L.width / 2
  const y0 = L.bottomY
  const y1 = L.bottomY + L.height
  const xb = L.depthBottom / 2
  const xt = L.depthTop / 2
  const cx = L.centerX
  const full = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ] as const
  const plain = [
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
  ] as const
  // 前（+X）。前から見て左が +Z なので、u は +Z の端から始める
  face(pos, uv, rows, nrm, [[cx + xb, y0, hw], [cx + xb, y0, -hw], [cx + xt, y1, -hw], [cx + xt, y1, hw]], full, SIGN_ROW_LAMP_TEXT)
  // 後ろ（-X）。後ろから見て左が -Z
  face(pos, uv, rows, nrm, [[cx - xb, y0, -hw], [cx - xb, y0, hw], [cx - xt, y1, hw], [cx - xt, y1, -hw]], full, SIGN_ROW_LAMP_TEXT)
  face(pos, uv, rows, nrm, [[cx - xt, y1, -hw], [cx - xt, y1, hw], [cx + xt, y1, hw], [cx + xt, y1, -hw]], plain, SIGN_ROW_LAMP_BODY)
  face(pos, uv, rows, nrm, [[cx - xb, y0, hw], [cx + xb, y0, hw], [cx + xt, y1, hw], [cx - xt, y1, hw]], plain, SIGN_ROW_LAMP_BODY)
  face(pos, uv, rows, nrm, [[cx + xb, y0, -hw], [cx - xb, y0, -hw], [cx - xt, y1, -hw], [cx + xt, y1, -hw]], plain, SIGN_ROW_LAMP_BODY)

  // 台座（屋根との継ぎ目を隠す暗い板）
  const bx = xb + 0.015
  const bz = hw + 0.015
  const by0 = y0 - 0.006
  const by1 = y0 + 0.008
  face(pos, uv, rows, nrm, [[cx - bx, by1, -bz], [cx - bx, by1, bz], [cx + bx, by1, bz], [cx + bx, by1, -bz]], plain, SIGN_ROW_DARK)
  face(pos, uv, rows, nrm, [[cx + bx, by0, bz], [cx + bx, by0, -bz], [cx + bx, by1, -bz], [cx + bx, by1, bz]], plain, SIGN_ROW_DARK)
  face(pos, uv, rows, nrm, [[cx - bx, by0, -bz], [cx - bx, by0, bz], [cx - bx, by1, bz], [cx - bx, by1, -bz]], plain, SIGN_ROW_DARK)
  face(pos, uv, rows, nrm, [[cx - bx, by0, bz], [cx + bx, by0, bz], [cx + bx, by1, bz], [cx - bx, by1, bz]], plain, SIGN_ROW_DARK)
  face(pos, uv, rows, nrm, [[cx + bx, by0, -bz], [cx - bx, by0, -bz], [cx - bx, by1, -bz], [cx + bx, by1, -bz]], plain, SIGN_ROW_DARK)

  const S = STATUS_SIGN
  const [sx0, sx1] = S.x
  const [sy0, sy1] = S.y
  const [sz0, sz1] = S.z
  // 表示板の前（+X）。前から見て左が +Z
  face(pos, uv, rows, nrm, [[sx1, sy0, sz1], [sx1, sy0, sz0], [sx1, sy1, sz0], [sx1, sy1, sz1]], full, -1)
  face(pos, uv, rows, nrm, [[sx0, sy0, sz0], [sx0, sy0, sz1], [sx0, sy1, sz1], [sx0, sy1, sz0]], plain, SIGN_ROW_DARK)
  face(pos, uv, rows, nrm, [[sx0, sy1, sz0], [sx0, sy1, sz1], [sx1, sy1, sz1], [sx1, sy1, sz0]], plain, SIGN_ROW_DARK)
  face(pos, uv, rows, nrm, [[sx0, sy0, sz1], [sx1, sy0, sz1], [sx1, sy1, sz1], [sx0, sy1, sz1]], plain, SIGN_ROW_DARK)
  face(pos, uv, rows, nrm, [[sx1, sy0, sz0], [sx0, sy0, sz0], [sx0, sy1, sz0], [sx1, sy1, sz0]], plain, SIGN_ROW_DARK)

  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2))
  g.setAttribute('signRow', new THREE.Float32BufferAttribute(rows, 1))
  g.computeBoundingBox()
  return g
}

