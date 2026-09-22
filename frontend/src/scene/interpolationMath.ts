/** フレーム間補間の計算だけを集めた純粋モジュール（`npm run verify` から読む）。 */

import type { NpcPedestrianState } from '../types/protocol'

/** これ以上動いたら補間せず瞬間移動（respawn / 歩行者の回し直し）[m] */
export const TELEPORT_DISTANCE_M = 20

/** 角度差を (-pi, pi] に畳む。heading は最短回りで補間しなければならない */
export function angleDelta(from: number, to: number): number {
  const d = to - from
  return Math.atan2(Math.sin(d), Math.cos(d))
}

/** 角度の最短回り補間 */
export function lerpAngle(from: number, to: number, t: number): number {
  return from + angleDelta(from, to) * t
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** NPC 歩行者の補間結果。 */
export interface PedestrianPose {
  x: number
  y: number
  heading: number
  /** 手足の振りの位相 [rad] */
  stride: number
  crossing: boolean
}

/** 使い回し用の空ポーズを作る（毎フレームのアロケーションを避ける） */
export function createPedestrianPose(): PedestrianPose {
  return { x: 0, y: 0, heading: 0, stride: 0, crossing: false }
}

/** NPC 歩行者を補間して out に書き込む。 */
export function samplePedestrian(
  curr: NpcPedestrianState,
  prev: NpcPedestrianState | undefined,
  alpha: number,
  out: PedestrianPose,
): void {
  out.crossing = curr.crossing
  const jumped =
    prev === undefined ||
    (curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2 >
      TELEPORT_DISTANCE_M * TELEPORT_DISTANCE_M
  if (jumped) {
    out.x = curr.x
    out.y = curr.y
    out.heading = curr.heading
    out.stride = curr.stride
    return
  }
  out.x = lerp(prev.x, curr.x, alpha)
  out.y = lerp(prev.y, curr.y, alpha)
  out.heading = lerpAngle(prev.heading, curr.heading, alpha)
  // 位相は 2π で折り返すので、角度と同じ扱いにしないと手足が逆回りする
  out.stride = lerpAngle(prev.stride, curr.stride, alpha)
}
