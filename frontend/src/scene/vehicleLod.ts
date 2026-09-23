/** 車両の細かさの段（近景・遠景）を選ぶ。**React から切り離した純粋モジュール。** */

/** 遠景（外板を粗く、内装・メーター・ペダルを省く） */
export const LOD_FAR = 0
/** 近景（細かい形と内装） */
export const LOD_NEAR = 1

/** 近景に入る距離・出る距離 [m]。境目で行き来しないよう、出る方を遠くする */
export const LOD_ENTER_M = 45
export const LOD_EXIT_M = 58

/** いまの段と、最寄りの視点までの距離から次の段を決める。`forced` は運転席から見ている車など */
export function nextLod(current: number, distance: number, forced: boolean): number {
  if (forced) return LOD_NEAR
  if (current === LOD_NEAR) return distance > LOD_EXIT_M ? LOD_FAR : LOD_NEAR
  return distance < LOD_ENTER_M ? LOD_NEAR : LOD_FAR
}

/** 視点（three 空間）から車（ENU の x, y。高さは車体の中ほど）までの距離の最小値 */
export function nearestViewDistance(
  views: ReadonlyArray<readonly [number, number, number]>,
  enuX: number,
  enuY: number,
): number {
  let best = Infinity
  for (const v of views) {
    const d = Math.hypot(v[0] - enuX, v[2] + enuY, v[1] - 0.7)
    if (d < best) best = d
  }
  return best
}
