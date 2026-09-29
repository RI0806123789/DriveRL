/** 車車間通信（V2X）のリンク。近傍の選び方（モック）、描く線分の頂点、追跡中の車の表示の文言。 */

/** 届く距離 [m]（`backend/app/config.py` の `V2X_RANGE_M`） */
export const V2X_RANGE_M = 30
/** 受け取る相手の数（同じく `V2X_MAX_PEERS`） */
export const V2X_MAX_PEERS = 2
/** 線を引く高さ [m]（屋根の少し上） */
export const V2X_LINK_HEIGHT_M = 1.9

export interface LinkVehicle {
  id: number
  active: boolean
  x: number
  y: number
  v2xConnectedIds?: number[]
}

/** 車ごとに、届く距離の中で近い順に最大 `maxPeers` 台を選ぶ（バックエンドの `route_and_aggregate` と同じ規則）。 */
export function nearestPeers(
  vehicles: readonly LinkVehicle[],
  range = V2X_RANGE_M,
  maxPeers = V2X_MAX_PEERS,
): Map<number, number[]> {
  const out = new Map<number, number[]>()
  const alive = vehicles.filter((v) => v.active)
  for (const v of alive) {
    const near = alive
      .filter((o) => o.id !== v.id)
      .map((o) => ({ id: o.id, d: Math.hypot(o.x - v.x, o.y - v.y) }))
      .filter((o) => o.d <= range)
      .sort((a, b) => a.d - b.d || a.id - b.id)
      .slice(0, Math.max(0, maxPeers))
      .map((o) => o.id)
    if (near.length > 0) out.set(v.id, near)
  }
  return out
}

/** 線で結ぶ組（向きを区別しない。小さい id が先）。どちらかが走っていない・見当たらない組は外す。 */
export function linkPairs(vehicles: readonly LinkVehicle[]): Array<[number, number]> {
  const alive = new Set(vehicles.filter((v) => v.active).map((v) => v.id))
  const seen = new Set<string>()
  const pairs: Array<[number, number]> = []
  for (const v of vehicles) {
    if (!v.active || !v.v2xConnectedIds) continue
    for (const peer of v.v2xConnectedIds) {
      if (peer === v.id || !alive.has(peer)) continue
      const a = Math.min(v.id, peer)
      const b = Math.max(v.id, peer)
      const key = `${a}:${b}`
      if (seen.has(key)) continue
      seen.add(key)
      pairs.push([a, b])
    }
  }
  return pairs.sort((p, q) => p[0] - q[0] || p[1] - q[1])
}

/** 組ごとの線分の頂点を three の座標（x, 高さ, -y）で `out` に書き、書いた頂点の数を返す。位置が分からない組は飛ばす。 */
export function writeLinkSegments(
  pairs: ReadonlyArray<readonly [number, number]>,
  position: (id: number) => { x: number; y: number } | null,
  out: Float32Array,
  height = V2X_LINK_HEIGHT_M,
): number {
  let n = 0
  for (const [a, b] of pairs) {
    if ((n + 2) * 3 > out.length) break
    const pa = position(a)
    const pb = position(b)
    if (!pa || !pb) continue
    out.set([pa.x, height, -pa.y, pb.x, height, -pb.y], n * 3)
    n += 2
  }
  return n
}

/** 追跡中の車のチップの文言。受け取った相手がいなければ null（チップを出さない）。 */
export function v2xStatusText(links: readonly number[] | undefined): string | null {
  if (!links || links.length === 0) return null
  return `V2X: ${links.map((id, i) => (i === 0 ? `車両#${id}` : `#${id}`)).join('・')}とリンク中`
}
