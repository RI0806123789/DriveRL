/** 経路の矢印のジオメトリのキャッシュを、frameBuffer の経路の表に合わせる。React と three から切り離した純粋な処理。 */

import type { Vec2 } from '../types/protocol'

/** キャッシュ 1 件。rev は「このジオメトリを作った時点の経路の版」 */
export interface HeldRoute<T> {
  entry: T
  rev: number
}

/** 消えた車両のものを捨て、版が変わった車両のものだけ作り直す。キャッシュが変わったら true。 */
export function syncRouteCache<T>(
  cache: Map<number, HeldRoute<T>>,
  routes: ReadonlyMap<number, Vec2[]>,
  revisions: ReadonlyMap<number, number>,
  build: (id: number, points: Vec2[]) => T,
  retire: (entry: T) => void,
): boolean {
  let dirty = false
  for (const id of Array.from(cache.keys())) {
    if (!routes.has(id)) {
      const held = cache.get(id)
      if (held) retire(held.entry)
      cache.delete(id)
      dirty = true
    }
  }
  routes.forEach((points, id) => {
    const rev = revisions.get(id) ?? 0
    const current = cache.get(id)
    if (current && current.rev === rev) return
    if (current) retire(current.entry)
    cache.set(id, { rev, entry: build(id, points) })
    dirty = true
  })
  return dirty
}
