/** 経路の矢印のキャッシュ（scene/routeCache.ts）が、frameBuffer のリセットをまたいでも新しい経路へ作り直すかの単体テスト。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { describe, test } from 'node:test'
import { hooksUrl } from './support/tsxHooks.ts'

register(hooksUrl)
const { frameBuffer, pushFrame, resetFrameBuffer } = await import('../store/frameBuffer.ts')
const { syncRouteCache } = await import('../scene/routeCache.ts')

type Vec2 = [number, number]
type FrameMessage = Parameters<typeof pushFrame>[0]

let simTime = 0

function frame(vehicles: { id: number; active?: boolean; route?: Vec2[] }[]): FrameMessage {
  simTime += 0.05
  return {
    type: 'frame',
    tick: Math.round(simTime * 20),
    simTime,
    vehicles: vehicles.map(({ id, active = true, route }) => ({
      id, active, x: 0, y: 0, heading: 0, speed: 0, steer: 0, collided: false, reachedGoal: false, goal: [0, 0],
      ...(route ? { route } : {}),
    })),
    obstacles: [],
  } as unknown as FrameMessage
}

/** RouteLines の useFrame と同じ手順（routeVersion が変わったときだけ同期する）。 */
function makeRenderer() {
  const cache = new Map<number, { entry: Vec2[]; rev: number }>()
  const retired: Vec2[][] = []
  let built = 0
  let lastVersion = -1
  const draw = () => {
    if (frameBuffer.routeVersion === lastVersion) return false
    lastVersion = frameBuffer.routeVersion
    return syncRouteCache(
      cache,
      frameBuffer.routes,
      frameBuffer.routeRevisions,
      (_id, points) => {
        built += 1
        return points
      },
      (entry) => retired.push(entry),
    )
  }
  return { cache, retired, draw, built: () => built }
}

const A: Vec2[] = [[0, 0], [10, 0]]
const B: Vec2[] = [[100, 100], [200, 100]]

describe('経路の矢印のキャッシュ', () => {
  test('リセットの直後、次の描画の前に届いた新しい経路へ作り直す（同じ車両 ID）', () => {
    resetFrameBuffer()
    const r = makeRenderer()
    pushFrame(frame([{ id: 0, route: A }]))
    r.draw()
    assert.deepEqual(r.cache.get(0)?.entry, A)

    resetFrameBuffer()
    pushFrame(frame([{ id: 0, route: B }]))
    assert.deepEqual(frameBuffer.routes.get(0), B)
    r.draw()
    assert.deepEqual(r.cache.get(0)?.entry, B, '描画が古い経路のまま')
    assert.deepEqual(r.retired, [A], '古いジオメトリは解放に回す')
  })

  test('止まって経路が消え、次の描画の前に別の経路で走り出しても作り直す', () => {
    resetFrameBuffer()
    const r = makeRenderer()
    pushFrame(frame([{ id: 3, route: A }]))
    r.draw()
    pushFrame(frame([{ id: 3, active: false }]))
    pushFrame(frame([{ id: 3, route: B }]))
    r.draw()
    assert.deepEqual(r.cache.get(3)?.entry, B)
  })

  test('経路が変わらなければ作り直さない（毎フレームの再構築を増やさない）', () => {
    resetFrameBuffer()
    const r = makeRenderer()
    pushFrame(frame([{ id: 0, route: A }, { id: 1, route: B }]))
    r.draw()
    assert.equal(r.built(), 2)
    for (let i = 0; i < 10; i++) {
      pushFrame(frame([{ id: 0 }, { id: 1 }]))
      r.draw()
    }
    assert.equal(r.built(), 2)
    pushFrame(frame([{ id: 0 }, { id: 1, route: A }]))
    r.draw()
    assert.equal(r.built(), 3, '変わった 1 台だけ作り直す')
  })

  test('リセットで経路が消えた車両は捨てて解放に回す', () => {
    resetFrameBuffer()
    const r = makeRenderer()
    pushFrame(frame([{ id: 5, route: A }]))
    r.draw()
    resetFrameBuffer()
    assert.equal(r.draw(), true)
    assert.equal(r.cache.size, 0)
    assert.deepEqual(r.retired, [A])
  })
})
