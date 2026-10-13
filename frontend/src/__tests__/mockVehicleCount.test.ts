/** モックの予約枠パラメータと実働車両の操作を分けて検査する（#146）。 */

import assert from 'node:assert/strict'
import { test, type TestContext } from 'node:test'

import type { ServerMessage } from '../types/protocol.ts'
import { createMockTransport } from '../store/mockServer.ts'

function open(t: TestContext) {
  t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'] })
  t.mock.method(console, 'info', () => undefined)
  const messages: ServerMessage[] = []
  const transport = createMockTransport((json) => messages.push(JSON.parse(json) as ServerMessage))
  t.after(() => transport.close())
  const send = (message: object) => transport.send(JSON.stringify(message))
  const tick = (ms: number) => t.mock.timers.tick(ms)
  const last = <T extends ServerMessage['type']>(type: T) => [...messages].reverse().find(
    (message): message is Extract<ServerMessage, { type: T }> => message.type === type,
  )
  tick(100)
  const init = last('init')
  assert.ok(init)
  return { messages, send, tick, last, init }
}

function loadMap(m: ReturnType<typeof open>) {
  m.send({ type: 'load_map', presetId: 'ginza' })
  m.tick(1500)
  m.tick(100)
  const frame = m.last('frame')
  assert.ok(frame)
  return frame
}

test('マップ読み込み前も vehicleCount の値にかかわらず拒否し、error の後に同じ params を返す', (t) => {
  const m = open(t)
  for (const vehicleCount of [0, 1, 8, 99, -1, 1.5, '2', true, null, {}, [1]]) {
    const start = m.messages.length
    m.send({ type: 'set_params', params: { vehicleCount } })
    assert.deepEqual(m.messages.slice(start), [
      {
        type: 'error',
        code: 'INVALID_MESSAGE',
        message: 'params の値が許容範囲外です。無視した項目: vehicleCount',
      },
      { type: 'params', params: m.init.params },
    ])
  }
})

test('vehicleCount を含む部分更新でも、他のパラメータは適用して配り直す', (t) => {
  const m = open(t)
  const before = loadMap(m)
  const start = m.messages.length
  const patch = { simSpeed: 2, pedestrianCount: 3, weatherRain: 0.35, obeySignals: false }
  m.send({ type: 'set_params', params: { vehicleCount: 0, ...patch } })
  assert.deepEqual(m.messages.slice(start).map((message) => message.type), ['error', 'params'])
  assert.deepEqual(m.last('params')?.params, { ...m.init.params, ...patch })
  m.tick(100)
  const frame = m.last('frame')
  assert.ok(frame)
  assert.deepEqual(frame.vehicles.map((v) => v.active), before.vehicles.map((v) => v.active))
  assert.equal(frame.vehicles.filter((v) => v.active).length, m.init.config.maxVehicles)
  assert.equal(frame.pedestrians?.length, 3)
})

test('params がオブジェクトでなければ error だけを返し、次の更新も受け付ける', (t) => {
  const m = open(t)
  for (const params of [null, [], 2, 'vehicleCount']) {
    const start = m.messages.length
    m.send({ type: 'set_params', params })
    assert.deepEqual(m.messages.slice(start), [
      { type: 'error', code: 'INVALID_MESSAGE', message: 'params が不正です' },
    ])
  }
  m.send({ type: 'set_params', params: { simSpeed: 2 } })
  assert.equal(m.last('params')?.params.simSpeed, 2)
  assert.equal(m.last('params')?.params.vehicleCount, m.init.params.vehicleCount)
})

test('車両の除去と追加は実働状態だけを変え、別の params 更新でも除去済み車両を復活させない', (t) => {
  const m = open(t)
  const initial = loadMap(m)
  const id = initial.vehicles[3]!.id
  const start = m.messages.length
  m.send({ type: 'despawn_vehicle', id })
  m.tick(100)
  const removed = m.last('frame')
  assert.ok(removed)
  assert.equal(removed.vehicles.find((v) => v.id === id)?.active, false)
  assert.equal(removed.vehicles.filter((v) => v.active).length, m.init.params.vehicleCount - 1)
  assert.equal(m.messages.slice(start).filter((message) => message.type === 'params').length, 0)

  m.send({ type: 'set_params', params: { simSpeed: 2 } })
  assert.equal(m.last('params')?.params.vehicleCount, m.init.params.vehicleCount)
  m.tick(100)
  assert.equal(m.last('frame')?.vehicles.find((v) => v.id === id)?.active, false)

  const spawnedAt = m.messages.length
  m.send({ type: 'spawn_vehicle', x: 0, y: 0 })
  m.tick(100)
  const spawned = m.last('frame')
  assert.ok(spawned)
  assert.equal(spawned.vehicles.find((v) => v.id === id)?.active, true)
  assert.equal(spawned.vehicles.filter((v) => v.active).length, m.init.params.vehicleCount)
  assert.equal(m.messages.slice(spawnedAt).filter((message) => message.type === 'params').length, 0)
  m.send({ type: 'set_params', params: {} })
  assert.equal(m.last('params')?.params.vehicleCount, m.init.params.vehicleCount)
})

test('全車を除去しても予約枠を維持し、reset_episode は非アクティブの車両を復活させない', (t) => {
  const m = open(t)
  const frame = loadMap(m)
  const start = m.messages.length
  for (const v of frame.vehicles) m.send({ type: 'despawn_vehicle', id: v.id })
  m.tick(100)
  assert.equal(m.last('frame')?.vehicles.filter((v) => v.active).length, 0)
  m.send({ type: 'reset_episode' })
  m.tick(100)
  assert.equal(m.last('frame')?.vehicles.filter((v) => v.active).length, 0)
  assert.equal(m.messages.slice(start).filter((message) => message.type === 'params').length, 0)
  m.send({ type: 'set_params', params: { pedestrianCount: 0, vehicleCount: 1 } })
  m.tick(100)
  assert.equal(m.last('params')?.params.vehicleCount, m.init.params.vehicleCount)
  assert.equal(m.last('frame')?.vehicles.filter((v) => v.active).length, 0)
  m.send({ type: 'spawn_vehicle', x: 0, y: 0 })
  m.tick(100)
  assert.equal(m.last('frame')?.vehicles.filter((v) => v.active).length, 1)
  m.send({ type: 'reset_episode' })
  m.tick(100)
  assert.equal(m.last('frame')?.vehicles.filter((v) => v.active).length, 1)
  const resetAt = m.messages.length
  m.send({ type: 'reset_policy' })
  m.tick(100)
  assert.equal(m.last('frame')?.vehicles.filter((v) => v.active).length, 1)
  assert.equal(m.messages.slice(resetAt).filter((message) => message.type === 'params').length, 0)
  m.send({ type: 'set_params', params: {} })
  assert.equal(m.last('params')?.params.vehicleCount, m.init.params.vehicleCount)
})

test('空きスロットがない追加と範囲外 ID の除去は台数も params も変えない', (t) => {
  const m = open(t)
  const initial = loadMap(m)
  const start = m.messages.length
  m.send({ type: 'spawn_vehicle', x: 0, y: 0 })
  assert.equal(m.last('status')?.message, '空きスロットがありません')
  assert.equal(m.last('status')?.notice, true)
  for (const id of [-1, m.init.config.maxVehicles, 0.5]) m.send({ type: 'despawn_vehicle', id })
  m.tick(100)
  assert.deepEqual(m.last('frame')?.vehicles.map((v) => v.active), initial.vehicles.map((v) => v.active))
  assert.equal(m.messages.slice(start).filter((message) => message.type === 'params').length, 0)
})

test('学習の自動化中も vehicleCount を拒否し、探索対象以外の部分更新は受け付ける', (t) => {
  const m = open(t)
  loadMap(m)
  m.send({ type: 'start_auto_tune' })
  assert.equal(m.last('autotune')?.running, true)
  const before = m.last('params')
  assert.ok(before)
  const start = m.messages.length
  m.send({ type: 'set_params', params: { vehicleCount: 0, learningRate: 0.01, simSpeed: 2 } })
  assert.deepEqual(m.messages.slice(start).map((message) => message.type), ['status', 'error', 'params'])
  assert.deepEqual(m.last('params')?.params, { ...before.params, simSpeed: 2 })
  assert.equal(m.last('params')?.params.vehicleCount, m.init.params.vehicleCount)
})
