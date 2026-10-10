/** モック（store/mockServer.ts）が、認識器の学習中は実機と同じく物理と学習を止めるかの単体テスト（#128）。 */

import assert from 'node:assert/strict'
import { test, type TestContext } from 'node:test'

import { createMockTransport } from '../store/mockServer.ts'

type Msg = { type: string; [key: string]: unknown }

const REQUEST = {
  mode: 'collect',
  presetId: 'ginza',
  samples: 240,
  epochs: 1,
  batchSize: 32,
  width: 0.5,
  seed: 0,
  weatherMix: false,
  focusWeak: false,
}

function open(t: TestContext) {
  t.mock.timers.enable({ apis: ['setInterval', 'setTimeout', 'Date'] })
  t.mock.method(console, 'info', () => undefined)
  const messages: Msg[] = []
  const transport = createMockTransport((json) => messages.push(JSON.parse(json) as Msg))
  t.after(() => transport.close())
  const send = (msg: object) => transport.send(JSON.stringify(msg))
  const after = (start: number, type: string) => messages.slice(start).filter((m) => m.type === type)
  const last = (type: string) => [...messages].reverse().find((m) => m.type === type)
  return { messages, send, after, last, tick: (ms: number) => t.mock.timers.tick(ms) }
}

function startRunning(m: ReturnType<typeof open>) {
  m.tick(100)
  m.send({ type: 'load_map', presetId: 'ginza' })
  m.tick(1500)
  m.tick(2000)
  const frame = m.last('frame')
  assert.ok(frame, '地図を読んだ後に frame が届く')
  return frame
}

test('学習中は時刻も物理も進めず frame を送らない。学習の進み具合は届く', (t) => {
  const m = open(t)
  startRunning(m)
  m.send({ type: 'start_detector_training', request: REQUEST })
  const status = m.last('status')
  assert.equal(status?.simSuspended, true, '止めていると知らせる')
  const stoppedAt = m.messages.length
  const before = m.last('frame') as unknown as { tick: number; simTime: number }

  m.tick(3000)
  // 失敗したときに frame の中身を差分にしないよう、通数で比べる（全車ぶんの差分を作るとメモリが尽きる）
  assert.equal(m.after(stoppedAt, 'frame').length, 0, '止めている間は frame を送らない')
  assert.ok(m.after(stoppedAt, 'detector').length >= 5, '学習の進み具合は届き続ける')
  const metrics = m.after(stoppedAt, 'metrics')
  assert.ok(metrics.length >= 2, 'metrics は 1Hz で届き続ける')
  assert.ok(metrics.every((x) => x.updates === metrics[0].updates), '学習（PPO）の更新回数は増えない')
  assert.ok(metrics.every((x) => JSON.stringify(x) === JSON.stringify(metrics[0])), '同じ値を配り直す')

  // 中止すると、止めた所の続きから進む
  m.send({ type: 'cancel_detector_training' })
  assert.equal(m.last('status')?.simSuspended, false)
  const resumedAt = m.messages.length
  m.tick(100)
  const next = m.after(resumedAt, 'frame')[0] as unknown as { tick: number; simTime: number }
  assert.ok(next, '再開すると frame が届く')
  assert.equal(next.tick, before.tick + 1, '止めている間に tick を進めていない')
  assert.ok(Math.abs(next.simTime - before.simTime - 0.05) < 1e-9, '止めている間に simTime を進めていない')
  m.tick(1100)
  const resumedMetrics = m.after(resumedAt, 'metrics')
  assert.ok(resumedMetrics.some((x) => (x.updates as number) > (metrics[0].updates as number)), '再開すると学習も進む')
})

test('学習が終わると自動で再開する', (t) => {
  const m = open(t)
  startRunning(m)
  m.send({ type: 'start_detector_training', request: REQUEST })
  const stoppedAt = m.messages.length
  m.tick(8000) // 収集は 1 周 250ms × 24 周
  const done = m.after(stoppedAt, 'detector').at(-1)
  assert.equal(done?.running, false)
  assert.equal(m.last('status')?.simSuspended, false)
  const doneAt = m.messages.lastIndexOf(m.last('status') as Msg)
  assert.equal(m.messages.slice(stoppedAt, doneAt).filter((x) => x.type === 'frame').length, 0, '終わるまで frame は届かない')
  m.tick(200)
  assert.ok(m.after(doneAt, 'frame').length >= 2, '終わった後は frame が届く')
})
