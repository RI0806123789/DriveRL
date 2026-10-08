/** 古い問い合わせが会話と新しい問い合わせの pending を更新しないことを検査する。 */

import assert from 'node:assert/strict'
import { register } from 'node:module'
import { test } from 'node:test'
import { hooksUrl } from './support/tsxHooks.ts'

register(hooksUrl)
const { useConcierge } = await import('../store/concierge.ts')
const { useSimStore } = await import('../store/simStore.ts')

test('clear 後の古い成功・失敗は新しい履歴と pending を書き換えない', async (t) => {
  for (const fail of [false, true]) {
    await t.test(fail ? '通信失敗' : '成功', async (t) => {
      const calls: { resolve(response: Response): void; reject(error: Error): void }[] = []
      t.mock.method(globalThis, 'fetch', () => new Promise<Response>((resolve, reject) => calls.push({ resolve, reject })))
      const store = useConcierge.getState()
      store.clear()
      const old = store.ask({ message: '以前の質問', rideId: 'ride-A' })
      store.clear()
      const current = store.ask({ message: '新しい質問', rideId: 'ride-B' })
      if (fail) calls[0]!.reject(new Error('接続失敗'))
      else calls[0]!.resolve(Response.json({ ok: true, reply: '以前の返答' }))
      await old
      assert.equal(useConcierge.getState().pending, true)
      assert.deepEqual(useConcierge.getState().entries.map((entry) => entry.text), ['新しい質問'])
      calls[1]!.resolve(Response.json({ ok: true, reply: '新しい返答' }))
      await current
      assert.equal(useConcierge.getState().pending, false)
      assert.deepEqual(useConcierge.getState().entries.map((entry) => entry.text), ['新しい質問', '新しい返答'])
    })
  }
})

test('配車 ID とモードの変更で古い fetch を無効化し、迎車の引き継ぎは会話を保つ', async (t) => {
  let resolve!: (response: Response) => void
  let sent: unknown
  t.mock.method(globalThis, 'fetch', (_url: unknown, options?: RequestInit) => {
    sent = JSON.parse(String(options?.body))
    return new Promise<Response>((done) => { resolve = done })
  })
  const sim = useSimStore.getState()
  useSimStore.setState({ mode: 'taxi' })
  sim.applyServerMessage({ ...sim.taxi, type: 'taxi', phase: 'approaching', rideId: 'ride-A' })
  const old = useConcierge.getState().ask({ action: 'hurry', rideId: 'ride-A' })
  assert.deepEqual(sent, { action: 'hurry', rideId: 'ride-A' })
  sim.applyServerMessage({ ...sim.taxi, type: 'taxi', phase: 'approaching', rideId: 'ride-A', vehicleId: 2 })
  assert.equal(useConcierge.getState().pending, true)
  sim.applyServerMessage({ ...sim.taxi, type: 'taxi', phase: 'approaching', rideId: 'ride-B' })
  resolve(Response.json({ ok: true, reply: '古い返答' }))
  await old
  assert.deepEqual(useConcierge.getState().entries, [])
  const next = useConcierge.getState().ask({ action: 'comfort', rideId: 'ride-B' })
  useSimStore.getState().setMode('dev')
  resolve(Response.json({ ok: true, reply: '古い返答' }))
  await next
  assert.deepEqual(useConcierge.getState().entries, [])
  assert.equal(useConcierge.getState().pending, false)
})
