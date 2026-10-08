/** モデルの書き出し要求とブラウザへの保存を検証する。 */

import assert from 'node:assert/strict'
import { test } from 'node:test'
import { downloadModel } from '../store/exportModel.ts'
import type { ExportKind } from '../store/exportModel.ts'

const kinds: ExportKind[] = ['checkpoint', 'torchscript', 'pt2', 'keras']

test('各形式の書き出しは JSON の POST で要求し、応答を保存する', async (t) => {
  const documentDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'document')
  const windowDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'window')
  t.after(() => {
    if (documentDescriptor) Object.defineProperty(globalThis, 'document', documentDescriptor)
    else Reflect.deleteProperty(globalThis, 'document')
    if (windowDescriptor) Object.defineProperty(globalThis, 'window', windowDescriptor)
    else Reflect.deleteProperty(globalThis, 'window')
  })

  const downloads: string[] = []
  const anchor = { href: '', download: '', rel: '', click: () => downloads.push(anchor.download) }
  let attached = false
  Object.defineProperty(globalThis, 'document', { configurable: true, value: {
    createElement: (tag: string) => {
      assert.equal(tag, 'a')
      return anchor
    },
    body: {
      appendChild: (element: unknown) => {
        assert.equal(element, anchor)
        attached = true
      },
      removeChild: (element: unknown) => {
        assert.equal(element, anchor)
        attached = false
      },
    },
  } })
  Object.defineProperty(globalThis, 'window', { configurable: true, value: {
    setTimeout: (callback: () => void, delay: number) => {
      assert.equal(delay, 10_000)
      callback()
    },
  } })
  t.mock.method(URL, 'createObjectURL', () => 'blob:model')
  const revoke = t.mock.method(URL, 'revokeObjectURL', () => {})
  let requestIndex = 0
  const fetch = t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const kind = kinds[requestIndex++]
    assert.equal(input, `/api/export/${kind}`)
    assert.equal(init?.method, 'POST')
    const headers = new Headers(init?.headers)
    assert.equal(headers.get('Content-Type'), 'application/json')
    assert.equal(headers.get('Accept'), 'application/octet-stream')
    assert.deepEqual(JSON.parse(String(init?.body)), {})
    return new Response(new Blob(['model']), {
      headers: { 'Content-Disposition': `attachment; filename="${kind}.model"` },
    })
  })

  for (const kind of kinds) {
    assert.deepEqual(await downloadModel(kind), { ok: true, filename: `${kind}.model`, sizeBytes: 5 })
    assert.equal(attached, false)
    assert.equal(anchor.href, 'blob:model')
    assert.equal(anchor.rel, 'noopener')
  }
  assert.deepEqual(downloads, kinds.map((kind) => `${kind}.model`))
  assert.equal(fetch.mock.callCount(), kinds.length)
  assert.equal(revoke.mock.callCount(), kinds.length)
})

test('拒否された書き出しではダウンロードせず、サーバーの理由を返す', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(
    JSON.stringify({ error: 'このアドレスからの接続は許可されていません' }),
    { status: 403, headers: { 'Content-Type': 'application/json' } },
  ))
  assert.deepEqual(await downloadModel('checkpoint'), {
    ok: false, error: 'このアドレスからの接続は許可されていません',
  })
})
