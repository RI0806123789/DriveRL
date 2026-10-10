/** Service Worker（public/sw.js）が同じオリジンの別アプリのキャッシュに触れないかの単体テスト。 */

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, test } from 'node:test'
import vm from 'node:vm'

const ORIGIN = 'http://127.0.0.1:8000'
const SOURCE = readFileSync(new URL('../../public/sw.js', import.meta.url), 'utf8')

class FakeRequest {
  url: string
  method: string
  mode: string
  constructor(url: string, init: { method?: string; mode?: string } = {}) {
    this.url = new URL(url, ORIGIN).href
    this.method = init.method ?? 'GET'
    this.mode = init.mode ?? 'cors'
  }
}

class FakeResponse {
  body: string
  status: number
  type: string
  constructor(body: string, status = 200, type = 'basic') {
    this.body = body
    this.status = status
    this.type = type
  }
  clone(): FakeResponse {
    return new FakeResponse(this.body, this.status, this.type)
  }
  static error(): FakeResponse {
    return new FakeResponse('', 0, 'error')
  }
}

type Key = string | FakeRequest

function keyOf(request: Key): string {
  return typeof request === 'string' ? new URL(request, ORIGIN).href : request.url
}

class FakeCache {
  entries = new Map<string, FakeResponse>()
  doFetch: (r: Key) => Promise<FakeResponse>
  constructor(doFetch: (r: Key) => Promise<FakeResponse>) {
    this.doFetch = doFetch
  }
  async match(request: Key) {
    return this.entries.get(keyOf(request))
  }
  async put(request: Key, response: FakeResponse) {
    this.entries.set(keyOf(request), response)
  }
  async add(request: Key) {
    await this.put(request, await this.doFetch(request))
  }
  async keys() {
    return [...this.entries.keys()].map((url) => new FakeRequest(url))
  }
  async delete(request: Key) {
    return this.entries.delete(keyOf(request))
  }
}

interface Harness {
  stores: Map<string, FakeCache>
  dispatch: (type: string, extra?: Record<string, unknown>) => Promise<FakeResponse | undefined>
  setOnline: (online: boolean) => void
  fetched: string[]
}

function loadWorker(): Harness {
  const handlers = new Map<string, (event: unknown) => void>()
  const fetched: string[] = []
  let online = true
  const doFetch = async (request: Key) => {
    const url = keyOf(request)
    fetched.push(new URL(url).pathname)
    if (!online) throw new TypeError('offline')
    return new FakeResponse(`network:${new URL(url).pathname}`)
  }
  const stores = new Map<string, FakeCache>()
  const caches = {
    async open(name: string) {
      let store = stores.get(name)
      if (!store) {
        store = new FakeCache(doFetch)
        stores.set(name, store)
      }
      return store
    },
    async keys() {
      return [...stores.keys()]
    },
    async delete(name: string) {
      return stores.delete(name)
    },
    // 同じオリジンの全キャッシュを探す（実物と同じ）。sw.js がこれに頼ると別アプリの応答を拾う
    async match(request: Key) {
      for (const store of stores.values()) {
        const hit = await store.match(request)
        if (hit) return hit
      }
      return undefined
    },
  }
  const self = {
    location: { origin: ORIGIN },
    addEventListener: (type: string, fn: (event: unknown) => void) => handlers.set(type, fn),
    skipWaiting: async () => {},
    clients: { claim: async () => {} },
  }
  vm.runInNewContext(SOURCE, {
    self,
    caches,
    fetch: doFetch,
    Request: FakeRequest,
    Response: FakeResponse,
    URL,
    Promise,
    Boolean,
    TypeError,
  })

  const dispatch = async (type: string, extra: Record<string, unknown> = {}) => {
    const waits: Promise<unknown>[] = []
    let responded: Promise<FakeResponse> | undefined
    const event = {
      ...extra,
      waitUntil: (p: Promise<unknown>) => waits.push(p),
      respondWith: (p: Promise<FakeResponse>) => {
        responded = p
      },
    }
    handlers.get(type)?.(event)
    const response = responded ? await responded : undefined
    await Promise.all(waits)
    return response
  }
  return { stores, dispatch, setOnline: (v) => (online = v), fetched }
}

function shellVersion(): string {
  const m = SOURCE.match(/const CACHE_VERSION = '([^']+)'/)
  assert.ok(m, 'CACHE_VERSION が見つからない')
  return `driverl-shell-${m[1]}`
}

describe('Service Worker のキャッシュ', () => {
  test('activate は DriveRL の旧版だけを消し、最新版と別アプリのキャッシュを残す', async () => {
    const sw = loadWorker()
    const current = shellVersion()
    for (const name of ['driverl-shell-v4', 'driverl-shell-v7', 'other-app-cache', 'workbox-precache-v2']) {
      const store = new FakeCache(async () => new FakeResponse('x'))
      await store.put('/', new FakeResponse(name))
      sw.stores.set(name, store)
    }
    await sw.dispatch('install')
    await sw.dispatch('activate')
    assert.deepEqual([...sw.stores.keys()].sort(), [current, 'other-app-cache', 'workbox-precache-v2'].sort())
    assert.equal((await sw.stores.get('other-app-cache')!.match('/'))?.body, 'other-app-cache')
  })

  test('オフラインで開くと DriveRL のシェルを返す（通常の起動を保つ）', async () => {
    const sw = loadWorker()
    await sw.dispatch('install')
    await sw.dispatch('activate')
    sw.setOnline(false)
    const response = await sw.dispatch('fetch', { request: new FakeRequest('/', { mode: 'navigate' }) })
    assert.equal(response?.body, 'network:/')
  })

  test('オフラインでも別アプリが同じパスにキャッシュした HTML は返さない', async () => {
    const sw = loadWorker()
    const other = new FakeCache(async () => new FakeResponse('x'))
    await other.put('/', new FakeResponse('other-app-shell'))
    sw.stores.set('other-app-cache', other)
    sw.setOnline(false)
    const response = await sw.dispatch('fetch', { request: new FakeRequest('/', { mode: 'navigate' }) })
    assert.equal(response?.type, 'error')
  })

  test('別アプリが同じパスにキャッシュした JS は使わずネットワークから取る', async () => {
    const sw = loadWorker()
    const other = new FakeCache(async () => new FakeResponse('x'))
    await other.put('/assets/index-abc.js', new FakeResponse('other-app-js'))
    await other.put('/icon-192.png', new FakeResponse('other-app-icon'))
    sw.stores.set('other-app-cache', other)

    const asset = await sw.dispatch('fetch', { request: new FakeRequest('/assets/index-abc.js') })
    assert.equal(asset?.body, 'network:/assets/index-abc.js')

    const icon = await sw.dispatch('fetch', { request: new FakeRequest('/icon-192.png') })
    assert.equal(icon?.body, 'network:/icon-192.png')
    assert.equal((await sw.stores.get(shellVersion())!.match('/assets/index-abc.js'))?.body, 'network:/assets/index-abc.js')
  })
})
