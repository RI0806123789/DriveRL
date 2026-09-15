/** Service Worker（PWA のインストールと起動を成り立たせるための最小限）。 */

const CACHE_VERSION = 'v2'
const CACHE_NAME = `driverl-shell-${CACHE_VERSION}`

/** 先に取っておくもの。**ハッシュ付きの JS/CSS はここに書けない** */
const SHELL = ['/', '/icon-192.png', '/icon-512.png', '/icon-maskable-512.png']

/** `/assets/` に残してよい件数の上限。 */
const ASSET_KEEP = 24

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME)
      await Promise.all(
        SHELL.map((url) =>
          cache.add(new Request(url, { cache: 'reload' })).catch(() => {
          }),
        ),
      )
      await self.skipWaiting()
    })(),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys()
      await Promise.all(
        names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name)),
      )
      await self.clients.claim()
    })(),
  )
})

/** キャッシュに入れてよい応答か（部分応答・エラー・他オリジンは入れない） */
function isCacheable(response) {
  return Boolean(response) && response.status === 200 && response.type === 'basic'
}

/** `/assets/` のキャッシュを古いものから `ASSET_KEEP` 件まで減らす。 */
async function trimAssets(cache) {
  const keys = await cache.keys()
  const assets = keys.filter((request) => new URL(request.url).pathname.startsWith('/assets/'))
  const excess = assets.length - ASSET_KEEP
  if (excess <= 0) return
  await Promise.all(assets.slice(0, excess).map((request) => cache.delete(request)))
}

/** 取ってきた応答をキャッシュへ入れる。**`event.waitUntil` で寿命を伸ばす** */
function putInCache(event, request, response, { trim = false } = {}) {
  if (!isCacheable(response)) return
  const copy = response.clone()
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME)
      await cache.put(request, copy)
      if (trim) await trimAssets(cache)
    })(),
  )
}

self.addEventListener('fetch', (event) => {
  const request = event.request

  if (request.method !== 'GET') return

  const url = new URL(request.url)

  if (url.origin !== self.location.origin) return

  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws')) return

  if (url.pathname === '/manifest.webmanifest') return

  if (request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          const fresh = await fetch(request)
          putInCache(event, '/', fresh)
          return fresh
        } catch {
          const cached = (await caches.match('/')) ?? (await caches.match(request))
          if (cached) return cached
          return Response.error()
        }
      })(),
    )
    return
  }

  if (url.pathname.startsWith('/assets/')) {
    event.respondWith(
      (async () => {
        const cached = await caches.match(request)
        if (cached) return cached
        const fresh = await fetch(request)
        putInCache(event, request, fresh, { trim: true })
        return fresh
      })(),
    )
    return
  }

  event.respondWith(
    (async () => {
      const cached = await caches.match(request)
      const network = fetch(request)
        .then((fresh) => {
          putInCache(event, request, fresh)
          return fresh
        })
        .catch(() => null)
      return cached ?? (await network) ?? Response.error()
    })(),
  )
})
