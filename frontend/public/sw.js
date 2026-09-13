/**
 * Service Worker（PWA のインストールと起動を成り立たせるための最小限）。
 *
 * ★ **このアプリはオフラインでは動かない。** 走行も学習もバックエンド
 *   （FastAPI + PPO スレッド）の中で起きていて、画面は WebSocket で
 *   受け取ったものを描いているだけだからである。ここでキャッシュするのは
 *   「起動を速くする」「インストール可能にする」ためのシェル（HTML と JS/CSS）
 *   だけで、**オフラインで開いても「バックエンドに接続していません」の
 *   ベールが出る**（それが正しい表示）。
 *
 * ライブラリは使わない（workbox も vite-plugin-pwa も入れない）。
 * この 1 ファイルで完結させ、規則を全部ここに書いておく。
 *
 * ---------------------------------------------------------------------------
 * 守らなければならないこと
 * ---------------------------------------------------------------------------
 *
 * ★ 1. **`/api/` には絶対に触らない。** モデルの書き出し（GET /api/export/...）は
 *   数十 MB のバイナリ、読み込み（POST /api/import）は本体がアップロード。
 *   キャッシュに入れれば古い重みを配り、`respondWith` を挟むだけでも
 *   ダウンロードの進捗が壊れる。GET 以外も同じ理由で素通しする。
 *
 * ★ 2. **HTML は必ずネットワーク優先。** `index.html` はファイル名が固定なのに
 *   中身（参照する `index-<ハッシュ>.js`）が毎ビルド変わる。キャッシュを先に
 *   返すと、**再ビルドしたのに既に消えた JS を指したままの画面**が出る。
 *   これは実際に起きた事故で、バックエンド側も同じ理由で index.html にだけ
 *   `Cache-Control: no-cache` を付けている（`app/main.py` の `_FrontendStatic`）。
 *   **その対策をこちらで台無しにしないこと。**
 *
 * ★ 3. **`/assets/` だけはキャッシュ優先でよい。** Vite の出力は内容が変われば
 *   ファイル名が変わる（`index-CA4O_nSs.css`）ので、同じ名前なら中身も同じ。
 *
 * ★ 4. **`CACHE_VERSION` は生成物の中身を変えたら上げること。**
 *   `backend/app/map/loader.py` の `CACHE_VERSION` と同じ約束。上げないと
 *   古いシェルが残り続ける（オンラインなら 2 の通り HTML は新しくなるので
 *   実害は出にくいが、オフライン時の表示だけが古いまま取り残される）。
 */

const CACHE_VERSION = 'v1'
const CACHE_NAME = `driverl-shell-${CACHE_VERSION}`

/**
 * 先に取っておくもの。**ハッシュ付きの JS/CSS はここに書けない**
 * （名前がビルドごとに変わるため）。それらは初回の表示時に拾う。
 */
const SHELL = ['/', '/manifest.webmanifest', '/icon-192.png', '/icon-512.png']

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME)
      // 1 つでも失敗すると addAll ごと落ちるので、個別に入れて取りこぼしを許す
      await Promise.all(
        SHELL.map((url) =>
          cache.add(new Request(url, { cache: 'reload' })).catch(() => {
            /* バックエンドが落ちている最中の更新など。次回の表示で拾う */
          }),
        ),
      )
      // 新しい SW を待たせない。中身はハッシュ付きなので、
      // 古いページが新しい SW に管理されても参照先は変わらない
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

self.addEventListener('fetch', (event) => {
  const request = event.request

  // ★ GET 以外は素通し（POST /api/import を絶対に横取りしない）
  if (request.method !== 'GET') return

  const url = new URL(request.url)

  // 他オリジン（CDN も外部 API も使っていないので通常は来ない）は素通し
  if (url.origin !== self.location.origin) return

  // ★ API と WebSocket は素通し。ここを通すとモデルの入出力が壊れる
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws')) return

  // ★ HTML（ナビゲーション）はネットワーク優先。
  //   落ちているときだけキャッシュのシェルを返す
  if (request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          const fresh = await fetch(request)
          if (isCacheable(fresh)) {
            const cache = await caches.open(CACHE_NAME)
            cache.put('/', fresh.clone())
          }
          return fresh
        } catch {
          const cached = (await caches.match('/')) ?? (await caches.match(request))
          if (cached) return cached
          // シェルすら無い（初回からオフライン）。ブラウザの既定の失敗表示に任せる
          return Response.error()
        }
      })(),
    )
    return
  }

  // ★ ハッシュ付きのアセットはキャッシュ優先（同じ名前なら中身も同じ）
  if (url.pathname.startsWith('/assets/')) {
    event.respondWith(
      (async () => {
        const cached = await caches.match(request)
        if (cached) return cached
        const fresh = await fetch(request)
        if (isCacheable(fresh)) {
          const cache = await caches.open(CACHE_NAME)
          cache.put(request, fresh.clone())
        }
        return fresh
      })(),
    )
    return
  }

  // それ以外（アイコン・マニフェスト）は「キャッシュを返しつつ裏で更新」。
  // どれも小さく、古くても表示が壊れないもの
  event.respondWith(
    (async () => {
      const cached = await caches.match(request)
      const network = fetch(request)
        .then(async (fresh) => {
          if (isCacheable(fresh)) {
            const cache = await caches.open(CACHE_NAME)
            cache.put(request, fresh.clone())
          }
          return fresh
        })
        .catch(() => null)
      return cached ?? (await network) ?? Response.error()
    })(),
  )
})
