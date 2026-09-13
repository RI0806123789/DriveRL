/**
 * Service Worker の登録（PWA）。
 *
 * 実体は `frontend/public/sw.js`（ライブラリを足さず手書き）。ここは
 * 「いつ登録するか」だけを持つ。
 *
 * ★ **開発中（Vite）は登録しない。** dev サーバーはモジュールを 1 つずつ配り、
 *   HMR で差し替える。そこに SW を挟むと、保存しても古いモジュールが
 *   返り続ける（しかも「効かないのは HMR のせいか SW のせいか」が
 *   画面から切り分けられない）。本番ビルドでのみ動かす。
 *
 * ★ **開発中に前回の登録が残っていたら剥がす。** 一度でも本番ビルドを
 *   同じオリジンで開いていると SW が居座り、`npm run dev` に切り替えた後も
 *   ページを支配し続ける。**ポートが同じなら（8000 で `-Build`、その後
 *   5173 で開発、ではなく `run.py` 経由で同じ 8000 を使い回した場合）
 *   これが実際に起こる。**
 */

/** 本番ビルドで Service Worker を登録する（開発中は剥がす） */
export function setupServiceWorker(): void {
  if (typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return

  if (!import.meta.env.PROD) {
    void navigator.serviceWorker
      .getRegistrations()
      .then((registrations) => {
        for (const registration of registrations) {
          console.info('[pwa] 開発中なので Service Worker を解除しました')
          void registration.unregister()
        }
      })
      .catch(() => {
        /* 解除できなくても開発は続けられる */
      })
    return
  }

  // `load` を待つ。初回表示と SW の登録・シェルの取得が同時に走ると、
  // 3D の初期化（three と drei で 1.2MB）と帯域を取り合う
  window.addEventListener('load', () => {
    void navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch((e) => {
      // 登録できなくてもアプリは動く（オフラインで開けなくなるだけ）。
      // ただし黙らせない：インストールできない原因がここにしか出ない
      console.warn('[pwa] Service Worker を登録できませんでした', e)
    })
  })
}
