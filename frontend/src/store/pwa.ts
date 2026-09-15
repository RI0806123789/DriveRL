/** Service Worker の登録（PWA）。 */

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
      })
    return
  }

  window.addEventListener('load', () => {
    void navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch((e) => {
      console.warn('[pwa] Service Worker を登録できませんでした', e)
    })
  })
}
