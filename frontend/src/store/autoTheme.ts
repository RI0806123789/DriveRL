/**
 * 配色を日の出・日の入りに連動させる。
 *
 * **利用者の設定項目は無い。** 時計と暦だけで決まる。
 *
 * 場所は「いま走らせている街」を使う。銀座を走らせているなら東京の日の出で切り替わる。
 * こうしているのは、
 *   - 画面に映っているのがその街だから（空の明るさと街並みが食い違わない）
 *   - 位置情報の許可を求めずに済むから（`navigator.geolocation` は使わない）
 *   - サーバーが `init` で `centerLat` / `centerLon` を送ってくるので、追加の通信が要らないから
 *
 * マップを読み込む前は座標が分からないので、タイムゾーンから経度を逆算して代用する
 * （`themeClock.timezoneLocation()`）。
 *
 * 切り替わるのはパネル UI（`<html data-theme>` → `tokens.css`）と
 * 3D シーン（`scene/palette.ts`）の**両方**。同じ `theme` を見ている。
 */

import { useEffect } from 'react'
import { applyTheme, useSimStore } from './simStore'
import { nextThemeChange, themeFor, timezoneLocation, type SunLocation } from './themeClock'

/** setTimeout が扱える上限（約 24.8 日）。超えると即座に発火してしまう */
const MAX_TIMEOUT_MS = 2_147_483_647

/** 次の切り替わりが無いとき（白夜・極夜）に様子を見る間隔 */
const FALLBACK_RECHECK_MS = 6 * 3600_000

/**
 * いま基準にすべき座標。
 *
 * 読み込み済みのプリセット → 一覧の先頭 → タイムゾーンからの推定、の順に落ちる。
 */
function currentLocation(): SunLocation {
  const { presets, status, pendingPresetId } = useSimStore.getState()
  const id = status.presetId ?? pendingPresetId
  if (presets.length > 0) {
    const hit = (id ? presets.find((p) => p.id === id) : undefined) ?? presets[0]
    if (hit && Number.isFinite(hit.centerLat) && Number.isFinite(hit.centerLon)) {
      return { lat: hit.centerLat, lon: hit.centerLon, source: hit.name }
    }
  }
  return timezoneLocation()
}

/**
 * 日の出・日の入りに合わせて配色を切り替え続ける。
 *
 * ポーリングはしない。**次に切り替わる時刻を計算して、そこ 1 点にタイマーを張る。**
 * 一日に 2 回しか起きないので、タブを開きっぱなしにしても負荷にならない。
 */
export function useAutoTheme(): void {
  // プリセットが届く／マップが切り替わると座標が変わるので、そのときも張り直す
  const presetCount = useSimStore((s) => s.presets.length)
  const presetId = useSimStore((s) => s.status.presetId)

  useEffect(() => {
    let timer: number | undefined
    let cancelled = false

    const tick = (): void => {
      if (cancelled) return
      const where = currentLocation()
      const now = new Date()

      const theme = themeFor(now, where)
      // ★ applyTheme は差分の有無によらず必ず呼ぶ。
      //   ストアの初期値（bootstrapTheme）が既に正しいと差分が無いので、
      //   「違うときだけ」にすると **DOM への反映だけが漏れる**。
      //   3D はストアを見るので明るくなり、パネルだけ既定（暗い方）に
      //   取り残される、という片側だけの不整合になる（実機で再現した）。
      applyTheme(theme)
      if (useSimStore.getState().theme !== theme) {
        useSimStore.setState({ theme })
      }

      // 次の境目ちょうどに起こす。1 秒だけ余裕を足すのは、
      // 起きた瞬間がまだ切り替わり前（境界の丸め）になるのを防ぐため
      const next = nextThemeChange(now, where)
      const wait = next
        ? Math.min(MAX_TIMEOUT_MS, Math.max(1000, next.getTime() - now.getTime() + 1000))
        : FALLBACK_RECHECK_MS
      timer = window.setTimeout(tick, wait)
    }

    tick()

    // スリープ復帰やタブの再表示では、張ったタイマーが遅れて発火することがある。
    // 表に戻った時点で必ず見直す。
    const onVisible = (): void => {
      if (document.visibilityState !== 'visible') return
      if (timer !== undefined) window.clearTimeout(timer)
      tick()
    }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [presetCount, presetId])
}
