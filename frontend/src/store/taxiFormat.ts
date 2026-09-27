/** 配車の表示に使う文字列の整形。**React から切り離した純粋モジュール。** */

/** 残り時間を「1 分 20 秒」の形にする */
export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'まもなく'
  const total = Math.round(seconds)
  if (total < 60) return `${total} 秒`
  return `${Math.floor(total / 60)} 分 ${total % 60} 秒`
}
