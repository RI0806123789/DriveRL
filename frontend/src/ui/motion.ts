/** 操作パネルのモーション共通処理。 */

/** OS の「視差効果を減らす」設定。 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return false
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/** startRipple が受け取れる最小限の形（React の PointerEvent もこれを満たす） */
interface PointerLike {
  currentTarget: HTMLElement
  clientX: number
  clientY: number
}

/** 押した場所から広がる波紋（Material 3 の state layer）。pointerdown に挿す。 */
export function startRipple(e: PointerLike): void {
  const el = e.currentTarget
  if (!el) return
  if ((el as HTMLButtonElement).disabled) return
  if (prefersReducedMotion()) return

  const rect = el.getBoundingClientRect()
  if (rect.width === 0 || rect.height === 0) return

  const x = e.clientX - rect.left
  const y = e.clientY - rect.top
  const diameter = 2 * Math.hypot(Math.max(x, rect.width - x), Math.max(y, rect.height - y))

  el.style.setProperty('--m3-ripple-x', `${x}px`)
  el.style.setProperty('--m3-ripple-y', `${y}px`)
  el.style.setProperty('--m3-ripple-d', `${diameter}px`)
  restartAnimation(el, 'is-rippling')
}

/** CSS アニメーションを頭から再生し直す。 */
export function restartAnimation(el: HTMLElement, className: string): void {
  el.classList.remove(className)
  void el.offsetWidth
  el.classList.add(className)
}

/** バネで目標値に追いつく 1 自由度のシミュレーション。 */
export const SPRING_K = 260
export const SPRING_C = 28

export interface Spring1D {
  /** 現在値 */
  x: number
  /** 速度 */
  v: number
}

/** 半陰的オイラー法で 1 ステップ進める。dt は秒 */
export function stepSpring(s: Spring1D, target: number, dt: number): void {
  s.v += (-SPRING_K * (s.x - target) - SPRING_C * s.v) * dt
  s.x += s.v * dt
}

/** 目標との差も速度も十分小さいか（ここまで来たら止めてよい） */
export function springSettled(s: Spring1D, target: number): boolean {
  return Math.abs(s.x - target) < 0.25 && Math.abs(s.v) < 2
}
