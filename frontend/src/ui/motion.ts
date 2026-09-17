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

/** CSS アニメーションを頭から再生し直す（押した瞬間だけの演出用）。
 *
 * `void el.offsetWidth` は同期レイアウトを強制するが、これを呼ぶのは
 * `startRipple`（pointerdown のときだけ）なので 1 回ぶんは無視できる。
 * **毎フレーム・毎秒のように繰り返し呼ぶ経路からは使わないこと**
 * （`ValueFlash` 用には `restartValueFlash()` がある）。
 */
export function restartAnimation(el: HTMLElement, className: string): void {
  el.classList.remove(className)
  void el.offsetWidth
  el.classList.add(className)
}

/** `.m3-valueflash.is-flash` が流すアニメーション名。**出典は global.css。**
 * ここを増減させたら CSS 側も直すこと（名前だけの重複なので、ずれても
 * 「光らなくなる」という見える形で出る）。 */
const VALUE_FLASH_ANIMATIONS = new Set(['m3-value-up', 'm3-value-down', 'm3-value-flat'])

/** 数字の入れ替え演出を頭から流し直す。**毎秒のメトリクス更新ごとに呼ばれる。**
 *
 * ★ `restartAnimation()` を使わないこと。`ValueFlash` はパネル 1 枚に 9 個あり、
 * メトリクスは毎秒届くので、`void el.offsetWidth` だと毎秒 10〜15 回の強制リフローを
 * 3D と同じメインスレッドで踏む。
 * ★ クラスを外して `requestAnimationFrame` で付け直す形にもしないこと。
 * **タブが前面でないと rAF は回らないので、クラスが外れたまま戻らない。**
 * ★ 走っているアニメーションを名前で選ぶこと。`el.getAnimations()` はその要素の
 * アニメーションを全部返すので、名前を見ないと無関係なものまで巻き戻す。
 */
export function restartValueFlash(el: HTMLElement, className: string): void {
  el.classList.add(className)
  for (const animation of el.getAnimations()) {
    const name = (animation as CSSAnimation).animationName
    if (name !== undefined && VALUE_FLASH_ANIMATIONS.has(name)) {
      animation.currentTime = 0
      animation.play()
    }
  }
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
