/**
 * 操作パネルのモーション共通処理。
 *
 * ここに置くのは「押した」「変わった」「移った」をユーザーに返すための仕掛けだけで、
 * 見せ場を作るためのものは置かない。装飾のためのアニメーションは、
 * 20Hz で届くフレームを読みながら操作する画面ではノイズにしかならない。
 *
 * ★ 動かしてよいのは transform と opacity だけ。操作パネルは 3D と同じ
 *   メインスレッドに乗っているので、幅・高さ・影を毎フレーム変える演出を足すと
 *   シミュレータ側のフレームが落ちる。
 *   例外はタブのインジケータ（絶対配置の 1 要素だけなのでレイアウトが波及しない）。
 */

/**
 * OS の「視差効果を減らす」設定。
 * 毎回 matchMedia を引くのは、設定変更を次の操作から拾えるようにするため
 * （起動時に 1 度だけ読むと、設定を変えてもリロードするまで効かない）。
 */
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

/**
 * 押した場所から広がる波紋（Material 3 の state layer）。pointerdown に挿す。
 *
 * ボタンが効いたかどうかを、色の変化ではなく「押した点」で返す。
 * 連打されても頭から再生されるよう、クラスを外してリフローを 1 度強制してから
 * 付け直している。
 */
export function startRipple(e: PointerLike): void {
  const el = e.currentTarget
  if (!el) return
  if ((el as HTMLButtonElement).disabled) return
  if (prefersReducedMotion()) return

  const rect = el.getBoundingClientRect()
  if (rect.width === 0 || rect.height === 0) return

  const x = e.clientX - rect.left
  const y = e.clientY - rect.top
  // 押した点から一番遠い角までを半径にすると、波紋が必ず要素を覆いきる。
  // 端を押したときだけ途中で止まる、という見え方を避けるため
  const diameter = 2 * Math.hypot(Math.max(x, rect.width - x), Math.max(y, rect.height - y))

  el.style.setProperty('--m3-ripple-x', `${x}px`)
  el.style.setProperty('--m3-ripple-y', `${y}px`)
  el.style.setProperty('--m3-ripple-d', `${diameter}px`)
  restartAnimation(el, 'is-rippling')
}

/**
 * CSS アニメーションを頭から再生し直す。
 *
 * ★ void el.offsetWidth を消さないこと。クラスを外して付け直すだけでは
 *   ブラウザが 2 つの変更をまとめてしまい、2 回目以降が一切再生されない。
 */
export function restartAnimation(el: HTMLElement, className: string): void {
  el.classList.remove(className)
  void el.offsetWidth
  el.classList.add(className)
}

/**
 * バネで目標値に追いつく 1 自由度のシミュレーション。
 *
 * タブのインジケータに使う。CSS の transition ではなくこれを使うのは、
 * 追う相手（選択中のタブ）自身が flex-grow のアニメーションで幅を変え続けており、
 * 目標が毎フレーム動くため。transition だと目標が変わるたびに再スタートがかかり、
 * 最後まで到達しないまま次に移ってずるずる遅れる。
 *
 * 係数は質量 1 とみなしたときの値。臨界減衰は c = 2√k（k=260 なら 32.2）で、
 * それより少し弱くして止まる直前にわずかに行き過ぎるようにしている
 * （M3 Expressive の「弾む」表現）。
 */
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
