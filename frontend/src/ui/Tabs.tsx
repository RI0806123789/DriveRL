/**
 * タブ（memo 4章: 右側の設定セクションはタブ方式で切り替える）。
 *
 * M3 Expressive として、選択中のタブは container 色になり少し横に広がる。
 * その container 色の下敷き（インジケータ）は絶対配置の 1 要素で、
 * 押されたタブへバネで滑っていく。
 *
 * ★ インジケータを CSS transition で動かさないこと。
 *   追う相手である選択中タブ自身が flex-grow: 1.35 のアニメーションで
 *   幅を変え続けるため、目標が毎フレーム動く。transition は目標が変わるたびに
 *   再スタートするので最後まで到達せず、タブが広がりきった頃に下敷きだけ
 *   左にずれたまま残る。毎フレーム測り直してバネで追う（motion.ts）。
 *
 * ★ 追従の開始を「別の useEffect から ref 越しに関数を呼ぶ」形にしないこと。
 *   StrictMode の二重マウントで、生きているインスタンスが ref に入れた関数を
 *   捨てられる側のクリーンアップが上書きし、タブを押しても下敷きが動かなくなる
 *   （実際にそうなった）。effect は 1 つにまとめ、selected を依存に持たせる。
 */

import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import { prefersReducedMotion, springSettled, startRipple, stepSpring } from './motion'
import type { Spring1D } from './motion'

export interface TabItem<T extends string> {
  id: T
  label: string
  icon?: ReactNode
}

export interface TabsProps<T extends string> {
  items: TabItem<T>[]
  value: T
  disabled?: boolean
  onChange: (id: T) => void
  /** aria-label */
  label?: string
}

export function Tabs<T extends string>({
  items,
  value,
  disabled = false,
  onChange,
  label = 'セクション',
}: TabsProps<T>) {
  const listRef = useRef<HTMLDivElement>(null)
  const barRef = useRef<HTMLSpanElement>(null)
  const btnRefs = useRef<Array<HTMLButtonElement | null>>([])

  // バネの状態は effect をまたいで持ち越す。ここを effect の中に置くと、
  // タブを押すたびに現在位置が失われて瞬間移動になる
  const spring = useRef<{ x: Spring1D; w: Spring1D; placed: boolean }>({
    x: { x: 0, v: 0 },
    w: { x: 0, v: 0 },
    placed: false,
  })

  const selected = items.findIndex((item) => item.id === value)

  useEffect(() => {
    const bar = barRef.current
    const list = listRef.current
    if (!bar || !list) return

    const st = spring.current
    let raf = 0
    let last = 0

    const measure = () => {
      const el = btnRefs.current[selected]
      if (!el) return null
      // ★ offsetLeft / offsetWidth（整数に丸められる）を使わないこと。
      //   flex-grow のアニメーションの終わり際はタブの幅が小数で動くので、
      //   丸めた値で追うと目標が止まったと判定され、1px ずれた位置に居残る。
      const a = el.getBoundingClientRect()
      const b = list.getBoundingClientRect()
      // インジケータは list を基準に absolute 配置してある（list に border は無い）
      return { x: a.left - b.left, w: a.width }
    }

    const apply = () => {
      bar.style.transform = `translate3d(${st.x.x}px, 0, 0)`
      bar.style.width = `${Math.max(0, st.w.x)}px`
      // パネルを畳むと幅 0 になる。潰れた下敷きが線として見えないよう消す
      bar.style.opacity = st.w.x > 1 ? '1' : '0'
    }

    const snap = () => {
      const t = measure()
      if (!t) return
      st.x.x = t.x
      st.x.v = 0
      st.w.x = t.w
      st.w.v = 0
      st.placed = true
      apply()
    }

    const step = (now: number) => {
      raf = 0
      const t = measure()
      if (!t) return
      // タブを切り替えた直後は前フレームからの間隔が開いていることがある。
      // 1/30 秒で頭打ちにしてバネを暴れさせない
      const dt = Math.min((now - last) / 1000, 1 / 30)
      last = now
      stepSpring(st.x, t.x, dt)
      stepSpring(st.w, t.w, dt)
      apply()
      if (springSettled(st.x, t.x) && springSettled(st.w, t.w)) {
        st.x.x = t.x
        st.x.v = 0
        st.w.x = t.w
        st.w.v = 0
        apply()
        return
      }
      raf = requestAnimationFrame(step)
    }

    const start = () => {
      // 初回配置と、視差効果を減らす設定のときは滑らせずに合わせる
      if (!st.placed || prefersReducedMotion()) {
        snap()
        return
      }
      if (raf) return
      last = performance.now()
      raf = requestAnimationFrame(step)
    }

    start()

    // タブ 1 枚 1 枚も見る。選択でタブの幅が flex-grow のアニメーションで変わり、
    // その最後の数フレームは小数以下しか動かないため、
    // 親（list）の寸法変化だけを見ているとバネが先に止まって数 px ずれて残る
    const ro = new ResizeObserver(() => start())
    ro.observe(list)
    for (const el of btnRefs.current) {
      if (el) ro.observe(el)
    }

    return () => {
      ro.disconnect()
      if (raf) cancelAnimationFrame(raf)
    }
  }, [selected])

  return (
    <div className="m3-tabs" role="tablist" aria-label={label} ref={listRef}>
      <span className="m3-tab-indicator" ref={barRef} aria-hidden />
      {items.map((item, i) => (
        <button
          key={item.id}
          ref={(el) => {
            btnRefs.current[i] = el
          }}
          type="button"
          role="tab"
          className="m3-tab m3-ripple"
          aria-selected={value === item.id}
          disabled={disabled}
          onPointerDown={startRipple}
          onClick={() => onChange(item.id)}
          title={item.label}
        >
          {item.icon}
          <span className="m3-tab-label">{item.label}</span>
        </button>
      ))}
    </div>
  )
}
