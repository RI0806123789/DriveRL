/** タブ（memo 4章: 右側の設定セクションはタブ方式で切り替える）。 */

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
      const a = el.getBoundingClientRect()
      const b = list.getBoundingClientRect()
      return { x: a.left - b.left, w: a.width }
    }

    const apply = () => {
      bar.style.transform = `translate3d(${st.x.x}px, 0, 0)`
      bar.style.width = `${Math.max(0, st.w.x)}px`
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
      if (!st.placed || prefersReducedMotion()) {
        snap()
        return
      }
      if (raf) return
      last = performance.now()
      raf = requestAnimationFrame(step)
    }

    start()

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
