/** 中身の文字が変わった瞬間だけ、数字を入れ替えるように見せる。 */

import { useEffect, useRef } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { restartValueFlash } from './motion'

export interface ValueFlashProps {
  children: ReactNode
  className?: string
  title?: string
  style?: CSSProperties
}

export function ValueFlash({ children, className = '', title, style }: ValueFlashProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const prev = useRef<string | null>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const now = el.textContent ?? ''
    const before = prev.current
    prev.current = now
    if (before === null || before === now) return
    el.dataset.dir = compareNumeric(before, now)
    restartValueFlash(el, 'is-flash')
  })

  return (
    <span ref={ref} className={`m3-valueflash ${className}`.trim()} title={title} style={style}>
      {children}
    </span>
  )
}

/** 表示文字列から増減を判定する。数値を拾えないときは向き無し（横から出す） */
function compareNumeric(before: string, after: string): 'up' | 'down' | 'flat' {
  const a = parseFirstNumber(before)
  const b = parseFirstNumber(after)
  if (a === null || b === null || a === b) return 'flat'
  return b > a ? 'up' : 'down'
}

/** 桁区切りのカンマを外してから最初の数値を拾う（"1,234 回" → 1234） */
function parseFirstNumber(s: string): number | null {
  const m = s.replace(/,/g, '').match(/-?\d+(?:\.\d+)?/)
  return m ? Number(m[0]) : null
}
