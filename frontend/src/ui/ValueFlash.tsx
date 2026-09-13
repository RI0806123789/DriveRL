/**
 * 中身の文字が変わった瞬間だけ、数字を入れ替えるように見せる。
 *
 * 学習の指標は 1Hz で届くが、数字が黙って書き換わるだけだと
 * 「いま更新された」のか「止まっている」のかが画面から読めない。
 * 更新回数が伸びているのか、到達率が上がったのか下がったのかを
 * 数字の出てくる向き（増えたら下から、減ったら上から）で示す。
 *
 * ★ 値そのものではなく DOM の textContent で比べている。
 *   呼び出し側の JSX（数値と単位が別ノードでも、?? '—' でも）をそのまま
 *   包めるようにするため。比較のために値の作り方を揃えさせない。
 */

import { useEffect, useRef } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { restartAnimation } from './motion'

export interface ValueFlashProps {
  children: ReactNode
  className?: string
  title?: string
  style?: CSSProperties
}

export function ValueFlash({ children, className = '', title, style }: ValueFlashProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const prev = useRef<string | null>(null)

  // 依存配列を置かない。DOM に出たあとの文字で比べたいので、
  // children を deps にしても（要素が毎回作り直される以上）意味がない。
  // 走るのは指標カードの十数個 × 1Hz なので、コストは無視できる
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const now = el.textContent ?? ''
    const before = prev.current
    prev.current = now
    // 初回は光らせない（開いた瞬間に全部が光ると、何が変わったのか分からない）
    if (before === null || before === now) return
    el.dataset.dir = compareNumeric(before, now)
    restartAnimation(el, 'is-flash')
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
