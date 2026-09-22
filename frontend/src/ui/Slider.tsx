/** Material 3 Expressive のスライダー。 */

import { useCallback, useEffect, useId, useRef, useState } from 'react'

export interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  disabled?: boolean
  /** 値の表示形式。省略時はそのまま */
  format?: (v: number) => string
  /** つまみを動かしている間、毎回呼ばれる */
  onChange: (v: number) => void
  /** 操作を終えたとき（pointerup / keyup）に呼ばれる。サーバー送信はここで */
  onCommit?: (v: number) => void
  /** ラベル右の補足（単位など） */
  hint?: string
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  disabled = false,
  format,
  onChange,
  onCommit,
  hint,
}: SliderProps) {
  const id = useId()
  const span = max - min
  const pct = span > 0 ? ((value - min) / span) * 100 : 0
  const shown = format ? format(value) : String(value)

  const [dragging, setDragging] = useState(false)
  useEffect(() => {
    if (!dragging) return
    const end = () => setDragging(false)
    window.addEventListener('pointerup', end)
    window.addEventListener('pointercancel', end)
    return () => {
      window.removeEventListener('pointerup', end)
      window.removeEventListener('pointercancel', end)
    }
  }, [dragging])

  // 同じ値を二度送らない。ただし**外から値が変わったら忘れる**
  // （サーバーが丸めた後に元の値へ戻す操作を落とさないため。code_review U-07 / E-09）。
  // ドラッグ中の onChange による value の変化で忘れてはいけないので、
  //   「自分が最後に送った値」と違う値が外から来たときだけ無効化する
  const committed = useRef<number | null>(value)
  const lastSeen = useRef(value)
  if (lastSeen.current !== value) {
    lastSeen.current = value
    if (!dragging && committed.current !== value) committed.current = null
  }
  const commit = useCallback(
    (v: number) => {
      if (committed.current === v) return
      committed.current = v
      onCommit?.(v)
    },
    [onCommit],
  )

  return (
    <div className="m3-slider" data-dragging={dragging ? 'true' : 'false'}>
      <div className="m3-slider-head">
        <label className="m3-slider-label" htmlFor={id}>
          {label}
          {hint && <span style={{ opacity: 0.7 }}>　{hint}</span>}
        </label>
        <span className="m3-slider-value">{shown}</span>
      </div>
      <input
        id={id}
        className="m3-slider-input"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        style={{ ['--m3-slider-pct' as string]: `${pct}%` }}
        onPointerDown={() => setDragging(true)}
        onChange={(e) => onChange(Number(e.target.value))}
        onPointerUp={(e) => commit(Number((e.target as HTMLInputElement).value))}
        onKeyUp={(e) => commit(Number((e.target as HTMLInputElement).value))}
        onBlur={(e) => commit(Number(e.target.value))}
      />
    </div>
  )
}
