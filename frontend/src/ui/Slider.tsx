/**
 * Material 3 Expressive のスライダー。
 * トラックは太く、つまみは細い縦棒（M3 Expressive のスタイル）。
 * ドラッグ中は onInput でローカル更新し、離した時（onChange 相当）にサーバーへ送る、
 * という使い方ができるよう onChange と onCommit を分けている。
 */

import { useId } from 'react'

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

  return (
    <div className="m3-slider">
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
        onChange={(e) => onChange(Number(e.target.value))}
        onPointerUp={(e) => onCommit?.(Number((e.target as HTMLInputElement).value))}
        onKeyUp={(e) => onCommit?.(Number((e.target as HTMLInputElement).value))}
        onBlur={(e) => onCommit?.(Number(e.target.value))}
      />
    </div>
  )
}
