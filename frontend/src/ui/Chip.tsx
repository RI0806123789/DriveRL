/**
 * チップ。状態表示（表示専用）と、選択トグル（ボタン）の両方に使う。
 * M3 Expressive では選択時に角丸が変化するので、押下で shape を変える。
 */

import type { ReactNode } from 'react'

export type ChipTone = 'neutral' | 'ok' | 'warning' | 'error'

export interface ChipProps {
  children: ReactNode
  icon?: ReactNode
  /** 指定するとボタンになる */
  onClick?: () => void
  selected?: boolean
  disabled?: boolean
  tone?: ChipTone
  small?: boolean
  title?: string
}

export function Chip({
  children,
  icon,
  onClick,
  selected,
  disabled = false,
  tone = 'neutral',
  small = false,
  title,
}: ChipProps) {
  const classes = [
    'm3-chip',
    tone !== 'neutral' ? `m3-chip--tone-${tone}` : '',
    small ? 'm3-chip--sm' : '',
  ]
    .filter(Boolean)
    .join(' ')

  if (onClick) {
    return (
      <button
        type="button"
        className={classes}
        data-selected={selected ? 'true' : 'false'}
        aria-pressed={selected}
        disabled={disabled}
        onClick={onClick}
        title={title}
      >
        {icon}
        {children}
      </button>
    )
  }

  return (
    <span className={classes} data-selected={selected ? 'true' : 'false'} title={title}>
      {icon}
      {children}
    </span>
  )
}
