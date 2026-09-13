/**
 * Material 3 Expressive のボタン。
 * 押下すると角丸が縮んでわずかに潰れ（形が変わる表現）、押した点から波紋が広がる。
 * 見た目は global.css の .m3-btn 系クラスで実装している。
 */

import type { ButtonHTMLAttributes, PointerEvent, ReactNode } from 'react'
import { startRipple } from './motion'

export type ButtonVariant = 'filled' | 'tonal' | 'outlined' | 'text' | 'danger'
export type ButtonSize = 'sm' | 'md' | 'lg'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  /** 幅いっぱいに広げる */
  block?: boolean
  /** ラベルの前に置くアイコン */
  icon?: ReactNode
}

export function Button({
  variant = 'filled',
  size = 'md',
  block = false,
  icon,
  children,
  className = '',
  type = 'button',
  onPointerDown,
  ...rest
}: ButtonProps) {
  const classes = [
    'm3-btn',
    'm3-ripple',
    `m3-btn--${variant}`,
    size === 'sm' ? 'm3-btn--sm' : size === 'lg' ? 'm3-btn--lg' : '',
    block ? 'm3-btn--block' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button
      type={type}
      className={classes}
      onPointerDown={(e: PointerEvent<HTMLButtonElement>) => {
        startRipple(e)
        onPointerDown?.(e)
      }}
      {...rest}
    >
      {icon}
      {children != null && <span>{children}</span>}
    </button>
  )
}
