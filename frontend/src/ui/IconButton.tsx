/** アイコンだけのボタン。ハンバーガーやカメラ切替に使う */

import type { ButtonHTMLAttributes, ReactNode } from 'react'

export type IconButtonVariant = 'standard' | 'filled' | 'surface'

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: IconButtonVariant
  large?: boolean
  /** アクセシビリティ用のラベル（必須） */
  label: string
  children: ReactNode
}

export function IconButton({
  variant = 'standard',
  large = false,
  label,
  children,
  className = '',
  type = 'button',
  ...rest
}: IconButtonProps) {
  const classes = [
    'm3-iconbtn',
    variant !== 'standard' ? `m3-iconbtn--${variant}` : '',
    large ? 'm3-iconbtn--lg' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button type={type} className={classes} aria-label={label} title={label} {...rest}>
      {children}
    </button>
  )
}
