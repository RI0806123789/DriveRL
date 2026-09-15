/** セクションのまとまり。 */

import type { ReactNode } from 'react'

export interface CardProps {
  title?: string
  icon?: ReactNode
  /** タイトル行の右端に置く要素 */
  action?: ReactNode
  variant?: 'filled' | 'outlined' | 'high'
  children: ReactNode
  className?: string
}

export function Card({
  title,
  icon,
  action,
  variant = 'filled',
  children,
  className = '',
}: CardProps) {
  const classes = [
    'm3-card',
    variant === 'outlined' ? 'm3-card--outlined' : variant === 'high' ? 'm3-card--high' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <section className={classes}>
      {(title || action) && (
        <header className="m3-card-header">
          {icon && <span className="m3-card-icon">{icon}</span>}
          {title && <h3 className="m3-card-title">{title}</h3>}
          {action}
        </header>
      )}
      {children}
    </section>
  )
}
