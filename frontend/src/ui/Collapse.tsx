/** 開閉に合わせて高さが伸び縮みするラッパー。 */

import type { ReactNode } from 'react'

export interface CollapseProps {
  open: boolean
  children: ReactNode
  /** 親の gap（px）。閉じたときにこの分を詰める。Card 内なら 10、panel-body 直下なら 12 */
  gap?: number
  className?: string
}

export function Collapse({ open, children, gap = 10, className = '' }: CollapseProps) {
  return (
    <div
      className={`m3-collapse ${className}`.trim()}
      data-open={open ? 'true' : 'false'}
      style={{ ['--m3-collapse-gap' as string]: `${gap}px` }}
      inert={!open}
    >
      <div className="m3-collapse-inner">{children}</div>
    </div>
  )
}
