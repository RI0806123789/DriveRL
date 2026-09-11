/**
 * タブ（memo 4章: 右側の設定セクションはタブ方式で切り替える）。
 * M3 Expressive として、選択中のタブは container 色になり少し横に広がる。
 */

import type { ReactNode } from 'react'

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
  return (
    <div className="m3-tabs" role="tablist" aria-label={label}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          className="m3-tab"
          aria-selected={value === item.id}
          disabled={disabled}
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
