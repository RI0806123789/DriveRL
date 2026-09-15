/** ドロップダウン。 */

import { useId } from 'react'
import { ChevronDownIcon } from './Icons'

export interface SelectOption {
  value: string
  label: string
  disabled?: boolean
}

export interface SelectProps {
  label?: string
  value: string
  options: SelectOption[]
  disabled?: boolean
  onChange: (value: string) => void
}

export function Select({ label, value, options, disabled = false, onChange }: SelectProps) {
  const id = useId()
  return (
    <div className="m3-select">
      {label && (
        <label className="m3-select-label" htmlFor={id}>
          {label}
        </label>
      )}
      <div className="m3-select-box">
        <select
          id={id}
          className="m3-select-native"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value} disabled={o.disabled}>
              {o.label}
            </option>
          ))}
        </select>
        <span className="m3-select-chevron">
          <ChevronDownIcon size={18} />
        </span>
      </div>
    </div>
  )
}
