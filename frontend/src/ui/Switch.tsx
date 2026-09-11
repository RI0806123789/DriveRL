/** Material 3 のスイッチ。ON でつまみが大きくなる（Expressive の形状変化） */

export interface SwitchProps {
  label: string
  /** ラベル下の補足説明 */
  description?: string
  checked: boolean
  disabled?: boolean
  onChange: (checked: boolean) => void
}

export function Switch({ label, description, checked, disabled = false, onChange }: SwitchProps) {
  return (
    <label className={`m3-switch${disabled ? ' m3-switch--disabled' : ''}`}>
      <span className="m3-switch-label">
        {label}
        {description && <span className="m3-switch-desc">{description}</span>}
      </span>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="m3-switch-track">
        <span className="m3-switch-thumb" />
      </span>
    </label>
  )
}
