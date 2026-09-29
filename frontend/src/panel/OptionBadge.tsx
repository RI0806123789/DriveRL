/** 階層型の方策がいま選んでいる意図の色付きバッジ（CRUISE 青 / FOLLOW 緑 / YIELD 黄 / STOP 赤）。意図が無ければ何も出さない。 */

import { DRIVE_OPTION_STYLES, normalizeDriveOption } from '../store/driveOption'

export function OptionBadge({ option }: { option: unknown }) {
  const value = normalizeDriveOption(option)
  if (value === null) return null
  const style = DRIVE_OPTION_STYLES[value]
  return (
    <span
      className={`option-badge ${style.className}`}
      data-option={value}
      title={`上位方策の意図: ${style.label}（${style.description}）`}
    >
      <span aria-hidden="true">{style.icon}</span>
      {value}
    </span>
  )
}
