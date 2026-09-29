/** 直近 10 秒に方策が選んでいた意図の割合（`metrics.optionShares`）。届いていなければ案内だけ出す。 */

import { optionShareRows } from '../store/driveOption'
import { OptionBadge } from './OptionBadge'

export function OptionShares({ shares }: { shares: unknown }) {
  const rows = optionShareRows(shares)
  if (rows === null) {
    return <div className="m3-note">方策が運転したステップがまだありません。</div>
  }
  return (
    <div className="option-shares">
      {rows.map((row) => (
        <div key={row.style.option} className="option-shares-row" data-option={row.style.option}>
          <OptionBadge option={row.style.option} />
          <div className="m3-bar">
            <div
              className="m3-bar-fill"
              style={{
                ['--m3-bar-value' as string]: String(row.share),
                background: `var(${row.style.token})`,
              }}
            />
          </div>
          <span className="m3-note">{row.percent}%</span>
        </div>
      ))}
    </div>
  )
}
