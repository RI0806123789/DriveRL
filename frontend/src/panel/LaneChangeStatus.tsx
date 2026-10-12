/** 車線変更の判断と、操作介入の有無を表示する。 */

import type { LaneChangeDecision } from '../types/protocol'
import { laneChangeSummary } from '../scene/assistLabels'

export function LaneChangeStatus({ decision }: { decision: LaneChangeDecision | undefined }) {
  const badge = laneChangeSummary(decision)
  if (!badge) return null
  return (
    <span
      className={`lane-change-status lane-change-status--${badge.level}`}
      title={badge.detail}
      data-state={decision?.state}
      data-reason={decision?.reason}
      data-imposed={decision?.imposed ? 'true' : 'false'}
    >
      <span>{badge.text}</span>
      <span className="lane-change-status-detail">{badge.detail}</span>
    </span>
  )
}
