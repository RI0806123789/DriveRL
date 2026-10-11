/** 交差点判断の理由・待機時間・根拠と、操作介入の有無を表示する。 */

import type { IntersectionDecision } from '../types/protocol'
import { intersectionSummary } from '../scene/assistLabels'

export function IntersectionStatus({ decision }: { decision: IntersectionDecision | undefined }) {
  const badge = intersectionSummary(decision)
  if (!badge || !decision) return null
  return (
    <span
      className={`intersection-status intersection-status--${badge.level}`}
      data-state={decision.state}
      data-reason={decision.reason}
      data-source={decision.source}
      data-imposed={decision.imposed ? 'true' : 'false'}
    >
      <span>{badge.text}</span>
      <span className="intersection-status-detail">{badge.detail}</span>
    </span>
  )
}
