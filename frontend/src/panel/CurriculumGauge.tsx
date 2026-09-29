/** ヒヤリハットの難易度のゲージと、起こした件数・回避率。 */

import { curriculumView } from '../store/curriculum'

export interface CurriculumGaugeProps {
  level: number | undefined
  triggered: number | undefined
  avoidedRate: number | null | undefined
}

export function CurriculumGauge({ level, triggered, avoidedRate }: CurriculumGaugeProps) {
  const view = curriculumView(level, triggered, avoidedRate)
  return (
    <div className="m3-col">
      <div className="m3-row">
        <span className="m3-stat-label">{view.levelText}</span>
      </div>
      <span className="m3-bar" role="meter" aria-label="Curriculum Level" aria-valuemin={0} aria-valuemax={1} aria-valuenow={view.gauge}>
        <span
          className="m3-bar-fill"
          style={{ ['--m3-bar-value' as string]: String(view.gauge), background: 'var(--m3-tertiary)' }}
        />
      </span>
      <div className="m3-statgrid">
        <div className="m3-stat">
          <span className="m3-stat-label">発生確率（条件のそろった場面）</span>
          <span className="m3-stat-value">{view.probabilityText}</span>
        </div>
        <div className="m3-stat">
          <span className="m3-stat-label">ヒヤリハット</span>
          <span className="m3-stat-value">{view.triggeredText}</span>
        </div>
        <div className="m3-stat">
          <span className="m3-stat-label">自力で回避</span>
          <span className="m3-stat-value" data-testid="incidents-avoided">{view.avoidedText}</span>
        </div>
      </div>
    </div>
  )
}
