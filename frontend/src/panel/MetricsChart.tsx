/** 学習指標の折れ線グラフ。 */

import { useId, useMemo } from 'react'
import { buildChart, VIEW_W, xForIndex } from './metricsChartMath'

export interface MetricsChartProps {
  title: string
  /**
   * 値の列。**その場で追記される配列を渡してよい**（code_review E-03）。
   * その場合は `revision` を一緒に渡して、再計算の契機にすること。
   */
  values: readonly number[]
  /** `values` を書き換えたときに増やす番号。省略時は配列の同一性で判断する */
  revision?: number
  color?: string
  /** 値の表示形式 */
  format?: (v: number) => string
  height?: number
  /** 0 の基準線を引く（報酬など正負をまたぐ指標で有効） */
  showZero?: boolean
  /** 縦線を引く添字（マップを切り替えた位置。code_review E-04） */
  marks?: readonly number[]
}

export function MetricsChart({
  title,
  values,
  revision,
  color = 'var(--m3-primary)',
  format = (v) => v.toFixed(3),
  height = 64,
  showZero = false,
  marks,
}: MetricsChartProps) {
  const chart = useMemo(
    () => buildChart(values, { height, showZero }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [values, revision, height, showZero],
  )

  const rawId = useId()
  const gradientId = useMemo(
    () => `m3-chart-grad-${rawId.replace(/[^a-zA-Z0-9-]/g, '')}`,
    [rawId],
  )

  return (
    <div className="m3-chart">
      <div className="m3-chart-head">
        <span className="m3-chart-title">{title}</span>
        {chart && (
          <span className="m3-chart-latest" style={{ color }}>
            {format(chart.latest)}
          </span>
        )}
      </div>

      {chart ? (
        <svg
          className="m3-chart-svg"
          viewBox={`0 0 ${VIEW_W} ${height}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`${title} の推移（${chart.count} 点、学習開始から）`}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.28" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>

          {marks?.map((i) =>
            i > 0 && i < values.length ? (
              <line
                key={i}
                x1={xForIndex(i, values.length)}
                x2={xForIndex(i, values.length)}
                y1="0"
                y2={height}
                stroke="var(--m3-outline)"
                strokeWidth="1"
                strokeDasharray="2 3"
                vectorEffect="non-scaling-stroke"
              />
            ) : null,
          )}

          {chart.zeroY !== null && (
            <line
              x1="0"
              x2={VIEW_W}
              y1={chart.zeroY}
              y2={chart.zeroY}
              stroke="var(--m3-outline-variant)"
              strokeWidth="1"
              strokeDasharray="4 4"
              vectorEffect="non-scaling-stroke"
            />
          )}

          <path d={chart.area} fill={`url(#${gradientId})`} />
          <path
            d={chart.line}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      ) : (
        <div className="m3-chart-empty">データを収集しています…</div>
      )}

      {chart && (
        <div className="m3-chart-scale m3-note m3-mono">
          <span>最小 {format(chart.min)}</span>
          <span>平均 {format(chart.mean)}</span>
          <span>最大 {format(chart.max)}</span>
        </div>
      )}
    </div>
  )
}
