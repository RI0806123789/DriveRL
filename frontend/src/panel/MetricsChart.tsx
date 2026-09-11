/**
 * 学習指標の折れ線グラフ。
 *
 * グラフ描画ライブラリを追加せず、SVG を自前で組み立てている
 * （依存を増やさない方針のため）。点数は最大 300（METRICS_CAPACITY）なので
 * パスを毎回作り直しても十分軽い。
 */

import { useId, useMemo } from 'react'

export interface MetricsChartProps {
  title: string
  values: number[]
  color?: string
  /** 値の表示形式 */
  format?: (v: number) => string
  height?: number
  /** 0 の基準線を引く（報酬など正負をまたぐ指標で有効） */
  showZero?: boolean
}

const VIEW_W = 300

export function MetricsChart({
  title,
  values,
  color = 'var(--m3-primary)',
  format = (v) => v.toFixed(3),
  height = 64,
  showZero = false,
}: MetricsChartProps) {
  const chart = useMemo(() => {
    const pts = values.filter((v) => Number.isFinite(v))
    if (pts.length < 2) return null

    let min = Math.min(...pts)
    let max = Math.max(...pts)
    if (showZero) {
      min = Math.min(min, 0)
      max = Math.max(max, 0)
    }
    // 平坦な系列でも線が真ん中に出るように、幅ゼロを避ける
    if (max - min < 1e-9) {
      const pad = Math.max(1e-6, Math.abs(max) * 0.1)
      min -= pad
      max += pad
    }

    const span = max - min
    const toY = (v: number) => height - ((v - min) / span) * height
    const stepX = VIEW_W / (pts.length - 1)

    let line = ''
    for (let i = 0; i < pts.length; i++) {
      line += `${i === 0 ? 'M' : 'L'}${(i * stepX).toFixed(2)},${toY(pts[i]).toFixed(2)}`
    }
    // 塗り用に下端まで閉じたパス
    const area = `${line}L${VIEW_W},${height}L0,${height}Z`

    return {
      line,
      area,
      min,
      max,
      latest: pts[pts.length - 1],
      zeroY: min <= 0 && max >= 0 ? toY(0) : null,
    }
  }, [values, height, showZero])

  // React 19 の useId は «r0» のように SVG の url(#...) では使えない文字を
  // 含むので、英数字とハイフンだけに正規化してから id にする
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
          aria-label={`${title} の推移`}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.28" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>

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
        <div className="m3-note m3-mono" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>最小 {format(chart.min)}</span>
          <span>最大 {format(chart.max)}</span>
        </div>
      )}
    </div>
  )
}
