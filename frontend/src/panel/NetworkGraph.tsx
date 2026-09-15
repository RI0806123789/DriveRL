/** 方策・価値ネットワークのノード図。 */

import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { send } from '../store/connection'
import { useSimStore } from '../store/simStore'
import { Button } from '../ui/Button'
import { Chip } from '../ui/Chip'
import type { NetworkLayer, NetworkMessage } from '../types/protocol'

const VIEW_W = 420
const VIEW_H = 250
/** 図の左右の端（観測の列と出力の列の中心） */
const X_FIRST = 42
const X_LAST = 392

/** 列の x 座標。**隠れ層の数に応じて本数が変わる。** */
function columnXs(columns: number): number[] {
  const n = Math.max(2, columns)
  const step = (X_LAST - X_FIRST) / (n - 1)
  return Array.from({ length: n }, (_, i) => X_FIRST + step * i)
}
/** 方策の枝と価値の枝の中心 y */
const POLICY_Y = 74
const VALUE_Y = 182
/** 1 列に描く円の数（実際の次元数はラベルで出す） */
const DOTS = 5
const DOT_GAP = 17
const R = 5.2

interface Column {
  x: number
  cy: number
  count: number
  label: string
}

/** 列の円の y 座標 */
function dotYs(cy: number, n: number): number[] {
  const span = (n - 1) * DOT_GAP
  return Array.from({ length: n }, (_, i) => cy - span / 2 + i * DOT_GAP)
}

/** 実際の次元数に応じて描く円の数（少ない層はそのまま描く） */
function dotCount(dim: number): number {
  return Math.min(DOTS, Math.max(1, dim))
}

/** 層を名前で引く */
function findLayer(layers: NetworkLayer[], name: string): NetworkLayer | undefined {
  return layers.find((l) => l.name === name)
}

/** 0..1 に潰す。対数にするのは、層ごとに桁が違うため */
function normalize(value: number, max: number): number {
  if (!(max > 0) || !Number.isFinite(value) || value <= 0) return 0
  return Math.min(1, Math.log10(1 + value * 9) / Math.log10(1 + max * 9))
}

function fmt(value: number, digits = 4): string {
  if (!Number.isFinite(value)) return '—'
  if (value === 0) return '0'
  if (Math.abs(value) < 10 ** -digits) return value.toExponential(1)
  return value.toFixed(digits)
}

interface BundleProps {
  from: Column
  to: Column
  layer: NetworkLayer | undefined
  /** 同じ図の中での相対的な強さ（線の太さに使う） */
  weightScale: number
  gradScale: number
  accent: string
}

const Bundle = memo(function Bundle({
  from,
  to,
  layer,
  weightScale,
  gradScale,
  accent,
}: BundleProps) {
  const fromYs = dotYs(from.cy, dotCount(from.count))
  const toYs = dotYs(to.cy, dotCount(to.count))

  const weight = layer ? normalize(layer.weightAbsMean, weightScale) : 0
  const grad = layer ? normalize(layer.gradNorm, gradScale) : 0

  const dur = grad > 0 ? Math.max(0.6, 2.6 - grad * 2.0) : 0

  return (
    <g>
      {fromYs.map((y1, i) =>
        toYs.map((y2, j) => {
          const mx = (from.x + to.x) / 2
          const d = `M ${from.x + R} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${to.x - R} ${y2}`
          return (
            <path
              key={`${i}-${j}`}
              d={d}
              fill="none"
              stroke={accent}
              strokeWidth={0.4 + weight * 1.1}
              strokeOpacity={0.10 + weight * 0.22}
            />
          )
        }),
      )}
      {dur > 0 &&
        fromYs.map((y1, i) => {
          const y2 = toYs[i % toYs.length]
          const mx = (from.x + to.x) / 2
          const d = `M ${from.x + R} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${to.x - R} ${y2}`
          return (
            <path
              key={`flow-${i}`}
              d={d}
              fill="none"
              stroke={accent}
              strokeWidth={1.4}
              strokeOpacity={0.55}
              strokeDasharray="5 22"
              strokeLinecap="round"
            >
              <animate
                attributeName="stroke-dashoffset"
                from="27"
                to="0"
                dur={`${dur}s`}
                repeatCount="indefinite"
              />
            </path>
          )
        })}
    </g>
  )
})

const NodeColumn = memo(function NodeColumn({
  col,
  accent,
  title,
}: {
  col: Column
  accent: string
  title: string
}) {
  const n = dotCount(col.count)
  const ys = dotYs(col.cy, n)
  const truncated = col.count > n
  return (
    <g>
      {ys.map((y, i) => (
        <circle
          key={i}
          cx={col.x}
          cy={y}
          r={R}
          fill="var(--m3-surface-container-highest)"
          stroke={accent}
          strokeWidth={1.3}
        />
      ))}
      {truncated && (
        <text
          x={col.x}
          y={ys[ys.length - 1] + 15}
          textAnchor="middle"
          fontSize={11}
          fill="var(--m3-on-surface-variant)"
        >
          ⋮
        </text>
      )}
      <text
        x={col.x}
        y={ys[0] - 12}
        textAnchor="middle"
        fontSize={9.5}
        fill="var(--m3-on-surface-variant)"
      >
        {title}
      </text>
      <text
        x={col.x}
        y={ys[ys.length - 1] + (truncated ? 28 : 16)}
        textAnchor="middle"
        fontSize={10}
        fontWeight={600}
        fill="var(--m3-on-surface)"
      >
        {col.count}
      </text>
    </g>
  )
})

/** 探索ノイズのゲージ。上限に張り付いていたら方策が潰れているサイン */
function NoiseGauge({ net }: { net: NetworkMessage }) {
  const lo = net.logStdMin
  const hi = net.logStdMax
  const span = hi - lo || 1
  return (
    <div className="netgraph-noise">
      <div className="netgraph-noise-head">
        <span>探索ノイズ（log_std）</span>
        <span className="netgraph-noise-value">
          σ = {net.actionStd.map((s) => s.toFixed(3)).join(' / ')}
        </span>
      </div>
      <div className="netgraph-gauge">
        <div className="netgraph-gauge-track" />
        {net.logStd.map((v, i) => {
          const t = Math.min(1, Math.max(0, (v - lo) / span))
          return (
            <div
              key={i}
              className="netgraph-gauge-pin"
              style={{ left: `${t * 100}%` }}
              title={`log_std[${i}] = ${v.toFixed(4)}`}
            />
          )
        })}
      </div>
      <div className="netgraph-gauge-scale">
        <span>{lo}</span>
        <span>{hi}（上限）</span>
      </div>
      <div className="m3-note">
        上限に張り付くと、行動が範囲いっぱいのノイズに埋もれて
        <strong>方策が実質ランダム</strong>になります。
      </div>
    </div>
  )
}

/** 選べる層の数と幅。サーバー側の許容範囲（1〜4 層 / 16〜512）に収めてある */
const LAYER_CHOICES = [1, 2, 3, 4] as const
const WIDTH_CHOICES = [32, 64, 128, 256] as const

/** 隠れ層の構成を変える。 */
function Architecture({ net }: { net: NetworkMessage }) {
  const current = net.hiddenSizes
  const currentKey = current.join(',')

  const [layers, setLayers] = useState(current.length)
  const [width, setWidth] = useState(current[0] ?? 128)
  const [confirming, setConfirming] = useState(false)

  useEffect(() => {
    setLayers(current.length || 1)
    setWidth(current[0] ?? 128)
    setConfirming(false)
  }, [currentKey])

  const wanted = useMemo(
    () => Array.from({ length: layers }, () => width),
    [layers, width],
  )
  const unchanged = wanted.length === current.length && wanted.every((w, i) => w === current[i])

  const shape = (sizes: readonly number[]) =>
    [net.obsDim, ...sizes, net.actionDim].join(' → ')

  return (
    <div className="netgraph-arch">
      <div className="netgraph-noise-head">
        <span>隠れ層の構成</span>
        <span className="netgraph-noise-value">{current.join(' × ') || '—'}</span>
      </div>

      <div className="netgraph-arch-shape">現在 {shape(current)}</div>

      <div className="netgraph-arch-row">
        <span className="netgraph-arch-label">層の数</span>
        {LAYER_CHOICES.map((n) => (
          <Chip
            key={n}
            small
            selected={layers === n}
            onClick={() => {
              setLayers(n)
              setConfirming(false)
            }}
          >
            {n}
          </Chip>
        ))}
      </div>

      <div className="netgraph-arch-row">
        <span className="netgraph-arch-label">各層の幅</span>
        {WIDTH_CHOICES.map((w) => (
          <Chip
            key={w}
            small
            selected={width === w}
            onClick={() => {
              setWidth(w)
              setConfirming(false)
            }}
          >
            {w}
          </Chip>
        ))}
      </div>

      {!unchanged && (
        <div className="netgraph-arch-shape is-next">変更後 {shape(wanted)}</div>
      )}

      <div className="m3-note">
        深くするほど表現力は上がりますが、<strong>1 ステップの計算時間が伸びて
        実効倍速が落ちます</strong>。まず今の構成で学習が進むか見てから変えるのが確実です。
      </div>

      {confirming ? (
        <div className="netgraph-arch-confirm">
          <div className="m3-note">
            <strong>重みは引き継げません。</strong>層の形が変わるため、
            学習は 0 からやり直しになります（{net.updates.toLocaleString()} 更新ぶんを捨てます）。
            残したい場合は先に「モデルの書き出し」で保存してください。
          </div>
          <div className="m3-row">
            <Button
              variant="filled"
              size="sm"
              onClick={() => {
                send({ type: 'set_network', hiddenSizes: wanted })
                setConfirming(false)
              }}
            >
              作り直す
            </Button>
            <Button variant="text" size="sm" onClick={() => setConfirming(false)}>
              やめる
            </Button>
          </div>
        </div>
      ) : (
        <Button
          variant="outlined"
          size="sm"
          disabled={unchanged}
          onClick={() => setConfirming(true)}
        >
          {unchanged ? 'この構成で動いています' : 'この構成に変える'}
        </Button>
      )}
    </div>
  )
}

export function NetworkGraph() {
  const net = useSimStore((s) => s.network)

  const historyRef = useRef<Array<{ updates: number; at: number }>>([])
  const [rate, setRate] = useState(0)
  const [recentlyUpdated, setRecentlyUpdated] = useState(false)
  useEffect(() => {
    if (!net) return
    const now = performance.now()
    const hist = historyRef.current
    if (hist.length > 0 && net.updates < hist[hist.length - 1].updates) hist.length = 0
    hist.push({ updates: net.updates, at: now })
    while (hist.length > 2 && now - hist[0].at > 20_000) hist.shift()
    if (hist.length < 2) return
    const first = hist[0]
    const last = hist[hist.length - 1]
    const dt = (last.at - first.at) / 1000
    if (dt > 0) setRate((last.updates - first.updates) / dt)
    setRecentlyUpdated(last.updates > first.updates)
  }, [net])

  const view = useMemo(() => {
    if (!net) return null
    const layers = net.layers
    const hidden = net.hiddenSizes.length > 0 ? net.hiddenSizes : [128]

    const weightScale = Math.max(...layers.map((l) => l.weightAbsMean), 1e-6)
    const gradScale = Math.max(...layers.map((l) => l.gradNorm), 1e-9)

    const xs = columnXs(hidden.length + 2)

    const obs: Column = {
      x: xs[0],
      cy: (POLICY_Y + VALUE_Y) / 2,
      count: net.obsDim,
      label: '観測',
    }
    const policyHidden: Column[] = hidden.map((count, i) => ({
      x: xs[i + 1],
      cy: POLICY_Y,
      count,
      label: '隠れ' + (i + 1),
    }))
    const valueHidden: Column[] = hidden.map((count, i) => ({
      x: xs[i + 1],
      cy: VALUE_Y,
      count,
      label: '隠れ' + (i + 1),
    }))
    const act: Column = {
      x: xs[xs.length - 1],
      cy: POLICY_Y,
      count: net.actionDim,
      label: '行動',
    }
    const val: Column = { x: xs[xs.length - 1], cy: VALUE_Y, count: 1, label: '価値' }

    /** 列と列の「つなぎ」を作る。 */
    const linksFor = (role: 'policy' | 'value', hiddenCols: Column[], out: Column) => {
      const trunk = role === 'policy' ? 'policy_trunk' : 'value_trunk'
      const head = role === 'policy' ? 'mu_head' : 'value_head'
      const cols = [obs, ...hiddenCols, out]
      return cols.slice(0, -1).map((from, i) => ({
        key: role + '-' + i,
        from,
        to: cols[i + 1],
        layer: findLayer(layers, i < hiddenCols.length ? trunk + '.' + i * 2 : head),
      }))
    }

    const totalGrad = layers.reduce((a, l) => a + l.gradNorm, 0)

    return {
      layers,
      weightScale,
      gradScale,
      obs,
      policyHidden,
      valueHidden,
      act,
      val,
      policyLinks: linksFor('policy', policyHidden, act),
      valueLinks: linksFor('value', valueHidden, val),
      totalGrad,
    }
  }, [net])

  if (!net || !view) {
    return (
      <div className="m3-note">
        マップを読み込むと、ここに学習中のネットワークが表示されます。
      </div>
    )
  }

  const POLICY = 'var(--m3-primary)'
  const VALUE = 'var(--m3-tertiary)'

  const alive = recentlyUpdated && view.totalGrad > 0

  return (
    <div className="netgraph">
      <div className="netgraph-status">
        <span className={`netgraph-pulse${alive ? ' is-alive' : ''}`} aria-hidden />
        <span className="netgraph-status-text">
          {alive ? '重みが更新されています' : '更新が止まっています'}
        </span>
        <span className="netgraph-status-num">
          {net.updates.toLocaleString()} 更新
        </span>
        <span className="netgraph-status-num">
          {rate > 0 ? `${(1 / rate).toFixed(1)} 秒に 1 回` : '—'}
        </span>
      </div>

      <svg
        className="netgraph-svg"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        role="img"
        aria-label="方策ネットワークと価値ネットワークの構成図"
      >
        {view.policyLinks.map((link) => (
          <Bundle key={link.key} from={link.from} to={link.to} layer={link.layer}
                  weightScale={view.weightScale} gradScale={view.gradScale} accent={POLICY} />
        ))}
        {view.valueLinks.map((link) => (
          <Bundle key={link.key} from={link.from} to={link.to} layer={link.layer}
                  weightScale={view.weightScale} gradScale={view.gradScale} accent={VALUE} />
        ))}

        <NodeColumn col={view.obs} accent="var(--m3-outline)" title="観測" />
        {view.policyHidden.map((col, i) => (
          <NodeColumn key={'ph-' + i} col={col} accent={POLICY} title={col.label} />
        ))}
        <NodeColumn col={view.act} accent={POLICY} title="行動" />
        {view.valueHidden.map((col, i) => (
          <NodeColumn key={'vh-' + i} col={col} accent={VALUE} title={col.label} />
        ))}
        <NodeColumn col={view.val} accent={VALUE} title="価値" />

        <text x={X_LAST} y={POLICY_Y - 42} textAnchor="end" fontSize={10}
              fontWeight={700} fill={POLICY}>
          方策（アクセル・操舵）
        </text>
        <text x={X_LAST} y={VALUE_Y + 52} textAnchor="end" fontSize={10}
              fontWeight={700} fill={VALUE}>
          価値（この状況の見込み）
        </text>
      </svg>

      <div className="netgraph-legend">
        <span><i className="netgraph-key netgraph-key--w" />線の太さ = 重みの大きさ</span>
        <span><i className="netgraph-key netgraph-key--g" />流れ = 勾配（止まったら学習も止まっている）</span>
      </div>

      <table className="netgraph-table">
        <thead>
          <tr>
            <th>層</th>
            <th>形</th>
            <th>|w| 平均</th>
            <th>勾配</th>
            <th>Δ（直前の更新）</th>
          </tr>
        </thead>
        <tbody>
          {view.layers.map((l) => (
            <tr key={l.name} className={l.role === 'value' ? 'is-value' : 'is-policy'}>
              <td>{l.name}</td>
              <td className="netgraph-dim">{l.inDim}→{l.outDim}</td>
              <td>{fmt(l.weightAbsMean, 5)}</td>
              <td>{fmt(l.gradNorm, 5)}</td>
              <td className={l.deltaNorm > 0 ? 'is-moving' : undefined}>{fmt(l.deltaNorm, 6)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <NoiseGauge net={net} />
      <Architecture net={net} />
    </div>
  )
}
