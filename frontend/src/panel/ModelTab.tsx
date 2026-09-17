/** 「モデル作成」タブ。**画像認識の CNN（認識器）をここから学習する。** */

import { useEffect, useMemo, useState } from 'react'
import { send } from '../store/connection'
import { formatBytes } from '../store/exportModel'
import { useSimStore } from '../store/simStore'
import type { DetectorHistoryPoint, DetectorMessage, DetectorMode } from '../types/protocol'
import { MetricsChart } from './MetricsChart'
import { Button } from '../ui/Button'
import { Card } from '../ui/Card'
import { Chip } from '../ui/Chip'
import { Collapse } from '../ui/Collapse'
import { Select } from '../ui/Select'
import { Slider } from '../ui/Slider'
import { Switch } from '../ui/Switch'
import { ValueFlash } from '../ui/ValueFlash'
import {
  BrainIcon,
  CameraIcon,
  CheckIcon,
  GaugeIcon,
  InfoIcon,
  PauseIcon,
  PlayIcon,
  TuneIcon,
  WarningIcon,
} from '../ui/Icons'

/** 何をするか。バックエンドの `detector_job.MODES` と同じ 3 通り */
const MODES: Array<{ id: DetectorMode; label: string; hint: string }> = [
  { id: 'full', label: '収集して学習', hint: '教師データを集めてから学習します（既定）' },
  { id: 'collect', label: '収集だけ', hint: '教師データを集めて保存し、学習はしません' },
  { id: 'train', label: '学習だけ', hint: '保存済みの教師データで学習し直します' },
]

/** 段階の表示名。`DetectorMessage['state']` と 1 対 1 */
const STATE_LABEL: Record<DetectorMessage['state'], string> = {
  idle: '待機中',
  preparing: '準備中',
  evaluating: '弱点を測定中',
  collecting: '教師データを収集中',
  training: '学習中',
  saving: '保存中',
  done: '完了',
  error: '失敗',
  cancelled: '中断',
}

/** 検出クラスの日本語名。**バックエンドの `percep.DetClass` のメンバ名がキー。** */
const CLASS_LABEL: Record<string, string> = {
  TRAFFIC_LIGHT: '信号機',
  SPEED_SIGN: '速度標識',
  VEHICLE: '車両',
  OBSTACLE: '障害物',
  LANE: '車線',
}

/** 天候プリセットの日本語名。**バックエンドの `percep.weather.PRESETS` のキー。** */
const WEATHER_LABEL: Record<string, string> = {
  clear: '晴れ',
  drizzle: '小雨',
  rain: '雨',
  fog: '霧',
  heavy_fog: '濃霧',
}

/** 集める枚数から、おおよその所要時間を見積もる（銀座での実測が元）。 */
function estimateMinutes(
  samples: number,
  epochs: number,
  mode: DetectorMode,
  width: number,
): number {
  const collect = mode === 'train' ? 0 : samples / 120
  const train = mode === 'collect' ? 0 : (samples / 2400) * 15 * epochs * width * width
  return (collect + train) / 60
}

function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  const m = Math.floor(s / 60)
  return m > 0 ? `${m}分${String(s % 60).padStart(2, '0')}秒` : `${s}秒`
}

const EMPTY_HISTORY: DetectorHistoryPoint[] = []

export function ModelTab() {
  const detector = useSimStore((s) => s.detector)
  const presets = useSimStore((s) => s.presets)
  const status = useSimStore((s) => s.status)
  const connection = useSimStore((s) => s.connection)

  const limits = detector?.limits
  const running = detector?.running ?? false

  const req = detector?.request
  const [mode, setMode] = useState<DetectorMode>(req?.mode ?? 'full')
  const [samples, setSamples] = useState(req?.samples ?? 2400)
  const [epochs, setEpochs] = useState(req?.epochs ?? 12)
  const [batchSize, setBatchSize] = useState(req?.batchSize ?? 32)
  const [width, setWidth] = useState(req?.width ?? 1)
  const [seed, setSeed] = useState(req?.seed ?? 0)
  const [presetId, setPresetId] = useState<string | null>(req?.presetId ?? null)
  const [weatherMix, setWeatherMix] = useState(req?.weatherMix ?? true)
  const [focusWeak, setFocusWeak] = useState(req?.focusWeak ?? true)

  useEffect(() => {
    if (presetId === null && status.presetId) setPresetId(status.presetId)
  }, [presetId, status.presetId])

  const effectivePresetId = presetId ?? status.presetId ?? presets[0]?.id ?? null

  const disabled = connection !== 'open'
  const collects = mode !== 'train'
  const trains = mode !== 'collect'

  const handleStart = () => {
    send({
      type: 'start_detector_training',
      request: {
        mode,
        presetId: collects ? effectivePresetId : null,
        samples,
        epochs,
        batchSize,
        width,
        seed,
        weatherMix,
        focusWeak,
      },
    })
  }


  const model = detector?.model
  const datasetFile = detector?.datasetFile
  const dataset = detector?.dataset
  const evaluation = detector?.evaluation
  const seedMin = limits?.seedMin ?? 0
  const seedMax = limits?.seedMax ?? 999999
  const history = detector?.history ?? EMPTY_HISTORY
  const estimate = estimateMinutes(samples, epochs, mode, width)

  return (
    <>
      <Card title="いまの認識器" icon={<CameraIcon size={16} />}>
        <div className="m3-statgrid">
          <div className="m3-stat">
            <span className="m3-stat-label">観測の出どころ</span>
            <ValueFlash
              className={`m3-stat-value${model?.inUse ? ' m3-stat-value--accent' : ''}`}
              style={{ fontSize: 13 }}
            >
              {model?.inUse ? 'CNN の認識結果' : '真値フォールバック'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">モデルの有無</span>
            <ValueFlash className="m3-stat-value" style={{ fontSize: 13 }}>
              {model?.exists ? formatBytes(model.sizeBytes) : 'まだありません'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">最終更新</span>
            <ValueFlash className="m3-stat-value" style={{ fontSize: 12 }}>
              {model?.modifiedAt ?? '—'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">教師データ</span>
            <ValueFlash className="m3-stat-value" style={{ fontSize: 13 }}>
              {datasetFile?.exists ? formatBytes(datasetFile.sizeBytes) : 'まだありません'}
            </ValueFlash>
          </div>
        </div>

        <div className="m3-note">
          車両は<strong>擬似カメラ画像を CNN で認識した結果だけを見て走ります</strong>。
          認識器がまだ無いあいだは、シミュレーターの真値から作った「理想の検出結果」で
          代用しています（これは逃げ道ではなく、
          <strong>教師データを作っているのと同じ経路</strong>です）。
          <br />
          ここで学習すると <code className="m3-mono">backend/data/detector/detector.keras</code>{' '}
          が作られ、<strong>サーバーを再起動しなくてもその場で載せ替わります</strong>。
        </div>
      </Card>

      <Card title="学習の設定" icon={<TuneIcon size={16} />}>
        <div className="m3-row m3-row--wrap">
          {MODES.map((m) => (
            <Chip
              key={m.id}
              selected={mode === m.id}
              disabled={disabled || running}
              onClick={() => setMode(m.id)}
              title={m.hint}
            >
              {m.label}
            </Chip>
          ))}
        </div>
        <div className="m3-note">{MODES.find((m) => m.id === mode)?.hint}</div>

        <Collapse open={collects}>
          <div className="m3-col">
            <Select
              label="教師データを集めるエリア"
              value={effectivePresetId ?? ''}
              disabled={disabled || running || presets.length === 0}
              options={presets.map((p) => ({
                value: p.id,
                label: p.id === status.presetId ? `${p.name}（読み込み済み）` : p.name,
              }))}
              onChange={setPresetId}
            />
            <div className="m3-note">
              読み込み済みのエリアを選ぶと、そのマップをそのまま使うので待ち時間がありません。
              別のエリアを選ぶと、先に OpenStreetMap から取得します（初回は 10〜60 秒）。
            </div>

            <Slider
              label="集める枚数"
              hint="多いほど良いがメモリを食う"
              value={samples}
              min={limits?.samplesMin ?? 200}
              max={limits?.samplesMax ?? 4800}
              step={200}
              disabled={disabled || running}
              format={(v) => `${v.toLocaleString()} 枚`}
              onChange={setSamples}
            />
            <div className="m3-field">
              <label className="m3-field-label" htmlFor="detector-seed">
                乱数種
                <span style={{ opacity: 0.7 }}>　変えると別の教師データが集まる</span>
              </label>
              <div className="m3-field-row">
                <input
                  id="detector-seed"
                  className="m3-numberinput"
                  type="number"
                  inputMode="numeric"
                  min={seedMin}
                  max={seedMax}
                  step={1}
                  value={seed}
                  disabled={disabled || running}
                  onChange={(e) => {
                    const v = Number(e.target.value)
                    if (Number.isFinite(v)) setSeed(Math.max(seedMin, Math.min(seedMax, Math.round(v))))
                  }}
                />
                <Button
                  variant="tonal"
                  size="sm"
                  disabled={disabled || running}
                  onClick={() => setSeed(Math.floor(Math.random() * (seedMax - seedMin + 1)) + seedMin)}
                >
                  ランダム
                </Button>
              </div>
            </div>
            <div className="m3-note">
              同じ種なら何度回しても<strong>まったく同じ画</strong>が集まります。
              クラス内訳が偏っていたら、種を変えてもう一度集めると散らばりが変わります。
            </div>

            <hr className="m3-divider" />
            <Switch
              label="天候を混ぜて集める"
              description="晴れだけでなく、小雨・雨・霧・濃霧の画も教師データに入れます"
              checked={weatherMix}
              disabled={disabled || running}
              onChange={setWeatherMix}
            />
            <Switch
              label="弱点を狙って集める"
              description="集める前にいまの認識器を採点し、成績の低いクラスと天候を多めに集めます"
              checked={focusWeak}
              disabled={disabled || running}
              onChange={setFocusWeak}
            />
            <div className="m3-note">
              弱点の測定は収集の前に数秒かかります。認識器がまだ 1 つも無いときは
              採点する相手がいないので、そのまま一様に集めます。
              <br />
              走行中の天候（「シミュレーション」タブ）とは無関係で、
              <strong>収集は収集で天候を選び直します</strong>。
            </div>
          </div>
        </Collapse>

        <Collapse open={trains}>
          <div className="m3-col">
            <Slider
              label="エポック数"
              hint="教師データを何周するか"
              value={epochs}
              min={limits?.epochsMin ?? 1}
              max={limits?.epochsMax ?? 60}
              step={1}
              disabled={disabled || running}
              onChange={setEpochs}
            />
            <Slider
              label="バッチサイズ"
              value={batchSize}
              min={limits?.batchMin ?? 8}
              max={limits?.batchMax ?? 128}
              step={8}
              disabled={disabled || running}
              onChange={setBatchSize}
            />
            <Slider
              label="モデルの大きさ"
              hint="チャンネル倍率。推論が重ければ下げる"
              value={width}
              min={limits?.widthMin ?? 0.25}
              max={limits?.widthMax ?? 2}
              step={0.25}
              disabled={disabled || running}
              format={(v) => `${v.toFixed(2)} 倍`}
              onChange={setWidth}
            />
          </div>
        </Collapse>

        <div className="m3-banner m3-banner--warning">
          <span className="m3-banner-icon">
            <WarningIcon size={16} />
          </span>
          <span>
            <span className="m3-banner-title">学習中はシミュレーションが止まります</span>
            収集も学習も CPU を使い切るため、走行と強化学習は
            <strong>完了・中断まで停止</strong>します（画面と操作は生きています）。
            この設定でおよそ <strong>{estimate < 1 ? '1 分未満' : `${Math.round(estimate)} 分`}</strong>
            かかる見込みです。
          </span>
        </div>

        {running ? (
          <Button
            variant="danger"
            block
            icon={<PauseIcon size={18} />}
            disabled={disabled}
            onClick={() => send({ type: 'cancel_detector_training' })}
          >
            学習を中止
          </Button>
        ) : (
          <Button
            variant="filled"
            block
            icon={<PlayIcon size={18} />}
            disabled={disabled || (collects && !effectivePresetId)}
            onClick={handleStart}
          >
            {MODES.find((m) => m.id === mode)?.label}を開始
          </Button>
        )}
        {running && (
          <div className="m3-note">
            中止しても<strong>すぐには止まりません</strong>。いま処理中のバッチ
            （またはステップ）の切れ目まで進んでから終わります。
            <strong>中断したモデルは保存しません</strong>ので、いまの認識器はそのまま残ります。
          </div>
        )}
      </Card>

      <Card title="進捗" icon={<BrainIcon size={16} />}>
        {!detector ? (
          <div className="m3-note">
            サーバーから認識器の状態がまだ届いていません。接続を確認してください。
          </div>
        ) : (
          <>
            <div className="m3-row">
              <Chip
                small
                tone={
                  detector.state === 'error'
                    ? 'error'
                    : detector.state === 'done'
                      ? 'ok'
                      : running
                        ? 'warning'
                        : 'neutral'
                }
              >
                {STATE_LABEL[detector.state]}
              </Chip>
              <span className="m3-grow m3-note">{detector.message}</span>
            </div>

            <div
              className={`m3-progress${running ? ' m3-progress--determinate' : ''}`}
              style={{ ['--m3-progress-value' as string]: String(detector.progress) }}
              role="progressbar"
              aria-valuenow={Math.round(detector.progress * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              hidden={!running && detector.state !== 'done'}
            />

            <div className="m3-statgrid">
              <div className="m3-stat">
                <span className="m3-stat-label">集めた枚数</span>
                <ValueFlash className="m3-stat-value">
                  {detector.samples > 0
                    ? `${detector.collected.toLocaleString()} / ${detector.samples.toLocaleString()}`
                    : '—'}
                </ValueFlash>
              </div>
              <div className="m3-stat">
                <span className="m3-stat-label">エポック</span>
                <ValueFlash className="m3-stat-value">
                  {detector.epochs > 0 ? `${detector.epoch} / ${detector.epochs}` : '—'}
                </ValueFlash>
              </div>
              <div className="m3-stat">
                <span className="m3-stat-label">経過時間</span>
                <ValueFlash className="m3-stat-value">
                  {detector.elapsedSec > 0 ? formatElapsed(detector.elapsedSec) : '—'}
                </ValueFlash>
              </div>
              <div className="m3-stat">
                <span className="m3-stat-label">パラメータ数</span>
                <ValueFlash className="m3-stat-value">
                  {detector.paramCount > 0 ? detector.paramCount.toLocaleString() : '—'}
                </ValueFlash>
              </div>
            </div>

            {history.length >= 2 && (
              <>
                <MetricsChart
                  title="損失（学習データ）"
                  values={history.map((h) => h.loss)}
                  color="var(--m3-primary)"
                  format={(v) => v.toFixed(4)}
                  height={48}
                />
                <MetricsChart
                  title="損失（検証データ）"
                  values={history.map((h) => h.valLoss)}
                  color="var(--m3-tertiary)"
                  format={(v) => v.toFixed(4)}
                  height={48}
                />
                <div className="m3-note">
                  検証データの損失だけが下がらなくなったら、教師データの量に対して
                  エポックを回しすぎています（過学習）。
                </div>
              </>
            )}

            <Collapse open={detector.state === 'done'}>
              <div className="m3-banner m3-banner--info">
                <span className="m3-banner-icon">
                  <CheckIcon size={16} />
                </span>
                <span>
                  <span className="m3-banner-title">学習が終わりました</span>
                  観測は新しい CNN の出力に切り替わっています。「表示」タブでカメラを
                  <strong>運転席</strong>にすると、何をどう認識しているかが見えます。
                </span>
              </div>
            </Collapse>

            <Collapse open={detector.warning !== ''}>
              <div className="m3-banner m3-banner--warning">
                <span className="m3-banner-icon">
                  <WarningIcon size={16} />
                </span>
                <span>
                  <span className="m3-banner-title">確認してください</span>
                  {detector.warning}
                </span>
              </div>
            </Collapse>

            <Collapse open={detector.state === 'error'}>
              <div className="m3-banner m3-banner--error">
                <span className="m3-banner-icon">
                  <WarningIcon size={16} />
                </span>
                <span>
                  <span className="m3-banner-title">学習に失敗しました</span>
                  {detector.message}
                </span>
              </div>
            </Collapse>
          </>
        )}
      </Card>

      {evaluation && (
        <Card title="収集前に測った弱点" icon={<GaugeIcon size={16} />} variant="outlined">
          <div className="m3-note">
            <strong>これは学習後ではなく、収集を始める前の認識器の成績です。</strong>
            この結果がそのまま次の収集の重みになり、
            成績の低いクラス・天候ほど多く集めます。
          </div>

          <div className="m3-row">
            <span className="m3-grow" style={{ fontSize: 13, fontWeight: 700 }}>
              {evaluation.weakest}
            </span>
          </div>

          <div className="m3-col">
            {evaluation.classes.map((c) => (
              <div className="m3-row" key={c.name}>
                <span className="m3-note" style={{ width: 72 }}>
                  {CLASS_LABEL[c.name] ?? c.name}
                </span>
                <span className="m3-bar m3-grow">
                  <span
                    className="m3-bar-fill"
                    style={{
                      width: `${Math.round(c.recall * 100)}%`,
                      background:
                        c.recall < 0.5 ? 'var(--m3-error)' : 'var(--m3-primary)',
                    }}
                  />
                </span>
                <span className="m3-note" style={{ width: 92, textAlign: 'right' }}>
                  {c.truth === 0
                    ? '写らず'
                    : `${Math.round(c.recall * 100)}%` +
                      (c.attributeTotal > 0
                        ? ` / 読み ${Math.round(c.attributeAccuracy * 100)}%`
                        : '')}
                </span>
              </div>
            ))}
          </div>

          <hr className="m3-divider" />
          <div className="m3-statgrid">
            {evaluation.weathers.map((w) => (
              <div className="m3-stat" key={w.name}>
                <span className="m3-stat-label">{WEATHER_LABEL[w.name] ?? w.name}</span>
                <span
                  className={`m3-stat-value${w.recall < 0.5 ? ' m3-stat-value--error' : ''}`}
                >
                  {w.truth === 0 ? '—' : `${Math.round(w.recall * 100)}%`}
                </span>
              </div>
            ))}
          </div>
          <div className="m3-note">
            {evaluation.samples.toLocaleString()} 枚で採点（
            {formatElapsed(evaluation.elapsedSec)}）。
            「読み」は検出できたもののうち、灯色・規制速度まで合っていた割合です。
          </div>
        </Card>
      )}

      {dataset && (
        <Card title="集めた教師データ" icon={<InfoIcon size={16} />} variant="outlined">
          <div className="m3-note">
            <strong>件数が 0 のクラスは、学習しても検出できるようになりません。</strong>
            症状は「走らせてみたら前の車を認識しない」という形でしか出ないので、
            ここで必ず確かめてください（収集中は車両を寄せ集め、
            前方にパイロンを置いて写るようにしています）。
          </div>
          <div className="m3-statgrid">
            {Object.entries(dataset.classCounts).map(([name, count]) => (
              <div className="m3-stat" key={name}>
                <span className="m3-stat-label">{CLASS_LABEL[name] ?? name}</span>
                <ValueFlash
                  className={`m3-stat-value${count === 0 ? ' m3-stat-value--error' : ''}`}
                >
                  {count.toLocaleString()}
                </ValueFlash>
              </div>
            ))}
          </div>
          {Object.keys(dataset.weatherCounts ?? {}).length > 0 && (
            <>
              <hr className="m3-divider" />
              <div className="m3-row m3-row--wrap">
                {Object.entries(dataset.weatherCounts).map(([name, count]) => (
                  <Chip key={name} small>
                    {WEATHER_LABEL[name] ?? name} {count.toLocaleString()} 枚
                  </Chip>
                ))}
              </div>
            </>
          )}
          <div className="m3-note">
            {dataset.samples.toLocaleString()} 枚 / 物体のあるセル{' '}
            {(dataset.objectCellRatio * 100).toFixed(2)}%（1 枚あたり{' '}
            {dataset.objectsPerImage.toFixed(1)} 個）
          </div>
        </Card>
      )}

      <Card title="コマンドから回す場合" icon={<InfoIcon size={16} />} variant="outlined">
        <div className="m3-note">
          同じ学習は CLI からも実行できます（中身は同じ実装です）。こちらは
          <strong>サーバーを止めてから</strong>実行してください。
          <br />
          <code className="m3-mono">
            .venv\Scripts\python.exe train_detector.py
            {collects && ` --preset ${effectivePresetId ?? 'ginza'}`}
            {collects && ` --samples ${samples} --seed ${seed}`}
            {trains && ` --epochs ${epochs} --batch-size ${batchSize} --width ${width}`}
            {mode === 'collect' && ' --collect-only'}
            {mode === 'train' && ' --train-only'}
          </code>
        </div>
      </Card>
    </>
  )
}
