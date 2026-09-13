/**
 * 「モデル作成」タブ。**画像認識の CNN（認識器）をここから学習する。**
 *
 * 以前は `backend/train_detector.py` をコマンドで叩くしかなく、
 * 「走らせる」と「認識器を作る」が別々の世界に分かれていた。中身は同じ実装
 * （`app/percep/trainer.py`）を呼んでいるので、ここから回してもコマンドから
 * 回しても出来上がるモデルは同じである。
 *
 * ★ **学習中はシミュレーションが止まる。**（docs/protocol.md 2.9）
 *   収集も学習も CPU を使い切るうえ、`percep/groundtruth.py` の静的キャッシュを
 *   走行側と取り合うため。止まるのは物理と PPO だけで、画面・通信は生きている。
 *   利用者が驚かないよう、開始前と実行中の両方で必ず明示すること。
 *
 * 進捗は `detector` メッセージ（進捗が動いたときだけ、最大 1Hz）で届く。
 * 押した直後だけはサーバーが即座に 1 通返すので、ボタンの反応が 1 秒遅れない。
 */

import { useEffect, useMemo, useState } from 'react'
import { send } from '../store/connection'
import { formatBytes } from '../store/exportModel'
import { useSimStore } from '../store/simStore'
import type { DetectorMessage, DetectorMode } from '../types/protocol'
import { MetricsChart } from './MetricsChart'
import { Button } from '../ui/Button'
import { Card } from '../ui/Card'
import { Chip } from '../ui/Chip'
import { Collapse } from '../ui/Collapse'
import { Select } from '../ui/Select'
import { Slider } from '../ui/Slider'
import { ValueFlash } from '../ui/ValueFlash'
import {
  BrainIcon,
  CameraIcon,
  CheckIcon,
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
  collecting: '教師データを収集中',
  training: '学習中',
  saving: '保存中',
  done: '完了',
  error: '失敗',
  cancelled: '中断',
}

/**
 * 検出クラスの日本語名。**バックエンドの `percep.DetClass` のメンバ名がキー。**
 * 知らない名前が来たらそのまま出す（クラスが増えても表示は壊れない）。
 */
const CLASS_LABEL: Record<string, string> = {
  TRAFFIC_LIGHT: '信号機',
  SPEED_SIGN: '速度標識',
  VEHICLE: '車両',
  OBSTACLE: '障害物',
  LANE: '車線',
}

/** 集める枚数から、おおよその所要時間を見積もる（銀座での実測が元） */
function estimateMinutes(samples: number, epochs: number, mode: DetectorMode): number {
  // 収集は 8 台ぶんを 1 回で描くので、実測でおよそ 120 枚/秒。
  // 学習は 2,400 枚 1 エポックがおよそ 15 秒。
  const collect = mode === 'train' ? 0 : samples / 120
  const train = mode === 'collect' ? 0 : (samples / 2400) * 15 * epochs
  return (collect + train) / 60
}

function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  const m = Math.floor(s / 60)
  return m > 0 ? `${m}分${String(s % 60).padStart(2, '0')}秒` : `${s}秒`
}

export function ModelTab() {
  const detector = useSimStore((s) => s.detector)
  const presets = useSimStore((s) => s.presets)
  const status = useSimStore((s) => s.status)
  const connection = useSimStore((s) => s.connection)

  const limits = detector?.limits
  const running = detector?.running ?? false

  // ---- 入力。サーバーの値域に合わせるが、握るのはこのコンポーネント ----
  const [mode, setMode] = useState<DetectorMode>('full')
  const [samples, setSamples] = useState(2400)
  const [epochs, setEpochs] = useState(12)
  const [batchSize, setBatchSize] = useState(32)
  const [width, setWidth] = useState(1)
  const [presetId, setPresetId] = useState<string | null>(null)

  // エリアの既定は「いま走らせているマップ」。読み込み済みのものと同じなら
  // サーバーはそれをそのまま使うので、地図の読み直しが起きない
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
      },
    })
  }

  const lossSeries = useMemo(
    () => ({
      loss: (detector?.history ?? []).map((h) => h.loss),
      valLoss: (detector?.history ?? []).map((h) => h.valLoss),
    }),
    [detector?.history],
  )

  const model = detector?.model
  const datasetFile = detector?.datasetFile
  const dataset = detector?.dataset
  const estimate = estimateMinutes(samples, epochs, mode)

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

            {lossSeries.loss.length >= 2 && (
              <>
                <MetricsChart
                  title="損失（学習データ）"
                  values={lossSeries.loss}
                  color="var(--m3-primary)"
                  format={(v) => v.toFixed(4)}
                  height={48}
                />
                <MetricsChart
                  title="損失（検証データ）"
                  values={lossSeries.valLoss}
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
                  className={`m3-stat-value${count === 0 ? '' : ' m3-stat-value--accent'}`}
                >
                  {count.toLocaleString()}
                </ValueFlash>
              </div>
            ))}
          </div>
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
            .venv\Scripts\python.exe train_detector.py --samples {samples} --epochs {epochs}
          </code>
        </div>
      </Card>
    </>
  )
}
