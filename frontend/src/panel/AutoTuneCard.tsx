/** 学習タブの「学習の自動化」（Optuna によるハイパーパラメータの探索）。 */

import type { AutotuneMessage } from '../types/protocol'
import { autotuneView } from '../store/autotune'
import type { AutotuneContext } from '../store/autotune'
import { Card } from '../ui/Card'
import { Chip } from '../ui/Chip'
import { InfoIcon, SparkleIcon } from '../ui/Icons'
import { Switch } from '../ui/Switch'

export interface AutoTuneCardProps extends AutotuneContext {
  tune: AutotuneMessage | null
  onToggle: (on: boolean) => void
}

export function AutoTuneCard({ tune, onToggle, ...ctx }: AutoTuneCardProps) {
  const view = autotuneView(tune, ctx)
  return (
    <Card
      title="学習の自動化"
      icon={<SparkleIcon size={16} />}
      action={view.badge ? <Chip tone="ok" small>自動調整中</Chip> : undefined}
    >
      <Switch
        label="学習の自動化（Optuna）"
        description="学習率・割引率・クリップ範囲・エントロピー係数と報酬の重みを、試行を繰り返して自動で探します。OFF にした瞬間、いちばん成績の良かった試行の設定と重みを保存し、そのまま学習を続けます"
        checked={view.checked}
        disabled={!view.canToggle}
        onChange={onToggle}
      />
      {view.blockedReason && !view.checked && (
        <div className="m3-note" data-testid="autotune-blocked">
          {view.blockedReason}
        </div>
      )}
      <div className="m3-col">
        <span className="m3-stat-label" data-testid="autotune-status">{view.statusText}</span>
        <span
          className="m3-bar"
          role="progressbar"
          aria-label="いまの試行の進み具合"
          aria-valuemin={0}
          aria-valuemax={1}
          aria-valuenow={view.progress}
        >
          <span
            className="m3-bar-fill"
            style={{ ['--m3-bar-value' as string]: String(view.progress), background: 'var(--m3-primary)' }}
          />
        </span>
      </div>
      <div className="m3-statgrid">
        <div className="m3-stat">
          <span className="m3-stat-label">試行</span>
          <span className="m3-stat-value" data-testid="autotune-trials">{view.trialsText}</span>
        </div>
        <div className="m3-stat">
          <span className="m3-stat-label">最良</span>
          <span className="m3-stat-value" data-testid="autotune-best">{view.bestText}</span>
        </div>
      </div>
      {view.history.length > 0 && (
        <div className="m3-row m3-row--wrap" aria-label="直近の試行">
          {view.history.map((item) => (
            <Chip key={item.trial} tone={item.tone} small title={item.title}>
              {item.text}
            </Chip>
          ))}
        </div>
      )}
      {view.checked && (
        <div className="m3-banner m3-banner--info">
          <span className="m3-banner-icon">
            <InfoIcon size={16} />
          </span>
          <span>
            自動調整中は上のスライダーを動かせません。保存・読み込み・初期化・エリアの切り替え・実用モード・
            認識器の学習も、OFF にするまでできません。
          </span>
        </div>
      )}
      <div className="m3-note">
        1 試行ごとに、重みを<strong>自動化を始めたときの状態へ巻き戻し、街を作り直してから</strong>約 77 秒ぶん学習させ、
        後半の走り（到達・衝突・逸脱・信号無視・止まったままでないか）で採点します。報酬の重みも探すので、採点に報酬の合計は使いません。
        見込みの薄い試行は途中で打ち切ります。
        <br />
        OFF にすると、元の重みを
        <code className="m3-mono"> backend/data/checkpoints/shared_policy.pt.before-autotune </code>
        に退避してから最良の試行の重みで上書きします。気に入らなければ、その退避ファイルを「モデルの読み込み」で選べば戻せます。
        探索の履歴は
        <code className="m3-mono"> backend/data/tuning/ </code>
        に残り、次に ON にしたときは続きから探します（エリアごとに別の履歴）。
      </div>
    </Card>
  )
}
