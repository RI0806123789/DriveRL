/** 「学習」タブ。 */

import { useRef, useState } from 'react'
import { send } from '../store/connection'
import { formatBytes, startExport, startImport, useModelTransfer } from '../store/exportModel'
import type { ExportKind } from '../store/exportModel'
import { useSimStore } from '../store/simStore'
import { isAutotuned } from '../store/autotune'
import { AssistRateChip } from './AssistRateChip'
import { AutoTuneCard } from './AutoTuneCard'
import { CurriculumGauge } from './CurriculumGauge'
import { MetricsChart } from './MetricsChart'
import { NetworkGraph } from './NetworkGraph'
import { OptionShares } from './OptionShares'
import { jerkText } from '../store/driveOption'
import { Button } from '../ui/Button'
import { Card } from '../ui/Card'
import { Collapse } from '../ui/Collapse'
import { Slider } from '../ui/Slider'
import { Switch } from '../ui/Switch'
import { ValueFlash } from '../ui/ValueFlash'
import {
  BrainIcon,
  ChartIcon,
  CheckIcon,
  DownloadIcon,
  InfoIcon,
  RefreshIcon,
  SaveIcon,
  TuneIcon,
  WarningIcon,
} from '../ui/Icons'

export function LearningTab() {
  const series = useSimStore((s) => s.metricsSeries)
  const revision = useSimStore((s) => s.metricsRevision)
  const metricsMarks = useSimStore((s) => s.metricsMarks)
  const latest = useSimStore((s) => s.latestMetrics)
  const params = useSimStore((s) => s.params)
  const patchParamsLocal = useSimStore((s) => s.patchParamsLocal)
  const learning = useSimStore((s) => s.status.learning)
  const suspended = useSimStore((s) => s.status.simSuspended ?? false)

  const usingMock = useSimStore((s) => s.usingMock)
  const connection = useSimStore((s) => s.connection)
  const tune = useSimStore((s) => s.autotune)
  const practical = useSimStore((s) => s.status.practicalMode ?? false)
  const mapLoaded = useSimStore((s) => s.status.mapLoaded)
  const tuning = tune?.running ?? false
  const auto = (key: Parameters<typeof isAutotuned>[1]) => isAutotuned(tune, key)

  const [confirmReset, setConfirmReset] = useState(false)
  // 実行中かどうかと結果はタブの外に置く（タブを替えても消えず、二重に送れない）
  const exporting = useModelTransfer((s) => s.exporting)
  const exportResult = useModelTransfer((s) => s.exportResult)
  const importing = useModelTransfer((s) => s.importing)
  const importResult = useModelTransfer((s) => s.importResult)
  const handleExport = (kind: ExportKind) => void startExport(kind)
  const handleImportFile = (file: File) => void startImport(file)

  const fileInputRef = useRef<HTMLInputElement>(null)

  const exportDisabled = usingMock || connection !== 'open'

  return (
    <>
      <Card title="学習の状況" icon={<BrainIcon size={16} />}>
        {!learning && (
          <div className="m3-note">
            {suspended
              ? '「モデル作成」タブで認識器を学習している間は、強化学習も止まります。完了・中断すると自動で再開します。'
              : 'マップを読み込むと学習が始まります。ここには開始後の推移が表示されます。'}
          </div>
        )}
        {params.onlineAssist && (
          <div className="m3-row m3-row--wrap">
            <AssistRateChip rate={latest?.assistRate} />
          </div>
        )}
        <div className="m3-statgrid">
          <div className="m3-stat">
            <span className="m3-stat-label">PPO 更新回数</span>
            <ValueFlash className="m3-stat-value">{latest?.updates.toLocaleString() ?? '—'}</ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">エピソード数</span>
            <ValueFlash className="m3-stat-value">{latest?.episodes.toLocaleString() ?? '—'}</ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">目的地到達率</span>
            <ValueFlash className="m3-stat-value m3-stat-value--accent">
              {latest ? `${(latest.goalRate * 100).toFixed(0)}%` : '—'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">衝突率</span>
            <ValueFlash className="m3-stat-value">
              {latest ? `${(latest.collisionRate * 100).toFixed(0)}%` : '—'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">うち歩行者</span>
            <ValueFlash
              className="m3-stat-value"
              style={
                (latest?.pedestrianCollisionRate ?? 0) > 0
                  ? { color: 'var(--m3-error)' }
                  : undefined
              }
            >
              {latest ? `${((latest.pedestrianCollisionRate ?? 0) * 100).toFixed(0)}%` : '—'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">ステップ/秒</span>
            <ValueFlash className="m3-stat-value">{latest?.stepsPerSec.toFixed(1) ?? '—'}</ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">信号無視/エピソード</span>
            <ValueFlash className="m3-stat-value">
              {latest?.signalViolations.toFixed(2) ?? '—'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">速度超過/エピソード</span>
            <ValueFlash className="m3-stat-value">
              {latest?.speedViolations.toFixed(2) ?? '—'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">車線逸脱（平均）</span>
            <ValueFlash className="m3-stat-value">
              {latest ? `${latest.laneDeviation.toFixed(2)} m` : '—'}
            </ValueFlash>
          </div>
          <div className="m3-stat">
            <span className="m3-stat-label">平均エピソード長</span>
            <ValueFlash className="m3-stat-value">{latest?.meanEpisodeLength.toFixed(0) ?? '—'}</ValueFlash>
          </div>
        </div>
        <div className="m3-note">
          学習初期は衝突や道路外への逸脱が多く、車の動きもぎこちないのが正常です。
          これは学習済みモデルの再生ではなく、いま重みが更新されている最中の挙動です。
          <br />
          <strong>衝突率</strong>のうち、相手が歩行者だった割合を「うち歩行者」に出しています。
          歩行者は信号に従って横断するので、<strong>これが増えているときは車が
          歩行者用信号の青（＝車両側の赤）を無視して交差点へ入っています</strong>。
          <br />
          <strong>Assist Rate</strong>は、学習中の車を経路追従（お手本）が代わりに運転した割合です。
          学習の始めは 100% で、経験が積もるにつれて 5% まで下がります。車線を外れかけたとき・
          ぶつかりそうなとき・赤信号へ突っ込みそうなとき・理由なく止まったままのときは、
          割合に関わらずお手本が代わります。
          <br />
          <strong>車線逸脱</strong>は走るべき車線の中心からどれだけ横にずれているかの平均です。
          市街地の車線幅はおおむね 3m なので、<strong>1.5m を超えると隣の車線や対向車線に
          はみ出している</strong>とみてよく、0.5m 以下なら車線をよく保てています。
        </div>
      </Card>

      {params.incidentCurriculum && (
        <Card title="ヒヤリハット（オートカリキュラム）" icon={<WarningIcon size={16} />}>
          <CurriculumGauge
            level={latest?.curriculumLevel}
            triggered={latest?.incidentsTriggered}
            avoidedRate={latest?.incidentsAvoidedRate}
          />
          <div className="m3-note">
            目的地に着けるようになり衝突が減ってくると（直近 20 エピソードで到達 85% 以上・衝突 10% 以下）、
            難易度が 5% ずつ上がり、学習中の車の前で<strong>歩行者が車道へ飛び出したり、前の車が急ブレーキを
            踏んだり</strong>するようになります。衝突が 25% 以上に増えると下がります。
            <br />
            <strong>自力で回避</strong>は、起こしてから 5 秒のあいだに衝突も逸脱もせず、お手本（経路追従）にも
            代わられなかった割合です。
          </div>
        </Card>
      )}

      <Card title="意図（階層型の方策）" icon={<TuneIcon size={16} />}>
        <OptionShares shares={latest?.optionShares} />
        <div className="m3-statgrid">
          <div className="m3-stat">
            <span className="m3-stat-label">加加速度（Jerk）</span>
            <ValueFlash className="m3-stat-value">{jerkText(latest?.jerkRms)}</ValueFlash>
          </div>
        </div>
        <div className="m3-note">
          上位方策が <strong>1 秒（20 ステップ）ごと</strong>に「巡航・追従・徐行・停止」のどれで走るかを決め、
          下位方策がその意図の下で毎ステップのアクセルと操舵を決めます。割合は直近 10 秒に方策が運転した
          ステップで選ばれていた意図です（お手本が代わったステップは数えません）。
          <br />
          <strong>加加速度</strong>は加速度の変わる速さで、アクセルとブレーキを小刻みに踏み替えるほど大きくなります。
        </div>
      </Card>

      <Card title="ネットワーク" icon={<BrainIcon size={16} />}>
        <div className="m3-note">
          方策（アクセル・操舵を決める側）と価値（その状況の見込みを評価する側）の
          2 本のネットワークです（図は下位方策。上位方策の意図を選ぶ側と、その価値の 2 本は下の表に並びます）。<strong>全車両がこの 1 つを共有</strong>していて、
          車両ごとに別々の学習はしていません。台数を増やすのは、
          学習者を増やすためではなく<strong>経験を集める速度を上げる</strong>ためです。
        </div>
        <NetworkGraph />
      </Card>

      <Card title="推移" icon={<ChartIcon size={16} />}>
        <MetricsChart
          title="平均エピソード報酬"
          values={series.reward}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-secondary)"
          format={(v) => v.toFixed(1)}
          showZero
        />
        <MetricsChart
          title="目的地到達率"
          values={series.goalRate}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-tertiary)"
          format={(v) => `${(v * 100).toFixed(0)}%`}
          height={48}
        />
        <MetricsChart
          title="信号無視（1 エピソードあたり）"
          values={series.violations}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-error)"
          format={(v) => v.toFixed(2)}
          height={48}
        />
        <MetricsChart
          title="歩行者との衝突（1 エピソードあたり）"
          values={series.pedestrianCollisions}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-error)"
          format={(v) => `${(v * 100).toFixed(0)}%`}
          height={48}
        />
        <MetricsChart
          title="車線逸脱（車線中心からの平均ずれ）"
          values={series.laneDeviation}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-warning)"
          format={(v) => `${v.toFixed(2)} m`}
          height={48}
        />
        <MetricsChart
          title="加加速度（Jerk の二乗平均平方根）"
          values={series.jerk}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-primary)"
          format={(v) => `${v.toFixed(1)} m/s³`}
          height={48}
        />
        <MetricsChart
          title="Assist Rate（お手本が運転した割合）"
          values={series.assistRate}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-warning)"
          format={(v) => `${(v * 100).toFixed(0)}%`}
          height={48}
        />
        {params.incidentCurriculum && (
          <>
            <MetricsChart
              title="Curriculum Level（ヒヤリハットの難易度）"
              values={series.curriculumLevel}
              revision={revision}
              marks={metricsMarks}
              color="var(--m3-tertiary)"
              format={(v) => `${(v * 100).toFixed(0)}%`}
              height={48}
            />
            <MetricsChart
              title="ヒヤリハットを自力で回避した割合"
              values={series.incidentsAvoidedRate}
              revision={revision}
              marks={metricsMarks}
              color="var(--m3-secondary)"
              format={(v) => `${(v * 100).toFixed(0)}%`}
              height={48}
            />
          </>
        )}
        <MetricsChart
          title="ポリシー損失"
          values={series.policyLoss}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-primary)"
          format={(v) => v.toFixed(4)}
          height={48}
          showZero
        />
        <MetricsChart
          title="価値損失"
          values={series.valueLoss}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-primary)"
          format={(v) => v.toFixed(3)}
          height={48}
        />
        <MetricsChart
          title="エントロピー"
          values={series.entropy}
          revision={revision}
          marks={metricsMarks}
          color="var(--m3-outline)"
          format={(v) => v.toFixed(3)}
          height={48}
        />
      </Card>

      <Card title="学習ハイパーパラメータ" icon={<TuneIcon size={16} />}>
        <div className="m3-note">
          {usingMock
            ? 'モック接続中は、ここの設定値を保存しません。'
            : 'ここと「報酬の重み」の設定値は、変えるたびにサーバーへ自動で保存され、再起動しても戻ります。' +
              '学習の自動化の間は、探索が選んだ値は OFF にして最良の試行が確定したときに保存します。'}
        </div>
        <Slider
          label="学習率"
          value={params.learningRate}
          auto={auto('learningRate')}
          min={1e-5}
          max={1e-3}
          step={1e-5}
          format={(v) => v.toExponential(1)}
          onChange={(v) => patchParamsLocal({ learningRate: v })}
          onCommit={(v) => send({ type: 'set_params', params: { learningRate: v } })}
        />
        <Slider
          label="割引率 γ"
          value={params.gamma}
          auto={auto('gamma')}
          min={0.8}
          max={0.999}
          step={0.001}
          format={(v) => v.toFixed(3)}
          onChange={(v) => patchParamsLocal({ gamma: v })}
          onCommit={(v) => send({ type: 'set_params', params: { gamma: v } })}
        />
        <Slider
          label="クリップ範囲 ε"
          value={params.clipRange}
          auto={auto('clipRange')}
          min={0.05}
          max={0.5}
          step={0.01}
          format={(v) => v.toFixed(2)}
          onChange={(v) => patchParamsLocal({ clipRange: v })}
          onCommit={(v) => send({ type: 'set_params', params: { clipRange: v } })}
        />
        <Slider
          label="エントロピー係数"
          hint="大きいほど探索的"
          value={params.entropyCoef}
          auto={auto('entropyCoef')}
          min={0}
          max={0.1}
          step={0.001}
          format={(v) => v.toFixed(3)}
          onChange={(v) => patchParamsLocal({ entropyCoef: v })}
          onCommit={(v) => send({ type: 'set_params', params: { entropyCoef: v } })}
        />
        <Switch
          label="お手本のアシスト（オンライン模倣）"
          description="学習中の車を経路追従がときどき代わりに運転し、その操作を模倣の教師にします。止まったまま固まるのと、学習の始めの衝突の繰り返しを防ぎます"
          checked={params.onlineAssist}
          onChange={(v) => {
            patchParamsLocal({ onlineAssist: v })
            send({ type: 'set_params', params: { onlineAssist: v } })
          }}
        />
        <Switch
          label="ヒヤリハットのオートカリキュラム"
          description="成績が上がると、学習中の車の前で歩行者の飛び出しや前走車の急ブレーキを起こします。報酬と終了条件はそのまま（ぶつかれば衝突の罰）で、実用モードでは起こしません"
          checked={params.incidentCurriculum}
          onChange={(v) => {
            patchParamsLocal({ incidentCurriculum: v })
            send({ type: 'set_params', params: { incidentCurriculum: v } })
          }}
        />
        <Switch
          label="車車間通信（V2X）"
          description="30m 以内の近い 2 台から、車速・右左折の意図・危険（歩行者・障害物）・交差点への近さを受け取り、観測の末尾 4 次元に足します。切ると 0（近くに車がいないときと同じ）"
          checked={params.v2xComm}
          onChange={(v) => {
            patchParamsLocal({ v2xComm: v })
            send({ type: 'set_params', params: { v2xComm: v } })
          }}
        />
      </Card>

      <Card title="報酬の重み" icon={<TuneIcon size={16} />} variant="outlined">
        <div className="m3-note">
          報酬は「目的地到達・衝突回避・経路進捗・信号順守・速度順守」で構成しています。
          車線は経路そのものが左側通行の車線に沿っているため、経路追従の報酬に含まれます。
        </div>
        <Slider
          label="目的地到達"
          value={params.rewardGoal}
          auto={auto('rewardGoal')}
          format={(v) => v.toFixed(0)}
          min={0}
          max={300}
          step={5}
          onChange={(v) => patchParamsLocal({ rewardGoal: v })}
          onCommit={(v) => send({ type: 'set_params', params: { rewardGoal: v } })}
        />
        <Slider
          label="衝突"
          value={params.rewardCollision}
          auto={auto('rewardCollision')}
          format={(v) => v.toFixed(0)}
          min={-300}
          max={0}
          step={5}
          onChange={(v) => patchParamsLocal({ rewardCollision: v })}
          onCommit={(v) => send({ type: 'set_params', params: { rewardCollision: v } })}
        />
        <Slider
          label="経路進捗"
          hint="1m 進むごと"
          value={params.rewardProgress}
          auto={auto('rewardProgress')}
          min={0}
          max={5}
          step={0.1}
          format={(v) => v.toFixed(1)}
          onChange={(v) => patchParamsLocal({ rewardProgress: v })}
          onCommit={(v) => send({ type: 'set_params', params: { rewardProgress: v } })}
        />
        <Slider
          label="道路外"
          hint="1 ステップごと"
          value={params.rewardOffroad}
          auto={auto('rewardOffroad')}
          min={-10}
          max={0}
          step={0.1}
          format={(v) => v.toFixed(1)}
          onChange={(v) => patchParamsLocal({ rewardOffroad: v })}
          onCommit={(v) => send({ type: 'set_params', params: { rewardOffroad: v } })}
        />
        <Slider
          label="信号無視"
          hint="赤信号で停止線を越えたとき"
          value={params.rewardSignal}
          auto={auto('rewardSignal')}
          format={(v) => v.toFixed(0)}
          min={-200}
          max={0}
          step={5}
          onChange={(v) => patchParamsLocal({ rewardSignal: v })}
          onCommit={(v) => send({ type: 'set_params', params: { rewardSignal: v } })}
        />
        <Slider
          label="速度超過"
          hint="規制速度を超え始めたとき"
          value={params.rewardOverspeed}
          auto={auto('rewardOverspeed')}
          format={(v) => v.toFixed(0)}
          min={-100}
          max={0}
          step={1}
          onChange={(v) => patchParamsLocal({ rewardOverspeed: v })}
          onCommit={(v) => send({ type: 'set_params', params: { rewardOverspeed: v } })}
        />
        <Slider
          label="時間ペナルティ"
          hint="1 ステップごと"
          value={params.rewardTime}
          auto={auto('rewardTime')}
          min={-1}
          max={0}
          step={0.01}
          format={(v) => v.toFixed(2)}
          onChange={(v) => patchParamsLocal({ rewardTime: v })}
          onCommit={(v) => send({ type: 'set_params', params: { rewardTime: v } })}
        />
      </Card>

      <Card title="チェックポイント" icon={<SaveIcon size={16} />}>
        <div className="m3-note">
          学習した重みはディスクに保存され、サーバーを再起動しても復元されます。
          20 回の更新ごとに自動保存もされます。
        </div>
        <div className="m3-row">
          <Button
            variant="tonal"
            icon={<SaveIcon size={16} />}
            disabled={tuning}
            onClick={() => send({ type: 'save_checkpoint' })}
          >
            今すぐ保存
          </Button>
          <Button
            variant="outlined"
            icon={<DownloadIcon size={16} />}
            disabled={tuning}
            onClick={() => send({ type: 'load_checkpoint' })}
          >
            読み込む
          </Button>
        </div>

        <hr className="m3-divider" />

        <Collapse open={confirmReset}>
          <div className="m3-confirm">
            <span className="m3-confirm-text">
              <WarningIcon size={16} /> 学習済みの重みをすべて捨てて、ゼロからやり直します。
              これまでの学習成果は復元できません。実行しますか？
            </span>
            <div className="m3-row">
              <Button
                variant="danger"
                icon={<RefreshIcon size={16} />}
                onClick={() => {
                  send({ type: 'reset_policy' })
                  setConfirmReset(false)
                }}
              >
                初期化する
              </Button>
              <Button variant="text" onClick={() => setConfirmReset(false)}>
                やめる
              </Button>
            </div>
          </div>
        </Collapse>
        <Collapse open={!confirmReset}>
          <Button
            variant="outlined"
            block
            icon={<RefreshIcon size={16} />}
            disabled={tuning}
            onClick={() => setConfirmReset(true)}
          >
            ポリシーを初期化
          </Button>
        </Collapse>
      </Card>

      <Card title="モデルの書き出し" icon={<DownloadIcon size={16} />}>
        <div className="m3-note">
          いま学習中の重みをファイルとして手元に保存します。書き出しはサーバー側の
          学習ループの切れ目で行われるので、<strong>学習は止まりません</strong>。
          どの形式にも観測ベクトルの構成と行動のスケールがメタデータとして埋め込まれます。
        </div>

        <Button
          variant="tonal"
          block
          icon={<DownloadIcon size={16} />}
          disabled={exportDisabled || exporting !== null}
          onClick={() => void handleExport('checkpoint')}
        >
          {exporting === 'checkpoint' ? '書き出し中…' : '重み一式（.pt）'}
        </Button>
        <div className="m3-note">
          重み・オプティマイザ状態・メタデータを含みます。「読み込む」で戻せば
          <strong>続きから学習を再開</strong>できます。バックアップや別マシンへの持ち出し向け。
        </div>

        <hr className="m3-divider" />

        <Button
          variant="tonal"
          block
          icon={<DownloadIcon size={16} />}
          disabled={exportDisabled || exporting !== null}
          onClick={() => void handleExport('pt2')}
        >
          {exporting === 'pt2' ? '書き出し中…' : 'torch.export（.pt2）'}
        </Button>
        <div className="m3-note">
          推論だけを切り出した自己完結の形式で、PyTorch が TorchScript の後継として勧めているものです。
          このプロジェクトのコードが無くても
          <code className="m3-mono"> torch.export.load() </code>
          だけで読めます。観測ベクトルの構成や行動のスケールは
          <code className="m3-mono"> metadata.json </code>
          として同梱されるので、受け取った側だけで使えます。モデルをトレースする間（1 秒足らず）、
          シミュレーションが一瞬止まります。
        </div>

        <hr className="m3-divider" />

        <Button
          variant="tonal"
          block
          icon={<DownloadIcon size={16} />}
          disabled={exportDisabled || exporting !== null}
          onClick={() => void handleExport('torchscript')}
        >
          {exporting === 'torchscript' ? '書き出し中…' : 'TorchScript（.torchscript.pt）'}
        </Button>
        <div className="m3-note">
          推論だけを切り出した自己完結の形式です。このプロジェクトのコードが無くても
          <code className="m3-mono"> torch.jit.load() </code>
          だけで読めます（メタデータの同梱は上と同じ）。PyTorch は TorchScript を非推奨にしているので、
          新しく使うなら torch.export（.pt2）を選んでください。
        </div>

        <hr className="m3-divider" />

        <Button
          variant="tonal"
          block
          icon={<DownloadIcon size={16} />}
          disabled={exportDisabled || exporting !== null}
          onClick={() => void handleExport('keras')}
        >
          {exporting === 'keras' ? '書き出し中…' : 'Keras（.keras）'}
        </Button>
        <div className="m3-note">
          同じネットワークを Keras 3 のモデルとして組み直したものです。
          <code className="m3-mono"> keras.saving.load_model() </code>
          だけで読め、標準の Dense 層しか使っていないので custom_objects は要りません。
          Keras / TensorFlow 系のツールで扱いたい場合に使ってください。
          初回だけ Keras の読み込みに数秒かかります。
        </div>

        <Collapse open={exportResult !== null}>
          <div
            className={`m3-banner ${exportResult?.ok ? 'm3-banner--info' : 'm3-banner--error'}`}
          >
            <span className="m3-banner-icon">
              {exportResult?.ok ? <CheckIcon size={16} /> : <WarningIcon size={16} />}
            </span>
            <span>
              {exportResult?.ok ? (
                <>
                  <span className="m3-banner-title">書き出しました</span>
                  <span className="m3-mono">{exportResult.filename}</span>
                  {exportResult.sizeBytes !== undefined && (
                    <>（{formatBytes(exportResult.sizeBytes)}）</>
                  )}
                </>
              ) : (
                <>
                  <span className="m3-banner-title">書き出しに失敗しました</span>
                  {exportResult?.error}
                </>
              )}
            </span>
          </div>
        </Collapse>

        {usingMock && (
          <div className="m3-banner m3-banner--warning">
            <span className="m3-banner-icon">
              <InfoIcon size={16} />
            </span>
            <span>
              モック接続中は書き出せません。実際のバックエンドに繋いでください。
            </span>
          </div>
        )}
      </Card>

      <Card title="モデルの読み込み" icon={<DownloadIcon size={16} />}>
        <div className="m3-note">
          書き出した「重み一式（.pt）」を選ぶと、<strong>その時点から学習を再開</strong>します。
          重みだけでなくオプティマイザの状態も戻すので、学習が仕切り直しになりません。
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pt,.before-autotune,application/octet-stream"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0]
            e.target.value = ''
            if (file) void handleImportFile(file)
          }}
        />

        <Button
          variant="filled"
          block
          icon={<SaveIcon size={16} />}
          disabled={exportDisabled || importing || tuning}
          onClick={() => fileInputRef.current?.click()}
        >
          {importing ? '読み込み中…' : 'ファイルを選んで学習を再開'}
        </Button>

        <div className="m3-note">
          いま学習中のモデルは、載せ替える直前に
          <code className="m3-mono"> backend/data/exports/ </code>
          へ自動でバックアップされます。間違ったファイルを選んでも学習成果は失われません。
          <br />
          torch.export 版（<code className="m3-mono">.pt2</code>）と TorchScript 版（
          <code className="m3-mono">.torchscript.pt</code>）は推論専用なので、学習の再開には使えません。
        </div>

        {importResult && (
          <div
            className={`m3-banner ${importResult.ok ? 'm3-banner--info' : 'm3-banner--error'}`}
          >
            <span className="m3-banner-icon">
              {importResult.ok ? <CheckIcon size={16} /> : <WarningIcon size={16} />}
            </span>
            <span>
              {importResult.ok && importResult.checkpoint ? (
                <>
                  <span className="m3-banner-title">
                    学習回数 {importResult.checkpoint.updates.toLocaleString()} 回の状態から再開します
                  </span>
                  <span className="m3-mono">{importResult.name}</span>
                  <br />
                  {importResult.checkpoint.exportedAt && (
                    <>書き出し日時: {importResult.checkpoint.exportedAt}<br /></>
                  )}
                  {importResult.checkpoint.presetName && (
                    <>学習したエリア: {importResult.checkpoint.presetName}<br /></>
                  )}
                  {!importResult.checkpoint.hasOptimizer && (
                    <>
                      ※ オプティマイザの状態が入っていないため、学習の立ち上がりが
                      一時的に鈍る場合があります
                      <br />
                    </>
                  )}
                  {importResult.backup && (
                    <>
                      直前のモデルは{' '}
                      <span className="m3-mono">{importResult.backup.filename}</span>
                      （{formatBytes(importResult.backup.sizeBytes)}）として退避しました
                    </>
                  )}
                </>
              ) : (
                <>
                  <span className="m3-banner-title">読み込めませんでした</span>
                  {importResult.error}
                </>
              )}
            </span>
          </div>
        )}
      </Card>

      <AutoTuneCard
        tune={tune}
        connected={connection === 'open'}
        practical={practical}
        suspended={suspended}
        mapLoaded={mapLoaded}
        onToggle={(on) => send({ type: on ? 'start_auto_tune' : 'stop_auto_tune_and_save' })}
      />
    </>
  )
}
