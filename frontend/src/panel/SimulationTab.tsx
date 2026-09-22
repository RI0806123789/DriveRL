/** 「シミュレーション」タブ。 */

import { useEffect, useState } from 'react'
import { send } from '../store/connection'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import type { InteractionMode } from '../store/simStore'
import { WEATHER_LABELS } from '../types/protocol'
import { vehicleColor } from '../scene/vehicleColors'
import { Button } from '../ui/Button'
import { Card } from '../ui/Card'
import { Chip } from '../ui/Chip'
import { Collapse } from '../ui/Collapse'
import { startRipple } from '../ui/motion'
import { Slider } from '../ui/Slider'
import { Switch } from '../ui/Switch'
import { ValueFlash } from '../ui/ValueFlash'
import {
  CarIcon,
  ConeIcon,
  FogIcon,
  InfoIcon,
  PauseIcon,
  PlayIcon,
  RefreshIcon,
  TargetIcon,
  TrashIcon,
  TuneIcon,
  WarningIcon,
} from '../ui/Icons'

/** 車両一覧の表示に使う 1 台分のスナップショット */
interface VehicleRow {
  id: number
  active: boolean
  speed: number
  goalDistance: number
  /** 経路の達成度 0.0〜1.0 */
  progress: number
  signalViolations: number
  laneDepartures: number
  collided: boolean
  reachedGoal: boolean
}

/** 一覧の表示に関わる値が変わったかどうか。 */
function sameRows(a: VehicleRow[], b: VehicleRow[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    const x = a[i]
    const y = b[i]
    if (
      x.id !== y.id ||
      x.active !== y.active ||
      x.collided !== y.collided ||
      x.reachedGoal !== y.reachedGoal ||
      x.signalViolations !== y.signalViolations ||
      x.laneDepartures !== y.laneDepartures ||
      x.speed.toFixed(1) !== y.speed.toFixed(1) ||
      Math.round(x.goalDistance) !== Math.round(y.goalDistance) ||
      Math.round(x.progress * 100) !== Math.round(y.progress * 100)
    ) {
      return false
    }
  }
  return true
}

const INTERACTIONS: Array<{ id: InteractionMode; label: string; icon: React.ReactNode }> = [
  { id: 'none', label: '操作なし', icon: <TargetIcon size={16} /> },
  { id: 'obstacle', label: '障害物を置く', icon: <ConeIcon size={16} /> },
  { id: 'vehicle', label: '車両を追加', icon: <CarIcon size={16} /> },
]

export function SimulationTab() {
  const params = useSimStore((s) => s.params)
  const config = useSimStore((s) => s.config)
  const weatherPresets = useSimStore((s) => s.weatherPresets)
  const status = useSimStore((s) => s.status)
  const interaction = useSimStore((s) => s.interaction)
  const obstacleRadius = useSimStore((s) => s.obstacleRadius)
  const patchParamsLocal = useSimStore((s) => s.patchParamsLocal)
  const setInteraction = useSimStore((s) => s.setInteraction)
  const setObstacleRadius = useSimStore((s) => s.setObstacleRadius)
  const setFollowTarget = useSimStore((s) => s.setFollowTarget)
  const followTarget = useSimStore((s) => s.followTarget)
  const panelOpen = useSimStore((s) => s.panelOpen)

  const paused = status.renderPaused
  const suspended = status.simSuspended ?? false

  const [rows, setRows] = useState<VehicleRow[]>([])
  const [obstacleCount, setObstacleCount] = useState(0)
  const [visibility, setVisibility] = useState(0)

  useEffect(() => {
    if (!panelOpen) return
    const timer = window.setInterval(() => {
      const curr = frameBuffer.curr
      const obstacles = frameBuffer.obstacles.length
      setObstacleCount((prev) => (prev === obstacles ? prev : obstacles))
      const reach = Math.round(frameBuffer.weather.visibility)
      setVisibility((prev) => (prev === reach ? prev : reach))
      if (!curr) {
        setRows((prev) => (prev.length === 0 ? prev : []))
        return
      }
      const next = curr.vehicles.map((v) => ({
        id: v.id,
        active: v.active,
        speed: v.speed,
        goalDistance: Math.hypot(v.goal[0] - v.x, v.goal[1] - v.y),
        progress: v.progress,
        signalViolations: v.signalViolations,
        laneDepartures: v.laneDepartures,
        collided: v.collided,
        reachedGoal: v.reachedGoal,
      }))
      setRows((prev) => (sameRows(prev, next) ? prev : next))
    }, 250)
    return () => window.clearInterval(timer)
  }, [panelOpen])

  const tracked = rows.find((r) => r.id === followTarget && r.active) ?? null
  const currentWeather = weatherPresets.find(
    (w) =>
      Math.abs(w.rain - params.weatherRain) < 0.02 &&
      Math.abs(w.fog - params.weatherFog) < 0.02,
  )

  return (
    <>
      <Card title="再生" icon={<PlayIcon size={16} />}>
        <div className="m3-row">
          <Button
            variant={paused ? 'filled' : 'tonal'}
            block
            icon={paused ? <PlayIcon size={18} /> : <PauseIcon size={18} />}
            onClick={() => send({ type: 'set_render_paused', paused: !paused })}
          >
            {paused ? '描画を再開' : '描画を一時停止'}
          </Button>
        </div>
        <div className="m3-banner m3-banner--info">
          <span className="m3-banner-icon">
            <InfoIcon size={16} />
          </span>
          <span>
            <span className="m3-banner-title">一時停止しても学習は止まりません</span>
            止まるのは画面の描画だけです。裏側では強化学習の重み更新が走り続けます
            （オンライン学習を常に継続する設計のため）。
          </span>
        </div>

        <Collapse open={suspended}>
          <div className="m3-banner m3-banner--warning">
            <span className="m3-banner-icon">
              <WarningIcon size={16} />
            </span>
            <span>
              <span className="m3-banner-title">いまは走行も学習も止まっています</span>
              {status.suspendReason ||
                '認識器の学習中はシミュレーションを止めています'}
              。「モデル作成」タブで進み具合を確認できます。完了・中断すると自動で再開します。
            </span>
          </div>
        </Collapse>
      </Card>

      <Card title="パラメータ" icon={<TuneIcon size={16} />}>
        <Slider
          label="車両数"
          hint="台"
          value={params.vehicleCount}
          min={1}
          max={config.maxVehicles}
          step={1}
          format={(v) => `${v} 台`}
          onChange={(v) => patchParamsLocal({ vehicleCount: v })}
          onCommit={(v) => send({ type: 'set_params', params: { vehicleCount: v } })}
        />
        <Slider
          label="歩行者"
          hint="人"
          value={params.pedestrianCount}
          min={0}
          max={config.maxPedestrians ?? 64}
          step={1}
          format={(v) => `${v} 人`}
          onChange={(v) => patchParamsLocal({ pedestrianCount: v })}
          onCommit={(v) => send({ type: 'set_params', params: { pedestrianCount: v } })}
        />
        <Slider
          label="シミュレーション速度"
          value={params.simSpeed}
          min={0.25}
          max={8}
          step={0.25}
          format={(v) => `${v.toFixed(2)} 倍`}
          onChange={(v) => patchParamsLocal({ simSpeed: v })}
          onCommit={(v) => send({ type: 'set_params', params: { simSpeed: v } })}
        />
        <Slider
          label="最高速度"
          value={params.maxSpeed}
          min={5}
          max={30}
          step={0.5}
          format={(v) => `${v.toFixed(1)} m/s（${(v * 3.6).toFixed(0)} km/h）`}
          onChange={(v) => patchParamsLocal({ maxSpeed: v })}
          onCommit={(v) => send({ type: 'set_params', params: { maxSpeed: v } })}
        />
      </Card>

      <Card title="交通ルール" icon={<InfoIcon size={16} />}>
        <Switch
          label="信号に従う"
          description="赤と、安全に止まれる黄で停止線の手前に止まります（道交法施行令 2 条）"
          checked={params.obeySignals}
          onChange={(v) => {
            patchParamsLocal({ obeySignals: v })
            send({ type: 'set_params', params: { obeySignals: v } })
          }}
        />
        <Switch
          label="標識に従う"
          description="最高速度標識の規制速度を超えないよう加速を抑えます（道交法 22 条）"
          checked={params.obeySpeedSigns}
          onChange={(v) => {
            patchParamsLocal({ obeySpeedSigns: v })
            send({ type: 'set_params', params: { obeySpeedSigns: v } })
          }}
        />
        <div className="m3-note">
          切ると信号を無視して走るようになり、赤信号を越えた罰だけが残ります。
          標識も同じで、切ると規制速度を超えた罰だけが残ります。
          守れるようになるかを学習に任せたい場合に使ってください。
          <br />
          車線（左側通行・右左折時の寄せ）は経路そのものが車線に沿っているため、
          この切り替えとは無関係に常に守られます。
        </div>
      </Card>

      <Card title="天候" icon={<FogIcon size={16} />}>
        <div className="m3-row m3-row--wrap">
          {weatherPresets.map((w) => (
            <Chip
              key={w.id}
              selected={!params.weatherAuto && currentWeather?.id === w.id}
              disabled={params.weatherAuto}
              onClick={() => {
                const patch = { weatherRain: w.rain, weatherFog: w.fog }
                patchParamsLocal(patch)
                send({ type: 'set_params', params: patch })
              }}
            >
              {WEATHER_LABELS[w.id] ?? w.id}
            </Chip>
          ))}
        </div>

        <Switch
          label="時間で自動に変える"
          description="シミュレーション内の時刻に沿って、晴れ → 雨 → 晴れ → 霧 とゆっくり巡ります"
          checked={params.weatherAuto}
          onChange={(v) => {
            patchParamsLocal({ weatherAuto: v })
            send({ type: 'set_params', params: { weatherAuto: v } })
          }}
        />

        <div className="m3-row">
          <span className="m3-note m3-grow">
            いまの視程: <strong>{visibility > 0 ? `${visibility} m` : '—'}</strong>
          </span>
        </div>
        <div className="m3-note">
          雨は画を濁らせるだけで視程は縮めません。霧は視程を縮め、
          <strong>遠くのものは正解ラベルからも外れます</strong>
          （見えないものを当てさせると学習が成立しないため）。
          <br />
          報酬と終了条件は真値のままなので、霧で赤信号を見落として進めば
          そのまま信号無視として罰が入ります。
        </div>
      </Card>

      <Card title="介入" icon={<ConeIcon size={16} />}>
        <div className="m3-note">
          3D 画面をクリックして、走行中の環境に手を加えられます。学習は止まりません。
          <br />
          置いたパイロンを<strong>クリックすると、その 1 個だけ</strong>取り消せます。
        </div>
        <div className="m3-row m3-row--wrap">
          {INTERACTIONS.map((it) => (
            <Chip
              key={it.id}
              icon={it.icon}
              selected={interaction === it.id}
              onClick={() => setInteraction(it.id)}
            >
              {it.label}
            </Chip>
          ))}
        </div>

        <Collapse open={interaction === 'obstacle'}>
          <Slider
            label="障害物の半径"
            value={obstacleRadius}
            min={0.2}
            max={2}
            step={0.05}
            format={(v) => `${v.toFixed(2)} m`}
            onChange={setObstacleRadius}
          />
        </Collapse>

        <hr className="m3-divider" />
        <div className="m3-row">
          <span className="m3-note m3-grow">設置済みの障害物: {obstacleCount} 個</span>
          <Button
            variant="outlined"
            size="sm"
            icon={<TrashIcon size={16} />}
            disabled={obstacleCount === 0}
            onClick={() => send({ type: 'clear_obstacles' })}
          >
            すべて消す
          </Button>
        </div>
      </Card>

      <Card title="追跡中の車両" icon={<TargetIcon size={16} />}>
        {tracked ? (
          <>
            <div className="m3-row">
              <span
                className="m3-vehicle-swatch"
                style={{ background: vehicleColor(tracked.id), width: 14, height: 14 }}
              />
              <span style={{ fontSize: 15, fontWeight: 700 }}>車両 #{tracked.id}</span>
              <span className="m3-grow" />
              {tracked.collided && <Chip small tone="error">衝突</Chip>}
              {tracked.reachedGoal && <Chip small tone="ok">到達</Chip>}
            </div>

            <div className="m3-statgrid">
              <div className="m3-stat">
                <span className="m3-stat-label">車速</span>
                <span className="m3-stat-value">{tracked.speed.toFixed(1)} m/s</span>
                <span className="m3-note">{(tracked.speed * 3.6).toFixed(0)} km/h</span>
              </div>
              <div className="m3-stat">
                <span className="m3-stat-label">目的地まで</span>
                <span className="m3-stat-value">{tracked.goalDistance.toFixed(0)} m</span>
              </div>
              <div className="m3-stat">
                <span className="m3-stat-label">信号無視</span>
                <ValueFlash
                  className="m3-stat-value"
                  style={tracked.signalViolations > 0 ? { color: 'var(--m3-error)' } : undefined}
                >
                  {tracked.signalViolations} 回
                </ValueFlash>
              </div>
              <div className="m3-stat">
                <span className="m3-stat-label">車線逸脱</span>
                <ValueFlash
                  className="m3-stat-value"
                  style={tracked.laneDepartures > 0 ? { color: 'var(--m3-warning)' } : undefined}
                >
                  {tracked.laneDepartures} 回
                </ValueFlash>
              </div>
            </div>

            <div className="m3-slider-head">
              <span className="m3-slider-label">達成度</span>
              <span className="m3-slider-value">{(tracked.progress * 100).toFixed(0)} %</span>
            </div>
            <div className="m3-bar">
              <div
                className="m3-bar-fill"
                style={{
                  ['--m3-bar-value' as string]: String(
                    Math.max(0, Math.min(1, tracked.progress)),
                  ),
                  background: vehicleColor(tracked.id),
                }}
              />
            </div>
            <div className="m3-note">
              3D 画面ではこの車両の上に色付きのピンが立ちます。建物の陰に入っても
              見えるようにしてあるので、そのまま位置を追えます。
              達成度は経路上をどこまで進んだかの割合です。
            </div>
          </>
        ) : (
          <div className="m3-note">
            下の一覧から車両をクリックすると、その車両を追跡してピンで示します。
          </div>
        )}
      </Card>

      <Card title="車両" icon={<CarIcon size={16} />}>
        {rows.length === 0 ? (
          <div className="m3-note">走行データがまだありません。</div>
        ) : (
          <div className="m3-col">
            {rows
              .filter((r) => r.active)
              .map((r) => (
                <div
                  key={r.id}
                  className="m3-vehicle m3-ripple"
                  data-collided={r.collided ? 'true' : 'false'}
                  data-followed={followTarget === r.id ? 'true' : 'false'}
                  onPointerDown={startRipple}
                  onClick={() => setFollowTarget(r.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') setFollowTarget(r.id)
                  }}
                  title="クリックすると追跡します（3D 画面にピンが立ち、追従カメラの対象にもなります）"
                >
                  <span
                    className="m3-vehicle-swatch"
                    style={{ background: vehicleColor(r.id) }}
                  />
                  <span className="m3-vehicle-id">#{r.id}</span>
                  <span className="m3-vehicle-metrics">
                    <span>{r.speed.toFixed(1)} m/s</span>
                    <span>目的地まで {r.goalDistance.toFixed(0)} m</span>
                  </span>
                  {r.collided && <span>衝突</span>}
                  {r.reachedGoal && <span>到達</span>}
                </div>
              ))}
          </div>
        )}

        <hr className="m3-divider" />
        <Button
          variant="outlined"
          block
          icon={<RefreshIcon size={16} />}
          onClick={() => send({ type: 'reset_episode' })}
        >
          エピソードをリセット
        </Button>
        <div className="m3-note">
          全車両の出発地と目的地を選び直します。学習した重みはそのまま保持されます。
        </div>
      </Card>
    </>
  )
}
