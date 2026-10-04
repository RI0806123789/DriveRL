/** モックの学習の自動化。**実機と同じ段階（準備 → 試行 → 次の試行、OFF で最良を適用）を踏む**（試行は実機より短い） */

import type {
  AutotuneBest,
  AutotuneMessage,
  AutotunePhase,
  AutotuneTrial,
  SimParams,
  StatusPayload,
} from '../../types/protocol.ts'
import { makeRng } from './grid.ts'

interface MockDim {
  key: keyof SimParams
  low: number
  high: number
  log?: boolean
  /** 刻み（スライダーと同じ）。対数で探すものには無い */
  step?: number
}

/** 探索する値・範囲・刻み（`backend/app/runtime/autotune.py` の `SEARCH_SPACE` と同じ。test_tuning.py が突き合わせる） */
export const MOCK_SEARCH_SPACE: MockDim[] = [
  { key: 'learningRate', low: 1e-5, high: 1e-3, log: true },
  { key: 'gamma', low: 0.95, high: 0.999, step: 0.001 },
  { key: 'clipRange', low: 0.05, high: 0.4, step: 0.01 },
  { key: 'entropyCoef', low: 0, high: 0.01, step: 0.001 },
  { key: 'rewardGoal', low: 20, high: 300, step: 5 },
  { key: 'rewardCollision', low: -300, high: -20, step: 5 },
  { key: 'rewardProgress', low: 0.2, high: 3, step: 0.1 },
  { key: 'rewardOffroad', low: -5, high: -0.1, step: 0.1 },
  { key: 'rewardSignal', low: -200, high: -10, step: 5 },
  { key: 'rewardOverspeed', low: -50, high: -1, step: 1 },
  { key: 'rewardTime', low: -0.5, high: 0, step: 0.01 },
]

/** 刻みに乗せて、刻みの桁で丸める（0.30000000000000004 を出さない） */
function snap(value: number, d: MockDim): number {
  if (d.step === undefined) return value
  const k = Math.round((value - d.low) / d.step)
  const digits = Math.max(0, -Math.floor(Math.log10(d.step) + 1e-9))
  return Number((d.low + k * d.step).toFixed(digits))
}

export const MOCK_TUNED_KEYS = MOCK_SEARCH_SPACE.map((d) => d.key)

/** 1 試行のシミュレーション時間 [秒]（実機は 76.8 秒。見た目を確かめやすいよう短くしてある） */
const MOCK_TRIAL_SEC = 8
const MOCK_TRIAL_STEPS = 1536
const HISTORY_LEN = 20

export interface MockAutotuneHooks {
  sendAutotune(msg: AutotuneMessage): void
  sendStatus(patch: Partial<StatusPayload>): void
  /** params を書き換えて配る（スライダーが動く） */
  applyParams(patch: Partial<SimParams>): void
}

export class MockAutotune {
  /** 共有の乱数（`makeRng(20260905)`）とは別に回す。共有のほうを引くと、既存の操作列の送信が変わる */
  private readonly rng = makeRng(82)
  private readonly hooks: MockAutotuneHooks
  running = false
  private phase: AutotunePhase = 'idle'
  private trial: number | null = null
  private nextTrial = 0
  private progress = 0
  private finished = 0
  private current: Partial<SimParams> | null = null
  private currentScore = 0
  private best: AutotuneBest | null = null
  private history: AutotuneTrial[] = []
  private baseline: Partial<SimParams> = {}
  private message = '（モック）学習の自動化はまだ実行していません'

  constructor(hooks: MockAutotuneHooks) {
    this.hooks = hooks
  }

  /** その params のうち、探索中に外から変えてはいけないキーを除く */
  strip(patch: Partial<SimParams>): { patch: Partial<SimParams>; blocked: boolean } {
    if (!this.running) return { patch, blocked: false }
    const out: Partial<SimParams> = { ...patch }
    let blocked = false
    for (const key of MOCK_TUNED_KEYS) {
      if (key in out) {
        delete out[key]
        blocked = true
      }
    }
    return { patch: out, blocked }
  }

  start(params: SimParams, mapLoaded: boolean, practical: boolean): void {
    if (this.running) {
      this.hooks.sendStatus({ message: '（モック）すでに学習の自動化を実行中です' })
      return
    }
    if (!mapLoaded) {
      this.hooks.sendStatus({ message: '（モック）マップを読み込んでから学習の自動化を始めてください' })
      return
    }
    if (practical) {
      this.hooks.sendStatus({ message: '（モック）実用モードの間は学習の自動化を始められません' })
      return
    }
    this.baseline = Object.fromEntries(MOCK_TUNED_KEYS.map((k) => [k, params[k]]))
    this.running = true
    this.finished = 0
    this.best = null
    this.history = []
    this.beginTrial()
    this.hooks.sendStatus({ message: '（モック）学習の自動化を始めました' })
  }

  stop(): void {
    if (!this.running) {
      this.hooks.sendStatus({ message: '（モック）学習の自動化は実行していません' })
      return
    }
    const best = this.best
    this.hooks.applyParams(best ? best.params : this.baseline)
    this.message = best
      ? `（モック）試行 #${best.trial}（スコア ${best.score.toFixed(3)}）のパラメータと重みを適用して保存しました`
      : '（モック）完了した試行が無かったため、自動化の前の値に戻しました'
    // 実機と同じく、止めたら探索 1 回ぶんの記録は空に戻す（結果は message に残る）
    this.running = false
    this.phase = 'idle'
    this.trial = null
    this.current = null
    this.progress = 0
    this.finished = 0
    this.best = null
    this.history = []
    this.send()
    this.hooks.sendStatus({ message: this.message })
  }

  /** シミュレーション時間を dt [秒] 進める */
  step(dt: number): void {
    if (!this.running || this.phase !== 'running') return
    this.progress += dt / MOCK_TRIAL_SEC
    if (this.progress < 1) return
    this.finishTrial()
    this.beginTrial()
  }

  private beginTrial(): void {
    const patch: Partial<SimParams> = {}
    for (const d of MOCK_SEARCH_SPACE) {
      const u = this.rng()
      const value = d.log
        ? Math.exp(Math.log(d.low) + u * (Math.log(d.high) - Math.log(d.low)))
        : snap(d.low + u * (d.high - d.low), d)
      ;(patch as Record<string, number>)[d.key] = value
    }
    // それらしい山（学習率 3e-4・割引率 0.99 の近く）に揺らぎを足したもの
    const lr = Math.log10(patch.learningRate ?? 3e-4) + 3.5
    const gamma = ((patch.gamma ?? 0.99) - 0.99) * 30
    this.currentScore = 0.35 - 0.25 * lr * lr - gamma * gamma + (this.rng() - 0.5) * 0.1
    this.trial = this.nextTrial
    this.nextTrial += 1
    this.current = patch
    this.progress = 0
    this.phase = 'running'
    this.message = `（モック）試行 #${this.trial} を走らせています`
    this.hooks.applyParams(patch)
    this.send()
  }

  private finishTrial(): void {
    if (this.trial === null || this.current === null) return
    const score = this.currentScore
    this.history = [...this.history, { trial: this.trial, score, outcome: 'complete' as const }].slice(-HISTORY_LEN)
    this.finished += 1
    if (this.best === null || score > this.best.score) {
      this.best = { trial: this.trial, score, params: { ...this.current } }
    }
  }

  send(): void {
    this.hooks.sendAutotune({
      type: 'autotune',
      available: true,
      running: this.running,
      phase: this.phase,
      studyName: this.running ? 'live-mock' : null,
      trial: this.running ? this.trial : null,
      trialProgress: this.running ? Math.min(1, this.progress) : 0,
      trialSteps: MOCK_TRIAL_STEPS,
      finishedTrials: this.finished,
      priorTrials: 0,
      tunedKeys: MOCK_TUNED_KEYS,
      current: this.running ? this.current : null,
      best: this.best,
      history: this.history,
      message: this.message,
    })
  }
}
